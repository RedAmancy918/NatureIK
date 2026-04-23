import torch
from diffusion_policy.model.differentiable_ik_model import TTGLoss


def guided_inference(
    policy,
    obs_dict,
    ik_helper,
    target_pose_dict,
    action_stats,
    guidance_scale: float = 10.0,
    dq_prev: torch.Tensor | None = None,   # [B, 6] 上一步执行的 delta_q，来自 NatureIKSolver 历史 buffer
    ttg_loss: TTGLoss | None = None,        # 共享 TTGLoss 实例，None 时使用默认权重
    guidance_start_frac: float = 0.5,       # 只在后 50% 的去噪步加 guidance（前期噪声大，加了没用）
    device: str = "cuda:0",
):
    """Diffusion 去噪循环 + TTG（Test-Time Guidance）。

    在每一个去噪步：
      A. 用当前 latent 预测 x0 → FK → 计算 TTG loss（L_pose + L_hist + L_smooth）
      B. 用 TTG 梯度修正 latent
      C. 正常执行 scheduler.step

    Args:
        policy:              IKDiffuserPolicy 或 DiffusionResNetLowdimPolicy
        obs_dict:            {'obs': Tensor [B, n_obs_steps, obs_dim]}
        ik_helper:           DifferentiableIKHelper 实例（含 FK + 归一化器）
        target_pose_dict:    {'pos': [B,3], 'rot': [B,3,3] (可选)}
        action_stats:        动作反归一化统计量
        guidance_scale:      TTG 梯度缩放系数（越大几何纠偏越强，建议 5~20）
        dq_prev:             历史 delta_q，用于 L_hist；None 时跳过历史一致性项
        ttg_loss:            TTGLoss 实例；None 时使用默认权重
        guidance_start_frac: 从第几分之几的步数开始加 guidance（0=全程，0.5=后半段）
        device:              推理设备
    """
    if ttg_loss is None:
        ttg_loss = TTGLoss()

    # ── 1. 归一化观测，提取 conditioning ─────────────────────────────────────
    nobs = policy.normalizer.normalize(obs_dict)

    if hasattr(policy.model, 'cond_encoder'):
        global_cond = policy.model.cond_encoder(nobs)
    else:
        global_cond = nobs['obs']

    B = global_cond.shape[0]

    # ── 2. 初始化随机噪声 latent ──────────────────────────────────────────────
    latents = torch.randn(
        (B, policy.horizon, policy.action_dim),
        device=device,
    )

    # ── 3. 配置 scheduler ────────────────────────────────────────────────────
    policy.noise_scheduler.set_timesteps(policy.num_inference_steps)
    total_steps  = len(policy.noise_scheduler.timesteps)
    guide_from   = int(total_steps * guidance_start_frac)  # 从第 guide_from 步开始加

    # ── 4. 去噪循环 ──────────────────────────────────────────────────────────
    for step_idx, t in enumerate(policy.noise_scheduler.timesteps):

        # ── A. TTG 梯度修正（后半段才加，前期噪声过大效果差）────────────────
        if guidance_scale > 0 and step_idx >= guide_from:
            ik_grad = ik_helper.compute_guidance_gradient(
                latents          = latents,
                t                = t,
                model            = policy.model,
                noise_scheduler  = policy.noise_scheduler,
                target_pose_dict = target_pose_dict,
                action_stats     = action_stats,
                global_cond      = global_cond,
                dq_prev          = dq_prev,
                ttg_loss         = ttg_loss,
            )
            latents = latents - guidance_scale * ik_grad

        # ── B. 正常去噪步 ────────────────────────────────────────────────────
        with torch.no_grad():
            noise_pred = policy.model(latents, t, global_cond=global_cond)
            if isinstance(noise_pred, dict):
                noise_pred = noise_pred['sample']
            latents = policy.noise_scheduler.step(noise_pred, t, latents).prev_sample

    # ── 5. 反归一化，返回物理单位的动作 ──────────────────────────────────────
    action = policy.normalizer['action'].unnormalize(latents)
    return action