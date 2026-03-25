import torch

def guided_inference(
    policy, 
    obs_dict, 
    ik_helper, 
    target_pose_dict, 
    action_stats, 
    guidance_scale=10.0, 
    device="cuda:0"
):
    """
    手动执行 Diffusion 去噪循环，并加入 IK 引导
    """
    # 1. 预处理
    nobs = policy.normalizer.normalize(obs_dict)
    
    # 准备 Condition
    # 适配不同的模型结构 (Transformer vs UNet)
    if hasattr(policy.model, 'cond_encoder'):
        global_cond = policy.model.cond_encoder(nobs)
    else:
        # 标准 Lowdim UNet
        global_cond = nobs['obs']
        
    B = global_cond.shape[0]
    
    # 初始化随机噪声 (Latents)
    latents = torch.randn(
        (B, policy.horizon, policy.action_dim), 
        device=device
    )
    
    # 设置 Scheduler
    policy.noise_scheduler.set_timesteps(policy.num_inference_steps)

    # 2. 去噪循环
    for t in policy.noise_scheduler.timesteps:
        
        # --- A. 计算 IK 梯度 (Guidance) ---
        if guidance_scale > 0:
            # 注意：这里需要确保 ik_helper.compute_guidance_gradient 支持 global_cond 参数
            ik_grad = ik_helper.compute_guidance_gradient(
                latents=latents,
                t=t,
                model=policy.model,
                noise_scheduler=policy.noise_scheduler,
                target_pose_dict=target_pose_dict,
                action_stats=action_stats,
                global_cond=global_cond 
            )
            latents = latents - guidance_scale * ik_grad

        # --- B. 正常预测 ---
        with torch.no_grad():
            noise_pred = policy.model(
                latents, 
                t, 
                global_cond=global_cond
            )
            
            # 兼容字典输出
            if isinstance(noise_pred, dict):
                noise_pred = noise_pred['sample']

            latents = policy.noise_scheduler.step(
                noise_pred, t, latents
            ).prev_sample

    # 3. 后处理
    action = policy.normalizer['action'].unnormalize(latents)
    return action