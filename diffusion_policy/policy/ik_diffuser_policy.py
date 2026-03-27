"""
IKDiffuserPolicy
================
将 IKDiffuserModel（cross-attention Transformer 扩散主干）包装成训练/推理 Policy。

与 DeltaJoint（ResNet/UNet）方案的区别
-------------------------------------
- 输入格式：obs = ee_pose（7 维末端位姿），action = joint_angle（6 维绝对关节）
  而非 DeltaJoint 的 delta_q。
- 末端位姿通过 cross-attention 条件化，而非展平后作为 global_cond。
- horizon 固定为 1（单步 IK），不做时序预测。

归一化策略（混合归一化）
------------------------
- 位置（xyz, [0:3]）：标准 LinearNorm。
- 四元数（[3:7]）：scale=1, offset=0（直通），保持方向空间不失真。
  该设置在 Workspace.run() 里强制写入，这里 normalizer 的默认行为不会破坏它。
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.ik_transformer import IKDiffuserModel


class IKDiffuserPolicy(nn.Module):
    """
    IK 扩散 Policy。

    参数
    ----
    noise_scheduler: DDPMScheduler（DDPM，epsilon prediction）
    n_embd:  嵌入维度（默认 256）
    n_layer: Transformer Block 层数（默认 4）
    n_head:  注意力头数（默认 8）
    num_inference_steps: 推理时使用的扩散步数，None 则等于训练时的步数
    """

    def __init__(
        self,
        noise_scheduler: DDPMScheduler,
        n_embd: int = 256,
        n_layer: int = 4,
        n_head: int = 8,
        num_inference_steps: int = None,
        **kwargs,
    ):
        super().__init__()

        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = (
            num_inference_steps
            or noise_scheduler.config.num_train_timesteps
        )

        self.model = IKDiffuserModel(
            joint_dim=6,
            ee_dim=7,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )

        # Normalizer 在 Workspace.run() 里用 dataset.get_normalizer() 填充；
        # 此处仅作占位，Policy 自身不负责拟合统计量。
        self.normalizer = LinearNormalizer()

    # ------------------------------------------------------------------
    # 训练接口
    # ------------------------------------------------------------------
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        batch 格式（来自 ParquetDeltaJointDataset 或 IKParquetDataset）：
          batch['obs']['state_quat'] : (B, 1, 7)  末端位姿（条件）
          batch['action']            : (B, 1, 6)  关节角（目标）

        返回 epsilon-prediction MSE loss。
        """
        cond_data = batch['obs']['state_quat']   # (B, 1, 7) 或 (B, 7)
        target_data = batch['action']            # (B, 1, 6) 或 (B, 6)

        # 归一化
        ncond = self.normalizer['obs/state_quat'].normalize(cond_data)
        ntarget = self.normalizer['action'].normalize(target_data)

        B = ntarget.shape[0]
        device = ntarget.device

        noise = torch.randn_like(ntarget)        # (B, 1, 6)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        noisy_target = self.noise_scheduler.add_noise(ntarget, noise, timesteps)

        # 模型输出 (B, 1, 6)
        pred = self.model(noisy_target, timesteps, cond=ncond)

        loss = F.mse_loss(pred, noise)
        return loss

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict 格式：
          obs_dict['obs']['state_quat'] : (B, 1, 7) 或 (B, 7)

        或（兼容 eval_ik.py 的调用方式）：
          obs_dict['state_quat']        : (B, 1, 7) 或 (B, 7)

        返回：
          {'action': (B, 6)}  反归一化后的关节角
        """
        # 兼容两种 key 结构
        if 'obs' in obs_dict:
            cond_data = obs_dict['obs']['state_quat']
        else:
            cond_data = obs_dict['state_quat']

        # 统一为 (B, 1, 7)
        if cond_data.dim() == 2:
            cond_data = cond_data.unsqueeze(1)

        ncond = self.normalizer['obs/state_quat'].normalize(cond_data)
        B = ncond.shape[0]
        device = ncond.device

        # 从标准高斯噪声开始去噪
        nq = torch.randn((B, 1, 6), device=device)

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        with torch.no_grad():
            for t in self.noise_scheduler.timesteps:
                batched_t = torch.full((B,), t.item(), device=device, dtype=torch.long)
                model_output = self.model(nq, batched_t, cond=ncond)
                nq = self.noise_scheduler.step(model_output, t, nq).prev_sample

        # 反归一化，squeeze 为 (B, 6)
        action = self.normalizer['action'].unnormalize(nq)
        return {'action': action.squeeze(1)}

    # ------------------------------------------------------------------
    # Normalizer 加载（Workspace 调用）
    # ------------------------------------------------------------------
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
