from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

class DiffusionResNetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: nn.Module,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            **kwargs):
        super().__init__()
        
        # ResNet 架构通常强制要求使用 global_cond，因为 MLP 很难处理 local spatial conditioning
        assert obs_as_global_cond, "ResNet policy currently requires obs_as_global_cond=True"
        
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0, # 因为是 global cond，mask 不需要管 obs
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            t_tensor = t
            if not isinstance(t_tensor, torch.Tensor):
                t_tensor = torch.tensor(t, dtype=torch.long)
            t_tensor = t_tensor.to(trajectory.device)
            # 1. apply conditioning (impainting)
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. step
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        trajectory[condition_mask] = condition_data[condition_mask]        
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'obs' in obs_dict
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # ResNet 强制走 global_cond 路径
        device = nobs.device # 确保 global_cond 和 cond_data 在同一设备
        global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
        shape = (B, T, Da)
        if self.pred_action_steps_only:
            shape = (B, self.n_action_steps, Da)
        
        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize
        action_pred = self.normalizer['action'].unnormalize(nsample)

        # get action
        start = 0 # 因为 pred_action_steps_only
        if not self.pred_action_steps_only:
            start = To
            if self.oa_step_convention: start = To - 1
        
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        return {
            'action': action,
            'action_pred': action_pred
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # global cond 准备
        global_cond = obs[:,:self.n_obs_steps,:].reshape(obs.shape[0], -1)
        
        if self.pred_action_steps_only:
            To = self.n_obs_steps
            start = To
            if self.oa_step_convention: start = To - 1
            end = start + self.n_action_steps
            trajectory = action[:,start:end]
        else:
            trajectory = action

        # 采样噪声
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (trajectory.shape[0],), device=trajectory.device
        ).long()
        
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        
        # 预测
        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)

        target = noise if self.noise_scheduler.config.prediction_type == 'epsilon' else trajectory
        loss = F.mse_loss(pred, target)
        return loss