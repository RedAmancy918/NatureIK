from __future__ import annotations

import pathlib
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

try:
    from flow_matching.path.affine import CondOTProbPath
    from flow_matching.solver.ode_solver import ODESolver
    from flow_matching.utils.model_wrapper import ModelWrapper
except ModuleNotFoundError:
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    candidate_roots = [
        repo_root / "flow_matching",
        repo_root.parent / "flow_matching",
    ]
    for flow_matching_root in candidate_roots:
        affine_path = flow_matching_root / "flow_matching" / "path" / "affine.py"
        if affine_path.is_file():
            sys.path.insert(0, str(flow_matching_root))
            break
    from flow_matching.path.affine import CondOTProbPath
    from flow_matching.solver.ode_solver import ODESolver
    from flow_matching.utils.model_wrapper import ModelWrapper


class _ResNetVelocityWrapper(ModelWrapper):
    def __init__(self, model: nn.Module, time_scale: float):
        super().__init__(model)
        self.time_scale = time_scale

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        else:
            t = t.to(device=x.device, dtype=x.dtype)

        if t.ndim == 0:
            t = t.expand(x.shape[0])
        elif t.ndim == 1 and t.shape[0] == 1:
            t = t.expand(x.shape[0])

        return self.model(
            x,
            timestep=t * self.time_scale,
            local_cond=extras.get("local_cond"),
            global_cond=extras.get("global_cond"),
        )


class FlowResNetLowdimPolicy(BaseLowdimPolicy):
    """
    Flow Matching version of the lowdim ResNet IK policy.

    This keeps the same dataset / global conditioning interface as the diffusion
    variant, but replaces DDPM noise prediction with conditional flow matching on
    a straight CondOT path.
    """

    def __init__(
        self,
        model: nn.Module,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int = 4,
        sampling_method: str = "midpoint",
        obs_as_local_cond: bool = False,
        obs_as_global_cond: bool = False,
        pred_action_steps_only: bool = False,
        oa_step_convention: bool = False,
        time_scale: float = 100.0,
        **kwargs,
    ):
        super().__init__()

        assert obs_as_global_cond, "Flow ResNet policy currently requires obs_as_global_cond=True"
        if sampling_method not in {"euler", "midpoint"}:
            raise ValueError(f"Unsupported sampling_method={sampling_method!r}")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        self.model = model
        self.path = CondOTProbPath()
        self.velocity_model = _ResNetVelocityWrapper(
            model=self.model,
            time_scale=time_scale,
        )
        self.solver = ODESolver(velocity_model=self.velocity_model)
        self.normalizer = LinearNormalizer()

        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.sampling_method = sampling_method
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.time_scale = time_scale
        self.kwargs = kwargs

    def _prepare_global_cond(self, nobs: torch.Tensor) -> torch.Tensor:
        return nobs[:, : self.n_obs_steps].reshape(nobs.shape[0], -1)

    def _predict_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | float,
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.velocity_model(
            x=x_t,
            t=t,
            global_cond=global_cond,
        )

    def conditional_sample(
        self,
        shape,
        global_cond: Optional[torch.Tensor] = None,
        generator=None,
    ) -> torch.Tensor:
        trajectory = torch.randn(
            size=shape,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        return self.solver.sample(
            x_init=trajectory,
            step_size=1.0 / self.num_inference_steps,
            method=self.sampling_method,
            time_grid=torch.tensor(
                [0.0, 1.0],
                device=trajectory.device,
                dtype=trajectory.dtype,
            ),
            return_intermediates=False,
            enable_grad=False,
            global_cond=global_cond,
        )

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "obs" in obs_dict
        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        B, _, _ = nobs.shape
        To = self.n_obs_steps
        T = self.horizon
        Da = self.action_dim

        global_cond = self._prepare_global_cond(nobs)
        shape = (B, T, Da)
        if self.pred_action_steps_only:
            shape = (B, self.n_action_steps, Da)

        nsample = self.conditional_sample(
            shape=shape,
            global_cond=global_cond,
        )

        action_pred = self.normalizer["action"].unnormalize(nsample)

        start = 0
        if not self.pred_action_steps_only:
            start = To
            if self.oa_step_convention:
                start = To - 1

        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            "action": action,
            "action_pred": action_pred,
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch["obs"]
        action = nbatch["action"]

        global_cond = self._prepare_global_cond(obs)

        if self.pred_action_steps_only:
            start = self.n_obs_steps
            if self.oa_step_convention:
                start = start - 1
            end = start + self.n_action_steps
            trajectory = action[:, start:end]
        else:
            trajectory = action

        x_1 = trajectory
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=x_1.device, dtype=x_1.dtype)
        path_sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)

        pred_velocity = self._predict_velocity(
            x_t=path_sample.x_t,
            t=path_sample.t,
            global_cond=global_cond,
        )
        loss = F.mse_loss(pred_velocity, path_sample.dx_t)
        return loss
