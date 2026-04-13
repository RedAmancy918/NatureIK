"""Pure MLP baseline for IK regression.

This module intentionally does not depend on MIP flow matching. It only reuses
the same normalized observation / action tensors produced by IKParquetDataset.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class PureMLPIK(nn.Module):
    """Simple supervised MLP for IK.

    Input:
        obs: (B, obs_steps, obs_dim)
    Output:
        action: (B, horizon, act_dim)
    """

    def __init__(
        self,
        obs_dim: int,
        obs_steps: int,
        act_dim: int,
        horizon: int,
        hidden_dims: Sequence[int] = (1024, 1024, 512),
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.obs_steps = int(obs_steps)
        self.act_dim = int(act_dim)
        self.horizon = int(horizon)
        self.input_dim = self.obs_dim * self.obs_steps
        self.output_dim = self.act_dim * self.horizon

        dims = [self.input_dim, *[int(d) for d in hidden_dims]]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
            layers.append(nn.Linear(in_dim, out_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(_make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        last_dim = dims[-1]
        self.head = nn.Linear(last_dim, self.output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 3:
            raise ValueError(
                f"Expected obs with shape (B, obs_steps, obs_dim), got {tuple(obs.shape)}"
            )
        x = obs.flatten(start_dim=1)
        x = self.backbone(x)
        x = self.head(x)
        return x.view(obs.shape[0], self.horizon, self.act_dim)
