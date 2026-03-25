import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )
    def forward(self, x):
        return self.block(x)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size=3),
            Conv1dBlock(out_channels, out_channels, kernel_size=3),
        ])
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2) 
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        cond_emb = self.cond_encoder(cond).unsqueeze(-1)
        scale, shift = cond_emb.chunk(2, dim=1)
        out = out * (scale + 1) + shift
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

class ConditionalResNet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, hidden_dim=256, n_blocks=6):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(128),
            nn.Linear(128, 256),
            nn.Mish(),
            nn.Linear(256, 256),
        )
        combined_cond_dim = 256 + global_cond_dim
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            ConditionalResidualBlock1D(hidden_dim, hidden_dim, combined_cond_dim)
            for _ in range(n_blocks)
        ])
        self.output_proj = nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1)

    def forward(self, sample, timestep, local_cond=None, global_cond=None):
        if not isinstance(timestep, torch.Tensor):
            timesteps = torch.tensor(timestep, dtype=torch.long, device=sample.device)
        else:
            timesteps = timestep.to(sample.device)
            
        B = sample.shape[0]
        if len(timesteps.shape) == 0:
            timesteps = timesteps.expand(B)
        elif len(timesteps.shape) == 1 and timesteps.shape[0] == 1:
            timesteps = timesteps.expand(B)
            
        t_emb = self.time_mlp(timesteps)
        cond = torch.cat([t_emb, global_cond], dim=-1) if global_cond is not None else t_emb
            
        x = sample.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, cond)
        x = self.output_proj(x)
        return x.transpose(1, 2)