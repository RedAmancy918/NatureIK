"""
IK Diffusion Transformer（IKDiffuserModel）
==========================================
架构：带 cross-attention 的小型 Transformer 扩散去噪网络。

与 ResNet 方案的对比
------------------
- ResNet：观测展平后直接作为 global_cond 输入每个残差块（FiLM 调制）。
- IKDiffuserModel：末端位姿（7 维）通过 cross-attention 作为 key/value，
  关节噪声 token 作为 query，让模型以更灵活的注意力机制对齐位姿条件。

数据流
------
sample (B, 1, 6)  → joint_emb → (B, 1, n_embd)
cond   (B, 1, 7)  → ee_emb    → (B, 1, n_embd)
timestep (B,)     → time_emb  → (B, 1, n_embd)  加到 joint token 上

N × IKTransformerBlock(query=joint token, key/value=ee cond token)
  → LayerNorm → MultiheadAttention → 残差
  → LayerNorm → MLP(GELU) → 残差

ln_f → head (Linear) → 噪声预测 (B, 1, 6)
"""
import math
import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb


class IKTransformerBlock(nn.Module):
    """
    单个 Transformer 块：cross-attention（关节 query，位姿 cond 作为 key/value）+ MLP。

    参数
    ----
    n_embd: 嵌入维度
    n_head: 注意力头数（必须整除 n_embd）
    dropout: dropout 概率（默认 0.1）
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=n_embd, num_heads=n_head, batch_first=True, dropout=dropout
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, T_q, n_embd)  关节 token（query）
        cond: (B, T_k, n_embd)  末端位姿 token（key/value）
        """
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.cross_attn(query=x, key=cond, value=cond)
        x = residual + attn_out

        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        return x


class IKDiffuserModel(nn.Module):
    """
    IK 扩散去噪主干网络。

    参数
    ----
    joint_dim:  关节维度（默认 6，对应 airbot 单臂）
    ee_dim:     末端位姿维度（默认 7，xyz + quat）
    n_embd:     嵌入/隐藏维度
    n_layer:    Transformer Block 数量
    n_head:     注意力头数
    dropout:    dropout 概率
    """

    def __init__(
        self,
        joint_dim: int = 6,
        ee_dim: int = 7,
        n_embd: int = 256,
        n_layer: int = 4,
        n_head: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 关节噪声嵌入
        self.joint_emb = nn.Linear(joint_dim, n_embd)
        # 末端位姿条件嵌入
        self.ee_emb = nn.Linear(ee_dim, n_embd)

        # 时间步嵌入：Sinusoidal → 2 × Linear → GELU
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(n_embd),
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd),
        )

        # N 层 cross-attention Block
        self.blocks = nn.ModuleList(
            [IKTransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        # 输出头：预测 6 维噪声（epsilon prediction）
        self.head = nn.Linear(n_embd, joint_dim)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数
        ----
        sample:   (B, 1, joint_dim) 或 (B, joint_dim)  带噪关节
        timestep: (B,)                                  扩散时间步
        cond:     (B, 1, ee_dim)  或 (B, ee_dim)        末端位姿条件

        返回
        ----
        (B, 1, joint_dim)  预测的噪声
        """
        # 统一维度为 (B, 1, D)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
        if sample.dim() == 2:
            sample = sample.unsqueeze(1)

        # 关节 token
        x = self.joint_emb(sample)   # (B, 1, n_embd)
        # 位姿条件 token
        c = self.ee_emb(cond)        # (B, 1, n_embd)

        # 时间步 token 加到关节 token 上（广播）
        t_emb = self.time_emb(timestep).unsqueeze(1)  # (B, 1, n_embd)
        x = x + t_emb

        for block in self.blocks:
            x = block(x, c)

        x = self.ln_f(x)
        return self.head(x)  # (B, 1, joint_dim)
