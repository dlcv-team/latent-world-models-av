"""Spatial-Temporal DiT for patch-token future prediction.

Processes H*S tokens where H=horizon steps, S=spatial patches per frame.
Uses 2D spatial + 1D temporal learned positional embeddings.
Self-attention sees all H*S tokens (captures spatial AND temporal patterns).

Action-sequence conditioning: per-horizon-step action broadcast to all S
tokens in that step via adaLN-Zero.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimestepEmbedding(nn.Module):
    def __init__(self, cond_dim: int = 384):
        super().__init__()
        self.cond_dim = cond_dim
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half_dim = self.cond_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=timestep.device, dtype=torch.float32)
            / half_dim
        )
        args = timestep.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


def _modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


class DiTBlock(nn.Module):
    def __init__(self, dim=384, cond_dim=384, n_heads=6, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, dim))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.adaln_linear = nn.Linear(cond_dim, 6 * dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D) where T = H*S (all spatio-temporal tokens)
        cond: (B, T, D) per-token conditioning (action-seq broadcast to spatial)
        """
        mod = self.adaln_linear(cond)  # (B, T, 6*D)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mod.chunk(6, dim=-1)
        h = _modulate(self.norm_attn(x), shift_a, scale_a)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_a * self.drop(attn_out)
        h = _modulate(self.norm_mlp(x), shift_m, scale_m)
        x = x + gate_m * self.drop(self.mlp(h))
        return x


class FourierActionEmbedding(nn.Module):
    def __init__(self, action_dim=2, n_frequencies=64, base=2.0, out_dim=384):
        super().__init__()
        freqs = base ** torch.arange(n_frequencies, dtype=torch.float32) * torch.pi
        self.register_buffer("freqs", freqs)
        fourier_dim = action_dim * 2 * n_frequencies
        self.proj = nn.Sequential(
            nn.Linear(fourier_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """action: (B, H, 2) -> (B, H, out_dim)"""
        if action.dim() == 2:
            action = action.unsqueeze(1)
        x = action.unsqueeze(-1) * self.freqs
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = x.flatten(-2)
        return self.proj(x)


class SpatialTemporalDiT(nn.Module):
    """DiT operating on spatial patch tokens across prediction horizons.

    Parameters
    ----------
    z_dim : int
        Per-token embedding dimension (384).
    cond_dim : int
        Conditioning dimension (384).
    n_blocks : int
        Number of DiT transformer blocks.
    n_heads : int
        Number of attention heads.
    horizon : int
        Number of future prediction steps.
    n_spatial : int
        Number of spatial tokens per frame (49 for 7x7, 64 for 8x8).
    mlp_ratio : float
        MLP expansion ratio in transformer blocks.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        z_dim: int = 384,
        cond_dim: int = 384,
        n_blocks: int = 4,
        n_heads: int = 6,
        horizon: int = 16,
        n_spatial: int = 49,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.horizon = horizon
        self.n_spatial = n_spatial
        self.n_tokens = horizon * n_spatial  # Total sequence length

        # Input projection
        self.input_proj = nn.Linear(z_dim, z_dim)

        # Positional embeddings: separate spatial and temporal
        # Spatial: learned per-patch position (shared across horizon steps)
        self.spatial_pos = nn.Parameter(torch.zeros(1, n_spatial, z_dim))
        # Temporal: learned per-step position (shared across spatial positions)
        self.temporal_pos = nn.Parameter(torch.zeros(1, horizon, z_dim))

        # Conditioning
        self.timestep_embed = TimestepEmbedding(cond_dim)
        self.z_t_proj = nn.Linear(z_dim, cond_dim)  # Projects mean-pooled z_t

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(z_dim, cond_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_blocks)
        ])

        # Final output
        self.final_norm = nn.LayerNorm(z_dim, elementwise_affine=False)
        self.final_adaln = nn.Linear(cond_dim, 3 * z_dim)
        self.final_linear = nn.Linear(z_dim, z_dim)

        # Initialize
        nn.init.normal_(self.spatial_pos, std=0.02)
        nn.init.normal_(self.temporal_pos, std=0.02)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def forward(
        self,
        x_noisy: torch.Tensor,       # (B, H*S, D) noisy future spatial tokens
        z_t_spatial: torch.Tensor,    # (B, S, D) current spatial tokens
        a_embed: torch.Tensor,        # (B, H, D) per-step action embeddings
        timestep: torch.Tensor,       # (B,) diffusion timestep
    ) -> torch.Tensor:
        B, HS, D = x_noisy.shape
        H, S = self.horizon, self.n_spatial
        assert HS == H * S, f"Expected {H*S} tokens, got {HS}"

        # Build positional embeddings: spatial + temporal, broadcast to H*S
        # spatial_pos: (1, S, D) -> (1, H, S, D) -> (1, H*S, D)
        # temporal_pos: (1, H, D) -> (1, H, S, D) -> (1, H*S, D)
        sp = self.spatial_pos.unsqueeze(1).expand(-1, H, -1, -1).reshape(1, H * S, D)
        tp = self.temporal_pos.unsqueeze(2).expand(-1, -1, S, -1).reshape(1, H * S, D)
        pos = sp + tp

        # Project noisy input and add positional embeddings
        x = self.input_proj(x_noisy) + pos

        # Build per-token conditioning:
        # Global: timestep + mean-pooled z_t
        z_t_pooled = z_t_spatial.mean(dim=1)  # (B, D)
        cond_global = self.timestep_embed(timestep) + self.z_t_proj(z_t_pooled)  # (B, D)

        # Per-step action: (B, H, D) -> broadcast to (B, H, S, D) -> (B, H*S, D)
        a_broadcast = a_embed.unsqueeze(2).expand(-1, -1, S, -1).reshape(B, H * S, D)

        # Combined conditioning: global + per-token action
        cond = cond_global.unsqueeze(1).expand(-1, H * S, -1) + a_broadcast  # (B, H*S, D)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, cond)

        # Final output
        mod = self.final_adaln(cond)
        shift, scale, gate = mod.chunk(3, dim=-1)
        x = gate * self.final_linear(_modulate(self.final_norm(x), shift, scale))
        return x  # (B, H*S, D)
