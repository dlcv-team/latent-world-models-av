"""Diffusion Transformer (DiT) for latent future prediction.

Predicts 4 future latent embeddings as a 4-token sequence using
self-attention with adaLN-Zero conditioning (Peebles & Xie, 2023).
The model takes noised target tokens, a current-frame embedding from a
frozen encoder, an action embedding from
:class:`~models.fourier_embed.FourierActionEmbedding`, and a diffusion
timestep, and predicts the noise added to the target tokens.

**Architecture overview:**

1. Project noisy input tokens via a learned linear layer.
2. Form a conditioning vector by summing a sinusoidal timestep
   embedding, a projected current-frame embedding, and the action
   embedding.
3. Pass the projected tokens through ``n_blocks`` DiT blocks, each
   using adaLN-Zero modulation from the conditioning vector.
4. Apply a final adaLN-modulated LayerNorm + linear projection to
   produce the noise prediction.

Hyperparameters are stored in ``configs/dit.yaml``; the recommended
constructor is :meth:`LatentDiT.from_dit_config`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Default architectural constants (mirror configs/dit.yaml)
# ---------------------------------------------------------------------------

DEFAULT_Z_DIM = 384
DEFAULT_COND_DIM = 384
DEFAULT_N_BLOCKS = 4
DEFAULT_N_HEADS = 6
DEFAULT_HORIZON = 4
DEFAULT_MLP_RATIO = 4.0
DEFAULT_DROPOUT = 0.0


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a 2-layer MLP.

    Uses the standard sin/cos positional encoding pattern from
    *Attention Is All You Need* (Vaswani et al., 2017) to map scalar
    integer timesteps to dense vectors, then projects through
    ``Linear -> SiLU -> Linear``.

    Parameters
    ----------
    cond_dim
        Output conditioning dimension.
    """

    def __init__(self, cond_dim: int = DEFAULT_COND_DIM) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """Embed integer timesteps ``(B,)`` to ``(B, cond_dim)``.

        Parameters
        ----------
        timestep
            Integer diffusion timestep tensor of shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Conditioning vector of shape ``(B, cond_dim)``.
        """
        half_dim = self.cond_dim // 2
        # log-spaced frequencies: exp(-log(10000) * i / (d/2))
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=timestep.device, dtype=torch.float32)
            / half_dim
        )
        args = timestep.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, d/2)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, d)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Modulation helper
# ---------------------------------------------------------------------------


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaLN modulation: ``x * (1 + scale) + shift``."""
    return x * (1.0 + scale) + shift


# ---------------------------------------------------------------------------
# DiT block
# ---------------------------------------------------------------------------


class DiTBlock(nn.Module):
    """Single transformer block with adaLN-Zero conditioning.

    Implements the adaptive layer-norm zero (adaLN-Zero) design from
    Peebles & Xie (2023): a conditioning vector produces per-sample
    shift, scale, and gate parameters for both the attention and MLP
    sub-layers.  The gate parameters are initialized to zero so each
    fresh block acts as an identity function.

    Parameters
    ----------
    dim
        Token embedding dimension.
    cond_dim
        Conditioning vector dimension.
    n_heads
        Number of attention heads.
    mlp_ratio
        MLP hidden-layer width as a multiple of ``dim``.
    dropout
        Dropout rate applied after attention and MLP.
    """

    def __init__(
        self,
        dim: int = DEFAULT_Z_DIM,
        cond_dim: int = DEFAULT_COND_DIM,
        n_heads: int = DEFAULT_N_HEADS,
        mlp_ratio: float = DEFAULT_MLP_RATIO,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        # Attention sub-layer
        self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # MLP sub-layer
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )
        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = nn.Identity()

        # adaLN modulation: produces 6 vectors from conditioning
        self.adaln_linear = nn.Linear(cond_dim, 6 * dim)
        # Zero-initialize so gates start at 0 (identity block)
        nn.init.zeros_(self.adaln_linear.weight)
        nn.init.zeros_(self.adaln_linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply one DiT block.

        Parameters
        ----------
        x
            Token sequence of shape ``(B, T, dim)``.
        cond
            Conditioning vector of shape ``(B, cond_dim)``.

        Returns
        -------
        torch.Tensor
            Output token sequence of shape ``(B, T, dim)``.
        """
        # Compute 6 modulation parameters
        mod = self.adaln_linear(cond).unsqueeze(1)  # (B, 1, 6*dim)
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = mod.chunk(6, dim=-1)  # each (B, 1, dim)

        # Attention path
        h = _modulate(self.norm_attn(x), shift_attn, scale_attn)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_attn * self.drop(attn_out)

        # MLP path
        h = _modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.drop(self.mlp(h))

        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class LatentDiT(nn.Module):
    """Diffusion Transformer for latent future prediction.

    Predicts noise added to ``horizon`` future latent tokens using
    self-attention with adaLN-Zero conditioning from the current-frame
    embedding, action embedding, and diffusion timestep.

    Parameters
    ----------
    z_dim
        Latent embedding dimension.  Defaults to 384 (project-wide
        ``target_embedding_dim``).
    cond_dim
        Conditioning vector dimension.  Defaults to 384.
    n_blocks
        Number of stacked DiT blocks.  Defaults to 4.
    n_heads
        Number of attention heads per block.  Defaults to 6.
    horizon
        Number of future latent tokens to predict.  Defaults to 4.
    mlp_ratio
        MLP hidden-layer width as a multiple of ``z_dim``.
    dropout
        Dropout rate.
    """

    def __init__(
        self,
        z_dim: int = DEFAULT_Z_DIM,
        cond_dim: int = DEFAULT_COND_DIM,
        n_blocks: int = DEFAULT_N_BLOCKS,
        n_heads: int = DEFAULT_N_HEADS,
        horizon: int = DEFAULT_HORIZON,
        mlp_ratio: float = DEFAULT_MLP_RATIO,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.n_blocks = n_blocks
        self.horizon = horizon

        # Input projection
        self.input_proj = nn.Linear(z_dim, z_dim)

        # Conditioning
        self.timestep_embed = TimestepEmbedding(cond_dim)
        self.z_t_proj = nn.Linear(z_dim, cond_dim)

        # DiT blocks
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=z_dim,
                    cond_dim=cond_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        # Final layer with its own adaLN modulation
        self.final_norm = nn.LayerNorm(z_dim, elementwise_affine=False)
        self.final_adaln = nn.Linear(cond_dim, 3 * z_dim)  # shift, scale, gate
        nn.init.zeros_(self.final_adaln.weight)
        nn.init.zeros_(self.final_adaln.bias)
        self.final_linear = nn.Linear(z_dim, z_dim)

    def forward(
        self,
        x_noisy: torch.Tensor,
        z_t: torch.Tensor,
        a_embed: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise from noised future tokens and conditioning.

        Parameters
        ----------
        x_noisy
            Noised target tokens of shape ``(B, horizon, z_dim)``.
        z_t
            Current-frame encoder embedding of shape ``(B, z_dim)``.
        a_embed
            Action embedding of shape ``(B, z_dim)``.
        timestep
            Integer diffusion timestep of shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Predicted noise of shape ``(B, horizon, z_dim)``.
        """
        # Conditioning vector: sum of timestep, projected z_t, and action
        cond = (
            self.timestep_embed(timestep)
            + self.z_t_proj(z_t)
            + a_embed
        )  # (B, cond_dim)

        # Project input tokens
        x = self.input_proj(x_noisy)  # (B, horizon, z_dim)

        # DiT blocks
        for block in self.blocks:
            x = block(x, cond)

        # Final adaLN-modulated output
        mod = self.final_adaln(cond).unsqueeze(1)  # (B, 1, 3*z_dim)
        shift, scale, gate = mod.chunk(3, dim=-1)
        x = gate * self.final_linear(_modulate(self.final_norm(x), shift, scale))

        return x

    @classmethod
    def from_dit_config(
        cls,
        cfg_path: Optional[Union[str, Path]] = None,
    ) -> "LatentDiT":
        """Construct from a YAML config file.

        Parameters
        ----------
        cfg_path
            Path to the DiT config YAML.  Defaults to
            ``configs/dit.yaml`` relative to the repository root.

        Returns
        -------
        LatentDiT
            A freshly initialised model with the config's hyperparameters.
        """
        import yaml

        if cfg_path is None:
            cfg_path = Path(__file__).resolve().parent.parent / "configs" / "dit.yaml"
        else:
            cfg_path = Path(cfg_path)

        with cfg_path.open("r") as fh:
            raw = yaml.safe_load(fh)

        dit_cfg = raw["dit"]
        return cls(
            z_dim=int(dit_cfg["z_dim"]),
            cond_dim=int(dit_cfg["cond_dim"]),
            n_blocks=int(dit_cfg["n_blocks"]),
            n_heads=int(dit_cfg["n_heads"]),
            horizon=int(dit_cfg["horizon"]),
            mlp_ratio=float(dit_cfg["mlp_ratio"]),
            dropout=float(dit_cfg["dropout"]),
        )
