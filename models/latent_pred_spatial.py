"""Per-token shared MLP baseline for spatial-token future prediction.

For each spatial position s, the MLP receives:
  - z_t[s]: the current spatial token at position s (384-d)
  - z_t_pool: mean-pooled current frame (384-d global context)
  - a_embed: action embedding for this horizon step (384-d)

This gives each token access to its own spatial info, global context, and
the action, but WITHOUT cross-token interaction. This is the fairest
comparison to DiT, which adds cross-token self-attention.

The MLP is shared across all spatial positions (parameter sharing, like
a 1x1 convolution), ensuring fair parameter count.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialMLPPredictor(nn.Module):
    """Per-token shared MLP for spatial future prediction.

    Parameters
    ----------
    z_dim : int
        Per-token embedding dimension (384).
    a_dim : int
        Action embedding dimension (384).
    horizon : int
        Number of future prediction steps.
    n_spatial : int
        Number of spatial tokens per frame.
    hidden : int
        MLP hidden dimension.
    dropout : float
        Dropout for regularization.
    """

    def __init__(
        self,
        z_dim: int = 384,
        a_dim: int = 384,
        horizon: int = 16,
        n_spatial: int = 49,
        hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.horizon = horizon
        self.n_spatial = n_spatial

        # Input: z_t[s] (384) + z_t_pool (384) + a_embed_step (384) = 1152
        input_dim = z_dim + z_dim + a_dim

        # Shared MLP across all spatial positions
        # Predicts residual: delta_z[s] = z_future[s] - z_t[s]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, z_dim),
        )

    def forward(
        self,
        z_t_spatial: torch.Tensor,    # (B, S, D) current spatial tokens
        a_embed: torch.Tensor,        # (B, H, D) per-step action embeddings
    ) -> torch.Tensor:
        """Predict future spatial tokens.

        Returns
        -------
        z_hat : (B, H*S, D)
            Predicted future spatial tokens (as residuals added to z_t).
        """
        B, S, D = z_t_spatial.shape
        H = self.horizon

        # Global context: mean-pool current spatial tokens
        z_t_pool = z_t_spatial.mean(dim=1)  # (B, D)

        # For each horizon step and spatial position:
        # input = concat(z_t[s], z_t_pool, a_embed[h])
        outputs = []
        for h in range(H):
            a_h = a_embed[:, h, :]  # (B, D)
            # Broadcast to all spatial positions
            z_pool_exp = z_t_pool.unsqueeze(1).expand(-1, S, -1)  # (B, S, D)
            a_h_exp = a_h.unsqueeze(1).expand(-1, S, -1)          # (B, S, D)

            x = torch.cat([z_t_spatial, z_pool_exp, a_h_exp], dim=-1)  # (B, S, 3D)
            delta = self.net(x)  # (B, S, D) -- shared MLP across S positions
            z_hat_h = z_t_spatial + delta  # residual prediction
            outputs.append(z_hat_h)

        # Stack: (B, H, S, D) -> reshape to (B, H*S, D)
        z_hat = torch.stack(outputs, dim=1).reshape(B, H * S, D)
        return z_hat
