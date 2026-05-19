"""Latent-space world model for the P1 prediction pipeline.

The predictor takes the current-frame latent state ``z_t`` (384-d) and
an action embedding ``a_embed`` (384-d, from
:class:`~models.fourier_embed.FourierActionEmbedding`) and predicts the
next ``horizon`` latent states ``z_{t+1}, ..., z_{t+horizon}`` via a
3-layer MLP.  It never generates pixels.

The same class is used for both conditioned and unconditional variants:

- **Conditioned**: ``a_embed`` is a real Fourier action embedding.
- **Unconditional**: ``a_embed`` is zeroed before the forward pass.

Both variants are trained separately with MSE loss against real future
encoder embeddings.  The difference in cosine similarity between the two
(DeltaCosSim) measures whether the action signal adds prediction value.

Hyperparameters are pinned in
``configs/canonical.yaml::latent_predictor`` and should not be tuned;
``LatentPredictor.from_canonical`` is the recommended constructor.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from config import CanonicalConfig, load_canonical

# Default architectural constants.  Mirror
# configs/canonical.yaml::latent_predictor so that callers can construct
# a LatentPredictor without importing the config (e.g. in a unit test);
# ``from_canonical`` is the path that reads the config and is
# authoritative.
DEFAULT_Z_DIM = 384
DEFAULT_A_DIM = 384
DEFAULT_HORIZON = 4
DEFAULT_HIDDEN = 512  # from architecture string: "Linear(768,512) -> ..."


class LatentPredictor(nn.Module):
    """Three-layer MLP: ``Linear -> GELU -> Linear -> GELU -> Linear``.

    Parameters
    ----------
    z_dim
        Encoder embedding dimension.  Defaults to 384 (project-wide
        ``target_embedding_dim``).
    a_dim
        Action embedding dimension (output of
        :class:`~models.fourier_embed.FourierActionEmbedding`).
        Defaults to 384.
    horizon
        Number of future latent states to predict.  Defaults to 4.
    hidden
        Hidden width of the MLP.  Defaults to 512 per canonical config
        architecture string.

    Notes
    -----
    The model input is ``cat(z_t, a_embed)`` with shape ``(B, z_dim +
    a_dim)`` and the output is reshaped to ``(B, horizon, z_dim)`` so
    each predicted future frame gets its own 384-d latent vector.
    """

    def __init__(
        self,
        z_dim: int = DEFAULT_Z_DIM,
        a_dim: int = DEFAULT_A_DIM,
        horizon: int = DEFAULT_HORIZON,
        hidden: int = DEFAULT_HIDDEN,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.a_dim = int(a_dim)
        self.horizon = int(horizon)
        self.hidden = int(hidden)

        self.net = nn.Sequential(
            nn.Linear(self.z_dim + self.a_dim, self.hidden),  # 768 -> 512
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),  # 512 -> 512
            nn.GELU(),
            nn.Linear(self.hidden, self.z_dim * self.horizon),  # 512 -> 1536
        )

    @classmethod
    def from_canonical(
        cls, cfg: Optional[CanonicalConfig] = None
    ) -> "LatentPredictor":
        """Construct from ``configs/canonical.yaml``.

        Parameters
        ----------
        cfg
            Optional pre-loaded :class:`CanonicalConfig`. If omitted, the
            canonical config is loaded from disk via :func:`load_canonical`.
        """
        if cfg is None:
            cfg = load_canonical()
        lp_cfg = cfg.latent_predictor()
        fae_cfg = lp_cfg["fourier_action_embed"]
        return cls(
            z_dim=cfg.target_embedding_dim,
            a_dim=int(fae_cfg["out_dim"]),
            horizon=int(lp_cfg["prediction_horizon"]),
            # hidden=512 is encoded in the architecture string
            # "Linear(768,512) -> GELU -> Linear(512,512) -> ..."
            # rather than as a separate YAML key.
            hidden=DEFAULT_HIDDEN,
        )

    def forward(
        self, z_t: torch.Tensor, a_embed: torch.Tensor
    ) -> torch.Tensor:
        """Predict future latents from current state + action embedding.

        Parameters
        ----------
        z_t
            ``(B, z_dim)`` — current-frame encoder embedding.
        a_embed
            ``(B, a_dim)`` — action embedding from
            :class:`~models.fourier_embed.FourierActionEmbedding`.
            Pass ``torch.zeros_like(a_embed)`` for the unconditional
            variant.

        Returns
        -------
        torch.Tensor
            ``(B, horizon, z_dim)`` — predicted future latent states.
        """
        x = torch.cat([z_t, a_embed], dim=-1)  # (B, z_dim + a_dim)
        return self.net(x).view(-1, self.horizon, self.z_dim)  # (B, H, z_dim)
