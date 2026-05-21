"""Fourier action embedding for the latent prediction pipeline (P1).

Maps a 2-d normalized action ``(steer_norm, accel_norm)`` to a dense
``out_dim``-d (default 384) embedding via sinusoidal Fourier features
followed by a two-layer MLP projection.  The output dimensionality
matches the project-wide ``target_embedding_dim`` so the action
embedding can be concatenated directly with a frozen encoder embedding
before being fed to the latent predictor (A16).

**Architecture:**

1. Multiply each action dimension by ``n_frequencies`` sinusoidal
   frequencies: ``freqs[k] = base^k * pi`` for ``k = 0 .. n_frequencies-1``.
2. Compute ``sin`` and ``cos`` for each, giving a
   ``(B, action_dim, 2 * n_frequencies)`` tensor.
3. Flatten to ``(B, action_dim * 2 * n_frequencies)`` (default 256-d).
4. Project through ``Linear -> GELU -> Linear`` to ``(B, out_dim)``.

Hyperparameters are pinned in
``configs/canonical.yaml::latent_predictor::fourier_action_embed``;
``FourierActionEmbedding.from_canonical`` is the recommended constructor.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from config import CanonicalConfig, load_canonical

# Default architectural constants.  Mirror
# configs/canonical.yaml::latent_predictor::fourier_action_embed so that
# callers can construct a FourierActionEmbedding without importing the
# config (e.g. in a unit test); ``from_canonical`` is the path that reads
# the config and is authoritative.
DEFAULT_N_FREQUENCIES = 64
DEFAULT_BASE = 2
DEFAULT_OUT_DIM = 384
DEFAULT_ACTION_DIM = 2  # (steer_norm, accel_norm)


class FourierActionEmbedding(nn.Module):
    """Sinusoidal Fourier features + MLP: ``(B, action_dim) -> (B, out_dim)``.

    Parameters
    ----------
    action_dim
        Number of action dimensions. Defaults to 2 (steer, accel).
    n_frequencies
        Number of sinusoidal frequency bands. Defaults to 64.
    base
        Geometric base for the frequency schedule. Frequency ``k`` is
        ``base ** k * pi``. Defaults to 2.
    out_dim
        Output embedding dimension. Defaults to 384 (project-wide
        ``target_embedding_dim``).

    Notes
    -----
    The frequency buffer ``freqs`` is registered via
    :meth:`~torch.nn.Module.register_buffer` so it follows the module to
    the correct device on ``.to(device)`` calls without appearing in
    :meth:`~torch.nn.Module.parameters`.  At high frequency indices
    (k > ~24) the ``sin``/``cos`` outputs become pseudo-random due to
    float32 precision limits; this is expected and acts as a learned hash.
    """

    def __init__(
        self,
        action_dim: int = DEFAULT_ACTION_DIM,
        n_frequencies: int = DEFAULT_N_FREQUENCIES,
        base: float = DEFAULT_BASE,
        out_dim: int = DEFAULT_OUT_DIM,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.n_frequencies = int(n_frequencies)
        self.base = float(base)
        self.out_dim = int(out_dim)

        freqs = self.base ** torch.arange(self.n_frequencies) * torch.pi
        self.register_buffer("freqs", freqs)

        fourier_dim = self.action_dim * 2 * self.n_frequencies
        self.proj = nn.Sequential(
            nn.Linear(fourier_dim, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

    @classmethod
    def from_canonical(
        cls, cfg: Optional[CanonicalConfig] = None
    ) -> "FourierActionEmbedding":
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
            n_frequencies=int(fae_cfg["n_frequencies"]),
            base=float(fae_cfg["base"]),
            out_dim=int(fae_cfg["out_dim"]),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """Map action ``(B, action_dim)`` to embedding ``(B, out_dim)``."""
        x = action.unsqueeze(-1) * self.freqs  # (B, action_dim, n_frequencies)
        feat = torch.cat([x.sin(), x.cos()], dim=-1)  # (B, action_dim, 2*n_freq)
        return self.proj(feat.view(feat.size(0), -1))  # (B, fourier_dim) -> (B, out_dim)
