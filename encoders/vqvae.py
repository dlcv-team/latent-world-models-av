"""VQ-VAE encoder wrapper with documented DINOv2-S/14 fallback.

No pretrained VQ-VAE / VQGAN checkpoint loads reliably across the team's
environments — pickle compatibility issues with public ``taming-transformers``
checkpoints, missing or stale weights on hub mirrors, etc. The documented
project policy (see the ``vqvae.fallback_policy`` field in
``configs/canonical.yaml``) is to substitute DINOv2-S/14 embeddings for
the VQ track and surface the substitution loudly so figure rendering and
report tables can mark VQ results as a fallback rather than independent
VQ evidence.

Public surface beyond the standard wrapper API:

* :class:`VQFallbackUsed` — UserWarning subclass emitted on construction
  whenever the wrapper falls back. Downstream code can install a
  ``warnings.simplefilter("error", VQFallbackUsed)`` to make accidental
  VQ usage a hard error in contexts where the fallback isn't acceptable.
* ``wrapper.fallback_active`` — currently always True. When a working VQ
  checkpoint is added, this will become False on successful primary loads.
* ``wrapper.fallback_caveat`` — the caption string figure renderers must
  include whenever VQ appears in a table or plot.

The primary-load probe is deliberately a stub today (returns "no working
VQ checkpoint available"). When a working loader becomes available, fill
in :meth:`_probe_primary_load` to attempt it; the fallback branch will
stop firing automatically.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch

from encoders.base import BaseEncoderWrapper

# DINOv2 is the documented fallback target; loaded via torch.hub the same
# way DINOv2S14Wrapper does.
_FALLBACK_REPO = "facebookresearch/dinov2"
_FALLBACK_MODEL_ID = "dinov2_vits14"
_FALLBACK_NATIVE_DIM = 384

# ImageNet stats per upstream dinov2/data/transforms.py — the fallback
# backbone expects exactly these.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

FALLBACK_CAVEAT = (
    "VQ-VAE checkpoint failed to load reproducibly; results shown are "
    "from DINOv2-S/14 embeddings as a documented fallback and do not "
    "represent independent VQ-VAE performance."
)


class VQFallbackUsed(UserWarning):
    """Emitted when :class:`VQVAEWrapper` substitutes DINOv2-S/14 for VQ."""


class VQVAEWrapper(BaseEncoderWrapper):
    """Frozen VQ encoder slot — currently routes to the DINOv2-S/14 fallback.

    Parameters
    ----------
    pretrained
        Forwarded to the DINOv2 fallback backbone. Default True for
        production (real DINOv2 features so probes train on something
        meaningful); pass False for fast unit tests that only exercise
        wrapper plumbing.
    target_dim
        Project-wide embedding dimension. Defaults to 384, matching the
        DINOv2 fallback's native dim — so no projection adapter is
        attached in fallback mode.

    Inputs are expected as ``(B, 3, 224, 224)`` float tensors in
    ``[0, 1]``. The wrapper applies ImageNet mean/std before forwarding.
    """

    FALLBACK_CAVEAT = FALLBACK_CAVEAT

    def __init__(
        self,
        *,
        pretrained: bool = True,
        target_dim: int = 384,
    ) -> None:
        self._pretrained = pretrained
        self._fallback_active, primary_error = self._probe_primary_load()

        if self._fallback_active:
            warnings.warn(
                f"VQVAEWrapper is using the DINOv2-S/14 fallback "
                f"({primary_error}); see configs/canonical.yaml for the "
                "documented policy. Tag every figure caption that includes "
                "this encoder with `wrapper.fallback_caveat`.",
                VQFallbackUsed,
                stacklevel=2,
            )
            # DINOv2's native dim already matches target_dim, no projection.
            super().__init__(
                native_dim=_FALLBACK_NATIVE_DIM,
                needs_projection=False,
                target_dim=target_dim,
            )
        else:  # pragma: no cover — primary path is currently unreachable
            raise NotImplementedError(
                "Primary VQ load succeeded but VQVAEWrapper has no primary "
                "_load implementation yet. Implement encoders/vqvae.py "
                "_load() to handle the success branch."
            )

    @staticmethod
    def _probe_primary_load() -> tuple[bool, Optional[str]]:
        """Decide whether to use the fallback. Returns ``(use_fallback, reason)``.

        Today this always returns ``(True, "no working VQ checkpoint "
        "available")``. When a reliable pretrained VQ checkpoint becomes
        loadable, replace the body with an actual attempt (try-import
        ``vector_quantize_pytorch``, try-load the checkpoint, return
        ``(False, None)`` on success).
        """
        return True, "no working pretrained VQ checkpoint available"

    def _load(self) -> None:
        # Only the fallback path is reachable today. When the primary path
        # is implemented, branch on ``self._fallback_active`` here.
        self.backbone = torch.hub.load(
            _FALLBACK_REPO,
            _FALLBACK_MODEL_ID,
            pretrained=self._pretrained,
            trust_repo=True,
            verbose=False,
        )
        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_image_mean", mean, persistent=False)
        self.register_buffer("_image_std", std, persistent=False)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._image_mean) / self._image_std
        return self.backbone(x)

    @property
    def fallback_active(self) -> bool:
        """True iff the wrapper is using the DINOv2 fallback backbone."""
        return self._fallback_active

    @property
    def fallback_caveat(self) -> str:
        """Caption text to include whenever VQ appears in a figure or table.

        Returns the empty string when the primary VQ path is active (which
        will only happen once a working checkpoint is wired in).
        """
        return FALLBACK_CAVEAT if self._fallback_active else ""
