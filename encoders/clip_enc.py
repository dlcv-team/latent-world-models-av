"""CLIP ViT-B/32 wrapper.

Loads ``ViT-B-32`` via open_clip with the OpenAI pretrained tag. The
visual encoder outputs 512-d, so the base class attaches a trainable
``nn.Linear(512, 384, bias=False)`` projection adapter at
``self.adapter``.

Only the visual encoder is retained. The full CLIP model also contains a
text transformer of comparable size, which is dead weight here — the
language-conditioned latent predictor that lands later loads its own CLIP
text encoder separately, so dropping it from this wrapper roughly halves
the per-encoder memory footprint without affecting any downstream code.

Applies the OpenAI CLIP normalization (mean/std distinct from ImageNet),
sourced from ``model.visual.image_mean`` / ``image_std``, which open_clip
2.x+ exposes as attributes on the visual transformer.
"""

from __future__ import annotations

from typing import Optional

import open_clip
import torch

from encoders.base import BaseEncoderWrapper

# The OpenAI CLIP checkpoint was trained with QuickGELU activations.
# open_clip's ``ViT-B-32`` config defaults to standard GELU, so loading
# the openai weights into it silently mis-applies the activation and
# shifts every embedding. ``ViT-B-32-quickgelu`` matches the trained
# activation exactly and eliminates open_clip's QuickGELU-mismatch
# warning at load time.
MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"
NATIVE_DIM = 512


class CLIPB32Wrapper(BaseEncoderWrapper):
    """Frozen CLIP ViT-B/32 visual encoder + 512→384 projection adapter.

    Parameters
    ----------
    pretrained
        open_clip pretrained tag. Defaults to ``"openai"`` (OpenAI weights).
        Pass ``None`` for random init — useful for fast unit tests that
        only exercise wrapper plumbing.
    target_dim
        Project-wide embedding dimension. Defaults to 384.

    Inputs are expected as ``(B, 3, 224, 224)`` float tensors in
    ``[0, 1]``. The wrapper applies OpenAI CLIP mean/std before
    forwarding.
    """

    def __init__(
        self,
        *,
        pretrained: Optional[str] = PRETRAINED,
        target_dim: int = 384,
    ) -> None:
        # _load() runs inside super().__init__(); stash the flag first.
        self._pretrained = pretrained
        super().__init__(
            native_dim=NATIVE_DIM,
            needs_projection=True,
            target_dim=target_dim,
        )

    def _load(self) -> None:
        # The two transforms returned alongside the model are discarded:
        # the dataset emits already-resized [0, 1] tensors and the wrapper
        # applies normalization in _encode below.
        full_model, _, _ = open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=self._pretrained,
        )
        # Keep only the visual transformer; see module docstring.
        self.backbone = full_model.visual

        mean = torch.tensor(
            self.backbone.image_mean, dtype=torch.float32
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self.backbone.image_std, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.register_buffer("_image_mean", mean, persistent=False)
        self.register_buffer("_image_std", std, persistent=False)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._image_mean) / self._image_std
        # Equivalent to full_model.encode_image(x) without optional L2 norm:
        # CLIP.encode_image is a thin wrapper around self.visual(x).
        return self.backbone(x)
