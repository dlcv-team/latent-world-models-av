"""DINOv2-S/14 wrapper.

Loads ``facebookresearch/dinov2 :: dinov2_vits14`` via ``torch.hub``. The
backbone's ``forward()`` returns the 384-d CLS token directly, so native
dim already matches ``target_dim`` and no projection adapter is attached.

Applies the official ImageNet normalization that DINOv2 was trained with
(per ``dinov2/data/transforms.py`` in the upstream repo). The constants are
hardcoded — unlike timm, ``torch.hub`` returns a plain ``nn.Module`` with no
attached pretraining metadata to query.

Patch size is 14 (not 16), so the input side must be a multiple of 14.
224 = 16 × 14 satisfies this; any other input size would need adjusting.
"""

from __future__ import annotations

import torch

from encoders.base import BaseEncoderWrapper

REPO = "facebookresearch/dinov2"
MODEL_ID = "dinov2_vits14"
NATIVE_DIM = 384

# Source of these constants: facebookresearch/dinov2 :: dinov2/data/transforms.py
#   IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
#   IMAGENET_DEFAULT_STD  = (0.229, 0.224, 0.225)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv2S14Wrapper(BaseEncoderWrapper):
    """Frozen DINOv2 ViT-S/14 encoder.

    Parameters
    ----------
    pretrained
        If True (default), load the official DINOv2 weights via
        ``torch.hub``. If False, the backbone is created with random init
        — useful for fast unit tests that only exercise wrapper plumbing.
    target_dim
        Project-wide embedding dimension. Defaults to 384, which matches
        DINOv2-S/14's native CLS-token dimension.

    Inputs are expected as ``(B, 3, 224, 224)`` float tensors in
    ``[0, 1]``. The wrapper applies ImageNet mean/std before forwarding.
    """

    def __init__(self, *, pretrained: bool = True, target_dim: int = 384) -> None:
        # _load() runs inside super().__init__(); stash the flag first.
        self._pretrained = pretrained
        super().__init__(
            native_dim=NATIVE_DIM,
            needs_projection=False,
            target_dim=target_dim,
        )

    def _load(self) -> None:
        # trust_repo=True suppresses the interactive "do you trust this
        # repo?" prompt that breaks non-interactive runs (CI, scripts).
        self.backbone = torch.hub.load(
            REPO,
            MODEL_ID,
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
