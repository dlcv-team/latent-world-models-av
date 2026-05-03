"""ViT-S/16 (ImageNet-pretrained) wrapper.

Loads timm's ``vit_small_patch16_224`` as the frozen backbone and exposes
its 384-d pre-head pooled CLS token as the encoder output. Native dim
already matches ``target_dim=384``, so no projection adapter is attached.

The wrapper applies the per-checkpoint image normalization (mean/std)
pulled from timm's ``pretrained_cfg`` — encoder-specific normalization
lives inside the wrapper rather than the dataset, since each checkpoint
expects its own mean/std and the dataset is encoder-agnostic.
"""

from __future__ import annotations

import timm
import timm.data
import torch

from encoders.base import BaseEncoderWrapper

MODEL_ID = "vit_small_patch16_224"
NATIVE_DIM = 384


class ViTS16Wrapper(BaseEncoderWrapper):
    """Frozen ViT-S/16 ImageNet encoder.

    Parameters
    ----------
    pretrained
        If True (default), load ImageNet weights via timm. If False, the
        backbone is created with random initialization — useful for fast
        unit tests that only exercise wrapper plumbing and don't need
        meaningful embeddings.
    target_dim
        Project-wide embedding dimension. Defaults to 384, which matches
        ViT-S/16's native CLS-token dimension; overriding it is unusual.

    Inputs are expected as ``(B, 3, 224, 224)`` float tensors in
    ``[0, 1]`` (the standard torchvision ToTensor output range). The
    wrapper applies the checkpoint's mean/std normalization in
    :meth:`_encode` before invoking the backbone.
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
        # num_classes=0 strips the ImageNet classification head, so
        # backbone(x) returns the 384-d pre-head pooled CLS token directly.
        self.backbone = timm.create_model(
            MODEL_ID,
            pretrained=self._pretrained,
            num_classes=0,
        )
        # Pull the mean/std this checkpoint was trained with rather than
        # hardcoding ImageNet stats — that way if timm switches to a new
        # pretrained variant in the future, normalization tracks the model.
        cfg = timm.data.resolve_data_config({}, model=self.backbone)
        mean = torch.tensor(cfg["mean"], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(cfg["std"], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_image_mean", mean, persistent=False)
        self.register_buffer("_image_std", std, persistent=False)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._image_mean) / self._image_std
        return self.backbone(x)
