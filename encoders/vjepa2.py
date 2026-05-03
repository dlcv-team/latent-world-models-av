"""V-JEPA2 ViT-L/16 video encoder wrapper.

Loads ``facebook/vjepa2-vitl-fpc64-256`` via Hugging Face transformers.
Native checkpoint expects 64-frame clips at 256×256. The team's data
pipeline emits 16-frame clips at 224×224, so the wrapper:

1. Bilinear-resizes each frame from 224 to 256 inside the wrapper, so
   the input matches the spatial resolution the encoder was trained at.
   We resize rather than letting the model interpolate positional
   embeddings spatially because the training distribution dominates
   benchmark fairness, and resize is cheap (one F.interpolate per clip).
2. Feeds 16 frames directly. The model handles temporal length
   differences via temporal positional embedding interpolation. We chose
   not to pad/duplicate to 64 because frame duplication produces
   unnatural input distributions (probed during the pilot — gave worse
   features than direct 16-frame input).
3. Mean-pools the ``(B, N_tokens, 1024)`` encoder output over the token
   axis to a single ``(B, 1024)`` vector before the 1024→384 projection
   adapter. V-JEPA2's ``AutoModel`` has no pooler_output (it returns
   ``None``), so we build our own pooled vector.

Inputs are expected as ``(B, T, 3, H, W)`` float tensors in ``[0, 1]``
— note the temporal axis, which differs from the four single-frame
encoders. ``T`` is expected to be 16 (the canonical clip length); other
values work but were not validated.

ImageNet mean/std applied before forwarding (matches the official
``VJEPA2VideoProcessor`` stats).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from encoders.base import BaseEncoderWrapper

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
NATIVE_DIM = 1024
NATIVE_INPUT_SIZE = 256  # spatial side the checkpoint was trained at

# ImageNet stats — pinned in tests. Same values as VJEPA2VideoProcessor.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VJEPA2Wrapper(BaseEncoderWrapper):
    """Frozen V-JEPA2 ViT-L/16 video encoder + 1024→384 projection adapter.

    Parameters
    ----------
    pretrained
        If True (default), load real V-JEPA2 weights via
        :func:`transformers.AutoModel.from_pretrained` (~1.5 GB checkpoint
        download on first use). If False, instantiate the model
        architecture with random init via
        :func:`AutoModel.from_config` — only the small config file is
        downloaded, useful for fast unit tests.
    target_dim
        Project-wide embedding dimension. Defaults to 384.

    Notes
    -----
    Unlike the four single-frame wrappers, the input tensor has a
    temporal axis: ``(B, T, 3, H, W)``. Spatial size H, W can be anything
    that survives bilinear resize to 256 (most realistically the canonical
    224×224); T is expected to be the canonical clip length (16) but
    other values work via temporal positional embedding interpolation.
    """

    def __init__(self, *, pretrained: bool = True, target_dim: int = 384) -> None:
        # _load() runs inside super().__init__(); stash the flag first.
        self._pretrained = pretrained
        super().__init__(
            native_dim=NATIVE_DIM,
            needs_projection=True,
            target_dim=target_dim,
        )

    def _load(self) -> None:
        if self._pretrained:
            self.backbone = AutoModel.from_pretrained(MODEL_ID)
        else:
            cfg = AutoConfig.from_pretrained(MODEL_ID)
            self.backbone = AutoModel.from_config(cfg)

        # Mean/std broadcast against (B, T, 3, H, W).
        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.register_buffer("_image_mean", mean, persistent=False)
        self.register_buffer("_image_std", std, persistent=False)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3, H, W) in [0, 1]
        b, t, c, h, w = x.shape

        if (h, w) != (NATIVE_INPUT_SIZE, NATIVE_INPUT_SIZE):
            # F.interpolate operates on 4-D tensors; flatten time into
            # batch, resize, then restore the temporal axis.
            x = x.reshape(b * t, c, h, w)
            x = F.interpolate(
                x,
                size=(NATIVE_INPUT_SIZE, NATIVE_INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            x = x.reshape(b, t, c, NATIVE_INPUT_SIZE, NATIVE_INPUT_SIZE)

        x = (x - self._image_mean) / self._image_std

        # AutoModel returns last_hidden_state of shape
        #   (B, (T/tubelet) * (NATIVE/patch)^2, hidden)
        # = (B, (T/2) * 16^2, 1024) at NATIVE_INPUT_SIZE=256, patch=16.
        # No pooler_output is provided; mean-pool over the token axis.
        out = self.backbone(pixel_values_videos=x)
        return out.last_hidden_state.mean(dim=1)
