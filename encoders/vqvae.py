"""VQ-VAE encoder wrapper using a vendored VQGAN encoder.

Loads the VQGAN ImageNet f16 16384 checkpoint from the University of
Heidelberg mirror. The encoder definition is vendored from
CompVis/taming-transformers (MIT license) in ``encoders/_vqgan_arch.py``.

If the checkpoint cannot be downloaded or loaded, falls back to
DINOv2-S/14 embeddings and emits a ``VQFallbackUsed`` warning per
the documented policy in ``configs/canonical.yaml``.

Resolution order for the checkpoint:
  1. ``$VQGAN_CKPT_PATH`` environment variable
  2. ``~/.cache/latent-world-models-av/vqgan_imagenet_f16_16384.ckpt``
  3. Heidelberg direct-binary download (935 MB)
"""

from __future__ import annotations

import hashlib
import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from encoders._vqgan_arch import Encoder, VQGAN_IMAGENET_F16_16384_CONFIG
from encoders.base import BaseEncoderWrapper

logger = logging.getLogger(__name__)

_HEIDELBERG_URL = (
    "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/"
    "files/?p=%2Fckpts%2Flast.ckpt&dl=1"
)
_CACHE_DIR = Path.home() / ".cache" / "latent-world-models-av"
_CACHE_FILENAME = "vqgan_imagenet_f16_16384.ckpt"
_EXPECTED_SIZE = 980_092_370

_NATIVE_DIM = 256
_NATIVE_RESOLUTION = 256

_FALLBACK_REPO = "facebookresearch/dinov2"
_FALLBACK_MODEL_ID = "dinov2_vits14"
_FALLBACK_NATIVE_DIM = 384

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

FALLBACK_CAVEAT = (
    "VQ-VAE checkpoint failed to load reproducibly; results shown are "
    "from DINOv2-S/14 embeddings as a documented fallback and do not "
    "represent independent VQ-VAE performance."
)


class VQFallbackUsed(UserWarning):
    """Emitted when VQVAEWrapper substitutes DINOv2-S/14 for VQ."""


class VQVAEWrapper(BaseEncoderWrapper):
    """Frozen VQGAN encoder projecting to 384-d embeddings.

    On successful checkpoint load, the forward pass resizes input from
    224 to 256 (VQGAN's native resolution), normalizes to [-1, 1],
    encodes to (B, 256, 16, 16), spatially mean-pools, and projects
    to 384 via a trainable linear adapter.

    When the checkpoint is unavailable, falls back to DINOv2-S/14
    with ImageNet normalization and no projection adapter (native 384-d).

    Parameters
    ----------
    pretrained
        If True, attempt to load the real VQGAN checkpoint. If False,
        build the vendored encoder with random weights (for fast tests).
    target_dim
        Project-wide embedding dimension. Default 384.
    """

    FALLBACK_CAVEAT = FALLBACK_CAVEAT

    def __init__(
        self,
        *,
        pretrained: bool = True,
        target_dim: int = 384,
    ) -> None:
        self._pretrained = pretrained
        self._fallback_active = False
        self._primary_error: Optional[str] = None

        if pretrained:
            self._fallback_active, self._primary_error = self._probe_primary_load()

        if self._fallback_active:
            warnings.warn(
                f"VQVAEWrapper is using the DINOv2-S/14 fallback "
                f"({self._primary_error}); see configs/canonical.yaml for "
                "the documented policy.",
                VQFallbackUsed,
                stacklevel=2,
            )
            super().__init__(
                native_dim=_FALLBACK_NATIVE_DIM,
                needs_projection=False,
                target_dim=target_dim,
            )
        else:
            super().__init__(
                native_dim=_NATIVE_DIM,
                needs_projection=True,
                target_dim=target_dim,
            )

    @staticmethod
    def _resolve_checkpoint_path() -> Optional[Path]:
        """Find or download the VQGAN checkpoint. Returns path or None."""
        env_path = os.environ.get("VQGAN_CKPT_PATH")
        if env_path:
            p = Path(env_path)
            if p.is_file():
                return p
            logger.warning("VQGAN_CKPT_PATH=%s does not exist", env_path)

        cached = _CACHE_DIR / _CACHE_FILENAME
        if cached.is_file() and cached.stat().st_size > 1_000_000:
            return cached

        try:
            logger.info(
                "Downloading VQGAN checkpoint from Heidelberg (%d MB)...",
                _EXPECTED_SIZE // 1_000_000,
            )
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_suffix(".tmp")
            _download_with_progress(_HEIDELBERG_URL, tmp)
            tmp.rename(cached)
            return cached
        except Exception as exc:
            logger.warning("VQGAN download failed: %s", exc)
            return None

    @staticmethod
    def _probe_primary_load() -> tuple[bool, Optional[str]]:
        """Try to locate the checkpoint. Returns (use_fallback, error_msg)."""
        path = VQVAEWrapper._resolve_checkpoint_path()
        if path is None:
            return True, "checkpoint not found and download failed"
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if "state_dict" not in ckpt:
                return True, f"checkpoint at {path} has no 'state_dict' key"
            return False, None
        except Exception as exc:
            return True, f"torch.load failed: {exc}"

    def _load(self) -> None:
        if self._fallback_active:
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
        else:
            encoder = Encoder(**VQGAN_IMAGENET_F16_16384_CONFIG)
            if self._pretrained:
                path = self._resolve_checkpoint_path()
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                sd = ckpt["state_dict"]
                encoder_sd = {}
                for k, v in sd.items():
                    if k.startswith("encoder."):
                        encoder_sd[k[len("encoder."):]] = v
                encoder.load_state_dict(encoder_sd, strict=True)
            self.backbone = encoder

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self._fallback_active:
            x = (x - self._image_mean) / self._image_std
            return self.backbone(x)
        else:
            x = F.interpolate(
                x, size=(_NATIVE_RESOLUTION, _NATIVE_RESOLUTION),
                mode="bilinear", align_corners=False, antialias=True,
            )
            x = 2.0 * x - 1.0
            z = self.backbone(x)
            return z.mean(dim=[2, 3])

    @property
    def fallback_active(self) -> bool:
        """True when using the DINOv2 fallback backbone."""
        return self._fallback_active

    @property
    def fallback_caveat(self) -> str:
        """Caption text for figures. Empty string when primary VQ is active."""
        return FALLBACK_CAVEAT if self._fallback_active else ""


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a large file with a tqdm progress bar."""
    import urllib.request
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        block_size = 1024 * 1024
        if tqdm and total:
            pbar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name)
        else:
            pbar = None
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                if pbar:
                    pbar.update(len(chunk))
        if pbar:
            pbar.close()
