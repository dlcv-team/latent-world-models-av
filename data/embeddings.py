"""Transparent loader for embedding files with HuggingFace fallback.

Supports two layouts:
  1. Full-dataset (default): per-encoder npz files in artifacts/full/embeddings/
     e.g. vit_s16.npz, clip_b32.npz, vjepa2_rep64.npz, ...
  2. Pilot (legacy): merged split files in artifacts/pilot/embeddings/

Download cascade:
  1. Local artifacts/full/embeddings/
  2. HuggingFace Hub (surlac/lwm-av-embeddings)
  3. Local artifacts/pilot/embeddings/ (fallback for legacy code)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FULL_DIR = _PROJECT_ROOT / "artifacts" / "full" / "embeddings"
_PILOT_DIR = _PROJECT_ROOT / "artifacts" / "pilot" / "embeddings"

_HF_REPO = "surlac/lwm-av-embeddings"

# "rep64" / "rep1" refer to V-JEPA2 checkpoint variants, NOT input frame counts.
# rep64 = facebook/vjepa2-vitl-fpc64-256 (fpc64 = pre-trained on 64-frame clips).
# rep1  = facebook/vjepa2-vitl-fpc1-256  (fpc1  = pre-trained on 1-frame clips).
# Our canonical input is 16 frames (canonical.yaml::dataset::vjepa2::clip_frames).
ENCODER_NAMES = [
    "vit_s16",
    "dino_vits14",
    "clip_b32",
    "vq_track",
    "vjepa2_rep64",
    "vjepa2_rep1",
]


def _download_from_hf(encoder_name: str, cache_dir: Path) -> Path:
    """Download a single encoder's npz from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading embeddings. "
            "Install with: pip install huggingface_hub"
        )

    fname = f"{encoder_name}.npz"
    local_path = hf_hub_download(
        repo_id=_HF_REPO,
        filename=fname,
        repo_type="dataset",
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
    )
    return Path(local_path)


def load_encoder_embedding(
    encoder_name: str,
    directory: Path | None = None,
) -> dict[str, np.ndarray]:
    """Load a single encoder's embedding file.

    Returns dict with keys: embeddings, splits, scene_names,
    steer_norms, accel_norms, tokens (encoder-specific).
    """
    if directory:
        d = Path(directory)
    else:
        d = _FULL_DIR

    local_path = d / f"{encoder_name}.npz"

    if not local_path.exists() and directory is None:
        # Try HF download
        print(f"[embeddings] {encoder_name}.npz not found locally, downloading from HF ...")
        try:
            local_path = _download_from_hf(encoder_name, _FULL_DIR)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load {encoder_name} embeddings. "
                f"Not found at {local_path} and HF download failed: {e}\n"
                f"Run: python scripts/upload_hf.py to populate HF, or "
                f"download manually from https://huggingface.co/datasets/{_HF_REPO}"
            ) from e

    with np.load(local_path, allow_pickle=True) as f:
        return dict(f)


def load_all_embeddings(
    directory: Path | None = None,
    encoders: list[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Load embeddings for all (or specified) encoders.

    Returns:
        Dict mapping encoder_name -> {embeddings, splits, scene_names, ...}
    """
    enc_list = encoders or ENCODER_NAMES
    return {
        name: load_encoder_embedding(name, directory)
        for name in enc_list
    }


# Legacy API -- kept for backward compatibility with pilot code
_PILOT_SPLIT_FILES = [
    "camfront_keyframes_core.npz",
    "camfront_keyframes_vjepa2.npz",
]


def load_embeddings(directory: Path | None = None) -> dict[str, np.ndarray]:
    """Load all encoder embeddings (legacy pilot format).

    For new code, prefer load_encoder_embedding() or load_all_embeddings().
    """
    d = Path(directory) if directory else _PILOT_DIR
    merged: dict[str, np.ndarray] = {}
    for fname in _PILOT_SPLIT_FILES:
        path = d / fname
        if path.exists():
            with np.load(path, allow_pickle=True) as f:
                merged.update(dict(f))
    return merged
