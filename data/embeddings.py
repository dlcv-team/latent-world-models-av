"""Transparent loader for split embedding files."""

from pathlib import Path

import numpy as np

_EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "pilot" / "embeddings"

_SPLIT_FILES = [
    "camfront_keyframes_core.npz",
    "camfront_keyframes_vjepa2.npz",
]


def load_embeddings(directory: Path | None = None) -> dict[str, np.ndarray]:
    """Load all encoder embeddings, merging split files transparently."""
    d = Path(directory) if directory else _EMBEDDINGS_DIR
    merged = {}
    for fname in _SPLIT_FILES:
        with np.load(d / fname, allow_pickle=True) as f:
            merged.update(dict(f))
    return merged
