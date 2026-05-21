"""Tests for :class:`data.temporal.TemporalEmbeddingDataset`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.temporal import TemporalEmbeddingDataset

# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

_EMBED_DIM = 384
_HORIZON = 4


def _make_synthetic(
    n_scenes: int = 3,
    frames_per_scene: int = 10,
    embed_dim: int = _EMBED_DIM,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Build synthetic embedding arrays that mimic the real NPZ schema."""
    rng = np.random.RandomState(seed)
    n = n_scenes * frames_per_scene
    embeddings = rng.randn(n, embed_dim).astype(np.float32)
    steer_norms = rng.uniform(-1, 1, n).astype(np.float32)
    accel_norms = rng.uniform(-1, 1, n).astype(np.float32)

    scene_names = np.array(
        [f"scene-{s:04d}" for s in range(n_scenes) for _ in range(frames_per_scene)]
    )
    # Timestamps: 500_000 us apart (2 Hz) within each scene
    timestamps_us = np.array(
        [
            1_000_000_000 + s * 100_000_000 + f * 500_000
            for s in range(n_scenes)
            for f in range(frames_per_scene)
        ],
        dtype=np.int64,
    )
    # First scene = train, second = val, third = test
    splits = np.array(
        [
            ["train"] * frames_per_scene
            + ["val"] * frames_per_scene
            + ["test"] * frames_per_scene
        ]
    ).flatten()

    return {
        "embeddings": embeddings,
        "steer_norms": steer_norms,
        "accel_norms": accel_norms,
        "scene_names": scene_names,
        "timestamps_us": timestamps_us,
        "splits": splits,
    }


@pytest.fixture
def synth():
    return _make_synthetic()


# ---------------------------------------------------------------------------
# Length and shapes
# ---------------------------------------------------------------------------


def test_dataset_length(synth):
    """Each scene with 10 frames yields 10 - 4 = 6 valid windows."""
    ds = TemporalEmbeddingDataset(**synth, split="train", horizon=_HORIZON)
    # 1 train scene x (10 - 4) = 6
    assert len(ds) == 6


def test_getitem_shapes(synth):
    ds = TemporalEmbeddingDataset(**synth, split="train", horizon=_HORIZON)
    sample = ds[0]
    assert sample["z_t"].shape == (_EMBED_DIM,)
    assert sample["action"].shape == (2,)
    assert sample["z_future"].shape == (_HORIZON, _EMBED_DIM)


# ---------------------------------------------------------------------------
# Scene boundary safety
# ---------------------------------------------------------------------------


def test_no_scene_boundary_crossing():
    """All 5 frames in each sequence must belong to the same scene."""
    # 2 train scenes, 8 frames each
    data = _make_synthetic(n_scenes=2, frames_per_scene=8)
    # Override splits: both scenes are train
    data["splits"] = np.array(["train"] * 16)

    ds = TemporalEmbeddingDataset(**data, split="train", horizon=_HORIZON)
    # Expected: 2 scenes x (8 - 4) = 8 sequences
    assert len(ds) == 8

    # Verify each sequence stays within a scene by checking that
    # z_future embeddings match the expected contiguous slice
    for i in range(len(ds)):
        sample = ds[i]
        start_idx = ds._valid_indices[i]
        # z_future should be embeddings[start+1 : start+5]
        expected_future = ds._embeddings[start_idx + 1 : start_idx + 1 + _HORIZON]
        assert torch.allclose(sample["z_future"], expected_future)


# ---------------------------------------------------------------------------
# Temporal ordering
# ---------------------------------------------------------------------------


def test_temporal_ordering(synth):
    """Timestamps must be strictly increasing within each sequence."""
    ds = TemporalEmbeddingDataset(**synth, split="train", horizon=_HORIZON)
    for i in range(len(ds)):
        start_idx = ds._valid_indices[i]
        ts_window = ds.timestamps[start_idx : start_idx + 1 + _HORIZON]
        diffs = ts_window[1:] - ts_window[:-1]
        assert (diffs > 0).all(), f"Non-increasing timestamps at index {i}"


# ---------------------------------------------------------------------------
# Split filtering
# ---------------------------------------------------------------------------


def test_split_filtering(synth):
    """Only the requested split's data should appear."""
    ds_train = TemporalEmbeddingDataset(**synth, split="train", horizon=_HORIZON)
    ds_val = TemporalEmbeddingDataset(**synth, split="val", horizon=_HORIZON)
    ds_test = TemporalEmbeddingDataset(**synth, split="test", horizon=_HORIZON)

    # Each split has 1 scene x 10 frames -> 6 valid windows
    assert len(ds_train) == 6
    assert len(ds_val) == 6
    assert len(ds_test) == 6


def test_invalid_split_raises(synth):
    with pytest.raises(ValueError, match="No samples found"):
        TemporalEmbeddingDataset(**synth, split="nonexistent")


# ---------------------------------------------------------------------------
# from_encoder classmethod
# ---------------------------------------------------------------------------


def test_from_encoder_classmethod():
    """Constructs from encoder name using real NPZ if available."""
    full_dir = Path(__file__).resolve().parent.parent / "artifacts" / "full" / "embeddings"
    if not (full_dir / "vjepa2_rep64.npz").exists():
        pytest.skip("Full-dataset embeddings not available locally")

    ds = TemporalEmbeddingDataset.from_encoder("vjepa2_rep64", split="train")
    assert len(ds) > 0
    sample = ds[0]
    d = ds.embed_dim
    assert sample["z_t"].shape == (d,)
    assert sample["z_future"].shape == (4, d)


# ---------------------------------------------------------------------------
# Synthetic roundtrip
# ---------------------------------------------------------------------------


def test_synthetic_data():
    """Works end-to-end with hand-built numpy arrays."""
    data = _make_synthetic(n_scenes=1, frames_per_scene=6)
    data["splits"] = np.array(["train"] * 6)
    ds = TemporalEmbeddingDataset(**data, split="train", horizon=2)
    # 1 scene x (6 - 2) = 4 valid windows
    assert len(ds) == 4
    sample = ds[0]
    assert sample["z_future"].shape == (2, _EMBED_DIM)


# ---------------------------------------------------------------------------
# DataLoader compatibility
# ---------------------------------------------------------------------------


def test_dataloader_compatible(synth):
    """Dataset works when wrapped in a DataLoader."""
    ds = TemporalEmbeddingDataset(**synth, split="train", horizon=_HORIZON)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["z_t"].shape == (4, _EMBED_DIM)
    assert batch["action"].shape == (4, 2)
    assert batch["z_future"].shape == (4, _HORIZON, _EMBED_DIM)
