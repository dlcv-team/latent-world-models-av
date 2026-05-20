"""Tests for :mod:`data.z_hat` transparent loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from data.z_hat import load_z_hat, load_z_real


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_synthetic_tensor(
    directory: Path,
    filename: str,
    shape: tuple[int, ...] = (10, 4, 384),
) -> torch.Tensor:
    """Save a random tensor and return it for comparison."""
    t = torch.randn(*shape)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(t, directory / filename)
    return t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_from_local_dir(tmp_path):
    """Loads z_hat from an explicit local directory."""
    expected = _save_synthetic_tensor(tmp_path, "z_hat_conditioned.pt")
    loaded = load_z_hat("conditioned", directory=tmp_path)
    assert torch.equal(loaded, expected)


def test_load_z_real_from_local(tmp_path):
    """Loads z_real from an explicit local directory."""
    expected = _save_synthetic_tensor(tmp_path, "z_real.pt", shape=(10, 4, 384))
    loaded = load_z_real(directory=tmp_path)
    assert torch.equal(loaded, expected)


def test_invalid_variant_raises():
    """load_z_hat('invalid') raises ValueError."""
    with pytest.raises(ValueError, match="variant must be one of"):
        load_z_hat("invalid")


def test_missing_file_raises(tmp_path):
    """FileNotFoundError when file doesn't exist and no HF fallback."""
    # Point to an empty directory so local lookup fails;
    # provide explicit directory so HF fallback is skipped.
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_z_hat("conditioned", directory=empty_dir)


def test_load_unconditioned_variant(tmp_path):
    """Loads the unconditioned variant correctly."""
    expected = _save_synthetic_tensor(tmp_path, "z_hat_unconditioned.pt")
    loaded = load_z_hat("unconditioned", directory=tmp_path)
    assert torch.equal(loaded, expected)
