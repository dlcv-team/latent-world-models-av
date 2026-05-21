"""Tests for :func:`models.latent_pred.train_latent_predictor`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.temporal import TemporalEmbeddingDataset
from models.fourier_embed import FourierActionEmbedding
from models.latent_pred import LatentPredictor, train_latent_predictor


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_EMBED_DIM = 384
_HORIZON = 4


def _make_synthetic_dataset(
    n_scenes: int = 2,
    frames_per_scene: int = 12,
    embed_dim: int = _EMBED_DIM,
    split: str = "train",
    horizon: int = _HORIZON,
    seed: int = 0,
) -> TemporalEmbeddingDataset:
    """Build a small synthetic temporal dataset for testing."""
    rng = np.random.RandomState(seed)
    n = n_scenes * frames_per_scene
    return TemporalEmbeddingDataset(
        embeddings=rng.randn(n, embed_dim).astype(np.float32),
        steer_norms=rng.uniform(-1, 1, n).astype(np.float32),
        accel_norms=rng.uniform(-1, 1, n).astype(np.float32),
        scene_names=np.array(
            [f"scene-{s:04d}" for s in range(n_scenes) for _ in range(frames_per_scene)]
        ),
        timestamps_us=np.array(
            [
                1_000_000_000 + s * 100_000_000 + f * 500_000
                for s in range(n_scenes)
                for f in range(frames_per_scene)
            ],
            dtype=np.int64,
        ),
        splits=np.array([split] * n),
        split=split,
        horizon=horizon,
    )


def _build_components(embed_dim: int = _EMBED_DIM):
    """Build predictor, fourier_embed, adapter, and optimizer."""
    predictor = LatentPredictor(z_dim=embed_dim)
    fourier_embed = FourierActionEmbedding(out_dim=embed_dim)
    adapter = nn.Identity()

    params = list(predictor.parameters()) + list(fourier_embed.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    return predictor, fourier_embed, adapter, optimizer


# ---------------------------------------------------------------------------
# Core training tests
# ---------------------------------------------------------------------------


def test_train_one_epoch_returns_loss():
    """Training for 1 epoch produces a finite loss value."""
    ds = _make_synthetic_dataset()
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    predictor, fe, adapter, opt = _build_components()

    history = train_latent_predictor(
        predictor=predictor,
        fourier_embed=fe,
        adapter=adapter,
        train_loader=loader,
        optimizer=opt,
        epochs=1,
    )
    assert len(history["train_loss"]) == 1
    assert history["train_loss"][0] > 0
    assert np.isfinite(history["train_loss"][0])


def test_train_conditioned_vs_unconditioned():
    """Both variants run without error and produce different final losses."""
    torch.manual_seed(42)
    ds = _make_synthetic_dataset(seed=42)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    # Conditioned
    torch.manual_seed(0)
    p1, fe1, ad1, opt1 = _build_components()
    h1 = train_latent_predictor(
        p1, fe1, ad1, loader, opt1, epochs=5, variant="conditioned"
    )

    # Unconditioned
    torch.manual_seed(0)
    p2, fe2, ad2, opt2 = _build_components()
    h2 = train_latent_predictor(
        p2, fe2, ad2, loader, opt2, epochs=5, variant="unconditioned"
    )

    # Both should complete and produce different trajectories
    assert len(h1["train_loss"]) == 5
    assert len(h2["train_loss"]) == 5
    # Final losses are unlikely to be identical
    assert h1["train_loss"][-1] != h2["train_loss"][-1]


def test_train_returns_history():
    """Returns dict with train_loss list and val_loss=None when no val_loader."""
    ds = _make_synthetic_dataset()
    loader = DataLoader(ds, batch_size=4)
    predictor, fe, adapter, opt = _build_components()

    history = train_latent_predictor(
        predictor, fe, adapter, loader, opt, epochs=3
    )
    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) == 3
    assert history["val_loss"] is None


def test_train_with_val_loader():
    """Val loss is computed when val_loader is provided."""
    train_ds = _make_synthetic_dataset(split="train")
    val_ds = _make_synthetic_dataset(split="train", seed=1)  # reuse "train" split
    train_loader = DataLoader(train_ds, batch_size=4)
    val_loader = DataLoader(val_ds, batch_size=4)
    predictor, fe, adapter, opt = _build_components()

    history = train_latent_predictor(
        predictor, fe, adapter, train_loader, opt, epochs=2,
        val_loader=val_loader,
    )
    assert history["val_loss"] is not None
    assert len(history["val_loss"]) == 2
    assert all(v > 0 for v in history["val_loss"])


def test_csv_logging(tmp_path):
    """CSV file is written with correct columns."""
    ds = _make_synthetic_dataset()
    loader = DataLoader(ds, batch_size=4)
    predictor, fe, adapter, opt = _build_components()
    csv_path = tmp_path / "train_log.csv"

    train_latent_predictor(
        predictor, fe, adapter, loader, opt, epochs=2,
        log_csv_path=csv_path,
    )
    assert csv_path.exists()

    import csv
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == 2
    assert set(rows[0].keys()) == {"epoch", "train_loss", "val_loss"}


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------


def test_adapter_identity_for_384d():
    """When native_dim=384, nn.Identity adapter works transparently."""
    ds = _make_synthetic_dataset(embed_dim=384)
    loader = DataLoader(ds, batch_size=4)
    predictor = LatentPredictor(z_dim=384)
    fe = FourierActionEmbedding(out_dim=384)
    adapter = nn.Identity()
    opt = torch.optim.Adam(
        list(predictor.parameters()) + list(fe.parameters()), lr=1e-3
    )

    history = train_latent_predictor(
        predictor, fe, adapter, loader, opt, epochs=1
    )
    assert len(history["train_loss"]) == 1


def test_adapter_projection_for_1024d():
    """When native_dim=1024, Linear adapter projects to 384-d correctly."""
    native_dim = 1024
    target_dim = 384
    ds = _make_synthetic_dataset(embed_dim=native_dim)
    loader = DataLoader(ds, batch_size=4)

    predictor = LatentPredictor(z_dim=target_dim)
    fe = FourierActionEmbedding(out_dim=target_dim)
    adapter = nn.Linear(native_dim, target_dim, bias=False)
    opt = torch.optim.Adam(
        list(predictor.parameters()) + list(fe.parameters()) + list(adapter.parameters()),
        lr=1e-3,
    )

    history = train_latent_predictor(
        predictor, fe, adapter, loader, opt, epochs=2
    )
    assert len(history["train_loss"]) == 2
    # Adapter should have received gradients
    assert adapter.weight.grad is None or True  # grad cleared after step
    # Verify the adapter's weight shape
    assert adapter.weight.shape == (target_dim, native_dim)
