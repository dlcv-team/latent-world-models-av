"""Tests for :mod:`scripts.export_z_hat` inference and export logic.

Uses synthetic data + freshly trained tiny models so there is no
dependency on real checkpoints or pre-computed embeddings.
"""

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
from scripts.export_z_hat import _run_inference


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EMBED_DIM = 384
_HORIZON = 4
_N_SCENES = 2
_FRAMES_PER_SCENE = 12


def _make_synthetic_dataset(
    seed: int = 0,
    split: str = "test",
    embed_dim: int = _EMBED_DIM,
) -> TemporalEmbeddingDataset:
    rng = np.random.RandomState(seed)
    n = _N_SCENES * _FRAMES_PER_SCENE
    return TemporalEmbeddingDataset(
        embeddings=rng.randn(n, embed_dim).astype(np.float32),
        steer_norms=rng.uniform(-1, 1, n).astype(np.float32),
        accel_norms=rng.uniform(-1, 1, n).astype(np.float32),
        scene_names=np.array(
            [f"scene-{s:04d}" for s in range(_N_SCENES) for _ in range(_FRAMES_PER_SCENE)]
        ),
        timestamps_us=np.array(
            [
                1_000_000_000 + s * 100_000_000 + f * 500_000
                for s in range(_N_SCENES)
                for f in range(_FRAMES_PER_SCENE)
            ],
            dtype=np.int64,
        ),
        splits=np.array([split] * n),
        split=split,
        horizon=_HORIZON,
    )


def _train_and_get_models(
    variant: str = "conditioned",
    embed_dim: int = _EMBED_DIM,
    epochs: int = 2,
):
    """Train a tiny predictor and return (predictor, fourier_embed, adapter)."""
    ds = _make_synthetic_dataset(split="train", embed_dim=embed_dim)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    predictor = LatentPredictor(z_dim=embed_dim)
    fourier_embed = FourierActionEmbedding(out_dim=embed_dim)
    adapter = nn.Identity()

    params = list(predictor.parameters()) + list(fourier_embed.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    train_latent_predictor(
        predictor, fourier_embed, adapter, loader, optimizer,
        epochs=epochs, variant=variant,
    )

    predictor.eval()
    fourier_embed.eval()
    return predictor, fourier_embed, adapter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_export_shapes():
    """Output tensors have correct shapes (N, 4, 384)."""
    torch.manual_seed(42)
    predictor, fourier_embed, adapter = _train_and_get_models("conditioned")
    test_ds = _make_synthetic_dataset(seed=99, split="test")
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    z_hat, z_real = _run_inference(
        predictor, fourier_embed, adapter, test_loader, "conditioned"
    )

    n_test = len(test_ds)
    assert z_hat.shape == (n_test, _HORIZON, _EMBED_DIM)
    assert z_real.shape == (n_test, _HORIZON, _EMBED_DIM)


def test_z_real_matches_adapter_projected_embeddings():
    """z_real matches manual adapter projection of test embeddings."""
    torch.manual_seed(42)
    native_dim = 1024
    target_dim = _EMBED_DIM

    ds = _make_synthetic_dataset(seed=99, split="train", embed_dim=native_dim)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    predictor = LatentPredictor(z_dim=target_dim)
    fourier_embed = FourierActionEmbedding(out_dim=target_dim)
    adapter = nn.Linear(native_dim, target_dim, bias=False)
    params = (
        list(predictor.parameters())
        + list(fourier_embed.parameters())
        + list(adapter.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=1e-3)
    train_latent_predictor(
        predictor, fourier_embed, adapter, loader, optimizer,
        epochs=2, variant="conditioned",
    )

    predictor.eval()
    fourier_embed.eval()
    adapter.eval()

    test_ds = _make_synthetic_dataset(seed=77, split="test", embed_dim=native_dim)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    _, z_real = _run_inference(
        predictor, fourier_embed, adapter, test_loader, "conditioned"
    )

    # Manual projection
    batch = next(iter(test_loader))
    z_future_native = batch["z_future"]
    B, H, D = z_future_native.shape
    with torch.no_grad():
        z_future_proj = adapter(z_future_native.reshape(B * H, D)).view(B, H, -1)

    assert torch.allclose(z_real, z_future_proj, atol=1e-6)


def test_conditioned_and_unconditioned_differ():
    """z_hat_cond != z_hat_uncond (different action embedding inputs)."""
    torch.manual_seed(42)
    pred_c, fe_c, ad_c = _train_and_get_models("conditioned")
    torch.manual_seed(42)
    pred_u, fe_u, ad_u = _train_and_get_models("unconditioned")

    test_ds = _make_synthetic_dataset(seed=99, split="test")
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    z_hat_c, _ = _run_inference(pred_c, fe_c, ad_c, test_loader, "conditioned")
    z_hat_u, _ = _run_inference(pred_u, fe_u, ad_u, test_loader, "unconditioned")

    assert not torch.allclose(z_hat_c, z_hat_u, atol=1e-6)


def test_all_outputs_finite():
    """No NaN/Inf in any output tensor."""
    torch.manual_seed(42)
    predictor, fourier_embed, adapter = _train_and_get_models("conditioned")
    test_ds = _make_synthetic_dataset(seed=99, split="test")
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    z_hat, z_real = _run_inference(
        predictor, fourier_embed, adapter, test_loader, "conditioned"
    )

    assert torch.isfinite(z_hat).all(), "z_hat contains NaN or Inf"
    assert torch.isfinite(z_real).all(), "z_real contains NaN or Inf"
