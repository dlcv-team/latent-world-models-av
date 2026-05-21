"""Tests for :class:`models.latent_pred.LatentPredictor`."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from config import load_canonical
from models.latent_pred import (
    DEFAULT_A_DIM,
    DEFAULT_HIDDEN,
    DEFAULT_HORIZON,
    DEFAULT_Z_DIM,
    LatentPredictor,
)


# ---------------------------------------------------------------------------
# Forward pass / shape
# ---------------------------------------------------------------------------


def test_forward_shape():
    """Headline A16 spec: ``(8, 384) + (8, 384) -> (8, 4, 384)``, no NaN."""
    lp = LatentPredictor()
    z_t = torch.randn(8, 384)
    a_embed = torch.randn(8, 384)
    out = lp(z_t, a_embed)
    assert out.shape == (8, 4, 384)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("b", [1, 4, 16, 256])
def test_various_batch_sizes(b):
    lp = LatentPredictor().eval()
    out = lp(torch.randn(b, 384), torch.randn(b, 384))
    assert out.shape == (b, 4, 384)


def test_output_dtype_float32():
    lp = LatentPredictor()
    out = lp(torch.randn(2, 384), torch.randn(2, 384))
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Architecture verification
# ---------------------------------------------------------------------------


def test_layer_structure_matches_spec():
    """Architecture: Linear(768,512) -> GELU -> Linear(512,512) -> GELU -> Linear(512,1536)."""
    lp = LatentPredictor()
    layers = list(lp.net.children())
    assert len(layers) == 5
    # Layer 0: Linear(768, 512)
    assert isinstance(layers[0], nn.Linear)
    assert layers[0].in_features == 768
    assert layers[0].out_features == 512
    assert layers[0].bias is not None
    # Layer 1: GELU
    assert isinstance(layers[1], nn.GELU)
    # Layer 2: Linear(512, 512)
    assert isinstance(layers[2], nn.Linear)
    assert layers[2].in_features == 512
    assert layers[2].out_features == 512
    assert layers[2].bias is not None
    # Layer 3: GELU
    assert isinstance(layers[3], nn.GELU)
    # Layer 4: Linear(512, 1536)
    assert isinstance(layers[4], nn.Linear)
    assert layers[4].in_features == 512
    assert layers[4].out_features == 1536  # 384 * 4
    assert layers[4].bias is not None


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_all_params_require_grad():
    lp = LatentPredictor()
    params = list(lp.parameters())
    assert len(params) == 6  # 3 linears x (weight, bias)
    for name, p in lp.named_parameters():
        assert p.requires_grad, f"param {name!r} should be trainable"


def test_gradient_flows():
    """Backprop through forward produces non-zero gradients."""
    lp = LatentPredictor()
    z_t = torch.randn(4, 384)
    a_embed = torch.randn(4, 384)
    out = lp(z_t, a_embed)
    out.sum().backward()
    first_layer = lp.net[0]
    assert first_layer.weight.grad is not None
    assert first_layer.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_eval():
    """Same inputs give identical output (no dropout in this module)."""
    lp = LatentPredictor().eval()
    z_t = torch.randn(4, 384)
    a_embed = torch.randn(4, 384)
    a = lp(z_t, a_embed).detach().clone()
    b = lp(z_t, a_embed).detach().clone()
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_defaults_match_canonical_yaml():
    """Module-level defaults must mirror the canonical config values."""
    cfg = load_canonical()
    lp_cfg = cfg.latent_predictor()
    fae_cfg = lp_cfg["fourier_action_embed"]
    assert DEFAULT_Z_DIM == cfg.target_embedding_dim
    assert DEFAULT_A_DIM == int(fae_cfg["out_dim"])
    assert DEFAULT_HORIZON == int(lp_cfg["prediction_horizon"])
    # hidden is encoded in architecture string, verify it appears there
    arch_str = lp_cfg["architecture"]
    assert "512" in arch_str


def test_from_canonical_dimensions(cfg):
    lp = LatentPredictor.from_canonical(cfg)
    lp_cfg = cfg.latent_predictor()
    layers = list(lp.net.children())
    expected_in = cfg.target_embedding_dim + int(
        lp_cfg["fourier_action_embed"]["out_dim"]
    )
    expected_out = cfg.target_embedding_dim * int(
        lp_cfg["prediction_horizon"]
    )
    assert layers[0].in_features == expected_in
    assert layers[4].out_features == expected_out


def test_from_canonical_none_arg():
    """Calling without an arg should still return a valid predictor."""
    lp = LatentPredictor.from_canonical()
    assert isinstance(lp, LatentPredictor)
    out = lp(torch.randn(2, 384), torch.randn(2, 384))
    assert out.shape == (2, 4, 384)


# ---------------------------------------------------------------------------
# Unconditional variant / action sensitivity
# ---------------------------------------------------------------------------


def test_zero_action_embed_produces_valid_output():
    """Unconditional forward: zeros as action embedding gives finite output."""
    lp = LatentPredictor().eval()
    z_t = torch.randn(8, 384)
    a_embed = torch.zeros(8, 384)
    out = lp(z_t, a_embed)
    assert out.shape == (8, 4, 384)
    assert torch.isfinite(out).all()


def test_output_differs_with_different_actions():
    """Different action embeddings produce different predictions."""
    torch.manual_seed(42)
    lp = LatentPredictor().eval()
    z_t = torch.randn(4, 384)
    a1 = torch.randn(4, 384)
    a2 = torch.randn(4, 384)
    out1 = lp(z_t, a1)
    out2 = lp(z_t, a2)
    assert not torch.allclose(out1, out2)
