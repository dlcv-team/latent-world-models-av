"""Tests for :class:`models.fourier_embed.FourierActionEmbedding`."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from config import load_canonical
from models.fourier_embed import (
    DEFAULT_ACTION_DIM,
    DEFAULT_BASE,
    DEFAULT_N_FREQUENCIES,
    DEFAULT_OUT_DIM,
    FourierActionEmbedding,
)


# ---------------------------------------------------------------------------
# Forward pass / shape
# ---------------------------------------------------------------------------


def test_forward_shape():
    """Headline A15 spec: ``(8, 2) -> (8, 384)``, no NaN."""
    module = FourierActionEmbedding()
    action = torch.randn(8, 2)
    out = module(action)
    assert out.shape == (8, 384)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("b", [1, 4, 16, 256])
def test_various_batch_sizes(b):
    module = FourierActionEmbedding().eval()
    out = module(torch.randn(b, 2))
    assert out.shape == (b, 384)


def test_output_dtype_float32():
    module = FourierActionEmbedding()
    out = module(torch.randn(2, 2))
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Architecture verification
# ---------------------------------------------------------------------------


def test_layer_structure_matches_spec():
    """Architecture: Linear(256, 384) -> GELU -> Linear(384, 384)."""
    module = FourierActionEmbedding()
    layers = list(module.proj.children())
    assert len(layers) == 3
    assert isinstance(layers[0], nn.Linear)
    assert layers[0].in_features == 256  # action_dim * 2 * n_frequencies
    assert layers[0].out_features == 384
    assert layers[0].bias is not None
    assert isinstance(layers[1], nn.GELU)
    assert isinstance(layers[2], nn.Linear)
    assert layers[2].in_features == 384
    assert layers[2].out_features == 384
    assert layers[2].bias is not None


def test_freqs_buffer_shape_and_values():
    """Frequency buffer has shape ``(64,)`` with values ``2^k * pi``."""
    module = FourierActionEmbedding()
    assert module.freqs.shape == (64,)
    expected = 2.0 ** torch.arange(64) * torch.pi
    assert torch.allclose(module.freqs, expected)


def test_freqs_is_buffer_not_parameter():
    module = FourierActionEmbedding()
    buffer_names = dict(module.named_buffers())
    param_names = dict(module.named_parameters())
    assert "freqs" in buffer_names
    assert "freqs" not in param_names
    assert not module.freqs.requires_grad


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_all_proj_params_require_grad():
    module = FourierActionEmbedding()
    params = list(module.parameters())
    assert len(params) == 4  # 2 linears x (weight, bias)
    for name, p in module.named_parameters():
        assert p.requires_grad, f"param {name!r} should be trainable"


def test_gradient_flows():
    """Backprop through forward produces non-zero gradients on proj weights."""
    module = FourierActionEmbedding()
    action = torch.randn(4, 2)
    out = module(action)
    out.sum().backward()
    first_layer = module.proj[0]
    assert first_layer.weight.grad is not None
    assert first_layer.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_eval():
    """Same input gives identical output (no dropout in this module)."""
    module = FourierActionEmbedding().eval()
    action = torch.randn(4, 2)
    a = module(action).detach().clone()
    b = module(action).detach().clone()
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_defaults_match_canonical_yaml():
    """Module-level defaults must mirror the canonical config values."""
    cfg = load_canonical()
    lp_cfg = cfg.latent_predictor()
    fae_cfg = lp_cfg["fourier_action_embed"]
    assert DEFAULT_N_FREQUENCIES == int(fae_cfg["n_frequencies"])
    assert DEFAULT_BASE == float(fae_cfg["base"])
    assert DEFAULT_OUT_DIM == int(fae_cfg["out_dim"])


def test_from_canonical_dimensions(cfg):
    module = FourierActionEmbedding.from_canonical(cfg)
    lp_cfg = cfg.latent_predictor()
    fae_cfg = lp_cfg["fourier_action_embed"]
    layers = list(module.proj.children())
    expected_fourier_dim = DEFAULT_ACTION_DIM * 2 * int(fae_cfg["n_frequencies"])
    assert layers[0].in_features == expected_fourier_dim
    assert layers[0].out_features == int(fae_cfg["out_dim"])
    assert layers[2].out_features == int(fae_cfg["out_dim"])


def test_from_canonical_none_arg():
    """Calling without an arg should still return a valid module."""
    module = FourierActionEmbedding.from_canonical()
    assert isinstance(module, FourierActionEmbedding)
    out = module(torch.randn(2, 2))
    assert out.shape == (2, 384)


# ---------------------------------------------------------------------------
# Edge cases / numerical
# ---------------------------------------------------------------------------


def test_zero_input_finite():
    module = FourierActionEmbedding().eval()
    out = module(torch.zeros(4, 2))
    assert torch.isfinite(out).all()


def test_extreme_values_finite():
    module = FourierActionEmbedding().eval()
    action = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    out = module(action)
    assert torch.isfinite(out).all()
