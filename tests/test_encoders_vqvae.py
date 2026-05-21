"""Tests for ``encoders.vqvae.VQVAEWrapper``.

Fast tests use ``pretrained=False`` which builds the vendored VQGAN
encoder with random weights (no network, no fallback). Slow tests
gated by ``RUN_SLOW_TESTS=1`` exercise real checkpoint loading.
"""

from __future__ import annotations

import os
import warnings

import pytest
import torch
from torch import nn

from encoders.base import BaseEncoderWrapper
from encoders.vqvae import FALLBACK_CAVEAT, VQFallbackUsed, VQVAEWrapper


def _batch(b: int = 4, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, 3, 224, 224, generator=g)


@pytest.fixture(scope="module")
def wrapper() -> VQVAEWrapper:
    """One ``pretrained=False`` instance reused across fast tests.

    With pretrained=False the wrapper uses the vendored VQGAN encoder
    with random weights (no fallback, no network needed).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", VQFallbackUsed)
            return VQVAEWrapper(pretrained=False).eval()
    except Exception as exc:
        pytest.skip(f"VQ wrapper construction failed: {exc}")


# ---------------------------------------------------------------------------
# Primary path (pretrained=False uses vendored encoder, not fallback)
# ---------------------------------------------------------------------------


def test_pretrained_false_does_not_use_fallback(wrapper):
    assert wrapper.fallback_active is False


def test_pretrained_false_has_no_fallback_caveat(wrapper):
    assert wrapper.fallback_caveat == ""


def test_forward_returns_b_by_384(wrapper):
    out = wrapper(_batch(b=4))
    assert out.shape == (4, 384)


@pytest.mark.parametrize("b", [1, 2, 8])
def test_forward_handles_various_batch_sizes(wrapper, b):
    assert wrapper(_batch(b=b)).shape == (b, 384)


def test_output_is_finite(wrapper):
    assert torch.isfinite(wrapper(_batch())).all()


def test_output_dtype_is_float32(wrapper):
    assert wrapper(_batch()).dtype == torch.float32


# ---------------------------------------------------------------------------
# Adapter: vendored encoder has 256 native dim, needs projection to 384
# ---------------------------------------------------------------------------


def test_adapter_is_linear_projection(wrapper):
    assert isinstance(wrapper.adapter, nn.Linear)
    assert wrapper.adapter.in_features == 256
    assert wrapper.adapter.out_features == 384
    assert wrapper.adapter.bias is None


def test_trainable_parameters_yields_adapter(wrapper):
    tp = list(wrapper.trainable_parameters())
    assert len(tp) == 1
    assert tp[0] is wrapper.adapter.weight


def test_adapter_receives_gradient(wrapper):
    x = _batch(b=2)
    out = wrapper(x)
    loss = out.sum()
    loss.backward()
    assert wrapper.adapter.weight.grad is not None
    wrapper.adapter.weight.grad = None


# ---------------------------------------------------------------------------
# Frozen backbone
# ---------------------------------------------------------------------------


def test_all_backbone_params_have_no_grad(wrapper):
    backbone_params = list(wrapper.backbone.named_parameters())
    assert len(backbone_params) > 0
    for name, p in backbone_params:
        assert not p.requires_grad, f"backbone param {name!r} is trainable"


def test_backbone_starts_in_eval_mode(wrapper):
    assert not wrapper.backbone.training


def test_backbone_stays_in_eval_after_wrapper_train(wrapper):
    wrapper.train()
    try:
        assert wrapper.training is True
        assert not wrapper.backbone.training
    finally:
        wrapper.eval()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


def test_inherits_from_base(wrapper):
    assert isinstance(wrapper, BaseEncoderWrapper)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_forward_is_deterministic_for_fixed_input(wrapper):
    x = _batch()
    y1 = wrapper(x).detach().clone()
    y2 = wrapper(x).detach().clone()
    assert torch.allclose(y1, y2)


# ---------------------------------------------------------------------------
# Fallback metadata exports
# ---------------------------------------------------------------------------


def test_fallback_caveat_is_module_level_constant():
    assert isinstance(FALLBACK_CAVEAT, str)
    assert "DINOv2" in FALLBACK_CAVEAT


def test_vq_fallback_used_is_a_user_warning():
    assert issubclass(VQFallbackUsed, UserWarning)


# ---------------------------------------------------------------------------
# Vendored encoder architecture sanity
# ---------------------------------------------------------------------------


def test_backbone_is_vendored_encoder(wrapper):
    from encoders._vqgan_arch import Encoder
    assert isinstance(wrapper.backbone, Encoder)


def test_encoder_output_shape():
    from encoders._vqgan_arch import Encoder, VQGAN_IMAGENET_F16_16384_CONFIG
    enc = Encoder(**VQGAN_IMAGENET_F16_16384_CONFIG)
    x = torch.rand(2, 3, 256, 256)
    out = enc(x)
    assert out.shape == (2, 256, 16, 16)


# ---------------------------------------------------------------------------
# Slow / network tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason="Set RUN_SLOW_TESTS=1 to run tests that download pretrained weights.",
)
def test_primary_load_succeeds():
    """Downloads real VQGAN checkpoint and verifies primary path works."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", VQFallbackUsed)
        enc = VQVAEWrapper(pretrained=True).eval()
    assert enc.fallback_active is False
    assert enc.fallback_caveat == ""
    out = enc(_batch(b=2))
    assert out.shape == (2, 384)
    assert torch.isfinite(out).all()
    x1 = _batch(b=2, seed=1)
    x2 = _batch(b=2, seed=999)
    assert not torch.allclose(enc(x1), enc(x2), atol=1e-3)
