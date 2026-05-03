"""Tests for ``encoders.vits16.ViTS16Wrapper``.

The whole module is skipped when ``timm`` is not installed (the canonical
contract CI gate intentionally does not install heavy deps), so these
tests run locally and on any future encoder-specific CI job.

Fast tests use ``pretrained=False`` to skip the ~80 MB weight download.
The single ``RUN_SLOW_TESTS=1`` test below exercises the full
``pretrained=True`` path end-to-end.
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("timm")  # whole module skips cleanly without timm

from torch import nn  # noqa: E402  (deliberately after importorskip)

from encoders.base import BaseEncoderWrapper  # noqa: E402
from encoders.vits16 import MODEL_ID, NATIVE_DIM, ViTS16Wrapper  # noqa: E402


def _batch(b: int = 4, seed: int = 0) -> torch.Tensor:
    """Random ``[0, 1]`` images of the canonical shape."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, 3, 224, 224, generator=g)


@pytest.fixture(scope="module")
def wrapper() -> ViTS16Wrapper:
    """One ``pretrained=False`` instance reused across fast tests."""
    return ViTS16Wrapper(pretrained=False).eval()


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


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
# Frozen backbone — the second A4 requirement
# ---------------------------------------------------------------------------


def test_all_backbone_params_have_no_grad(wrapper):
    backbone_params = list(wrapper.backbone.named_parameters())
    # ViT-S/16 has ~22M params; if this is empty something is seriously
    # wrong with the timm load.
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
# Adapter — confirms no projection (native dim already matches target)
# ---------------------------------------------------------------------------


def test_no_projection_adapter(wrapper):
    assert isinstance(wrapper.adapter, nn.Identity)


def test_trainable_parameters_is_empty(wrapper):
    # No adapter, frozen backbone → optimizer sees nothing on the encoder.
    assert list(wrapper.trainable_parameters()) == []


def test_inherits_from_base(wrapper):
    assert isinstance(wrapper, BaseEncoderWrapper)


# ---------------------------------------------------------------------------
# Normalization buffers
# ---------------------------------------------------------------------------


def test_normalization_buffers_registered(wrapper):
    assert hasattr(wrapper, "_image_mean")
    assert hasattr(wrapper, "_image_std")
    assert wrapper._image_mean.shape == (1, 3, 1, 1)
    assert wrapper._image_std.shape == (1, 3, 1, 1)
    # std must be strictly positive or division explodes.
    assert (wrapper._image_std > 0).all()


def test_normalization_buffers_are_non_persistent(wrapper):
    # Should not be in state_dict — they're rederivable from timm at
    # load time, and persisting them would couple checkpoints to a
    # specific timm version.
    keys = wrapper.state_dict().keys()
    assert "_image_mean" not in keys
    assert "_image_std" not in keys


def test_normalization_actually_applied(wrapper):
    """Pre-normalized input vs raw [0,1] input must give different outputs."""
    raw = _batch()
    pre_normalized = (raw - wrapper._image_mean) / wrapper._image_std
    out_raw = wrapper(raw)
    out_pre = wrapper(pre_normalized)
    # If _encode skipped normalization, these would be equal. They mustn't be.
    assert not torch.allclose(out_raw, out_pre)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_forward_is_deterministic_for_fixed_input(wrapper):
    x = _batch()
    y1 = wrapper(x).detach().clone()
    y2 = wrapper(x).detach().clone()
    assert torch.allclose(y1, y2)


# ---------------------------------------------------------------------------
# Spec identity
# ---------------------------------------------------------------------------


def test_model_id_constant():
    assert MODEL_ID == "vit_small_patch16_224"


def test_native_dim_constant():
    assert NATIVE_DIM == 384


# ---------------------------------------------------------------------------
# Slow / network test — gated by env var so it stays out of fast runs
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason="Set RUN_SLOW_TESTS=1 to run tests that download pretrained weights.",
)
def test_pretrained_load_produces_nontrivial_embedding():
    """Sanity: with real ImageNet weights, distinct inputs give distinct outputs.

    A purely random Linear-from-pixels backbone could collapse two random
    inputs to nearly the same output by chance; real pretrained features
    should be distinguishable.
    """
    enc = ViTS16Wrapper(pretrained=True).eval()
    x1 = _batch(b=2, seed=1)
    x2 = _batch(b=2, seed=999)
    y1 = enc(x1)
    y2 = enc(x2)
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()
    assert not torch.allclose(y1, y2, atol=1e-3)
