"""Tests for ``encoders.vjepa2.VJEPA2Wrapper``.

Whole module skips when the ``transformers`` library isn't installed.

V-JEPA2 ViT-L is large (~300 M params); even a forward pass with random
weights takes a few seconds on CPU. Tests use small inputs (b=1, t=8)
where shape doesn't matter; the headline ``(4, 16, 3, 224, 224)`` test
runs once.

Fast tests use ``pretrained=False`` (random init via
``AutoModel.from_config``; only the small config file is downloaded).
The slow test gated by ``RUN_SLOW_TESTS=1`` downloads the real ~1.5 GB
checkpoint.
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("transformers")

from torch import nn  # noqa: E402

from encoders.base import BaseEncoderWrapper  # noqa: E402
from encoders.vjepa2 import (  # noqa: E402
    MODEL_ID,
    NATIVE_DIM,
    NATIVE_INPUT_SIZE,
    VJEPA2Wrapper,
)


def _clip(
    b: int = 1,
    t: int = 8,
    h: int = 224,
    w: int = 224,
    seed: int = 0,
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, t, 3, h, w, generator=g)


@pytest.fixture(scope="module")
def wrapper() -> VJEPA2Wrapper:
    """One ``pretrained=False`` instance reused across fast tests."""
    try:
        return VJEPA2Wrapper(pretrained=False).eval()
    except Exception as exc:
        pytest.skip(
            f"V-JEPA2 wrapper construction failed (no network and no "
            f"cached transformers config?): {exc}"
        )


# ---------------------------------------------------------------------------
# Output shape — the headline requirement
# ---------------------------------------------------------------------------


def test_forward_returns_b_by_384_at_canonical_shape(wrapper):
    """Spec test: ``(4, 16, 3, 224, 224) -> (4, 384)``. The expensive one."""
    out = wrapper(_clip(b=4, t=16))
    assert out.shape == (4, 384)


def test_forward_returns_b_by_384_at_small_shape(wrapper):
    out = wrapper(_clip(b=2, t=8))
    assert out.shape == (2, 384)


def test_output_is_finite(wrapper):
    assert torch.isfinite(wrapper(_clip())).all()


def test_output_dtype_is_float32(wrapper):
    assert wrapper(_clip()).dtype == torch.float32


# ---------------------------------------------------------------------------
# Temporal-axis input — the structural difference vs the 4 single-frame wrappers
# ---------------------------------------------------------------------------


def test_input_is_5d_with_temporal_axis(wrapper):
    """Spot-check: V-JEPA2 expects (B, T, 3, H, W), not (B, 3, H, W)."""
    # A 4-D input (no temporal axis) must NOT silently succeed.
    bad = torch.rand(2, 3, 224, 224)
    with pytest.raises(Exception):  # noqa: B017 — any exception is fine
        wrapper(bad)


def test_accepts_other_temporal_lengths(wrapper):
    """T != 16 should still produce (B, 384) via temporal pos interpolation."""
    out = wrapper(_clip(b=1, t=4))
    assert out.shape == (1, 384)


def test_accepts_native_spatial_size(wrapper):
    """No-op resize when input already at the model's native 256×256."""
    out = wrapper(_clip(b=1, t=8, h=NATIVE_INPUT_SIZE, w=NATIVE_INPUT_SIZE))
    assert out.shape == (1, 384)


# ---------------------------------------------------------------------------
# Frozen backbone
# ---------------------------------------------------------------------------


def test_all_backbone_params_have_no_grad(wrapper):
    backbone_params = list(wrapper.backbone.named_parameters())
    # ViT-L scale; if this is empty, the AutoModel load is broken.
    assert len(backbone_params) > 100
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
# Adapter — V-JEPA2 needs the largest projection of the five (1024 → 384)
# ---------------------------------------------------------------------------


def test_adapter_is_linear_1024_to_384_no_bias(wrapper):
    assert isinstance(wrapper.adapter, nn.Linear)
    assert wrapper.adapter.in_features == 1024
    assert wrapper.adapter.out_features == 384
    assert wrapper.adapter.bias is None


def test_adapter_params_require_grad(wrapper):
    params = list(wrapper.adapter.parameters())
    assert params
    for p in params:
        assert p.requires_grad


def test_trainable_parameters_yields_only_adapter(wrapper):
    expected = {id(p) for p in wrapper.adapter.parameters()}
    actual = {id(p) for p in wrapper.trainable_parameters()}
    assert actual == expected


def test_inherits_from_base(wrapper):
    assert isinstance(wrapper, BaseEncoderWrapper)


def test_adapter_receives_gradient_through_loss(wrapper):
    for p in wrapper.adapter.parameters():
        p.grad = None
    out = wrapper(_clip(b=1, t=8))
    out.sum().backward()
    grad = wrapper.adapter.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0
    for name, p in wrapper.backbone.named_parameters():
        assert p.grad is None, f"backbone param {name!r} accumulated a gradient"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalization_buffers_registered(wrapper):
    assert hasattr(wrapper, "_image_mean")
    assert hasattr(wrapper, "_image_std")
    # Shape broadcasts against (B, T, 3, H, W).
    assert wrapper._image_mean.shape == (1, 1, 3, 1, 1)
    assert wrapper._image_std.shape == (1, 1, 3, 1, 1)
    assert (wrapper._image_std > 0).all()


def test_normalization_uses_imagenet_stats(wrapper):
    """Sanity check: wrapper holds the standard ImageNet stats.

    This pins the wrapper to its own literal — it catches accidental edits
    (someone types 0.495 instead of 0.485) but does NOT verify the docstring
    claim that the values match the official ``VJEPA2VideoProcessor``. That
    upstream check lives in ``test_normalization_matches_upstream_processor``
    below, gated by ``RUN_SLOW_TESTS=1`` because it requires a network fetch.
    """
    expected_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    expected_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    assert torch.allclose(wrapper._image_mean, expected_mean)
    assert torch.allclose(wrapper._image_std, expected_std)


def test_normalization_buffers_are_non_persistent(wrapper):
    keys = wrapper.state_dict().keys()
    assert "_image_mean" not in keys
    assert "_image_std" not in keys


def test_normalization_actually_applied(wrapper):
    raw = _clip(b=1, t=8)
    pre_normalized = (raw - wrapper._image_mean) / wrapper._image_std
    out_raw = wrapper(raw)
    out_pre = wrapper(pre_normalized)
    assert not torch.allclose(out_raw, out_pre)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_forward_is_deterministic_for_fixed_input(wrapper):
    x = _clip(b=1, t=8)
    y1 = wrapper(x).detach().clone()
    y2 = wrapper(x).detach().clone()
    assert torch.allclose(y1, y2)


# ---------------------------------------------------------------------------
# Spec identity
# ---------------------------------------------------------------------------


def test_model_id_constant():
    assert MODEL_ID == "facebook/vjepa2-vitl-fpc64-256"


def test_native_dim_constant():
    assert NATIVE_DIM == 1024


def test_native_input_size_constant():
    assert NATIVE_INPUT_SIZE == 256


# ---------------------------------------------------------------------------
# Slow / network test — gated by env var. Downloads ~1.5 GB.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason=(
        "Set RUN_SLOW_TESTS=1 to run tests that download pretrained weights. "
        "Note: V-JEPA2 ViT-L is ~1.5 GB."
    ),
)
def test_pretrained_load_produces_nontrivial_embedding():
    enc = VJEPA2Wrapper(pretrained=True).eval()
    x1 = _clip(b=1, t=8, seed=1)
    x2 = _clip(b=1, t=8, seed=999)
    y1 = enc(x1)
    y2 = enc(x2)
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()
    assert not torch.allclose(y1, y2, atol=1e-3)


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason=(
        "Set RUN_SLOW_TESTS=1 to run tests that fetch from HuggingFace. "
        "VJEPA2VideoProcessor's preprocessor_config.json is small (~KB), "
        "but the test still needs network access."
    ),
)
def test_normalization_matches_upstream_processor():
    """The wrapper's normalization stats must match the official processor.

    Pulls ``VJEPA2VideoProcessor.from_pretrained(MODEL_ID)`` and compares
    its ``image_mean`` / ``image_std`` against what the wrapper hardcodes.
    This is the real contract the docstring claims; the fast test only
    pins the wrapper to its own literal.
    """
    from transformers import VJEPA2VideoProcessor

    processor = VJEPA2VideoProcessor.from_pretrained(MODEL_ID)
    wrapper = VJEPA2Wrapper(pretrained=False).eval()

    expected_mean = torch.tensor(processor.image_mean).view(1, 1, 3, 1, 1)
    expected_std = torch.tensor(processor.image_std).view(1, 1, 3, 1, 1)
    assert torch.allclose(wrapper._image_mean, expected_mean), (
        f"wrapper mean {wrapper._image_mean.flatten().tolist()} != "
        f"upstream {processor.image_mean}"
    )
    assert torch.allclose(wrapper._image_std, expected_std), (
        f"wrapper std {wrapper._image_std.flatten().tolist()} != "
        f"upstream {processor.image_std}"
    )
