"""Tests for ``encoders.clip_enc.CLIPB32Wrapper``.

Whole module skips when open_clip isn't installed (the canonical contract
CI gate intentionally doesn't pull heavy deps).

Fast tests use ``pretrained=None`` (random init, no download). open_clip
emits a single ``WARNING:root:No pretrained weights loaded ...`` line in
that mode — that's expected and intentionally not suppressed (it would
mask a real misconfiguration in production code). The single
``RUN_SLOW_TESTS=1`` test exercises the full ``pretrained="openai"`` path.
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("open_clip")  # whole module skips cleanly without it

from torch import nn  # noqa: E402

from encoders.base import BaseEncoderWrapper  # noqa: E402
from encoders.clip_enc import (  # noqa: E402
    MODEL_NAME,
    NATIVE_DIM,
    PRETRAINED,
    CLIPB32Wrapper,
)


def _batch(b: int = 4, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, 3, 224, 224, generator=g)


@pytest.fixture(scope="module")
def wrapper() -> CLIPB32Wrapper:
    """One ``pretrained=None`` instance reused across fast tests."""
    return CLIPB32Wrapper(pretrained=None).eval()


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
# Frozen backbone
# ---------------------------------------------------------------------------


def test_all_backbone_params_have_no_grad(wrapper):
    backbone_params = list(wrapper.backbone.named_parameters())
    # CLIP ViT-B/32's visual transformer has ~88M params; if this is empty,
    # the backbone wasn't loaded.
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
# Adapter — CLIP DOES need projection (512 → 384)
# ---------------------------------------------------------------------------


def test_adapter_is_linear_512_to_384_no_bias(wrapper):
    assert isinstance(wrapper.adapter, nn.Linear)
    assert wrapper.adapter.in_features == 512
    assert wrapper.adapter.out_features == 384
    assert wrapper.adapter.bias is None  # bias=False matches encoder convention


def test_adapter_params_require_grad(wrapper):
    params = list(wrapper.adapter.parameters())
    assert params, "adapter should expose at least one parameter"
    for p in params:
        assert p.requires_grad


def test_trainable_parameters_yields_only_adapter_params(wrapper):
    expected = {id(p) for p in wrapper.adapter.parameters()}
    actual = {id(p) for p in wrapper.trainable_parameters()}
    assert actual == expected


def test_inherits_from_base(wrapper):
    assert isinstance(wrapper, BaseEncoderWrapper)


def test_adapter_receives_gradient_through_loss(wrapper):
    """Confirms backprop reaches the adapter despite the no_grad encoder."""
    for p in wrapper.adapter.parameters():
        p.grad = None
    out = wrapper(_batch())
    out.sum().backward()
    grad = wrapper.adapter.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0
    for name, p in wrapper.backbone.named_parameters():
        assert p.grad is None, f"backbone param {name!r} accumulated a gradient"


# ---------------------------------------------------------------------------
# Text encoder dropped (memory optimization vs spec)
# ---------------------------------------------------------------------------


def test_text_encoder_is_dropped(wrapper):
    """Wrapper retains only the visual transformer; text side is not loaded.

    These attributes (``token_embedding``, ``transformer``, ``text_projection``)
    live on the parent ``CLIP`` module, not on its ``visual`` child. Their
    absence on ``self.backbone`` confirms we kept only the visual half.
    """
    assert not hasattr(wrapper.backbone, "token_embedding")
    assert not hasattr(wrapper.backbone, "text_projection")


# ---------------------------------------------------------------------------
# Normalization — CLIP-specific, NOT ImageNet
# ---------------------------------------------------------------------------


def test_normalization_buffers_registered(wrapper):
    assert hasattr(wrapper, "_image_mean")
    assert hasattr(wrapper, "_image_std")
    assert wrapper._image_mean.shape == (1, 3, 1, 1)
    assert wrapper._image_std.shape == (1, 3, 1, 1)
    assert (wrapper._image_std > 0).all()


def test_normalization_uses_openai_clip_stats(wrapper):
    """OpenAI CLIP-specific normalization, distinct from ImageNet."""
    expected_mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073]
    ).view(1, 3, 1, 1)
    expected_std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711]
    ).view(1, 3, 1, 1)
    assert torch.allclose(wrapper._image_mean, expected_mean)
    assert torch.allclose(wrapper._image_std, expected_std)


def test_normalization_differs_from_imagenet(wrapper):
    """Sanity: don't accidentally apply ImageNet stats to CLIP."""
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert not torch.allclose(wrapper._image_mean, imagenet_mean, atol=1e-3)
    assert not torch.allclose(wrapper._image_std, imagenet_std, atol=1e-3)


def test_normalization_buffers_are_non_persistent(wrapper):
    keys = wrapper.state_dict().keys()
    assert "_image_mean" not in keys
    assert "_image_std" not in keys


def test_normalization_actually_applied(wrapper):
    raw = _batch()
    pre_normalized = (raw - wrapper._image_mean) / wrapper._image_std
    out_raw = wrapper(raw)
    out_pre = wrapper(pre_normalized)
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


def test_model_name_uses_quickgelu_variant():
    """OpenAI weights were trained with QuickGELU; the non-quickgelu config
    silently mis-applies activations. See module docstring for rationale.
    """
    assert MODEL_NAME == "ViT-B-32-quickgelu"


def test_pretrained_tag_is_openai():
    assert PRETRAINED == "openai"


def test_native_dim_constant():
    assert NATIVE_DIM == 512


# ---------------------------------------------------------------------------
# Slow / network test — gated by env var
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason="Set RUN_SLOW_TESTS=1 to run tests that download pretrained weights.",
)
def test_pretrained_load_produces_nontrivial_embedding():
    """With real OpenAI CLIP weights, distinct inputs give distinct outputs."""
    enc = CLIPB32Wrapper(pretrained="openai").eval()
    x1 = _batch(b=2, seed=1)
    x2 = _batch(b=2, seed=999)
    y1 = enc(x1)
    y2 = enc(x2)
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()
    assert not torch.allclose(y1, y2, atol=1e-3)
