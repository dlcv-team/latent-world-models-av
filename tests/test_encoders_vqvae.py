"""Tests for ``encoders.vqvae.VQVAEWrapper``.

The wrapper currently always routes to its DINOv2-S/14 fallback (no
working pretrained VQ checkpoint is available). These tests pin that
behavior, verify the wrapper plumbing under fallback mode, and confirm
the fallback metadata downstream code (figure renderers) will rely on.

Fast tests use ``pretrained=False`` to skip the ~80 MB DINOv2 weight
download. The single ``RUN_SLOW_TESTS=1`` test exercises the full
``pretrained=True`` path.

Tests that need a constructed wrapper depend on a fixture that catches
``torch.hub`` failures (no network AND no cached repo) and skips with a
clear message.
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

    Suppresses the expected :class:`VQFallbackUsed` warning during fixture
    construction; tests that specifically check the warning use a separate
    construction below so ``pytest.warns`` can observe it.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", VQFallbackUsed)
            warnings.filterwarnings(
                "ignore", message=".*xFormers is not available.*"
            )
            return VQVAEWrapper(pretrained=False).eval()
    except Exception as exc:
        pytest.skip(
            f"VQ wrapper construction failed (no network and no cached "
            f"DINOv2 hub repo?): {exc}"
        )


# ---------------------------------------------------------------------------
# Fallback signaling — the headline correctness for this wrapper
# ---------------------------------------------------------------------------


def test_construction_emits_vq_fallback_warning():
    """The wrapper must loudly announce that it's substituting DINOv2."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*xFormers is not available.*"
        )
        with pytest.warns(VQFallbackUsed, match="DINOv2"):
            VQVAEWrapper(pretrained=False)


def test_fallback_active_is_true(wrapper):
    assert wrapper.fallback_active is True


def test_fallback_caveat_returns_documented_text(wrapper):
    caveat = wrapper.fallback_caveat
    assert caveat == FALLBACK_CAVEAT
    # Substantive content checks — these are what the report/figures rely on.
    assert "DINOv2" in caveat
    assert "fallback" in caveat
    assert "independent VQ-VAE" in caveat


def test_fallback_caveat_is_module_level_constant():
    """Renderers can import FALLBACK_CAVEAT directly without instantiating."""
    assert isinstance(FALLBACK_CAVEAT, str)
    assert "DINOv2" in FALLBACK_CAVEAT


def test_vq_fallback_used_is_a_user_warning():
    """Downstream code can configure warnings filters by category."""
    assert issubclass(VQFallbackUsed, UserWarning)


# ---------------------------------------------------------------------------
# Output shape — same contract as every other encoder wrapper
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
# Adapter — no projection in fallback mode (DINOv2 native = target dim)
# ---------------------------------------------------------------------------


def test_no_projection_adapter_in_fallback_mode(wrapper):
    assert isinstance(wrapper.adapter, nn.Identity)


def test_trainable_parameters_is_empty(wrapper):
    assert list(wrapper.trainable_parameters()) == []


def test_inherits_from_base(wrapper):
    assert isinstance(wrapper, BaseEncoderWrapper)


# ---------------------------------------------------------------------------
# Fallback backbone identity — confirms we really did load DINOv2
# ---------------------------------------------------------------------------


def test_fallback_backbone_is_dinov2_architecture(wrapper):
    """The fallback backbone must be DINOv2-S/14, not some other 384-d ViT.

    Checks specific DINOv2 attributes rather than relying on type names
    (which are torch.hub-cached and could shift across versions).
    """
    # DINOv2 ViT-S/14 has embed_dim=384 and patch_size=14.
    assert getattr(wrapper.backbone, "embed_dim", None) == 384
    # The patch_embed module exposes patch_size as a tuple.
    patch_size = getattr(wrapper.backbone.patch_embed, "patch_size", None)
    assert patch_size == (14, 14) or patch_size == 14


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalization_buffers_registered(wrapper):
    assert hasattr(wrapper, "_image_mean")
    assert hasattr(wrapper, "_image_std")
    assert wrapper._image_mean.shape == (1, 3, 1, 1)
    assert wrapper._image_std.shape == (1, 3, 1, 1)
    assert (wrapper._image_std > 0).all()


def test_normalization_uses_imagenet_stats(wrapper):
    """DINOv2 fallback expects the same ImageNet stats as DINOv2S14Wrapper."""
    expected_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    expected_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.allclose(wrapper._image_mean, expected_mean)
    assert torch.allclose(wrapper._image_std, expected_std)


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
# Slow / network test — gated by env var
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason="Set RUN_SLOW_TESTS=1 to run tests that download pretrained weights.",
)
def test_pretrained_load_produces_nontrivial_embedding():
    """With real DINOv2 weights (the active fallback target), distinct
    inputs give distinct outputs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VQFallbackUsed)
        warnings.filterwarnings(
            "ignore", message=".*xFormers is not available.*"
        )
        enc = VQVAEWrapper(pretrained=True).eval()
    x1 = _batch(b=2, seed=1)
    x2 = _batch(b=2, seed=999)
    y1 = enc(x1)
    y2 = enc(x2)
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()
    assert not torch.allclose(y1, y2, atol=1e-3)
