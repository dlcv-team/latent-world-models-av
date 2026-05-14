"""Tests for ``encoders.dinov2.DINOv2S14Wrapper``.

DINOv2 is loaded via ``torch.hub``, which clones
``facebookresearch/dinov2`` to ``~/.cache/torch/hub`` on first run. If
hub access fails (no network AND no cached repo), tests that need a
constructed wrapper are skipped with a clear message via the ``wrapper``
fixture; tests that don't need the model (e.g., constant identities)
still run.

Fast tests use ``pretrained=False`` to skip the ~80 MB weight download.
The single ``RUN_SLOW_TESTS=1`` test exercises the full pretrained path.
"""

from __future__ import annotations

import os
import warnings

import pytest
import torch
from torch import nn

from encoders.base import BaseEncoderWrapper
from encoders.dinov2 import MODEL_ID, NATIVE_DIM, REPO, DINOv2S14Wrapper


def _batch(b: int = 4, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, 3, 224, 224, generator=g)


@pytest.fixture(scope="module")
def wrapper() -> DINOv2S14Wrapper:
    """One ``pretrained=False`` instance reused across fast tests."""
    try:
        with warnings.catch_warnings():
            # xFormers warnings are info, not errors — DINOv2 falls back
            # to standard attention. Suppressed here just to keep test
            # output readable.
            warnings.filterwarnings(
                "ignore", message=".*xFormers is not available.*"
            )
            return DINOv2S14Wrapper(pretrained=False).eval()
    except Exception as exc:
        pytest.skip(
            f"DINOv2 hub load failed (no network and no cached repo?): {exc}"
        )


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
    # DINOv2-S/14 has ~22M params; if this is empty, the hub load is broken.
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
    assert list(wrapper.trainable_parameters()) == []


def test_inherits_from_base(wrapper):
    assert isinstance(wrapper, BaseEncoderWrapper)


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
    """Per upstream dinov2/data/transforms.py."""
    expected_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    expected_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.allclose(wrapper._image_mean, expected_mean)
    assert torch.allclose(wrapper._image_std, expected_std)


def test_normalization_buffers_are_non_persistent(wrapper):
    # Same rationale as ViT-S/16: rederivable at load time, no point
    # bloating probe checkpoints with them.
    keys = wrapper.state_dict().keys()
    assert "_image_mean" not in keys
    assert "_image_std" not in keys


def test_normalization_actually_applied(wrapper):
    """Pre-normalized input vs raw [0,1] input must give different outputs."""
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


def test_repo_constant():
    assert REPO == "facebookresearch/dinov2"


def test_model_id_constant():
    assert MODEL_ID == "dinov2_vits14"


def test_native_dim_constant():
    assert NATIVE_DIM == 384


def test_input_side_is_compatible_with_patch_14():
    """224 must be divisible by patch size (14) for clean tokenization."""
    assert 224 % 14 == 0


# ---------------------------------------------------------------------------
# Slow / network test — gated by env var
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason="Set RUN_SLOW_TESTS=1 to run tests that download pretrained weights.",
)
def test_pretrained_load_produces_nontrivial_embedding():
    """With real pretrained weights, distinct inputs give distinct outputs."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*xFormers is not available.*")
        enc = DINOv2S14Wrapper(pretrained=True).eval()
    x1 = _batch(b=2, seed=1)
    x2 = _batch(b=2, seed=999)
    y1 = enc(x1)
    y2 = enc(x2)
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()
    assert not torch.allclose(y1, y2, atol=1e-3)
