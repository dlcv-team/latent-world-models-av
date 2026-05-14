"""Tests for ``encoders.base.BaseEncoderWrapper``.

Properties verified:

* Output shape is ``(B, target_dim)`` regardless of native_dim, with or
  without projection.
* Every backbone parameter has ``requires_grad=False`` after construction.
* Adapter parameters have ``requires_grad=True`` when a projection is
  attached, and the adapter receives gradients through a downstream loss
  (confirming the ``torch.no_grad`` block does not break adapter training).
* ``Identity`` adapter is used when no projection is needed; in that case
  ``trainable_parameters()`` yields nothing.
* Backbone stays in eval mode regardless of ``train()`` / ``eval()`` calls
  on the wrapper.
* Misconfiguration (``needs_projection=False`` with dimension mismatch) and
  subclasses that forget to assign ``self.backbone`` raise clear errors.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from encoders.base import BaseEncoderWrapper


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _DummyEncoder(BaseEncoderWrapper):
    """Stand-in for a real frozen backbone: a Linear projecting flat pixels."""

    def _load(self) -> None:
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, self.native_dim),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class _MissingBackboneEncoder(BaseEncoderWrapper):
    """Buggy subclass that forgets to assign self.backbone."""

    def _load(self) -> None:  # bug: nothing assigned
        return

    def _encode(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return x


def _batch(b: int = 4) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(b, 3, 224, 224)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


def test_forward_returns_target_dim_with_projection():
    enc = _DummyEncoder(native_dim=512, needs_projection=True)
    out = enc(_batch())
    assert out.shape == (4, 384)


def test_forward_returns_target_dim_without_projection():
    enc = _DummyEncoder(native_dim=384, needs_projection=False)
    out = enc(_batch())
    assert out.shape == (4, 384)


@pytest.mark.parametrize("native", [256, 384, 512, 768, 1024])
def test_forward_handles_all_canonical_native_dims(native):
    enc = _DummyEncoder(native_dim=native, needs_projection=(native != 384))
    out = enc(_batch(b=2))
    assert out.shape == (2, 384)


# ---------------------------------------------------------------------------
# Frozen backbone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "native_dim,needs_projection",
    [(384, False), (512, True), (1024, True)],
)
def test_backbone_params_have_no_grad(native_dim, needs_projection):
    enc = _DummyEncoder(native_dim=native_dim, needs_projection=needs_projection)
    backbone_params = list(enc.backbone.named_parameters())
    assert backbone_params, "test fixture should expose at least one backbone param"
    for name, p in backbone_params:
        assert not p.requires_grad, f"backbone param {name!r} is trainable"


def test_backbone_starts_in_eval_mode():
    enc = _DummyEncoder(native_dim=512, needs_projection=True)
    assert not enc.backbone.training


# ---------------------------------------------------------------------------
# Trainable adapter
# ---------------------------------------------------------------------------


def test_adapter_is_linear_no_bias_when_projection_needed():
    enc = _DummyEncoder(native_dim=512, needs_projection=True)
    assert isinstance(enc.adapter, nn.Linear)
    assert enc.adapter.in_features == 512
    assert enc.adapter.out_features == 384
    assert enc.adapter.bias is None  # bias=False matches encoder convention


def test_adapter_params_require_grad_when_present():
    enc = _DummyEncoder(native_dim=512, needs_projection=True)
    params = list(enc.adapter.parameters())
    assert params, "adapter should expose at least one parameter"
    for p in params:
        assert p.requires_grad


def test_adapter_is_identity_when_not_needed():
    enc = _DummyEncoder(native_dim=384, needs_projection=False)
    assert isinstance(enc.adapter, nn.Identity)


def test_trainable_parameters_yields_only_adapter_params():
    enc = _DummyEncoder(native_dim=512, needs_projection=True)
    expected = {id(p) for p in enc.adapter.parameters()}
    actual = {id(p) for p in enc.trainable_parameters()}
    assert actual == expected


def test_trainable_parameters_is_empty_without_adapter():
    enc = _DummyEncoder(native_dim=384, needs_projection=False)
    assert list(enc.trainable_parameters()) == []


def test_adapter_receives_gradient_through_downstream_loss():
    """Gradient must reach the adapter even though the encoder runs under no_grad."""
    enc = _DummyEncoder(native_dim=512, needs_projection=True)
    out = enc(_batch())
    out.sum().backward()

    grad = enc.adapter.weight.grad
    assert grad is not None, "adapter weight has no .grad after backward()"
    assert torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0, "adapter gradient is identically zero"

    for name, p in enc.backbone.named_parameters():
        assert p.grad is None, f"backbone param {name!r} accumulated a gradient"


def test_no_grad_path_when_no_adapter():
    """Without an adapter, the wrapper output has no gradient path at all."""
    enc = _DummyEncoder(native_dim=384, needs_projection=False)
    out = enc(_batch())
    assert out.requires_grad is False


# ---------------------------------------------------------------------------
# Eval-mode locking
# ---------------------------------------------------------------------------


def test_backbone_stays_in_eval_after_wrapper_train():
    enc = _DummyEncoder(native_dim=512, needs_projection=True)
    enc.train()
    assert enc.training is True
    assert not enc.backbone.training, "backbone should remain in eval mode"


def test_backbone_stays_in_eval_after_train_then_eval():
    enc = _DummyEncoder(native_dim=512, needs_projection=True)
    enc.train()
    enc.eval()
    assert not enc.backbone.training


# ---------------------------------------------------------------------------
# Misconfiguration
# ---------------------------------------------------------------------------


def test_needs_projection_false_with_dim_mismatch_raises():
    with pytest.raises(ValueError, match="needs_projection=False"):
        _DummyEncoder(native_dim=512, needs_projection=False)


def test_subclass_missing_backbone_raises():
    with pytest.raises(RuntimeError, match="must assign self.backbone"):
        _MissingBackboneEncoder(native_dim=512, needs_projection=True)


# ---------------------------------------------------------------------------
# Determinism sanity check
# ---------------------------------------------------------------------------


def test_forward_is_deterministic_for_fixed_input():
    """With a frozen backbone in eval mode, the wrapper must be deterministic."""
    torch.manual_seed(0)
    enc = _DummyEncoder(native_dim=512, needs_projection=True)
    enc.eval()

    x = _batch()
    y1 = enc(x).detach().clone()
    y2 = enc(x).detach().clone()

    assert torch.allclose(y1, y2)
