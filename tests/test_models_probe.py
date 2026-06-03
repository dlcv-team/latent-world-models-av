"""Tests for :class:`models.probe.ActionProbe` and :func:`train_probe`."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import load_canonical
from models.probe import (
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_INPUT_DIM,
    DEFAULT_OUTPUT_DIM,
    ActionProbe,
    train_probe,
)


# ---------------------------------------------------------------------------
# Synthetic helpers — keep tests independent of nuScenes
# ---------------------------------------------------------------------------


class _IdentityEncoder(nn.Module):
    """Encoder stub that returns its input unchanged.

    Lets us test ``train_probe`` without a real wrapper. Has no params,
    matches the ``encoder(image) -> embedding`` contract for
    pre-computed embeddings.
    """

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return iter(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _LinearLabelDataset(Dataset):
    """Synthetic dataset where action = W @ embedding + b (a learnable map)."""

    def __init__(
        self,
        n: int = 64,
        input_dim: int = 384,
        output_dim: int = 2,
        seed: int = 0,
        return_dict: bool = False,
    ) -> None:
        g = torch.Generator().manual_seed(seed)
        self.embeddings = torch.randn(n, input_dim, generator=g)
        # Fixed but non-trivial label map.
        true_w = torch.randn(input_dim, output_dim, generator=g) * 0.1
        true_b = torch.randn(output_dim, generator=g) * 0.05
        self.actions = self.embeddings @ true_w + true_b
        self.return_dict = return_dict

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int):
        if self.return_dict:
            return {
                "image": self.embeddings[idx],
                "actions": self.actions[idx],
                "sample_token": f"tok-{idx:04d}",
                "scene_token": f"scene-tok-{idx % 2:04d}",
            }
        return self.embeddings[idx], self.actions[idx]


# ---------------------------------------------------------------------------
# ActionProbe — architecture and forward
# ---------------------------------------------------------------------------


def test_forward_8x384_to_8x2():
    """Headline A9 spec: ``(8, 384) -> (8, 2)``, no NaN."""
    probe = ActionProbe()
    x = torch.randn(8, 384)
    out = probe(x)
    assert out.shape == (8, 2)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("b", [1, 4, 16, 256])
def test_forward_handles_various_batch_sizes(b):
    probe = ActionProbe().eval()
    out = probe(torch.randn(b, 384))
    assert out.shape == (b, 2)


def test_default_constants_match_canonical_yaml_constants():
    """Module-level defaults must mirror the canonical config values.

    If someone changes one without the other, this guards against silent drift.
    """
    cfg = load_canonical()
    probe_cfg = cfg.probe()
    assert DEFAULT_INPUT_DIM == cfg.target_embedding_dim
    assert DEFAULT_HIDDEN_DIM == int(probe_cfg["hidden_dim"])
    assert DEFAULT_DROPOUT == float(probe_cfg["dropout"])
    assert DEFAULT_OUTPUT_DIM == int(probe_cfg["output_dim"])


def test_layer_structure_matches_spec():
    """Architecture: Linear(384,256) -> GELU -> Dropout(0.1) -> Linear(256,2)."""
    probe = ActionProbe()
    layers = list(probe.net.children())
    assert len(layers) == 4
    assert isinstance(layers[0], nn.Linear)
    assert layers[0].in_features == 384
    assert layers[0].out_features == 256
    assert layers[0].bias is not None  # probe DOES use bias
    assert isinstance(layers[1], nn.GELU)
    assert isinstance(layers[2], nn.Dropout)
    assert pytest.approx(layers[2].p) == 0.1
    assert isinstance(layers[3], nn.Linear)
    assert layers[3].in_features == 256
    assert layers[3].out_features == 2
    assert layers[3].bias is not None


def test_all_probe_params_require_grad():
    probe = ActionProbe()
    params = list(probe.parameters())
    assert len(params) == 4  # 2 linears × (weight, bias)
    for name, p in probe.named_parameters():
        assert p.requires_grad, f"probe param {name!r} should be trainable"


def test_output_dtype_is_float32():
    probe = ActionProbe()
    out = probe(torch.randn(2, 384))
    assert out.dtype == torch.float32


def test_eval_mode_forward_is_deterministic():
    """In eval mode dropout is off, so the same input must give the same output."""
    probe = ActionProbe().eval()
    x = torch.randn(4, 384)
    a = probe(x).detach().clone()
    b = probe(x).detach().clone()
    assert torch.allclose(a, b)


def test_train_mode_dropout_is_active():
    """Sanity check: train-mode dropout introduces variance."""
    probe = ActionProbe().train()
    x = torch.randn(64, 384)
    torch.manual_seed(0)
    a = probe(x).detach().clone()
    torch.manual_seed(1)
    b = probe(x).detach().clone()
    assert not torch.allclose(a, b)


# ---------------------------------------------------------------------------
# from_canonical
# ---------------------------------------------------------------------------


def test_from_canonical_dimensions_match_config():
    cfg = load_canonical()
    probe = ActionProbe.from_canonical(cfg)
    layers = list(probe.net.children())
    assert layers[0].in_features == cfg.target_embedding_dim
    assert layers[0].out_features == cfg.probe()["hidden_dim"]
    assert pytest.approx(layers[2].p) == cfg.probe()["dropout"]
    assert layers[3].out_features == cfg.probe()["output_dim"]


def test_from_canonical_loads_config_when_none_passed():
    """Calling without an arg should still return a valid probe."""
    probe = ActionProbe.from_canonical()
    assert isinstance(probe, ActionProbe)
    out = probe(torch.randn(2, 384))
    assert out.shape == (2, 2)


# ---------------------------------------------------------------------------
# train_probe — synthetic regression
# ---------------------------------------------------------------------------


def _make_loaders(
    n_train: int = 64,
    n_val: int = 16,
    input_dim: int = 384,
    return_dict: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_ds = _LinearLabelDataset(
        n=n_train, input_dim=input_dim, seed=0, return_dict=return_dict
    )
    val_ds = _LinearLabelDataset(
        n=n_val, input_dim=input_dim, seed=42, return_dict=return_dict
    )
    return (
        DataLoader(train_ds, batch_size=16, shuffle=True),
        DataLoader(val_ds, batch_size=16, shuffle=False),
    )


def test_train_probe_loss_decreases_on_synthetic_regression():
    """Synthetic linear-label regression must train down quickly."""
    torch.manual_seed(0)
    encoder = _IdentityEncoder().eval()
    probe = ActionProbe(input_dim=384)
    train_loader, _ = _make_loaders()
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

    history = train_probe(
        encoder=encoder,
        probe=probe,
        train_loader=train_loader,
        optimizer=opt,
        epochs=30,
    )

    losses = history["train_loss"]
    assert len(losses) == 30
    assert all(l > 0 for l in losses)
    # Final epoch should be substantially better than the first. The 10×
    # bound is intentionally tight: Adam on this near-linear target should
    # easily clear it in 30 epochs, and a looser bound (e.g. 2×) would let
    # subtle regressions slip through — only stepping every other batch, or
    # accidentally training one of the two Linear layers, still halves loss.
    assert losses[-1] < 0.1 * losses[0], (
        f"loss did not decrease enough: first={losses[0]:.4f}, "
        f"last={losses[-1]:.4f}"
    )


def test_train_probe_returns_no_val_history_by_default():
    encoder = _IdentityEncoder().eval()
    probe = ActionProbe(input_dim=384)
    train_loader, _ = _make_loaders()
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

    history = train_probe(
        encoder=encoder,
        probe=probe,
        train_loader=train_loader,
        optimizer=opt,
        epochs=2,
    )
    assert history["val_loss"] is None


def test_train_probe_computes_val_loss_when_loader_provided():
    encoder = _IdentityEncoder().eval()
    probe = ActionProbe(input_dim=384)
    train_loader, val_loader = _make_loaders()
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

    history = train_probe(
        encoder=encoder,
        probe=probe,
        train_loader=train_loader,
        optimizer=opt,
        epochs=3,
        val_loader=val_loader,
    )
    assert history["val_loss"] is not None
    assert len(history["val_loss"]) == len(history["train_loss"]) == 3
    for v in history["val_loss"]:
        assert v > 0 and torch.isfinite(torch.tensor(v))


def test_train_probe_accepts_dict_batches():
    """Dataset's dict-style batches must work without an explicit unpacker."""
    encoder = _IdentityEncoder().eval()
    probe = ActionProbe(input_dim=384)
    train_loader, _ = _make_loaders(return_dict=True)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

    history = train_probe(
        encoder=encoder,
        probe=probe,
        train_loader=train_loader,
        optimizer=opt,
        epochs=2,
    )
    assert len(history["train_loss"]) == 2
    assert all(torch.isfinite(torch.tensor(l)) for l in history["train_loss"])


def test_train_probe_writes_log_csv(tmp_path):
    """sidecar contract: per-epoch CSV with epoch, train_loss, val_loss."""
    encoder = _IdentityEncoder().eval()
    probe = ActionProbe(input_dim=384)
    train_loader, val_loader = _make_loaders()
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    log_path = tmp_path / "outputs" / "train_log.csv"

    history = train_probe(
        encoder=encoder,
        probe=probe,
        train_loader=train_loader,
        optimizer=opt,
        epochs=4,
        val_loader=val_loader,
        log_csv_path=log_path,
    )

    assert log_path.exists()
    with log_path.open() as fh:
        rows = list(csv.reader(fh))

    assert rows[0] == ["epoch", "train_loss", "val_loss"]
    assert len(rows) == 5  # header + 4 epoch rows
    for i, row in enumerate(rows[1:]):
        epoch_idx, train_str, val_str = row
        assert int(epoch_idx) == i
        assert float(train_str) == pytest.approx(
            history["train_loss"][i], rel=1e-6
        )
        assert float(val_str) == pytest.approx(
            history["val_loss"][i], rel=1e-6
        )


def test_train_probe_writes_log_csv_with_blank_val_when_no_val_loader(tmp_path):
    encoder = _IdentityEncoder().eval()
    probe = ActionProbe(input_dim=384)
    train_loader, _ = _make_loaders()
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    log_path = tmp_path / "train_log.csv"

    train_probe(
        encoder=encoder,
        probe=probe,
        train_loader=train_loader,
        optimizer=opt,
        epochs=2,
        log_csv_path=log_path,
    )

    with log_path.open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["epoch", "train_loss", "val_loss"]
    for row in rows[1:]:
        assert row[2] == ""


def test_train_probe_rejects_zero_epochs():
    encoder = _IdentityEncoder().eval()
    probe = ActionProbe(input_dim=384)
    train_loader, _ = _make_loaders()
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="epochs"):
        train_probe(
            encoder=encoder,
            probe=probe,
            train_loader=train_loader,
            optimizer=opt,
            epochs=0,
        )


# ---------------------------------------------------------------------------
# Frozen-encoder integration — confirms the no_grad boundary still works
# ---------------------------------------------------------------------------


def test_train_probe_keeps_real_encoder_backbone_frozen():
    """After training, every backbone param of a real encoder is still frozen."""
    pytest.importorskip("timm")
    from encoders.vits16 import ViTS16Wrapper

    encoder = ViTS16Wrapper(pretrained=False).eval()
    probe = ActionProbe(input_dim=384)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

    # Tiny synthetic image-shaped loader.
    g = torch.Generator().manual_seed(0)
    images = torch.rand(8, 3, 224, 224, generator=g)
    actions = torch.randn(8, 2, generator=g)

    class _ImageActionDS(Dataset):
        def __len__(self):
            return images.shape[0]

        def __getitem__(self, idx):
            return images[idx], actions[idx]

    loader = DataLoader(_ImageActionDS(), batch_size=4)

    train_probe(
        encoder=encoder,
        probe=probe,
        train_loader=loader,
        optimizer=opt,
        epochs=2,
    )

    # Backbone params must still be frozen.
    for name, p in encoder.backbone.named_parameters():
        assert not p.requires_grad, (
            f"backbone param {name!r} became trainable after train_probe"
        )


def test_train_probe_updates_clip_adapter_when_passed_to_optimizer():
    """For projection encoders, gradient must reach the trainable adapter.

    Confirms ``BaseEncoderWrapper``'s ``no_grad`` boundary doesn't accidentally
    cut the adapter off from the loss.
    """
    pytest.importorskip("open_clip")
    from encoders.clip_enc import CLIPB32Wrapper

    encoder = CLIPB32Wrapper(pretrained=None).eval()
    probe = ActionProbe(input_dim=384)
    opt = torch.optim.Adam(
        list(probe.parameters()) + list(encoder.trainable_parameters()),
        lr=1e-3,
    )

    adapter_before = encoder.adapter.weight.detach().clone()

    g = torch.Generator().manual_seed(0)
    images = torch.rand(4, 3, 224, 224, generator=g)
    actions = torch.randn(4, 2, generator=g)

    class _ImageActionDS(Dataset):
        def __len__(self):
            return images.shape[0]

        def __getitem__(self, idx):
            return images[idx], actions[idx]

    loader = DataLoader(_ImageActionDS(), batch_size=2)

    train_probe(
        encoder=encoder,
        probe=probe,
        train_loader=loader,
        optimizer=opt,
        epochs=2,
    )

    adapter_after = encoder.adapter.weight.detach().clone()
    delta = (adapter_after - adapter_before).abs().sum().item()
    assert delta > 0, (
        "CLIP adapter weights did not move during training; gradient "
        "did not reach the adapter."
    )
