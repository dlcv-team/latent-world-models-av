"""Tests for :class:`models.bc_baseline.BCBaseline` and :func:`train_bc`."""

from __future__ import annotations

import csv
from typing import Iterator

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import load_canonical
from models import precompute_embeddings
from models.bc_baseline import (
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_INPUT_DIM,
    DEFAULT_OUTPUT_DIM,
    BCBaseline,
    train_bc,
)


# ---------------------------------------------------------------------------
# Synthetic helpers — keep tests independent of nuScenes and pretrained weights
# ---------------------------------------------------------------------------


class _IdentityEncoder(nn.Module):
    """Encoder stub that returns its input unchanged.

    Lets us exercise :func:`train_bc` without instantiating a real wrapper.
    """

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return iter(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _LinearLabelDataset(Dataset):
    """Synthetic dataset where ``actions = W @ embedding + b``.

    Train and val splits should share ``(W, b)`` so that what the head
    learns on train transfers to val; ``_make_loaders`` enforces this.
    """

    def __init__(
        self,
        n: int,
        input_dim: int,
        true_w: torch.Tensor,
        true_b: torch.Tensor,
        sample_seed: int,
        return_dict: bool = False,
    ) -> None:
        g = torch.Generator().manual_seed(sample_seed)
        self.embeddings = torch.randn(n, input_dim, generator=g)
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
            }
        return self.embeddings[idx], self.actions[idx]


def _make_loaders(
    n_train: int = 128,
    n_val: int = 32,
    input_dim: int = 384,
    output_dim: int = 2,
    batch_size: int = 16,
    return_dict: bool = False,
    label_seed: int = 17,
    train_sample_seed: int = 0,
    val_sample_seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(label_seed)
    true_w = torch.randn(input_dim, output_dim, generator=g) * 0.1
    true_b = torch.randn(output_dim, generator=g) * 0.05

    train_ds = _LinearLabelDataset(
        n=n_train,
        input_dim=input_dim,
        true_w=true_w,
        true_b=true_b,
        sample_seed=train_sample_seed,
        return_dict=return_dict,
    )
    val_ds = _LinearLabelDataset(
        n=n_val,
        input_dim=input_dim,
        true_w=true_w,
        true_b=true_b,
        sample_seed=val_sample_seed,
        return_dict=return_dict,
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def _adam(bc: BCBaseline, lr: float = 1e-3) -> torch.optim.Adam:
    return torch.optim.Adam(bc.parameters(), lr=lr)


# ---------------------------------------------------------------------------
# BCBaseline — architecture and forward
# ---------------------------------------------------------------------------


def test_forward_8x384_to_8x2():
    """Headline C3 spec: ``(8, 384) -> (8, 2)``, no NaN."""
    bc = BCBaseline()
    x = torch.randn(8, 384)
    out = bc(x)
    assert out.shape == (8, 2)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("b", [1, 4, 16, 256])
def test_forward_handles_various_batch_sizes(b):
    bc = BCBaseline().eval()
    out = bc(torch.randn(b, 384))
    assert out.shape == (b, 2)


def test_default_constants_match_canonical():
    """Module-level defaults must mirror the canonical config values."""
    cfg = load_canonical()
    probe_cfg = cfg.probe()
    assert DEFAULT_INPUT_DIM == cfg.target_embedding_dim
    assert DEFAULT_HIDDEN_DIM == int(probe_cfg["hidden_dim"])
    assert DEFAULT_DROPOUT == float(probe_cfg["dropout"])
    assert DEFAULT_OUTPUT_DIM == int(probe_cfg["output_dim"])


def test_layer_structure_matches_spec():
    """Architecture: ``Linear(384,256) -> GELU -> Dropout(0.1) -> Linear(256,2)``."""
    bc = BCBaseline()
    layers = list(bc.net.children())
    assert len(layers) == 4
    assert isinstance(layers[0], nn.Linear)
    assert layers[0].in_features == 384
    assert layers[0].out_features == 256
    assert layers[0].bias is not None
    assert isinstance(layers[1], nn.GELU)
    assert isinstance(layers[2], nn.Dropout)
    assert pytest.approx(layers[2].p) == 0.1
    assert isinstance(layers[3], nn.Linear)
    assert layers[3].in_features == 256
    assert layers[3].out_features == 2
    assert layers[3].bias is not None


def test_all_bc_params_require_grad():
    bc = BCBaseline()
    params = list(bc.parameters())
    assert len(params) == 4  # 2 Linears, each with weight + bias
    for name, p in bc.named_parameters():
        assert p.requires_grad, f"BC param {name!r} should be trainable"


def test_eval_mode_forward_is_deterministic():
    bc = BCBaseline().eval()
    x = torch.randn(4, 384)
    a = bc(x).detach().clone()
    b = bc(x).detach().clone()
    assert torch.allclose(a, b)


def test_train_mode_dropout_is_active():
    bc = BCBaseline().train()
    x = torch.randn(64, 384)
    torch.manual_seed(0)
    a = bc(x).detach().clone()
    torch.manual_seed(1)
    b = bc(x).detach().clone()
    assert not torch.allclose(a, b)


def test_gradients_flow_through_both_linears():
    bc = BCBaseline()
    loss = (bc(torch.randn(8, 384)) ** 2).mean()
    loss.backward()
    for name, p in bc.named_parameters():
        assert p.grad is not None, f"no gradient on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient on {name}"


# ---------------------------------------------------------------------------
# from_canonical
# ---------------------------------------------------------------------------


def test_from_canonical_dimensions_match_config():
    cfg = load_canonical()
    bc = BCBaseline.from_canonical(cfg)
    layers = list(bc.net.children())
    assert layers[0].in_features == cfg.target_embedding_dim
    assert layers[0].out_features == cfg.probe()["hidden_dim"]
    assert pytest.approx(layers[2].p) == cfg.probe()["dropout"]
    assert layers[3].out_features == cfg.probe()["output_dim"]


def test_from_canonical_loads_config_when_none_passed():
    bc = BCBaseline.from_canonical()
    assert isinstance(bc, BCBaseline)
    out = bc(torch.randn(2, 384))
    assert out.shape == (2, 2)


# ---------------------------------------------------------------------------
# train_bc
# ---------------------------------------------------------------------------


def test_train_bc_loss_decreases_on_synthetic_regression():
    """Synthetic linear-label regression must drive train loss down sharply."""
    torch.manual_seed(0)
    encoder = _IdentityEncoder().eval()
    bc = BCBaseline(input_dim=384)
    train_loader, val_loader = _make_loaders()

    history = train_bc(
        encoder=encoder,
        bc_model=bc,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=_adam(bc),
        epochs=30,
        patience=30,
    )

    train_losses = history["train_loss"]
    val_losses = history["val_loss"]
    assert len(train_losses) == 30
    assert len(val_losses) == 30
    assert all(l > 0 for l in train_losses)
    assert all(torch.isfinite(torch.tensor(l)) for l in val_losses)
    assert train_losses[-1] < 0.1 * train_losses[0], (
        f"train loss did not decrease enough: "
        f"first={train_losses[0]:.4f}, last={train_losses[-1]:.4f}"
    )


def test_train_bc_accepts_dict_batches():
    """Dataset-style ``{'image', 'actions'}`` dict batches must work."""
    encoder = _IdentityEncoder().eval()
    bc = BCBaseline(input_dim=384)
    train_loader, val_loader = _make_loaders(return_dict=True)

    history = train_bc(
        encoder=encoder,
        bc_model=bc,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=_adam(bc),
        epochs=2,
        patience=5,
    )
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2


def test_train_bc_early_stops_when_val_plateaus():
    """Constant-target val data plateaus immediately → early stop fires."""
    torch.manual_seed(0)
    encoder = _IdentityEncoder().eval()
    bc = BCBaseline(input_dim=384)
    train_loader, _ = _make_loaders()
    fixed_x = torch.zeros(4, 384)
    fixed_y = torch.zeros(4, 2)
    val_loader = DataLoader(list(zip(fixed_x, fixed_y)), batch_size=4)

    history = train_bc(
        encoder=encoder,
        bc_model=bc,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=_adam(bc),
        epochs=50,
        patience=2,
    )
    assert history["stopped_early"]
    assert len(history["train_loss"]) < 50
    assert history["best_epoch"] < len(history["train_loss"])


def test_train_bc_restores_best_epoch_weights():
    """After training, ``bc_model``'s val loss must equal ``best_val_loss``."""
    torch.manual_seed(0)
    encoder = _IdentityEncoder().eval()
    bc = BCBaseline(input_dim=384)
    train_loader, val_loader = _make_loaders()

    history = train_bc(
        encoder=encoder,
        bc_model=bc,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=_adam(bc),
        epochs=10,
        patience=10,
    )

    bc.eval()
    loss_fn = nn.MSELoss()
    total = 0.0
    n = 0
    with torch.no_grad():
        for x, y in val_loader:
            pred = bc(encoder(x))
            total += loss_fn(pred, y).item() * y.shape[0]
            n += y.shape[0]
    post_train_val_loss = total / n

    assert post_train_val_loss == pytest.approx(history["best_val_loss"], rel=1e-5)


def test_train_bc_default_patience_comes_from_canonical():
    """Omitting ``patience`` must behave the same as passing the canonical value."""
    cfg = load_canonical()
    canonical_patience = int(cfg.bc()["early_stopping_patience"])
    assert canonical_patience == 10

    def _run(patience):
        torch.manual_seed(0)
        bc = BCBaseline(input_dim=384)
        train_loader, val_loader = _make_loaders()
        return train_bc(
            encoder=_IdentityEncoder().eval(),
            bc_model=bc,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=_adam(bc),
            epochs=15,
            patience=patience,
        )

    default = _run(patience=None)
    explicit = _run(patience=canonical_patience)
    assert default["train_loss"] == explicit["train_loss"]
    assert default["val_loss"] == explicit["val_loss"]
    assert default["stopped_early"] == explicit["stopped_early"]


def test_train_bc_uses_supplied_cfg_for_patience_default(cfg):
    """Passing ``cfg=`` must avoid triggering an implicit ``load_canonical()``.

    Regression test for the hidden I/O concern raised in PR review:
    callers that already have the config in hand should be able to hand
    it in instead of forcing ``train_bc`` to re-read disk.
    """
    canonical_patience = int(cfg.bc()["early_stopping_patience"])

    def _run(**kwargs):
        torch.manual_seed(0)
        bc = BCBaseline(input_dim=384)
        train_loader, val_loader = _make_loaders()
        return train_bc(
            encoder=_IdentityEncoder().eval(),
            bc_model=bc,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=_adam(bc),
            epochs=15,
            **kwargs,
        )

    via_cfg = _run(cfg=cfg)
    via_explicit = _run(patience=canonical_patience)
    assert via_cfg["train_loss"] == via_explicit["train_loss"]
    assert via_cfg["val_loss"] == via_explicit["val_loss"]
    assert via_cfg["stopped_early"] == via_explicit["stopped_early"]


def test_train_bc_canonical_hyperparams_locked(cfg):
    """Spec contract: lr=1e-3, patience=10, epochs=50, MSE, Adam, wd=0."""
    bc_cfg = cfg.bc()
    assert bc_cfg["learning_rate"] == 1.0e-3
    assert bc_cfg["early_stopping_patience"] == 10
    assert bc_cfg["epochs"] == 50
    assert bc_cfg["loss"] == "mse"
    assert bc_cfg["optimizer"] == "adam"
    assert bc_cfg["weight_decay"] == 0.0


def test_train_bc_writes_log_csv(tmp_path):
    """Sidecar contract: per-epoch CSV with ``epoch, train_loss, val_loss``."""
    encoder = _IdentityEncoder().eval()
    bc = BCBaseline(input_dim=384)
    train_loader, val_loader = _make_loaders()
    log_path = tmp_path / "outputs" / "bc_train_log.csv"

    history = train_bc(
        encoder=encoder,
        bc_model=bc,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=_adam(bc),
        epochs=4,
        patience=10,
        log_csv_path=log_path,
    )

    assert log_path.exists()
    with log_path.open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["epoch", "train_loss", "val_loss"]
    assert len(rows) == 5
    for i, row in enumerate(rows[1:]):
        epoch_idx, train_str, val_str = row
        assert int(epoch_idx) == i
        assert float(train_str) == pytest.approx(history["train_loss"][i], rel=1e-6)
        assert float(val_str) == pytest.approx(history["val_loss"][i], rel=1e-6)


def test_train_bc_rejects_zero_epochs():
    encoder = _IdentityEncoder().eval()
    bc = BCBaseline(input_dim=384)
    train_loader, val_loader = _make_loaders()
    with pytest.raises(ValueError, match="epochs"):
        train_bc(
            encoder=encoder,
            bc_model=bc,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=_adam(bc),
            epochs=0,
        )


def test_train_bc_rejects_zero_patience():
    encoder = _IdentityEncoder().eval()
    bc = BCBaseline(input_dim=384)
    train_loader, val_loader = _make_loaders()
    with pytest.raises(ValueError, match="patience"):
        train_bc(
            encoder=encoder,
            bc_model=bc,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=_adam(bc),
            epochs=5,
            patience=0,
        )


# ---------------------------------------------------------------------------
# Pre-computed-embedding training path
# ---------------------------------------------------------------------------


class _CountingEncoder(nn.Module):
    """Identity encoder that tracks how many images it has been asked to embed.

    Used to verify that ``precompute_embeddings`` runs the encoder exactly
    once per sample regardless of how many BC epochs follow.
    """

    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
        self.images_seen = 0

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return iter(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        self.images_seen += x.shape[0]
        return x


def test_precompute_embeddings_runs_encoder_once_per_sample():
    """Sanity check: ``precompute_embeddings`` is a single sweep over the loader."""
    encoder = _CountingEncoder().eval()
    train_loader, _ = _make_loaders(
        n_train=64, n_val=16, input_dim=384, batch_size=16
    )

    cached = precompute_embeddings(encoder, train_loader)

    assert encoder.images_seen == 64
    assert encoder.forward_calls == 64 // 16  # one call per batch
    assert cached.tensors[0].shape == (64, 384)
    assert cached.tensors[1].shape == (64, 2)


def test_precompute_embeddings_supports_dict_batches():
    """The dict-style schema from ``NuScenesFrameDataset`` must work."""
    encoder = _IdentityEncoder().eval()
    loader, _ = _make_loaders(n_train=32, n_val=8, batch_size=8, return_dict=True)

    cached = precompute_embeddings(encoder, loader)

    assert cached.tensors[0].shape == (32, 384)
    assert cached.tensors[1].shape == (32, 2)


def _make_deterministic_loaders(
    n_train: int = 64,
    n_val: int = 16,
    batch_size: int = 16,
) -> tuple[DataLoader, DataLoader]:
    """Like ``_make_loaders`` but with ``shuffle=False`` so two passes over the
    same dataset produce the same batch sequence.

    Required for the cached-vs-live equivalence test: an order-randomized
    train loader would put the same samples in different gradient updates
    on the two paths and the trajectories would naturally diverge.
    """
    g = torch.Generator().manual_seed(17)
    true_w = torch.randn(384, 2, generator=g) * 0.1
    true_b = torch.randn(2, generator=g) * 0.05
    train_ds = _LinearLabelDataset(
        n=n_train,
        input_dim=384,
        true_w=true_w,
        true_b=true_b,
        sample_seed=0,
    )
    val_ds = _LinearLabelDataset(
        n=n_val,
        input_dim=384,
        true_w=true_w,
        true_b=true_b,
        sample_seed=42,
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def test_train_bc_on_precomputed_embeddings_matches_live_encoder():
    """Caching embeddings up front must yield the same training trajectory.

    This is the fast path called out in the PR review: pre-compute once,
    train the head on cached tensors. Setting ``dropout=0`` lets us drop
    the only stochastic op inside the head and demand byte-identical loss
    curves — exactly what callers should observe if they swap a live
    encoder for a cached :class:`~torch.utils.data.TensorDataset` built
    from :func:`precompute_embeddings`. (A non-zero dropout would still
    converge equivalently in expectation, but the per-step trajectories
    would diverge because :class:`~torch.utils.data.DataLoader` consumes
    one RNG value per ``__iter__`` call for worker seeding and the cached
    path issues two extra iter calls during pre-computation.)
    """
    cfg = load_canonical()
    train_loader, val_loader = _make_deterministic_loaders()

    def _run_live():
        torch.manual_seed(0)
        bc = BCBaseline(input_dim=384, dropout=0.0)
        return train_bc(
            encoder=_IdentityEncoder().eval(),
            bc_model=bc,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=_adam(bc),
            epochs=5,
            patience=10,
            cfg=cfg,
        )

    def _run_cached():
        torch.manual_seed(0)
        bc = BCBaseline(input_dim=384, dropout=0.0)
        encoder = _IdentityEncoder().eval()
        cached_train = precompute_embeddings(encoder, train_loader)
        cached_val = precompute_embeddings(encoder, val_loader)
        cached_train_loader = DataLoader(
            cached_train, batch_size=16, shuffle=False
        )
        cached_val_loader = DataLoader(cached_val, batch_size=16, shuffle=False)
        return train_bc(
            encoder=nn.Identity(),
            bc_model=bc,
            train_loader=cached_train_loader,
            val_loader=cached_val_loader,
            optimizer=_adam(bc),
            epochs=5,
            patience=10,
            cfg=cfg,
        )

    live = _run_live()
    cached = _run_cached()
    for k in ("train_loss", "val_loss"):
        for live_v, cached_v in zip(live[k], cached[k]):
            assert cached_v == pytest.approx(live_v, rel=1e-6, abs=1e-8)


def test_train_bc_on_precomputed_embeddings_only_runs_encoder_once():
    """End-to-end: caching must amortize the encoder forward over all epochs."""
    encoder = _CountingEncoder().eval()
    train_loader, val_loader = _make_loaders(
        n_train=32, n_val=8, batch_size=8
    )

    cached_train = precompute_embeddings(encoder, train_loader)
    cached_val = precompute_embeddings(encoder, val_loader)
    images_seen_after_cache = encoder.images_seen
    assert images_seen_after_cache == 32 + 8

    bc = BCBaseline(input_dim=384)
    train_bc(
        encoder=nn.Identity(),  # head trains on cached tensors only
        bc_model=bc,
        train_loader=DataLoader(cached_train, batch_size=8),
        val_loader=DataLoader(cached_val, batch_size=8),
        optimizer=_adam(bc),
        epochs=20,
        patience=5,
    )

    # The encoder must NOT be called again during training.
    assert encoder.images_seen == images_seen_after_cache


# ---------------------------------------------------------------------------
# Frozen-encoder integration — only runs when timm is available
# ---------------------------------------------------------------------------


def test_train_bc_keeps_real_encoder_backbone_frozen():
    pytest.importorskip("timm")
    from encoders.vits16 import ViTS16Wrapper

    encoder = ViTS16Wrapper(pretrained=False).eval()
    bc = BCBaseline(input_dim=384)

    g = torch.Generator().manual_seed(0)
    images = torch.rand(8, 3, 224, 224, generator=g)
    actions = torch.randn(8, 2, generator=g)

    class _ImageActionDS(Dataset):
        def __len__(self):
            return images.shape[0]

        def __getitem__(self, idx):
            return images[idx], actions[idx]

    loader = DataLoader(_ImageActionDS(), batch_size=4)

    train_bc(
        encoder=encoder,
        bc_model=bc,
        train_loader=loader,
        val_loader=loader,
        optimizer=torch.optim.Adam(
            list(bc.parameters()) + list(encoder.trainable_parameters()), lr=1e-3
        ),
        epochs=2,
        patience=5,
    )

    for name, p in encoder.backbone.named_parameters():
        assert not p.requires_grad, (
            f"backbone param {name!r} became trainable after train_bc"
        )
