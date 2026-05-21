"""Behavioral cloning baseline head, trained with early stopping.

(Member 3 task C3.)

The BC head is architecturally identical to
:class:`models.probe.ActionProbe` — the same
``Linear(384, 256) -> GELU -> Dropout(0.1) -> Linear(256, 2)`` MLP
mapping a 384-d encoder embedding to a canonical normalized
``(steer_norm, accel_norm)``. The two heads are kept as separate
classes because the canonical config splits them
(``probe`` vs ``bc_baseline``) and because the training regime
differs: the probe trains for a fixed 50 epochs without early stopping,
while the BC baseline early-stops on validation MSE with patience 10.

Hyperparameters are pinned in ``configs/canonical.yaml::bc_baseline``;
``BCBaseline.from_canonical`` is the recommended constructor.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Optional, Union

import torch
from torch import nn

from config import CanonicalConfig, load_canonical
from models._train_utils import _BatchLike, _epoch_loss

# Mirror canonical.yaml so tests can spin up a head without loading config.
# from_canonical is the real path.
DEFAULT_INPUT_DIM = 384
DEFAULT_HIDDEN_DIM = 256
DEFAULT_DROPOUT = 0.1
DEFAULT_OUTPUT_DIM = 2  # steer, accel


class BCBaseline(nn.Module):
    """Two-layer MLP BC head: ``Linear -> GELU -> Dropout -> Linear``.

    Parameters
    ----------
    input_dim
        Encoder embedding dimension. Defaults to 384 (project-wide
        ``target_embedding_dim``).
    hidden_dim
        Hidden width of the MLP. Defaults to 256 per canonical config.
    dropout
        Dropout probability after the GELU activation. Defaults to 0.1
        per canonical config.
    output_dim
        Number of action dimensions. Defaults to 2 (steer, accel).

    Notes
    -----
    Bias is enabled on both Linear layers, matching the probe convention.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
        output_dim: int = DEFAULT_OUTPUT_DIM,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.output_dim = int(output_dim)

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    @classmethod
    def from_canonical(
        cls, cfg: Optional[CanonicalConfig] = None
    ) -> "BCBaseline":
        """Construct from ``configs/canonical.yaml``.

        The BC head reuses the probe's layer dims by design: the
        ``bc_baseline`` block names only the architecture string and the
        training schedule, so dims (hidden, dropout, output) are pulled
        from the ``probe`` block.

        Parameters
        ----------
        cfg
            Optional pre-loaded :class:`CanonicalConfig`. If omitted, the
            canonical config is loaded from disk via :func:`load_canonical`.
        """
        if cfg is None:
            cfg = load_canonical()
        probe_cfg = cfg.probe()
        return cls(
            input_dim=cfg.target_embedding_dim,
            hidden_dim=int(probe_cfg["hidden_dim"]),
            dropout=float(probe_cfg["dropout"]),
            output_dim=int(probe_cfg["output_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map embedding ``(B, input_dim)`` to action prediction ``(B, output_dim)``."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_bc(
    encoder: nn.Module,
    bc_model: BCBaseline,
    train_loader: Iterable[_BatchLike],
    val_loader: Iterable[_BatchLike],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    *,
    patience: Optional[int] = None,
    cfg: Optional[CanonicalConfig] = None,
    device: Optional[torch.device] = None,
    log_csv_path: Optional[Union[Path, str]] = None,
) -> dict:
    """Train ``bc_model`` on top of a frozen ``encoder`` with early stopping.

    Parameters
    ----------
    encoder
        A :class:`encoders.base.BaseEncoderWrapper` instance (or any
        ``nn.Module`` that maps an input batch to a 2-D embedding tensor
        and whose backbone is already frozen). Gradients flow only into
        the encoder's adapter (when present) and into the BC head.

        For the full nuScenes-subset BC sweep (6 encoders × 3 seeds), the
        cheaper pattern is to pre-compute embeddings once with
        :func:`models.precompute_embeddings`, build a
        :class:`torch.utils.data.DataLoader` over the resulting
        :class:`torch.utils.data.TensorDataset`, and pass an
        :class:`torch.nn.Identity` here. Each early-stopping epoch then
        costs ~MLP-forward time instead of an encoder forward — typically
        2-3 orders of magnitude faster. See
        :mod:`models._train_utils` for the adapter caveat.
    bc_model
        The :class:`BCBaseline` head to train.
    train_loader, val_loader
        Iterables yielding either ``{"image": ..., "actions": ...}``
        dicts (the :class:`data.dataset.NuScenesFrameDataset` schema) or
        ``(image, actions)`` tuples. ``val_loader`` is required because
        the early-stopping signal comes from it.

        Batch size is the caller's responsibility — the canonical value
        is ``cfg.bc()["batch_size"]`` (256), which should be used when
        constructing the DataLoader. ``train_bc`` does not enforce it
        because the same loop is reused for unit tests with small synthetic
        batches.
    optimizer
        Constructed by the caller. The canonical pattern is ::

            optimizer = torch.optim.Adam(
                list(bc_model.parameters())
                + list(encoder.trainable_parameters()),
                lr=cfg.bc()["learning_rate"],
                weight_decay=cfg.bc()["weight_decay"],
            )

    epochs
        Maximum number of training epochs. Pass ``cfg.bc()["epochs"]``
        (50 in the canonical config) for the no-tuning run.
    patience
        Number of consecutive epochs without val-MSE improvement before
        training stops. If ``None``, the canonical default
        (``cfg.bc()["early_stopping_patience"]``, 10) is read either from
        the ``cfg`` argument (when provided) or from the on-disk canonical
        config via :func:`config.load_canonical`. Pass ``patience``
        explicitly to keep the call hermetic.
    cfg
        Optional pre-loaded :class:`CanonicalConfig`. Only consulted when
        ``patience`` is ``None``; provided so callers that already have
        the config in hand (e.g. test fixtures, evaluation pipelines)
        don't trigger an implicit disk read.
    device
        If provided, batches are moved to ``device`` before each forward
        pass. Encoder + bc_model are NOT moved here — caller should
        ``.to()`` them once before calling.
    log_csv_path
        If provided, write a per-epoch CSV with columns
        ``epoch, train_loss, val_loss`` (the §1.5 sidecar contract).

    Returns
    -------
    dict
        ``{
            "train_loss": list[float],
            "val_loss": list[float],
            "best_epoch": int,
            "best_val_loss": float,
            "stopped_early": bool,
        }``

        ``bc_model`` is reloaded to the best-epoch weights before
        returning, so callers can use the head directly without applying
        a separate checkpoint.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    patience = _resolve_patience(patience, cfg)

    loss_fn = nn.MSELoss()

    train_history: list[float] = []
    val_history: list[float] = []
    best_val_loss = math.inf
    best_epoch = -1
    best_state: Optional[dict[str, torch.Tensor]] = None
    epochs_since_improvement = 0
    stopped_early = False

    for epoch in range(epochs):
        train_loss = _epoch_loss(
            encoder=encoder,
            head=bc_model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            train=True,
        )
        val_loss = _epoch_loss(
            encoder=encoder,
            head=bc_model,
            loader=val_loader,
            optimizer=None,
            loss_fn=loss_fn,
            device=device,
            train=False,
        )
        train_history.append(train_loss)
        val_history.append(val_loss)

        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in bc_model.state_dict().items()
            }
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            stopped_early = True
            break

    if best_state is not None:
        target_device = next(bc_model.parameters()).device
        bc_model.load_state_dict(
            {k: v.to(target_device) for k, v in best_state.items()}
        )

    if log_csv_path is not None:
        _write_log_csv(log_csv_path, train_history, val_history)

    return {
        "train_loss": train_history,
        "val_loss": val_history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "stopped_early": stopped_early,
    }


def _resolve_patience(
    patience: Optional[int], cfg: Optional[CanonicalConfig]
) -> int:
    """Resolve early-stopping patience from the explicit arg or the canonical config.

    Centralizing this keeps :func:`train_bc` readable and makes the
    fallback-to-disk path explicit in one place.
    """
    if patience is None:
        if cfg is None:
            cfg = load_canonical()
        patience = int(cfg.bc()["early_stopping_patience"])
    else:
        patience = int(patience)
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")
    return patience


def _write_log_csv(
    path: Union[Path, str],
    train_history: list[float],
    val_history: list[float],
) -> None:
    """Write a per-epoch CSV log. Matches the schema used by ``train_probe``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, train_loss in enumerate(train_history):
            writer.writerow([i, f"{train_loss:.10g}", f"{val_history[i]:.10g}"])
