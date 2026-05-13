"""Action prediction probe: 2-layer MLP head trained on top of frozen encoders.

The probe takes a 384-d embedding from any of the five encoder wrappers and
maps it to a 2-d ``(steer_norm, accel_norm)`` action prediction in the
canonical normalized space (steer = clip(rad/6, [-1, 1]); accel =
clip(m_per_s2/10, [-1, 1])).

All five encoder wrappers in this repo project to the project-wide
``target_embedding_dim`` (384) before the probe sees them, so a single
probe class works for every encoder. Each encoder gets its own probe
instance trained from scratch — there is no shared probe head across
encoders, which is by design (we're benchmarking representations, not
heads).

Hyperparameters are pinned in ``configs/canonical.yaml::probe`` and
should not be tuned per-encoder; ``ActionProbe.from_canonical`` is the
recommended constructor.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import CanonicalConfig, load_canonical

# Default architectural constants. Mirror configs/canonical.yaml::probe so
# that callers can construct an ActionProbe without importing the config
# (e.g. in a unit test); ``from_canonical`` is the path that reads the
# config and is authoritative.
DEFAULT_INPUT_DIM = 384
DEFAULT_HIDDEN_DIM = 256
DEFAULT_DROPOUT = 0.1
DEFAULT_OUTPUT_DIM = 2  # (steer_norm, accel_norm)


class ActionProbe(nn.Module):
    """Two-layer MLP probe: ``Linear → GELU → Dropout → Linear``.

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
    Bias is enabled on both Linear layers (encoder *adapters* use
    ``bias=False`` for fair parameter counts across native dims; probes
    follow the standard regression-MLP convention with bias).
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
    ) -> "ActionProbe":
        """Construct from ``configs/canonical.yaml``.

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


_BatchLike = Union[
    Mapping[str, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]


def _extract_image_actions(
    batch: _BatchLike,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pull ``(image, actions)`` from either dict-style or tuple-style batches.

    Dict-style is what ``data.dataset.NuScenesFrameDataset`` emits
    (``{"image": ..., "actions": ..., "sample_token": ..., ...}``);
    tuple-style is what synthetic test loaders typically yield. Supporting
    both keeps the training loop decoupled from the dataset's exact return
    schema.
    """
    if isinstance(batch, Mapping):
        return batch["image"], batch["actions"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(
        f"Unsupported batch type: {type(batch).__name__}. "
        "Expected a Mapping with 'image' and 'actions' keys, or a "
        "(image, actions) tuple."
    )


def _epoch_loss(
    encoder: nn.Module,
    probe: ActionProbe,
    loader: Iterable[_BatchLike],
    optimizer: Optional[torch.optim.Optimizer],
    loss_fn: nn.Module,
    device: Optional[torch.device],
    train: bool,
) -> float:
    """Run one pass over ``loader``; step the optimizer if ``train`` is True.

    Returns the sample-weighted mean loss for the epoch. ``optimizer`` may
    be ``None`` for eval passes.
    """
    if train:
        probe.train()
    else:
        probe.eval()

    # Encoder is always frozen + always in eval mode (the wrapper's
    # ``train()`` override forces backbone.eval() — see encoders/base.py).
    # Calling .eval() here is just defensive.
    encoder.eval()

    total_loss_sum = 0.0
    total_n = 0

    grad_ctx: Any = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            image, action = _extract_image_actions(batch)
            if device is not None:
                image = image.to(device)
                action = action.to(device)

            embedding = encoder(image)
            prediction = probe(embedding)
            loss = loss_fn(prediction, action)

            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = action.shape[0]
            total_loss_sum += float(loss.detach().item()) * batch_size
            total_n += batch_size

    if total_n == 0:
        raise RuntimeError(
            "loader yielded zero samples; cannot compute mean epoch loss."
        )
    return total_loss_sum / total_n


def train_probe(
    encoder: nn.Module,
    probe: ActionProbe,
    train_loader: Iterable[_BatchLike],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    *,
    val_loader: Optional[Iterable[_BatchLike]] = None,
    device: Optional[torch.device] = None,
    log_csv_path: Optional[Union[Path, str]] = None,
) -> dict:
    """Train ``probe`` on top of a frozen ``encoder`` for ``epochs`` epochs.

    Parameters
    ----------
    encoder
        A :class:`encoders.base.BaseEncoderWrapper` instance (or any
        ``nn.Module`` that maps an input batch to a 2-D embedding tensor
        and whose backbone is already frozen). The training loop does not
        modify the encoder's frozen state — gradients flow only into the
        encoder's adapter (when present) and into the probe.
    probe
        The :class:`ActionProbe` to train.
    train_loader
        Iterable yielding either ``{"image": ..., "actions": ...}`` dicts
        (the :class:`data.dataset.NuScenesFrameDataset` schema) or
        ``(image, actions)`` tuples.
    optimizer
        Constructed by the caller. The canonical pattern is ::

            optimizer = torch.optim.Adam(
                list(probe.parameters())
                + list(encoder.trainable_parameters()),
                lr=cfg.probe()["learning_rate"],
                weight_decay=cfg.probe()["weight_decay"],
            )

        ``encoder.trainable_parameters()`` is the projection adapter for
        encoders that need it (CLIP, V-JEPA2, VQ-VAE in non-fallback
        mode) and an empty iterator otherwise. Constructing the optimizer
        outside this function keeps lr-schedule / weight-decay choices
        explicit at the call site.
    epochs
        Number of training epochs.
    val_loader
        Optional validation loader. When provided, mean val loss is
        computed (under ``torch.no_grad``) once per epoch.
    device
        If provided, batches are moved to ``device`` before each forward
        pass. Encoder + probe are NOT moved here — caller should ``.to()``
        them once before calling.
    log_csv_path
        If provided, write a per-epoch CSV with columns
        ``epoch, train_loss, val_loss`` (the ``val_loss`` cell is empty
        when no ``val_loader`` is supplied). This is the §1.5 sidecar
        contract — the in-memory return value is convenient but the CSV
        is the source of truth for downstream figure code.

    Returns
    -------
    dict
        ``{"train_loss": list[float], "val_loss": list[float] | None}``
        with one entry per epoch.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    loss_fn = nn.MSELoss()

    train_history: list[float] = []
    val_history: Optional[list[float]] = [] if val_loader is not None else None

    for _epoch in range(epochs):
        train_loss = _epoch_loss(
            encoder=encoder,
            probe=probe,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            train=True,
        )
        train_history.append(train_loss)

        if val_loader is not None:
            val_loss = _epoch_loss(
                encoder=encoder,
                probe=probe,
                loader=val_loader,
                optimizer=None,
                loss_fn=loss_fn,
                device=device,
                train=False,
            )
            assert val_history is not None
            val_history.append(val_loss)

    if log_csv_path is not None:
        _write_log_csv(log_csv_path, train_history, val_history)

    return {"train_loss": train_history, "val_loss": val_history}


def _write_log_csv(
    path: Union[Path, str],
    train_history: list[float],
    val_history: Optional[list[float]],
) -> None:
    """Write a per-epoch CSV log. Always emits the same column schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, train_loss in enumerate(train_history):
            val_cell = (
                "" if val_history is None else f"{val_history[i]:.10g}"
            )
            writer.writerow([i, f"{train_loss:.10g}", val_cell])
