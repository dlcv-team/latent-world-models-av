"""Latent-space world model for the P1 prediction pipeline.

The predictor takes the current-frame latent state ``z_t`` (384-d) and
an action embedding ``a_embed`` (384-d, from
:class:`~models.fourier_embed.FourierActionEmbedding`) and predicts the
next ``horizon`` latent states ``z_{t+1}, ..., z_{t+horizon}`` via a
3-layer MLP.  It never generates pixels.

The same class is used for both conditioned and unconditional variants:

- **Conditioned**: ``a_embed`` is a real Fourier action embedding.
- **Unconditional**: ``a_embed`` is zeroed before the forward pass.

Both variants are trained separately with MSE loss against real future
encoder embeddings.  The difference in cosine similarity between the two
(DeltaCosSim) measures whether the action signal adds prediction value.

Hyperparameters are pinned in
``configs/canonical.yaml::latent_predictor`` and should not be tuned;
``LatentPredictor.from_canonical`` is the recommended constructor.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import CanonicalConfig, load_canonical

# Default architectural constants.  Mirror
# configs/canonical.yaml::latent_predictor so that callers can construct
# a LatentPredictor without importing the config (e.g. in a unit test);
# ``from_canonical`` is the path that reads the config and is
# authoritative.
DEFAULT_Z_DIM = 384
DEFAULT_A_DIM = 384
DEFAULT_HORIZON = 4
DEFAULT_HIDDEN = 512  # from architecture string: "Linear(768,512) -> ..."


class LatentPredictor(nn.Module):
    """Three-layer MLP: ``Linear -> GELU -> Linear -> GELU -> Linear``.

    Parameters
    ----------
    z_dim
        Encoder embedding dimension.  Defaults to 384 (project-wide
        ``target_embedding_dim``).
    a_dim
        Action embedding dimension (output of
        :class:`~models.fourier_embed.FourierActionEmbedding`).
        Defaults to 384.
    horizon
        Number of future latent states to predict.  Defaults to 4.
    hidden
        Hidden width of the MLP.  Defaults to 512 per canonical config
        architecture string.

    Notes
    -----
    The model input is ``cat(z_t, a_embed)`` with shape ``(B, z_dim +
    a_dim)`` and the output is reshaped to ``(B, horizon, z_dim)`` so
    each predicted future frame gets its own 384-d latent vector.
    """

    def __init__(
        self,
        z_dim: int = DEFAULT_Z_DIM,
        a_dim: int = DEFAULT_A_DIM,
        horizon: int = DEFAULT_HORIZON,
        hidden: int = DEFAULT_HIDDEN,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.a_dim = int(a_dim)
        self.horizon = int(horizon)
        self.hidden = int(hidden)

        self.net = nn.Sequential(
            nn.Linear(self.z_dim + self.a_dim, self.hidden),  # 768 -> 512
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),  # 512 -> 512
            nn.GELU(),
            nn.Linear(self.hidden, self.z_dim * self.horizon),  # 512 -> 1536
        )

    @classmethod
    def from_canonical(
        cls, cfg: Optional[CanonicalConfig] = None
    ) -> "LatentPredictor":
        """Construct from ``configs/canonical.yaml``.

        Parameters
        ----------
        cfg
            Optional pre-loaded :class:`CanonicalConfig`. If omitted, the
            canonical config is loaded from disk via :func:`load_canonical`.
        """
        if cfg is None:
            cfg = load_canonical()
        lp_cfg = cfg.latent_predictor()
        fae_cfg = lp_cfg["fourier_action_embed"]
        return cls(
            z_dim=cfg.target_embedding_dim,
            a_dim=int(fae_cfg["out_dim"]),
            horizon=int(lp_cfg["prediction_horizon"]),
            # hidden=512 is encoded in the architecture string
            # "Linear(768,512) -> GELU -> Linear(512,512) -> ..."
            # rather than as a separate YAML key.
            hidden=DEFAULT_HIDDEN,
        )

    def forward(
        self, z_t: torch.Tensor, a_embed: torch.Tensor
    ) -> torch.Tensor:
        """Predict future latents from current state + action embedding.

        Parameters
        ----------
        z_t
            ``(B, z_dim)`` — current-frame encoder embedding.
        a_embed
            ``(B, a_dim)`` — action embedding from
            :class:`~models.fourier_embed.FourierActionEmbedding`.
            Pass ``torch.zeros_like(a_embed)`` for the unconditional
            variant.

        Returns
        -------
        torch.Tensor
            ``(B, horizon, z_dim)`` — predicted future latent states.
        """
        x = torch.cat([z_t, a_embed], dim=-1)  # (B, z_dim + a_dim)
        return self.net(x).view(-1, self.horizon, self.z_dim)  # (B, H, z_dim)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _epoch_loss_lp(
    predictor: LatentPredictor,
    fourier_embed: nn.Module,
    adapter: nn.Module,
    loader: Iterable[Mapping[str, torch.Tensor]],
    optimizer: Optional[torch.optim.Optimizer],
    variant: str,
    device: Optional[torch.device],
    train: bool,
) -> float:
    """Run one pass over ``loader``; step optimizer if ``train``."""
    if train:
        predictor.train()
        fourier_embed.train()
        if hasattr(adapter, "train"):
            adapter.train()
    else:
        predictor.eval()
        fourier_embed.eval()
        if hasattr(adapter, "eval"):
            adapter.eval()

    loss_fn = nn.MSELoss()
    total_loss_sum = 0.0
    total_n = 0

    grad_ctx: Any = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            z_t_native = batch["z_t"]
            action = batch["action"]
            z_future_native = batch["z_future"]

            if device is not None:
                z_t_native = z_t_native.to(device)
                action = action.to(device)
                z_future_native = z_future_native.to(device)

            # Adapter projection (identity when native_dim == target_dim)
            z_t = adapter(z_t_native)
            B, H, native_dim = z_future_native.shape
            z_future = adapter(
                z_future_native.reshape(B * H, native_dim)
            ).view(B, H, -1)

            # Action embedding
            a_embed = fourier_embed(action)
            if variant == "unconditioned":
                a_embed = torch.zeros_like(a_embed)

            z_hat = predictor(z_t, a_embed)
            loss = loss_fn(z_hat, z_future)

            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = z_t_native.shape[0]
            total_loss_sum += float(loss.detach().item()) * batch_size
            total_n += batch_size

    if total_n == 0:
        raise RuntimeError(
            "loader yielded zero samples; cannot compute mean epoch loss."
        )
    return total_loss_sum / total_n


def train_latent_predictor(
    predictor: LatentPredictor,
    fourier_embed: nn.Module,
    adapter: nn.Module,
    train_loader: Iterable[Mapping[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    *,
    variant: str = "conditioned",
    val_loader: Optional[Iterable[Mapping[str, torch.Tensor]]] = None,
    device: Optional[torch.device] = None,
    log_csv_path: Optional[Union[Path, str]] = None,
) -> dict:
    """Train ``predictor`` on temporal embedding sequences.

    Parameters
    ----------
    predictor
        The :class:`LatentPredictor` to train.
    fourier_embed
        A :class:`~models.fourier_embed.FourierActionEmbedding` instance.
        Its MLP parameters are included in the optimizer.
    adapter
        ``nn.Linear(native_dim, target_dim, bias=False)`` or
        ``nn.Identity()`` when no projection is needed.
    train_loader
        Iterable yielding dicts with ``z_t``, ``action``, ``z_future``.
    optimizer
        Constructed by the caller over predictor + fourier_embed +
        adapter parameters.
    epochs
        Number of training epochs.
    variant
        ``"conditioned"`` or ``"unconditioned"``. When unconditioned,
        the action embedding is zeroed before the predictor's forward
        pass.
    val_loader
        Optional validation loader.
    device
        Batches are moved to ``device`` before each forward pass.
    log_csv_path
        If provided, write a per-epoch CSV with columns
        ``epoch, train_loss, val_loss``.

    Returns
    -------
    dict
        ``{"train_loss": list[float], "val_loss": list[float] | None}``
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    train_history: list[float] = []
    val_history: Optional[list[float]] = [] if val_loader is not None else None

    for _epoch in range(epochs):
        train_loss = _epoch_loss_lp(
            predictor=predictor,
            fourier_embed=fourier_embed,
            adapter=adapter,
            loader=train_loader,
            optimizer=optimizer,
            variant=variant,
            device=device,
            train=True,
        )
        train_history.append(train_loss)

        if val_loader is not None:
            val_loss = _epoch_loss_lp(
                predictor=predictor,
                fourier_embed=fourier_embed,
                adapter=adapter,
                loader=val_loader,
                optimizer=None,
                variant=variant,
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
    """Write a per-epoch CSV log."""
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
