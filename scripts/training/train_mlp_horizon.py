"""Train MLP-fair predictors at longer horizons (DA9 Exp 3).

Fork of ``train_mlp_fair.py`` with a ``--horizon`` parameter.
Trains conditioned variant only.

Usage::

    python scripts/train_mlp_horizon.py --horizon 8
    python scripts/train_mlp_horizon.py --horizon 16 --encoders vit_s16 clip_b32
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import load_canonical
from evaluation.dit_utils import (
    DEFAULT_HORIZON,
    NATIVE_DIMS,
    TARGET_DIM,
    build_windows,
    load_embeddings,
    reconstruct_adapter_and_stats,
)
from models.fourier_embed import FourierActionEmbedding
from models.latent_pred import LatentPredictor

ENCODER_NAMES = sorted(NATIVE_DIMS.keys())
SEEDS = [0, 1, 2]


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _output_root(horizon: int) -> Path:
    return Path(f"outputs/latent_predictors_fair_h{horizon}")


def _train_one(
    encoder_name: str,
    seed: int,
    horizon: int,
    adapter: nn.Module,
    z_mean: torch.Tensor,
    z_std: torch.Tensor,
    train_windows: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_windows: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
) -> Path:
    """Train a single MLP-fair predictor at the given horizon."""
    variant = "conditioned"
    out_dir = _output_root(horizon) / encoder_name / variant / f"seed_{seed}"

    ckpt_path = out_dir / "checkpoint.pt"
    if ckpt_path.exists():
        print(f"  [skip] {ckpt_path} already exists")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    z_t_train, act_train, zf_train = train_windows

    # Normalize training data
    with torch.no_grad():
        z_t_norm = (adapter(z_t_train.to(device)) - z_mean) / z_std
        B, H, _ = zf_train.shape
        zf_norm = (
            adapter(zf_train.reshape(B * H, -1).to(device)).reshape(B, H, TARGET_DIM)
            - z_mean
        ) / z_std
        act_train_dev = act_train.to(device)

    # Validation data
    z_t_val_norm, zf_val_norm, act_val_dev = None, None, None
    if val_windows is not None:
        z_t_v, act_v, zf_v = val_windows
        with torch.no_grad():
            z_t_val_norm = (adapter(z_t_v.to(device)) - z_mean) / z_std
            Bv, Hv, _ = zf_v.shape
            zf_val_norm = (
                adapter(zf_v.reshape(Bv * Hv, -1).to(device)).reshape(Bv, Hv, TARGET_DIM)
                - z_mean
            ) / z_std
            act_val_dev = act_v.to(device)

    # Model -- override horizon in LatentPredictor
    cfg = load_canonical()
    predictor = LatentPredictor(
        z_dim=TARGET_DIM,
        a_dim=TARGET_DIM,
        horizon=horizon,
    ).to(device)
    fourier_embed = FourierActionEmbedding.from_canonical(cfg).to(device)

    optimizer = torch.optim.Adam(
        list(predictor.parameters()) + list(fourier_embed.parameters()),
        lr=lr,
    )
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(z_t_norm, act_train_dev, zf_norm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if z_t_val_norm is not None:
        val_ds = TensorDataset(z_t_val_norm, act_val_dev, zf_val_norm)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_hist: list[float] = []
    val_hist: list[float] = [] if val_loader is not None else None

    for epoch in range(epochs):
        predictor.train()
        fourier_embed.train()
        epoch_loss = 0.0
        epoch_n = 0

        for z_t_b, act_b, zf_b in train_loader:
            a_embed = fourier_embed(act_b)
            z_hat = predictor(z_t_b, a_embed)
            loss = loss_fn(z_hat, zf_b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * z_t_b.shape[0]
            epoch_n += z_t_b.shape[0]

        train_hist.append(epoch_loss / epoch_n)

        if val_loader is not None:
            predictor.eval()
            fourier_embed.eval()
            val_loss = 0.0
            val_n = 0
            with torch.no_grad():
                for z_t_b, act_b, zf_b in val_loader:
                    a_embed = fourier_embed(act_b)
                    z_hat = predictor(z_t_b, a_embed)
                    loss = loss_fn(z_hat, zf_b)
                    val_loss += loss.item() * z_t_b.shape[0]
                    val_n += z_t_b.shape[0]
            val_hist.append(val_loss / val_n)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            vl_str = f" val={val_hist[-1]:.6f}" if val_hist else ""
            print(
                f"  [epoch {epoch+1:3d}/{epochs}] "
                f"train={train_hist[-1]:.6f}{vl_str}"
            )

    # Save checkpoint
    native_dim = NATIVE_DIMS[encoder_name]
    needs_adapter = native_dim != TARGET_DIM

    checkpoint = {
        "predictor_state_dict": predictor.state_dict(),
        "fourier_embed_state_dict": fourier_embed.state_dict(),
        "adapter_state_dict": adapter.state_dict() if needs_adapter else {},
        "z_mean": z_mean.cpu(),
        "z_std": z_std.cpu(),
        "final_train_loss": train_hist[-1],
        "final_val_loss": val_hist[-1] if val_hist else None,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "horizon": horizon,
        "provenance": {
            "formulation": "absolute",
            "target": "z_future_norm",
            "space": "normalized (z - z_mean) / z_std",
            "adapter_source": "reconstructed_orthogonal",
            "n_train_windows": len(z_t_train),
            "n_val_windows": len(val_windows[0]) if val_windows else 0,
            "torch_version": torch.__version__,
            "git_sha": _git_sha(),
            "encoder_name": encoder_name,
            "variant": "conditioned",
            "seed": seed,
            "horizon": horizon,
            "source": "scripts/train_mlp_horizon.py",
        },
    }
    torch.save(checkpoint, ckpt_path)

    print(
        f"  [done] train={train_hist[-1]:.6f}"
        f"{f' val={val_hist[-1]:.6f}' if val_hist else ''}"
        f" -> {ckpt_path}"
    )
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP-fair predictors at longer horizons (DA9 Exp 3)."
    )
    parser.add_argument(
        "--horizon", type=int, required=True,
        help="Prediction horizon (e.g. 8, 16).",
    )
    parser.add_argument(
        "--encoders", nargs="+", default=None,
        help="Encoders to train (default: all 6).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Seeds to train (default: 0 1 2).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: auto-detect).",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device
        else (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    )

    encoders = args.encoders if args.encoders else ENCODER_NAMES
    horizon = args.horizon
    seeds = args.seeds

    cfg = load_canonical()
    lp_cfg = cfg.latent_predictor()
    epochs = int(lp_cfg["epochs"])
    lr = float(lp_cfg["learning_rate"])
    batch_size = int(lp_cfg["batch_size"])

    print("=" * 60)
    print(f"DA9 Exp 3: MLP-fair Training (horizon={horizon})")
    print(f"  encoders: {encoders}")
    print(f"  seeds: {seeds}")
    print(f"  device: {device}")
    print(f"  epochs={epochs} lr={lr} batch_size={batch_size}")
    print("=" * 60)

    t_start = time.time()

    for enc in encoders:
        print(f"\n{'='*40} {enc} {'='*40}")
        data = load_embeddings(enc)
        train_win = build_windows(data, "train", horizon)
        val_win = build_windows(data, "val", horizon)

        if train_win is None:
            print(f"  [ERROR] Missing train windows for {enc}")
            continue

        print(f"  Train: {len(train_win[0])} windows, Val: {len(val_win[0]) if val_win else 0}")

        for seed in seeds:
            print(f"\n  --- {enc} / seed={seed} / h={horizon} ---")
            # Reconstruct adapter using h=4 windows (same adapter as DA8)
            h4_train_win = build_windows(data, "train", DEFAULT_HORIZON)
            adapter, z_mean, z_std = reconstruct_adapter_and_stats(
                enc, seed, h4_train_win, device=device
            )

            _train_one(
                encoder_name=enc,
                seed=seed,
                horizon=horizon,
                adapter=adapter,
                z_mean=z_mean,
                z_std=z_std,
                train_windows=train_win,
                val_windows=val_win,
                device=device,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )

    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
