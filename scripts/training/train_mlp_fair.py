"""Train MLP predictors for fair DiT-vs-MLP comparison (DA7).

Trains :class:`~models.latent_pred.LatentPredictor` for all 6 encoders x
2 variants x 3 seeds using the **same** frozen orthogonal adapter and
per-element z_mean/z_std normalization as DiT. This isolates the
architecture/training-objective difference from adapter quality.

Before training, validates the reconstructed adapter against
``artifacts/full/rollout_results.json`` copy baseline (once per
encoder/seed). Aborts on mismatch.

Output: ``outputs/latent_predictors_fair/{encoder}/{variant}/seed_{seed}/``
  - ``checkpoint.pt`` -- predictor + fourier_embed + adapter + z_mean/z_std
  - ``train_log.csv`` -- per-epoch (epoch, train_loss, val_loss)

Usage
-----
    python scripts/train_mlp_fair.py                       # all 36 jobs
    python scripts/train_mlp_fair.py --encoders vit_s16     # single encoder
    python scripts/train_mlp_fair.py --validate-only        # validation only
    python scripts/train_mlp_fair.py --device cuda          # GPU training
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
    NATIVE_DIMS,
    TARGET_DIM,
    build_windows,
    diagnose_normalization,
    load_embeddings,
    reconstruct_adapter_and_stats,
    validate_adapter_copy_baseline,
)
from models.fourier_embed import FourierActionEmbedding
from models.latent_pred import LatentPredictor


ENCODER_NAMES = sorted(NATIVE_DIMS.keys())
SEEDS = [0, 1, 2]
VARIANTS = ["conditioned", "unconditioned"]
OUTPUT_ROOT = Path("outputs/latent_predictors_fair")


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


def _write_train_log(path: Path, train_hist: list, val_hist: list | None):
    """Write per-epoch CSV log."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, tl in enumerate(train_hist):
            vl = "" if val_hist is None else f"{val_hist[i]:.10g}"
            writer.writerow([i, f"{tl:.10g}", vl])


def _train_one(
    encoder_name: str,
    variant: str,
    seed: int,
    adapter: nn.Module,
    z_mean: torch.Tensor,
    z_std: torch.Tensor,
    train_windows: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_windows: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    validation_result: dict,
    norm_diagnostic: dict,
) -> Path:
    """Train a single MLP predictor in normalized space."""
    out_dir = OUTPUT_ROOT / encoder_name / variant / f"seed_{seed}"

    # Skip if checkpoint exists
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

        # Action embeddings
        act_train_dev = act_train.to(device)

    # Build val data
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

    # Build model
    cfg = load_canonical()
    predictor = LatentPredictor.from_canonical(cfg).to(device)
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
            if variant == "unconditioned":
                a_embed = torch.zeros_like(a_embed)

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
                    if variant == "unconditioned":
                        a_embed = torch.zeros_like(a_embed)
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
        "provenance": {
            "adapter_source": "reconstructed_orthogonal",
            "adapter_validation_atol": 1e-5,
            "adapter_validation_passed": validation_result["passed"],
            "matched_dit_copy_baseline": validation_result["passed"],
            "n_train_windows": len(z_t_train),
            "n_val_windows": len(val_windows[0]) if val_windows else 0,
            "n_test_windows": validation_result.get("n_test_windows", 0),
            "normalization_tokens": "z_t_plus_z_future_train",
            "normalization_diagnostic": norm_diagnostic,
            "torch_version": torch.__version__,
            "git_sha": _git_sha(),
            "encoder_name": encoder_name,
            "variant": variant,
            "seed": seed,
        },
    }
    torch.save(checkpoint, ckpt_path)

    # Write training log
    _write_train_log(out_dir / "train_log.csv", train_hist, val_hist)

    print(
        f"  [done] train={train_hist[-1]:.6f}"
        f"{f' val={val_hist[-1]:.6f}' if val_hist else ''}"
        f" -> {ckpt_path}"
    )
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP predictors for fair DiT-vs-MLP comparison."
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=ENCODER_NAMES,
        choices=ENCODER_NAMES,
        help="Encoders to train (default: all).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seeds to train (default: 0 1 2).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=VARIANTS,
        choices=VARIANTS,
        help="Variants to train (default: both).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: auto-detect cuda/mps/cpu).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run adapter/normalization validation without training.",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device
        else (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )

    cfg = load_canonical()
    lp_cfg = cfg.latent_predictor()
    epochs = int(lp_cfg["epochs"])
    lr = float(lp_cfg["learning_rate"])
    batch_size = int(lp_cfg["batch_size"])
    horizon = int(lp_cfg["prediction_horizon"])

    print("=" * 60)
    print("DA7: Fair MLP Training (frozen adapter + normalization)")
    print(f"  encoders: {args.encoders}")
    print(f"  seeds: {args.seeds}")
    print(f"  variants: {args.variants}")
    print(f"  device: {device}")
    print(f"  epochs={epochs} lr={lr} batch_size={batch_size}")
    if args.validate_only:
        print("  MODE: validate-only (no training)")
    print("=" * 60)

    t_start = time.time()
    failed_validations: list[str] = []

    for enc in args.encoders:
        print(f"\n{'='*40} {enc} {'='*40}")
        data = load_embeddings(enc)

        train_win = build_windows(data, "train", horizon)
        val_win = build_windows(data, "val", horizon)
        test_win = build_windows(data, "test", horizon)

        if train_win is None or test_win is None:
            print(f"  [ERROR] Missing train/test windows for {enc}")
            continue

        print(
            f"  windows: train={len(train_win[0])}, "
            f"val={len(val_win[0]) if val_win else 0}, "
            f"test={len(test_win[0])}"
        )

        for seed in args.seeds:
            print(f"\n  --- {enc} / seed={seed} ---")

            # Reconstruct adapter + stats (once per encoder/seed)
            adapter, z_mean, z_std = reconstruct_adapter_and_stats(
                enc, seed, train_win, device=device
            )

            # Validate adapter
            val_result = validate_adapter_copy_baseline(
                enc, seed, adapter, test_win, device=device
            )
            if val_result["passed"]:
                print(
                    f"  [OK] adapter validation passed "
                    f"(max_diff={val_result['max_copy_baseline_diff']:.2e})"
                )
            else:
                error = val_result.get("error", "")
                print(f"  [FAIL] adapter validation: {error}")
                if "per_horizon_diffs" in val_result:
                    for d in val_result["per_horizon_diffs"]:
                        print(
                            f"    {d['variant']} k={d['k']}: "
                            f"local={d['local']:.8f} dit={d['dit']:.8f} "
                            f"diff={d['diff']:.2e}"
                        )
                failed_validations.append(f"{enc}/seed={seed}")
                continue

            # Normalization diagnostic
            norm_diag = diagnose_normalization(
                z_mean, z_std, adapter, train_win, device=device
            )
            print(
                f"  [norm] mean_abs_mean={norm_diag['mean_abs_mean']:.2e}, "
                f"max_abs_mean={norm_diag['max_abs_mean']:.2e}, "
                f"mean_abs_std_dev={norm_diag['mean_abs_std_dev']:.2e}, "
                f"max_abs_std_dev={norm_diag['max_abs_std_dev']:.2e}"
            )
            if norm_diag["abort"]:
                print("  [ABORT] Normalization diagnostic failed!")
                failed_validations.append(f"{enc}/seed={seed} (normalization)")
                continue

            if args.validate_only:
                print("  [validate-only] Skipping training")
                continue

            # Train both variants
            for variant in args.variants:
                print(f"\n  >>> {enc}/{variant}/seed_{seed}")
                _train_one(
                    encoder_name=enc,
                    variant=variant,
                    seed=seed,
                    adapter=adapter,
                    z_mean=z_mean,
                    z_std=z_std,
                    train_windows=train_win,
                    val_windows=val_win,
                    device=device,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    validation_result=val_result,
                    norm_diagnostic=norm_diag,
                )

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Completed in {elapsed:.1f}s")

    if failed_validations:
        print(f"\n[FAILED] {len(failed_validations)} validation(s):")
        for fv in failed_validations:
            print(f"  - {fv}")
        sys.exit(1)
    else:
        print("[OK] All validations passed")


if __name__ == "__main__":
    main()
