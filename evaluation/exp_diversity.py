"""DA9 Experiment 1: Stochastic sampling diversity diagnostic.

Tests whether DiT-x0 samples diverse futures by running K=10 DDIM
rollouts with different initial noise x_T (eta=0). If all K trajectories
converge to the same output, the model has learned a unimodal conditional
and functions as a point predictor.

Usage::

    python -m evaluation.exp_diversity
    python -m evaluation.exp_diversity --encoders vit_s16 clip_b32
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import load_canonical
from evaluation.dit_utils import (
    DEFAULT_HORIZON,
    NATIVE_DIMS,
    TARGET_DIM,
    build_windows,
    load_embeddings,
)
from evaluation.tierb_eval import (
    EVAL_BATCH_SIZE,
    DIT_CKPT_ROOT,
    MLP_RESIDUAL_ROOT,
    load_dit_objective_checkpoint,
)
from evaluation.dit_eval import evaluate_mlp
from models.diffusion import CosineNoiseSchedule
from evaluation.warmstart_eval import DIFFUSION_CONFIG

CSV_PATH = Path("artifacts/full/da9_diversity.csv")

PILOT_ENCODERS = ["vit_s16", "clip_b32", "vjepa2_rep64"]
PILOT_SEEDS = [0]
K_SAMPLES = 10
ETA_VALUES = [0.0]  # primary test; eta>0 only if diversity ~0


def run_ddim_sample(
    dit, adapter, fourier_embed, z_mean, z_std,
    z_t_batch, act_batch,
    horizon: int, n_steps: int, device: torch.device,
    noise_seed: int,
) -> torch.Tensor:
    """Run single DDIM x0-prediction rollout with specified noise seed.

    Returns z_hat in unnormalized adapted space: (B, H, 384).
    """
    schedule = CosineNoiseSchedule(n_steps=DIFFUSION_CONFIG["n_train_steps"]).to(device)
    alphas_cumprod = schedule.alphas_cumprod.to(device)

    T = DIFFUSION_CONFIG["n_train_steps"]
    stride = T // n_steps
    timesteps = list(range(0, T, stride))[:n_steps]
    timesteps = list(reversed(timesteps))

    B = z_t_batch.shape[0]
    z_t_adapted = (adapter(z_t_batch) - z_mean) / z_std
    a_embed = fourier_embed(act_batch)

    # Start from noise with specific seed
    torch.manual_seed(noise_seed)
    x = torch.randn(B, horizon, TARGET_DIM, device=device)

    with torch.no_grad():
        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            model_out = dit(x, z_t=z_t_adapted, a_embed=a_embed, timestep=t)
            alpha_bar_t = alphas_cumprod[t_val]
            pred_x0 = model_out  # x0-prediction

            if i < len(timesteps) - 1:
                alpha_bar_prev = alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            noise_direction = (
                x - torch.sqrt(alpha_bar_t) * pred_x0
            ) / torch.sqrt(1.0 - alpha_bar_t + 1e-8)

            x = (
                torch.sqrt(alpha_bar_prev) * pred_x0
                + torch.sqrt(1.0 - alpha_bar_prev) * noise_direction
            )

    # Inverse transform to unnormalized adapted space
    z_hat = x * z_std + z_mean
    return z_hat


def run_diversity_analysis(
    encoders: list[str],
    seeds: list[int],
    device: torch.device,
    k_samples: int = K_SAMPLES,
    eta: float = 0.0,
) -> list[dict]:
    """Run diversity analysis for all encoders/seeds."""
    all_rows: list[dict] = []
    horizon = DEFAULT_HORIZON

    for enc in encoders:
        print(f"\n{'='*60} {enc} {'='*60}")
        data = load_embeddings(enc)
        test_win = build_windows(data, split="test", horizon=horizon)
        if test_win is None:
            print(f"  [skip] No test windows for {enc}")
            continue

        z_t_test, act_test, zf_test = test_win
        n_windows = len(z_t_test)
        print(f"  {n_windows} test windows, K={k_samples}")

        for seed in seeds:
            print(f"\n  --- seed {seed} ---")
            t0 = time.time()

            # Load DiT-x0
            dit, adapter, fourier, z_mean, z_std, prediction, residual = \
                load_dit_objective_checkpoint(enc, "conditioned__x0", seed, device)

            # Collect K samples
            # Process in batches to avoid OOM
            batch_size = EVAL_BATCH_SIZE
            all_samples = []  # list of K tensors, each (N, H, 384)

            for k in range(k_samples):
                noise_seed = seed * 1000 + k
                sample_parts = []

                for start in range(0, n_windows, batch_size):
                    end = min(start + batch_size, n_windows)
                    z_hat = run_ddim_sample(
                        dit, adapter, fourier, z_mean, z_std,
                        z_t_test[start:end].to(device),
                        act_test[start:end].to(device),
                        horizon=horizon, n_steps=50, device=device,
                        noise_seed=noise_seed,
                    )
                    sample_parts.append(z_hat.cpu())

                all_samples.append(torch.cat(sample_parts, dim=0))  # (N, H, 384)
                if (k + 1) % 5 == 0:
                    print(f"    Sample {k+1}/{k_samples} done")

            # Stack: (K, N, H, 384)
            samples = torch.stack(all_samples, dim=0)

            del dit, adapter, fourier
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Ground truth in unnormalized adapted space
            # Need to recompute via adapter from checkpoint
            # For simplicity, reload and compute
            _, adapter2, _, z_mean2, z_std2, _, _ = \
                load_dit_objective_checkpoint(enc, "conditioned__x0", seed, device)

            with torch.no_grad():
                B_f, H, _ = zf_test.shape
                zf_adapted = adapter2(
                    zf_test.reshape(B_f * H, -1).to(device)
                ).reshape(B_f, H, TARGET_DIM).cpu()
                z_t_adapted = adapter2(z_t_test.to(device)).cpu()

            del adapter2

            # MLP-fair baseline
            mlp_fair_r = evaluate_mlp(enc, "conditioned", seed, device=device)
            mlp_fair_cossim = mlp_fair_r["metrics"]["cossim_by_horizon"] if "error" not in mlp_fair_r else None

            # MLP-residual baseline
            from evaluation.tierb_eval import evaluate_residual_mlp
            mlp_res_r = evaluate_residual_mlp(enc, seed, device)
            mlp_res_cossim = mlp_res_r["cossim_by_horizon"]

            # Compute metrics per horizon step
            for k_h in range(horizon):
                h = k_h + 1
                zf_k = zf_adapted[:, k_h]  # (N, 384)
                z_t_orig = z_t_adapted  # (N, 384)

                # Per-sample CosSim with ground truth
                per_sample_cossim = []  # (K, N)
                for s_idx in range(k_samples):
                    cs = F.cosine_similarity(
                        samples[s_idx, :, k_h], zf_k, dim=-1
                    )
                    per_sample_cossim.append(cs)
                per_sample_cossim = torch.stack(per_sample_cossim, dim=0)  # (K, N)

                # Best-of-K: max across K
                best_of_k_cs, best_idx = per_sample_cossim.max(dim=0)  # (N,)

                # Gather best sample for norm ratio + MSE
                best_samples = torch.stack([
                    samples[best_idx[i], i, k_h] for i in range(n_windows)
                ])  # (N, 384)
                best_norm_ratio = (
                    best_samples.norm(dim=-1) / (zf_k.norm(dim=-1) + 1e-8)
                ).mean().item()
                best_mse = ((best_samples - zf_k) ** 2).mean(dim=-1).mean().item()

                # Mean-of-K
                mean_of_k_cs = per_sample_cossim.mean(dim=0).mean().item()

                # Pairwise distance: 1 - mean pairwise CosSim among K samples
                pairwise_cs_sum = 0.0
                n_pairs = 0
                for i in range(k_samples):
                    for j in range(i + 1, k_samples):
                        pw_cs = F.cosine_similarity(
                            samples[i, :, k_h], samples[j, :, k_h], dim=-1
                        ).mean().item()
                        pairwise_cs_sum += pw_cs
                        n_pairs += 1
                pairwise_distance = 1.0 - (pairwise_cs_sum / n_pairs) if n_pairs > 0 else 0.0

                # Spread: std of per-sample CosSim across K
                spread = per_sample_cossim.std(dim=0).mean().item()

                # Copy baseline
                copy_cs = F.cosine_similarity(z_t_orig, zf_k, dim=-1).mean().item()

                # Difficulty quartiles (per this horizon)
                copy_per_window = F.cosine_similarity(z_t_orig, zf_k, dim=-1)
                try:
                    quartile_labels = pd.qcut(
                        copy_per_window.numpy(), q=4,
                        labels=["Q1 (hardest)", "Q2", "Q3", "Q4 (easiest)"],
                    )
                except ValueError:
                    quartile_labels = pd.qcut(
                        pd.Series(copy_per_window.numpy()).rank(method="first"),
                        q=4,
                        labels=["Q1 (hardest)", "Q2", "Q3", "Q4 (easiest)"],
                    )

                # Per-quartile rows
                for q_label in ["Q1 (hardest)", "Q2", "Q3", "Q4 (easiest)"]:
                    q_mask = np.array(quartile_labels) == q_label
                    n_q = q_mask.sum()

                    row = {
                        "encoder": enc,
                        "seed": seed,
                        "eta": eta,
                        "quartile": q_label,
                        "horizon_step": h,
                        "best_of_k_cossim": round(best_of_k_cs[q_mask].mean().item(), 6),
                        "best_of_k_mse": round(
                            ((best_samples[q_mask] - zf_k[q_mask]) ** 2).mean(dim=-1).mean().item(), 6
                        ),
                        "best_of_k_norm_ratio": round(
                            (best_samples[q_mask].norm(dim=-1) /
                             (zf_k[q_mask].norm(dim=-1) + 1e-8)).mean().item(), 6
                        ),
                        "mean_of_k_cossim": round(
                            per_sample_cossim[:, q_mask].mean().item(), 6
                        ),
                        "mlp_fair_cossim": round(mlp_fair_cossim[k_h], 6) if mlp_fair_cossim else None,
                        "mlp_residual_cossim": round(mlp_res_cossim[k_h], 6),
                        "copy_baseline": round(
                            copy_per_window[q_mask].mean().item(), 6
                        ),
                        "pairwise_distance": round(pairwise_distance, 6),
                        "spread": round(spread, 6),
                        "n_windows": int(n_q),
                    }
                    all_rows.append(row)

            elapsed = time.time() - t0
            print(f"  {enc}/seed={seed}: {elapsed:.1f}s")
            print(f"    pairwise_distance (h1): {all_rows[-4*horizon]['pairwise_distance']:.6f}")
            print(f"    spread (h1): {all_rows[-4*horizon]['spread']:.6f}")

    return all_rows


def main():
    parser = argparse.ArgumentParser(description="DA9 Exp 1: Diversity diagnostic")
    parser.add_argument(
        "--encoders", nargs="+", default=None,
        help="Encoders (default: pilot set).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=PILOT_SEEDS,
    )
    parser.add_argument(
        "--k", type=int, default=K_SAMPLES,
        help="Number of samples per window.",
    )
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    encoders = args.encoders or PILOT_ENCODERS

    print("DA9 Experiment 1: Diversity Diagnostic")
    print(f"  encoders: {encoders}")
    print(f"  seeds: {args.seeds}")
    print(f"  K: {args.k}")
    print(f"  eta: {ETA_VALUES}")
    print(f"  device: {device}")

    all_rows = []
    for eta in ETA_VALUES:
        rows = run_diversity_analysis(encoders, args.seeds, device, args.k, eta)
        all_rows.extend(rows)

    if not all_rows:
        print("No results generated.")
        sys.exit(1)

    # Save CSV
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved CSV to {CSV_PATH} ({len(all_rows)} rows)")

    # Summary
    df = pd.DataFrame(all_rows)
    print("\n" + "=" * 80)
    print("Diversity Summary (h=1, all quartiles)")
    print("=" * 80)
    h1 = df[df["horizon_step"] == 1]
    for enc in encoders:
        enc_h1 = h1[h1["encoder"] == enc]
        if enc_h1.empty:
            continue
        pw_dist = enc_h1["pairwise_distance"].mean()
        spread = enc_h1["spread"].mean()
        best_k = enc_h1["best_of_k_cossim"].mean()
        mean_k = enc_h1["mean_of_k_cossim"].mean()
        mlp_res = enc_h1["mlp_residual_cossim"].mean()
        print(f"  {enc}: pairwise_dist={pw_dist:.6f}, spread={spread:.6f}")
        print(f"    Best-of-K={best_k:.4f}, Mean-of-K={mean_k:.4f}, MLP-res={mlp_res:.4f}")

        if pw_dist < 0.001:
            print(f"    -> FINDING: Unimodal conditional posterior (pairwise_dist ~ 0)")


if __name__ == "__main__":
    main()
