"""DA8 Tier B: Evaluate alternative DiT objectives and residual MLP.

Evaluates checkpoints from ``train_dit_objectives.py`` (B1: x0/v, B2:
residual) and ``train_mlp_residual.py`` (B2-ctrl) using DDIM sampling.

For each objective, the DDIM sampler must recover x0 differently:
- epsilon: pred_x0 = (x_t - sqrt(1-alpha)*eps) / sqrt(alpha)  [standard]
- x0: pred_x0 = model_output  [direct]
- v: pred_x0 = sqrt(alpha)*x_t - sqrt(1-alpha)*v  [Salimans & Ho 2022]

For residual models, the DDIM output is delta_z in normalized space.
The final prediction is: z_hat = (delta + z_t_norm) * z_std + z_mean

Usage::

    python -m evaluation.tierb_eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.dit_utils import (
    DEFAULT_HORIZON,
    NATIVE_DIMS,
    TARGET_DIM,
    build_windows,
    load_embeddings,
)
from models.diffusion import CosineNoiseSchedule, DDIMSampler
from models.fourier_embed import FourierActionEmbedding

# Reuse inline LatentDiT from warmstart_eval
from evaluation.warmstart_eval import LatentDiT, DIT_CONFIG, DIFFUSION_CONFIG, FOURIER_CONFIG

DIT_CKPT_ROOT = Path("outputs/dits")
MLP_RESIDUAL_ROOT = Path("outputs/latent_predictors_residual")
EVAL_BATCH_SIZE = 64


def load_dit_objective_checkpoint(
    encoder_name: str, variant_tag: str, seed: int, device: torch.device,
):
    """Load a DiT checkpoint from the objectives experiment.

    variant_tag examples: "conditioned__x0", "conditioned__v",
    "conditioned__residual"
    """
    ckpt_path = DIT_CKPT_ROOT / encoder_name / variant_tag / f"seed_{seed}" / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    z_mean = ckpt["z_mean"].to(device)
    z_std = ckpt["z_std"].to(device)
    prediction = ckpt.get("prediction", "epsilon")
    residual = ckpt.get("residual", False)

    native_dim = NATIVE_DIMS[encoder_name]
    needs_adapter = native_dim != TARGET_DIM
    if needs_adapter:
        adapter = nn.Linear(native_dim, TARGET_DIM, bias=False).to(device)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        for p in adapter.parameters():
            p.requires_grad_(False)
    else:
        adapter = nn.Identity().to(device)

    fourier_embed = FourierActionEmbedding(
        action_dim=2,
        n_frequencies=FOURIER_CONFIG["n_frequencies"],
        base=FOURIER_CONFIG["base"],
        out_dim=FOURIER_CONFIG["out_dim"],
    ).to(device)
    fourier_embed.load_state_dict(ckpt["fourier_embed_state_dict"])
    fourier_embed.eval()

    dit = LatentDiT(**DIT_CONFIG).to(device)
    dit.load_state_dict(ckpt["dit_state_dict"])
    dit.eval()

    return dit, adapter, fourier_embed, z_mean, z_std, prediction, residual


def evaluate_dit_objective(
    dit, adapter, fourier_embed, z_mean, z_std,
    z_t_test, act_test, zf_test,
    prediction: str, residual: bool,
    n_steps: int = 50, seed: int = 0, device: torch.device = torch.device("cpu"),
):
    """Run DDIM sampling with the appropriate x0 recovery for each objective."""
    horizon = DEFAULT_HORIZON
    schedule = CosineNoiseSchedule(n_steps=DIFFUSION_CONFIG["n_train_steps"]).to(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    test_ds = TensorDataset(z_t_test, act_test, zf_test)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    alphas_cumprod = schedule.alphas_cumprod.to(device)

    # Build DDIM timestep subsequence
    T = DIFFUSION_CONFIG["n_train_steps"]
    stride = T // n_steps
    timesteps = list(range(0, T, stride))[:n_steps]
    timesteps = list(reversed(timesteps))

    cossim_sums = [0.0] * horizon
    mse_sums = [0.0] * horizon
    z_hat_norm_sum = 0.0
    z_real_norm_sum = 0.0
    total = 0

    with torch.no_grad():
        for z_t_batch, act_batch, zf_batch in test_loader:
            z_t_batch = z_t_batch.to(device)
            act_batch = act_batch.to(device)
            zf_batch = zf_batch.to(device)
            B = z_t_batch.shape[0]

            B_f, H, _ = zf_batch.shape
            z_t_adapted = (adapter(z_t_batch) - z_mean) / z_std
            zf_adapted = (
                adapter(zf_batch.reshape(B_f * H, -1)).reshape(B_f, H, TARGET_DIM)
                - z_mean
            ) / z_std

            a_embed = fourier_embed(act_batch)

            # Start from pure noise
            x = torch.randn(B, horizon, TARGET_DIM, device=device)

            # DDIM loop with objective-specific x0 recovery
            for i, t_val in enumerate(timesteps):
                t = torch.full((B,), t_val, device=device, dtype=torch.long)
                model_out = dit(x, z_t=z_t_adapted, a_embed=a_embed, timestep=t)

                alpha_bar_t = alphas_cumprod[t_val]

                # Recover pred_x0 based on prediction type
                if prediction == "epsilon":
                    pred_x0 = (
                        x - torch.sqrt(1.0 - alpha_bar_t) * model_out
                    ) / torch.sqrt(alpha_bar_t)
                elif prediction == "x0":
                    pred_x0 = model_out
                elif prediction == "v":
                    # x0 = sqrt(alpha) * x_t - sqrt(1 - alpha) * v
                    pred_x0 = (
                        torch.sqrt(alpha_bar_t) * x
                        - torch.sqrt(1.0 - alpha_bar_t) * model_out
                    )
                else:
                    raise ValueError(f"Unknown prediction: {prediction}")

                # Next alpha_bar
                if i < len(timesteps) - 1:
                    t_prev = timesteps[i + 1]
                    alpha_bar_prev = alphas_cumprod[t_prev]
                else:
                    alpha_bar_prev = torch.tensor(1.0, device=device)

                # DDIM step (same for all objectives after x0 recovery)
                noise_direction = (
                    x - torch.sqrt(alpha_bar_t) * pred_x0
                ) / torch.sqrt(1.0 - alpha_bar_t + 1e-8)

                x = (
                    torch.sqrt(alpha_bar_prev) * pred_x0
                    + torch.sqrt(1.0 - alpha_bar_prev) * noise_direction
                )

            # x is now the denoised output in normalized space
            z_hat_norm = x

            # For residual models: add z_t back
            if residual:
                z_t_expanded = z_t_adapted.unsqueeze(1).expand(-1, horizon, -1)
                z_hat_norm = z_hat_norm + z_t_expanded

            # Inverse transform
            z_hat = z_hat_norm * z_std + z_mean
            zf_orig = zf_adapted * z_std + z_mean

            z_hat_norm_sum += z_hat.norm(dim=-1).sum().item()
            z_real_norm_sum += zf_orig.norm(dim=-1).sum().item()

            for k in range(horizon):
                cs = F.cosine_similarity(z_hat[:, k], zf_orig[:, k], dim=-1)
                mse = ((z_hat[:, k] - zf_orig[:, k]) ** 2).mean(dim=-1)
                cossim_sums[k] += cs.sum().item()
                mse_sums[k] += mse.sum().item()

            total += B

    return {
        "cossim_by_horizon": [s / total for s in cossim_sums],
        "mse_by_horizon": [s / total for s in mse_sums],
        "z_hat_norm_ratio": z_hat_norm_sum / (z_real_norm_sum + 1e-8),
        "n_test_windows": total,
    }


def evaluate_residual_mlp(
    encoder_name: str, seed: int, device: torch.device,
):
    """Evaluate the residual MLP (B2 fairness control)."""
    from models.latent_pred import LatentPredictor
    from config import load_canonical

    ckpt_path = MLP_RESIDUAL_ROOT / encoder_name / "conditioned" / f"seed_{seed}" / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    z_mean = ckpt["z_mean"].to(device)
    z_std = ckpt["z_std"].to(device)

    native_dim = NATIVE_DIMS[encoder_name]
    needs_adapter = native_dim != TARGET_DIM
    if needs_adapter:
        adapter = nn.Linear(native_dim, TARGET_DIM, bias=False).to(device)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        for p in adapter.parameters():
            p.requires_grad_(False)
    else:
        adapter = nn.Identity().to(device)

    cfg = load_canonical()
    fourier_embed = FourierActionEmbedding.from_canonical(cfg).to(device)
    fourier_embed.load_state_dict(ckpt["fourier_embed_state_dict"])
    fourier_embed.eval()

    predictor = LatentPredictor.from_canonical(cfg).to(device)
    predictor.load_state_dict(ckpt["predictor_state_dict"])
    predictor.eval()

    # Load test data
    data = load_embeddings(encoder_name)
    result = build_windows(data, split="test", horizon=DEFAULT_HORIZON)
    z_t_test, act_test, zf_test = result
    z_t_test = z_t_test.to(device)
    act_test = act_test.to(device)
    zf_test = zf_test.to(device)

    horizon = DEFAULT_HORIZON
    test_ds = TensorDataset(z_t_test, act_test, zf_test)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    cossim_sums = [0.0] * horizon
    mse_sums = [0.0] * horizon
    z_hat_norm_sum = 0.0
    z_real_norm_sum = 0.0
    total = 0

    with torch.no_grad():
        for z_t_batch, act_batch, zf_batch in test_loader:
            z_t_batch = z_t_batch.to(device)
            act_batch = act_batch.to(device)
            zf_batch = zf_batch.to(device)
            B, H, _ = zf_batch.shape

            z_t_adapted = (adapter(z_t_batch) - z_mean) / z_std
            zf_adapted = (
                adapter(zf_batch.reshape(B * H, -1)).reshape(B, H, TARGET_DIM)
                - z_mean
            ) / z_std

            a_embed = fourier_embed(act_batch)

            # Predict delta in normalized space
            z_hat_delta = predictor(z_t_adapted, a_embed)

            # Reconstruct absolute: z_hat = delta + z_t
            z_t_expanded = z_t_adapted.unsqueeze(1).expand(-1, H, -1)
            z_hat_norm = z_hat_delta + z_t_expanded

            # Inverse transform
            z_hat = z_hat_norm * z_std + z_mean
            zf_orig = zf_adapted * z_std + z_mean

            z_hat_norm_sum += z_hat.norm(dim=-1).sum().item()
            z_real_norm_sum += zf_orig.norm(dim=-1).sum().item()

            for k in range(horizon):
                cs = F.cosine_similarity(z_hat[:, k], zf_orig[:, k], dim=-1)
                mse = ((z_hat[:, k] - zf_orig[:, k]) ** 2).mean(dim=-1)
                cossim_sums[k] += cs.sum().item()
                mse_sums[k] += mse.sum().item()

            total += B

    return {
        "cossim_by_horizon": [s / total for s in cossim_sums],
        "mse_by_horizon": [s / total for s in mse_sums],
        "z_hat_norm_ratio": z_hat_norm_sum / (z_real_norm_sum + 1e-8),
        "n_test_windows": total,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="DA8 Tier B evaluation")
    parser.add_argument(
        "--encoders", nargs="+", default=None,
        help="Encoders to evaluate (default: all available).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2],
        help="Seeds to evaluate (default: 0 1 2).",
    )
    parser.add_argument(
        "--objectives", nargs="+", default=["x0"],
        help="DiT objectives to evaluate (default: x0).",
    )
    parser.add_argument(
        "--n-steps", type=int, default=50,
        help="DDIM sampling steps (default: 50).",
    )
    parser.add_argument(
        "--skip-mlp", action="store_true",
        help="Skip residual MLP evaluation.",
    )
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    all_encoders = sorted(NATIVE_DIMS.keys())
    encoders = args.encoders if args.encoders else all_encoders
    seeds = args.seeds
    n_steps = args.n_steps

    print("DA8 Tier B Evaluation (Full Sweep)")
    print(f"  encoders: {encoders}")
    print(f"  seeds: {seeds}")
    print(f"  objectives: {args.objectives}")
    print(f"  DDIM steps: {n_steps}, device: {device}")
    print()

    all_results = []  # list of dicts for CSV-style output

    for encoder in encoders:
        print(f"\n{'='*50} {encoder} {'='*50}")
        data = load_embeddings(encoder)
        test_win = build_windows(data, split="test", horizon=DEFAULT_HORIZON)
        if test_win is None:
            print(f"  [skip] No test windows for {encoder}")
            continue
        z_t_test = test_win[0].to(device)
        act_test = test_win[1].to(device)
        zf_test = test_win[2].to(device)

        for seed in seeds:
            # DiT objectives
            for obj_tag in args.objectives:
                variant_tag = f"conditioned__{obj_tag}"
                ckpt_path = (
                    DIT_CKPT_ROOT / encoder / variant_tag
                    / f"seed_{seed}" / "checkpoint.pt"
                )
                if not ckpt_path.exists():
                    print(f"  [skip] {encoder}/{variant_tag}/seed_{seed}: not found")
                    continue

                print(f"  [eval] {encoder}/dit_{obj_tag}/seed_{seed}...", end=" ")
                dit, adapter, fourier, z_mean, z_std, prediction, residual = \
                    load_dit_objective_checkpoint(encoder, variant_tag, seed, device)

                r = evaluate_dit_objective(
                    dit, adapter, fourier, z_mean, z_std,
                    z_t_test, act_test, zf_test,
                    prediction=prediction, residual=residual,
                    n_steps=n_steps, seed=seed, device=device,
                )
                print(
                    f"CosSim@h1={r['cossim_by_horizon'][0]:.4f}  "
                    f"norm={r['z_hat_norm_ratio']:.3f}"
                )
                all_results.append({
                    "encoder": encoder,
                    "model": f"dit_{obj_tag}",
                    "seed": seed,
                    **{f"cossim_h{k+1}": r["cossim_by_horizon"][k]
                       for k in range(len(r["cossim_by_horizon"]))},
                    **{f"mse_h{k+1}": r["mse_by_horizon"][k]
                       for k in range(len(r["mse_by_horizon"]))},
                    "z_hat_norm_ratio": r["z_hat_norm_ratio"],
                    "n_test": r["n_test_windows"],
                })

                # Free GPU memory
                del dit, adapter, fourier
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # Residual MLP
            if not args.skip_mlp:
                mlp_ckpt = (
                    MLP_RESIDUAL_ROOT / encoder / "conditioned"
                    / f"seed_{seed}" / "checkpoint.pt"
                )
                if mlp_ckpt.exists():
                    print(f"  [eval] {encoder}/mlp_residual/seed_{seed}...", end=" ")
                    r = evaluate_residual_mlp(encoder, seed, device)
                    print(
                        f"CosSim@h1={r['cossim_by_horizon'][0]:.4f}  "
                        f"norm={r['z_hat_norm_ratio']:.3f}"
                    )
                    all_results.append({
                        "encoder": encoder,
                        "model": "mlp_residual",
                        "seed": seed,
                        **{f"cossim_h{k+1}": r["cossim_by_horizon"][k]
                           for k in range(len(r["cossim_by_horizon"]))},
                        **{f"mse_h{k+1}": r["mse_by_horizon"][k]
                           for k in range(len(r["mse_by_horizon"]))},
                        "z_hat_norm_ratio": r["z_hat_norm_ratio"],
                        "n_test": r["n_test_windows"],
                    })

    # Summary table
    print("\n" + "=" * 90)
    print(
        f"{'Encoder':<16} {'Model':<16} {'Seed':>4} "
        f"{'CosSim@h1':>10} {'MSE@h1':>10} {'Norm Ratio':>12}"
    )
    print("-" * 90)
    for row in sorted(all_results, key=lambda x: (x["encoder"], x["model"], x["seed"])):
        print(
            f"{row['encoder']:<16} {row['model']:<16} {row['seed']:>4} "
            f"{row['cossim_h1']:>10.4f} {row['mse_h1']:>10.4f} "
            f"{row['z_hat_norm_ratio']:>12.3f}"
        )

    # Per-encoder mean across seeds
    print("\n--- Per-encoder mean (across seeds) ---")
    print(f"{'Encoder':<16} {'Model':<16} {'CosSim@h1':>10} {'MSE@h1':>10}")
    print("-" * 60)
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in all_results:
        grouped[(row["encoder"], row["model"])].append(row)
    for (enc, model), rows in sorted(grouped.items()):
        mean_cs = np.mean([r["cossim_h1"] for r in rows])
        mean_mse = np.mean([r["mse_h1"] for r in rows])
        print(f"{enc:<16} {model:<16} {mean_cs:>10.4f} {mean_mse:>10.4f}")

    # Save CSV
    import csv
    csv_path = Path("artifacts/full/da8_tierb_full.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSaved CSV to {csv_path}")

    # Save JSON
    out_path = Path("artifacts/full/da8_tierb_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "encoders": encoders,
            "seeds": seeds,
            "objectives": args.objectives,
            "n_steps": n_steps,
            "rows": all_results,
        }, f, indent=2)
    print(f"Saved JSON to {out_path}")


if __name__ == "__main__":
    main()
