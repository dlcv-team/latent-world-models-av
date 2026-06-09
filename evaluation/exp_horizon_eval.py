"""DA9 Experiment 3: Evaluate DiT-x0, MLP-fair, MLP-residual at longer horizons.

Evaluates models trained at h=8 and h=16. For h=4 baseline, reuses
existing DA8 Tier B results (no retraining).

Usage::

    python -m evaluation.exp_horizon_eval --pilot
    python -m evaluation.exp_horizon_eval --full
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
    reconstruct_adapter_and_stats,
)
from evaluation.tierb_eval import (
    EVAL_BATCH_SIZE,
    DIT_CKPT_ROOT,
)
from evaluation.warmstart_eval import LatentDiT, DIT_CONFIG, DIFFUSION_CONFIG, FOURIER_CONFIG
from models.diffusion import CosineNoiseSchedule
from models.fourier_embed import FourierActionEmbedding
from models.latent_pred import LatentPredictor

# Output paths
PILOT_CSV_PATH = Path("artifacts/full/da9_horizon_pilot.csv")
FULL_CSV_PATH = Path("artifacts/full/da9_horizon_full.csv")
DA8_CSV = Path("artifacts/full/da8_tierb_full.csv")

PILOT_ENCODERS = ["vit_s16", "clip_b32"]
FULL_ENCODERS = sorted(NATIVE_DIMS.keys())
HORIZONS = [8, 16]
SEEDS_PILOT = [0]
SEEDS_FULL = [0, 1, 2]


def load_da8_h4_results() -> dict[tuple[str, str, int], dict]:
    """Load DA8 h=4 results to avoid retraining."""
    ref = {}
    if not DA8_CSV.exists():
        return ref
    with open(DA8_CSV) as f:
        for row in csv.DictReader(f):
            key = (row["encoder"], row["model"], int(row["seed"]))
            ref[key] = {
                f"cossim_h{k}": float(row[f"cossim_h{k}"])
                for k in range(1, DEFAULT_HORIZON + 1)
            }
    return ref


def evaluate_dit_x0_horizon(
    encoder_name: str, seed: int, horizon: int,
    device: torch.device,
) -> dict | None:
    """Evaluate DiT-x0 checkpoint trained at given horizon.

    Returns per-horizon CosSim + copy baseline, or None if checkpoint missing.
    """
    variant_tag = f"conditioned__x0__h{horizon}"
    ckpt_path = DIT_CKPT_ROOT / encoder_name / variant_tag / f"seed_{seed}" / "checkpoint.pt"
    if not ckpt_path.exists():
        print(f"  [skip] Missing: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    z_mean = ckpt["z_mean"].to(device)
    z_std = ckpt["z_std"].to(device)
    prediction = ckpt.get("prediction", "x0")
    assert prediction == "x0", f"Expected x0, got {prediction}"

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

    dit = LatentDiT(**{**DIT_CONFIG, "horizon": horizon}).to(device)
    dit.load_state_dict(ckpt["dit_state_dict"])
    dit.eval()

    # Load test data at this horizon
    data = load_embeddings(encoder_name)
    test_win = build_windows(data, split="test", horizon=horizon)
    if test_win is None:
        return None

    z_t_test, act_test, zf_test = test_win
    z_t_test = z_t_test.to(device)
    act_test = act_test.to(device)
    zf_test = zf_test.to(device)

    schedule = CosineNoiseSchedule(n_steps=DIFFUSION_CONFIG["n_train_steps"]).to(device)
    alphas_cumprod = schedule.alphas_cumprod.to(device)

    n_steps = 50
    T = DIFFUSION_CONFIG["n_train_steps"]
    stride = T // n_steps
    timesteps = list(range(0, T, stride))[:n_steps]
    timesteps = list(reversed(timesteps))

    torch.manual_seed(seed)
    np.random.seed(seed)

    test_ds = TensorDataset(z_t_test, act_test, zf_test)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    cossim_sums = [0.0] * horizon
    copy_sums = [0.0] * horizon
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
            x = torch.randn(B, horizon, TARGET_DIM, device=device)

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

            z_hat = x * z_std + z_mean
            zf_orig = zf_adapted * z_std + z_mean
            z_t_unnorm = z_t_adapted * z_std + z_mean

            for k in range(horizon):
                cs = F.cosine_similarity(z_hat[:, k], zf_orig[:, k], dim=-1)
                cossim_sums[k] += cs.sum().item()
                copy_cs = F.cosine_similarity(z_t_unnorm, zf_orig[:, k], dim=-1)
                copy_sums[k] += copy_cs.sum().item()

            total += B

    del dit, adapter, fourier_embed
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "cossim_by_horizon": [s / total for s in cossim_sums],
        "copy_by_horizon": [s / total for s in copy_sums],
        "n_test_windows": total,
    }


def evaluate_mlp_fair_horizon(
    encoder_name: str, seed: int, horizon: int,
    device: torch.device,
) -> dict | None:
    """Evaluate MLP-fair checkpoint trained at given horizon."""
    ckpt_path = (
        Path(f"outputs/latent_predictors_fair_h{horizon}")
        / encoder_name / "conditioned" / f"seed_{seed}" / "checkpoint.pt"
    )
    if not ckpt_path.exists():
        print(f"  [skip] Missing: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    z_mean = ckpt["z_mean"].to(device)
    z_std = ckpt["z_std"].to(device)

    native_dim = NATIVE_DIMS[encoder_name]
    needs_adapter = native_dim != TARGET_DIM
    if needs_adapter and ckpt["adapter_state_dict"]:
        adapter = nn.Linear(native_dim, TARGET_DIM, bias=False).to(device)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        for p in adapter.parameters():
            p.requires_grad_(False)
    else:
        adapter = nn.Identity().to(device)

    cfg = load_canonical()
    predictor = LatentPredictor(
        z_dim=TARGET_DIM,
        a_dim=TARGET_DIM,
        horizon=horizon,
    ).to(device)
    predictor.load_state_dict(ckpt["predictor_state_dict"])
    predictor.eval()

    fourier_embed = FourierActionEmbedding.from_canonical(cfg).to(device)
    fourier_embed.load_state_dict(ckpt["fourier_embed_state_dict"])
    fourier_embed.eval()

    data = load_embeddings(encoder_name)
    test_win = build_windows(data, split="test", horizon=horizon)
    if test_win is None:
        return None

    z_t_test, act_test, zf_test = test_win
    z_t_test = z_t_test.to(device)
    act_test = act_test.to(device)
    zf_test = zf_test.to(device)

    test_ds = TensorDataset(z_t_test, act_test, zf_test)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    cossim_sums = [0.0] * horizon
    copy_sums = [0.0] * horizon
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
            z_hat_norm = predictor(z_t_adapted, a_embed)

            z_hat = z_hat_norm * z_std + z_mean
            zf_orig = zf_adapted * z_std + z_mean
            z_t_unnorm = z_t_adapted * z_std + z_mean

            for k in range(horizon):
                cs = F.cosine_similarity(z_hat[:, k], zf_orig[:, k], dim=-1)
                cossim_sums[k] += cs.sum().item()
                copy_cs = F.cosine_similarity(z_t_unnorm, zf_orig[:, k], dim=-1)
                copy_sums[k] += copy_cs.sum().item()

            total += B

    return {
        "cossim_by_horizon": [s / total for s in cossim_sums],
        "copy_by_horizon": [s / total for s in copy_sums],
        "n_test_windows": total,
    }


def evaluate_mlp_residual_horizon(
    encoder_name: str, seed: int, horizon: int,
    device: torch.device,
) -> dict | None:
    """Evaluate residual MLP checkpoint trained at given horizon."""
    ckpt_path = (
        Path(f"outputs/latent_predictors_residual_h{horizon}")
        / encoder_name / "conditioned" / f"seed_{seed}" / "checkpoint.pt"
    )
    if not ckpt_path.exists():
        print(f"  [skip] Missing: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    z_mean = ckpt["z_mean"].to(device)
    z_std = ckpt["z_std"].to(device)

    native_dim = NATIVE_DIMS[encoder_name]
    needs_adapter = native_dim != TARGET_DIM
    if needs_adapter and ckpt["adapter_state_dict"]:
        adapter = nn.Linear(native_dim, TARGET_DIM, bias=False).to(device)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        for p in adapter.parameters():
            p.requires_grad_(False)
    else:
        adapter = nn.Identity().to(device)

    cfg = load_canonical()
    predictor = LatentPredictor(
        z_dim=TARGET_DIM,
        a_dim=TARGET_DIM,
        horizon=horizon,
    ).to(device)
    predictor.load_state_dict(ckpt["predictor_state_dict"])
    predictor.eval()

    fourier_embed = FourierActionEmbedding.from_canonical(cfg).to(device)
    fourier_embed.load_state_dict(ckpt["fourier_embed_state_dict"])
    fourier_embed.eval()

    data = load_embeddings(encoder_name)
    test_win = build_windows(data, split="test", horizon=horizon)
    if test_win is None:
        return None

    z_t_test, act_test, zf_test = test_win
    z_t_test = z_t_test.to(device)
    act_test = act_test.to(device)
    zf_test = zf_test.to(device)

    test_ds = TensorDataset(z_t_test, act_test, zf_test)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    cossim_sums = [0.0] * horizon
    copy_sums = [0.0] * horizon
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
            z_hat_delta = predictor(z_t_adapted, a_embed)

            z_t_expanded = z_t_adapted.unsqueeze(1).expand(-1, H, -1)
            z_hat_norm = z_hat_delta + z_t_expanded

            z_hat = z_hat_norm * z_std + z_mean
            zf_orig = zf_adapted * z_std + z_mean
            z_t_unnorm = z_t_adapted * z_std + z_mean

            for k in range(horizon):
                cs = F.cosine_similarity(z_hat[:, k], zf_orig[:, k], dim=-1)
                cossim_sums[k] += cs.sum().item()
                copy_cs = F.cosine_similarity(z_t_unnorm, zf_orig[:, k], dim=-1)
                copy_sums[k] += copy_cs.sum().item()

            total += B

    return {
        "cossim_by_horizon": [s / total for s in cossim_sums],
        "copy_by_horizon": [s / total for s in copy_sums],
        "n_test_windows": total,
    }


def run_horizon_eval(
    encoders: list[str],
    horizons: list[int],
    seeds: list[int],
    device: torch.device,
) -> list[dict]:
    """Evaluate all models at all horizons. Include DA8 h=4 baseline."""
    da8_ref = load_da8_h4_results()
    all_rows: list[dict] = []

    for enc in encoders:
        print(f"\n{'='*60} {enc} {'='*60}")

        for seed in seeds:
            # h=4 from DA8 (no retraining)
            for model_name, da8_model_key in [
                ("dit_x0", "dit_x0"),
                ("mlp_residual", "mlp_residual"),
            ]:
                da8_key = (enc, da8_model_key, seed)
                if da8_key in da8_ref:
                    for k in range(DEFAULT_HORIZON):
                        all_rows.append({
                            "encoder": enc,
                            "seed": seed,
                            "horizon_trained": DEFAULT_HORIZON,
                            "horizon_step": k + 1,
                            "model": model_name,
                            "cossim": da8_ref[da8_key][f"cossim_h{k+1}"],
                            "source": "da8",
                        })

            for h in horizons:
                print(f"\n  --- {enc}/seed={seed}/h={h} ---")

                # DiT-x0
                dit_r = evaluate_dit_x0_horizon(enc, seed, h, device)
                if dit_r is not None:
                    for k in range(h):
                        all_rows.append({
                            "encoder": enc,
                            "seed": seed,
                            "horizon_trained": h,
                            "horizon_step": k + 1,
                            "model": "dit_x0",
                            "cossim": round(dit_r["cossim_by_horizon"][k], 6),
                            "copy_cossim": round(dit_r["copy_by_horizon"][k], 6),
                            "n_windows": dit_r["n_test_windows"],
                            "source": "da9",
                        })
                    print(f"    dit_x0: CosSim@h1={dit_r['cossim_by_horizon'][0]:.4f}")

                # MLP-fair
                mlp_r = evaluate_mlp_fair_horizon(enc, seed, h, device)
                if mlp_r is not None:
                    for k in range(h):
                        all_rows.append({
                            "encoder": enc,
                            "seed": seed,
                            "horizon_trained": h,
                            "horizon_step": k + 1,
                            "model": "mlp_fair",
                            "cossim": round(mlp_r["cossim_by_horizon"][k], 6),
                            "copy_cossim": round(mlp_r["copy_by_horizon"][k], 6),
                            "n_windows": mlp_r["n_test_windows"],
                            "source": "da9",
                        })
                    print(f"    mlp_fair: CosSim@h1={mlp_r['cossim_by_horizon'][0]:.4f}")

                # MLP-residual
                mlp_res_r = evaluate_mlp_residual_horizon(enc, seed, h, device)
                if mlp_res_r is not None:
                    for k in range(h):
                        all_rows.append({
                            "encoder": enc,
                            "seed": seed,
                            "horizon_trained": h,
                            "horizon_step": k + 1,
                            "model": "mlp_residual",
                            "cossim": round(mlp_res_r["cossim_by_horizon"][k], 6),
                            "copy_cossim": round(mlp_res_r["copy_by_horizon"][k], 6),
                            "n_windows": mlp_res_r["n_test_windows"],
                            "source": "da9",
                        })
                    print(f"    mlp_res: CosSim@h1={mlp_res_r['cossim_by_horizon'][0]:.4f}")

                # Copy baseline (just record from any model's copy_by_horizon)
                copy_source = dit_r or mlp_r or mlp_res_r
                if copy_source is not None:
                    for k in range(h):
                        all_rows.append({
                            "encoder": enc,
                            "seed": seed,
                            "horizon_trained": h,
                            "horizon_step": k + 1,
                            "model": "copy",
                            "cossim": round(copy_source["copy_by_horizon"][k], 6),
                            "n_windows": copy_source["n_test_windows"],
                            "source": "da9",
                        })

    return all_rows


def evaluate_gate(rows: list[dict], pilot_encoders: list[str]) -> dict:
    """Pre-registered gate decision for Exp 3 pilot.

    For each pilot encoder on Q1 of h=16 (approximated as overall mean
    since we haven't done quartile analysis on longer horizons yet):
    - If mlp_residual_h4 - dit_x0_h4 > 0.02: compute gap_closure
    - PASS if gap_closure >= 0.30
    - ALSO PASS if |dit_x0 - mlp_residual| < 0.02 AND mlp_residual > 0.60
    """
    df = pd.DataFrame(rows)

    gate_results = {}
    any_pass = False

    for enc in pilot_encoders:
        enc_df = df[df["encoder"] == enc]

        # h=4 baseline (last step, from DA8)
        h4_dit = enc_df[
            (enc_df["model"] == "dit_x0")
            & (enc_df["horizon_trained"] == 4)
            & (enc_df["horizon_step"] == 4)
        ]["cossim"].mean()

        h4_mlp_res = enc_df[
            (enc_df["model"] == "mlp_residual")
            & (enc_df["horizon_trained"] == 4)
            & (enc_df["horizon_step"] == 4)
        ]["cossim"].mean()

        # h=16, last step
        h16_dit = enc_df[
            (enc_df["model"] == "dit_x0")
            & (enc_df["horizon_trained"] == 16)
            & (enc_df["horizon_step"] == 16)
        ]["cossim"].mean()

        h16_mlp_res = enc_df[
            (enc_df["model"] == "mlp_residual")
            & (enc_df["horizon_trained"] == 16)
            & (enc_df["horizon_step"] == 16)
        ]["cossim"].mean()

        h4_gap = h4_mlp_res - h4_dit
        h16_gap = h16_mlp_res - h16_dit

        gate_info = {
            "encoder": enc,
            "h4_dit": round(h4_dit, 6),
            "h4_mlp_res": round(h4_mlp_res, 6),
            "h4_gap": round(h4_gap, 6),
            "h16_dit": round(h16_dit, 6) if not pd.isna(h16_dit) else None,
            "h16_mlp_res": round(h16_mlp_res, 6) if not pd.isna(h16_mlp_res) else None,
            "h16_gap": round(h16_gap, 6) if not pd.isna(h16_gap) else None,
        }

        passed = False

        # Gap closure check
        if h4_gap > 0.02 and not pd.isna(h16_gap):
            gap_closure = 1 - h16_gap / h4_gap
            gate_info["gap_closure"] = round(gap_closure, 4)
            if gap_closure >= 0.30:
                passed = True
                gate_info["pass_reason"] = "gap_closure >= 0.30"

        # Absolute proximity check
        if not passed and not pd.isna(h16_gap) and not pd.isna(h16_mlp_res):
            if abs(h16_gap) < 0.02 and h16_mlp_res > 0.60:
                passed = True
                gate_info["pass_reason"] = "absolute_proximity (gap < 0.02, floor > 0.60)"

        # DiT beats MLP
        if not passed and not pd.isna(h16_gap):
            if h16_gap < 0:
                passed = True
                gate_info["pass_reason"] = "dit_advantage"

        gate_info["passed"] = passed
        gate_results[enc] = gate_info
        if passed:
            any_pass = True

    return {
        "per_encoder": gate_results,
        "any_pass": any_pass,
    }


def main():
    parser = argparse.ArgumentParser(description="DA9 Exp 3: Horizon evaluation")
    parser.add_argument("--pilot", action="store_true", help="Pilot evaluation")
    parser.add_argument("--full", action="store_true", help="Full evaluation")
    parser.add_argument("--encoders", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    if args.full:
        encoders = args.encoders or FULL_ENCODERS
        seeds = args.seeds or SEEDS_FULL
        csv_path = FULL_CSV_PATH
    else:
        encoders = args.encoders or PILOT_ENCODERS
        seeds = args.seeds or SEEDS_PILOT
        csv_path = PILOT_CSV_PATH

    print("DA9 Experiment 3: Horizon Evaluation")
    print(f"  encoders: {encoders}")
    print(f"  horizons: {HORIZONS}")
    print(f"  seeds: {seeds}")
    print(f"  device: {device}")
    print(f"  output: {csv_path}")

    rows = run_horizon_eval(encoders, HORIZONS, seeds, device)

    if not rows:
        print("No results generated.")
        sys.exit(1)

    # Save CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    # Ensure all rows have same keys
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV to {csv_path} ({len(rows)} rows)")

    # Gate decision (pilot only)
    if not args.full:
        gate = evaluate_gate(rows, PILOT_ENCODERS)
        print("\n" + "=" * 60)
        print("GATE DECISION")
        print("=" * 60)
        for enc, info in gate["per_encoder"].items():
            status = "PASS" if info["passed"] else "FAIL"
            reason = info.get("pass_reason", "no criterion met")
            print(f"  {enc}: {status} ({reason})")
            print(f"    h4: DiT={info['h4_dit']}, MLP-res={info['h4_mlp_res']}, gap={info['h4_gap']}")
            if info.get("h16_dit") is not None:
                print(f"    h16: DiT={info['h16_dit']}, MLP-res={info['h16_mlp_res']}, gap={info['h16_gap']}")
                if "gap_closure" in info:
                    print(f"    gap_closure={info['gap_closure']}")

        print(f"\n  Overall: {'PASS -> expand to full' if gate['any_pass'] else 'FAIL -> proceed to Exp 1'}")

        # Save gate result
        import json
        gate_path = Path("artifacts/full/da9_gate_result.json")
        with open(gate_path, "w") as f:
            json.dump(gate, f, indent=2)
        print(f"  Gate result saved to {gate_path}")


if __name__ == "__main__":
    main()
