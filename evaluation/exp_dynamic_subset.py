"""DA9 Experiment 0: Dynamic subset analysis.

Mine existing DA8 models (DiT-x0, MLP-residual, MLP-fair, Copy) for
difficulty-dependent patterns. Split test windows into quartiles by
copy-baseline CosSim *per encoder and per horizon*, then report
per-quartile CosSim for all four models.

No retraining -- inference only on existing checkpoints.

Usage::

    python -m evaluation.exp_dynamic_subset [--encoders vit_s16 clip_b32]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from evaluation.dit_utils import (
    DEFAULT_HORIZON,
    NATIVE_DIMS,
    TARGET_DIM,
    build_windows,
    load_embeddings,
)
from evaluation.tierb_eval import (
    EVAL_BATCH_SIZE,
    evaluate_dit_objective,
    evaluate_residual_mlp,
    load_dit_objective_checkpoint,
)
from evaluation.dit_eval import evaluate_mlp

# Output paths
CSV_PATH = Path("artifacts/full/da9_dynamic_subset.csv")
AGG_CSV_PATH = Path("artifacts/full/da9_dynamic_subset_agg.csv")

ENCODERS = sorted(NATIVE_DIMS.keys())
SEEDS = [0, 1, 2]
HORIZON = DEFAULT_HORIZON
DA8_CSV = Path("artifacts/full/da8_tierb_full.csv")


def load_da8_reference() -> dict[tuple[str, str, int], dict[str, float]]:
    """Load DA8 Tier B aggregated results for verification.

    Returns dict keyed by (encoder, model, seed) -> {cossim_h1..h4}.
    """
    ref = {}
    with open(DA8_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["encoder"], row["model"], int(row["seed"]))
            ref[key] = {
                f"cossim_h{k}": float(row[f"cossim_h{k}"])
                for k in range(1, HORIZON + 1)
            }
    return ref


def run_exp0(
    encoders: list[str],
    seeds: list[int],
    device: torch.device,
) -> list[dict]:
    """Run Experiment 0: per-quartile analysis of all four models.

    Returns list of per-seed, per-quartile, per-horizon rows.
    """
    da8_ref = load_da8_reference()
    all_rows: list[dict] = []

    for enc in encoders:
        print(f"\n{'='*60} {enc} {'='*60}")
        t0 = time.time()

        # Load embeddings once per encoder
        data = load_embeddings(enc)
        test_win = build_windows(data, split="test", horizon=HORIZON)
        if test_win is None:
            print(f"  [skip] No test windows for {enc}")
            continue

        z_t_test, act_test, zf_test = test_win
        n_windows = len(z_t_test)
        print(f"  {n_windows} test windows")

        for seed in seeds:
            print(f"\n  --- seed {seed} ---")

            # 1. DiT-x0 with per-window output
            print(f"  [dit_x0] Loading checkpoint...", end=" ")
            dit, adapter, fourier, z_mean, z_std, prediction, residual = \
                load_dit_objective_checkpoint(enc, "conditioned__x0", seed, device)

            dit_result = evaluate_dit_objective(
                dit, adapter, fourier, z_mean, z_std,
                z_t_test.to(device), act_test.to(device), zf_test.to(device),
                prediction=prediction, residual=residual,
                n_steps=50, seed=seed, device=device,
                per_window=True,
            )
            dit_pw = dit_result["per_window_cossim"]      # list of (N,) arrays
            dit_copy_pw = dit_result["per_window_copy_cossim"]
            print(f"CosSim@h1={dit_result['cossim_by_horizon'][0]:.4f}")

            # Verify against DA8 reference
            da8_key = (enc, "dit_x0", seed)
            if da8_key in da8_ref:
                for k in range(HORIZON):
                    da8_val = da8_ref[da8_key][f"cossim_h{k+1}"]
                    our_val = dit_result["cossim_by_horizon"][k]
                    diff = abs(da8_val - our_val)
                    if diff > 0.001:
                        print(f"  WARNING: DA8 mismatch at h{k+1}: "
                              f"DA8={da8_val:.6f}, ours={our_val:.6f}, "
                              f"diff={diff:.6f}")

            del dit, adapter, fourier
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # 2. MLP-residual with per-window output
            print(f"  [mlp_residual] Evaluating...", end=" ")
            mlp_res_result = evaluate_residual_mlp(enc, seed, device, per_window=True)
            mlp_res_pw = mlp_res_result["per_window_cossim"]
            mlp_res_copy_pw = mlp_res_result["per_window_copy_cossim"]
            print(f"CosSim@h1={mlp_res_result['cossim_by_horizon'][0]:.4f}")

            # Verify MLP-residual against DA8
            da8_key_mlp = (enc, "mlp_residual", seed)
            if da8_key_mlp in da8_ref:
                for k in range(HORIZON):
                    da8_val = da8_ref[da8_key_mlp][f"cossim_h{k+1}"]
                    our_val = mlp_res_result["cossim_by_horizon"][k]
                    diff = abs(da8_val - our_val)
                    if diff > 0.001:
                        print(f"  WARNING: DA8 mismatch at h{k+1}: "
                              f"DA8={da8_val:.6f}, ours={our_val:.6f}, "
                              f"diff={diff:.6f}")

            # 3. MLP-fair (conditioned) with per-window output
            print(f"  [mlp_fair] Evaluating...", end=" ")
            mlp_fair_result = evaluate_mlp(
                enc, "conditioned", seed, device=device, per_window=True,
            )
            if "error" in mlp_fair_result:
                print(f"SKIP: {mlp_fair_result['error']}")
                continue
            mlp_fair_pw = mlp_fair_result["per_window_cossim"]
            mlp_fair_copy_pw = mlp_fair_result["per_window_copy_cossim"]
            print(f"CosSim@h1={mlp_fair_result['metrics']['cossim_by_horizon'][0]:.4f}")

            # 4. Copy baseline is already in the per-window arrays
            # Use dit_copy_pw as the canonical copy baseline (all should be identical)

            # Build per-quartile rows for each horizon
            for k in range(HORIZON):
                h = k + 1
                copy_arr = dit_copy_pw[k]  # (N,) copy baseline at this horizon
                dit_arr = dit_pw[k]
                mlp_res_arr = mlp_res_pw[k]
                mlp_fair_arr = mlp_fair_pw[k]

                # Define quartiles on THIS horizon's copy baseline (per plan)
                try:
                    quartile_labels = pd.qcut(
                        copy_arr, q=4,
                        labels=["Q1 (hardest)", "Q2", "Q3", "Q4 (easiest)"],
                    )
                except ValueError:
                    # Too many duplicate values for qcut; use rank-based
                    quartile_labels = pd.qcut(
                        pd.Series(copy_arr).rank(method="first"),
                        q=4,
                        labels=["Q1 (hardest)", "Q2", "Q3", "Q4 (easiest)"],
                    )

                df = pd.DataFrame({
                    "copy_cossim": copy_arr,
                    "dit_x0_cossim": dit_arr,
                    "mlp_residual_cossim": mlp_res_arr,
                    "mlp_fair_cossim": mlp_fair_arr,
                    "quartile": quartile_labels,
                })

                for q_label, grp in df.groupby("quartile", observed=False):
                    row = {
                        "encoder": enc,
                        "seed": seed,
                        "quartile": str(q_label),
                        "horizon": h,
                        "dit_x0_cossim": round(grp["dit_x0_cossim"].mean(), 6),
                        "mlp_fair_cossim": round(grp["mlp_fair_cossim"].mean(), 6),
                        "mlp_residual_cossim": round(grp["mlp_residual_cossim"].mean(), 6),
                        "copy_cossim": round(grp["copy_cossim"].mean(), 6),
                        "dit_minus_mlp_residual": round(
                            (grp["dit_x0_cossim"] - grp["mlp_residual_cossim"]).mean(), 6
                        ),
                        "dit_minus_mlp_fair": round(
                            (grp["dit_x0_cossim"] - grp["mlp_fair_cossim"]).mean(), 6
                        ),
                        "dit_minus_copy": round(
                            (grp["dit_x0_cossim"] - grp["copy_cossim"]).mean(), 6
                        ),
                        "n_windows": len(grp),
                    }
                    all_rows.append(row)

            print(f"  Quartile rows generated for seed {seed}")

        elapsed = time.time() - t0
        print(f"  {enc} done in {elapsed:.1f}s")

    return all_rows


def verify_aggregation(rows: list[dict], da8_ref: dict) -> bool:
    """Verify weighted mean of quartiles matches DA8 aggregates within 0.001."""
    passed = True
    df = pd.DataFrame(rows)

    for (enc, seed, h), grp in df.groupby(["encoder", "seed", "horizon"]):
        # Weighted mean across quartiles
        total_w = grp["n_windows"].sum()
        for model_col, da8_model in [
            ("dit_x0_cossim", "dit_x0"),
            ("mlp_residual_cossim", "mlp_residual"),
        ]:
            weighted_mean = (grp[model_col] * grp["n_windows"]).sum() / total_w
            da8_key = (enc, da8_model, int(seed))
            if da8_key in da8_ref:
                da8_val = da8_ref[da8_key][f"cossim_h{int(h)}"]
                diff = abs(weighted_mean - da8_val)
                if diff > 0.001:
                    print(f"  FAIL: {enc}/{da8_model}/seed{seed}/h{h}: "
                          f"weighted={weighted_mean:.6f}, DA8={da8_val:.6f}, "
                          f"diff={diff:.6f}")
                    passed = False

    if passed:
        print("\nVerification PASSED: all weighted quartile means match DA8 within 0.001")
    else:
        print("\nVerification FAILED: some weighted means deviate from DA8 by > 0.001")

    return passed


def aggregate_across_seeds(rows: list[dict]) -> list[dict]:
    """Average per-seed rows across seeds to produce seed-aggregated output."""
    df = pd.DataFrame(rows)
    numeric_cols = [
        "dit_x0_cossim", "mlp_fair_cossim", "mlp_residual_cossim",
        "copy_cossim", "dit_minus_mlp_residual", "dit_minus_mlp_fair",
        "dit_minus_copy",
    ]
    agg = df.groupby(["encoder", "quartile", "horizon"]).agg(
        {col: "mean" for col in numeric_cols}
        | {"n_windows": "first", "seed": "count"}
    ).rename(columns={"seed": "n_seeds"}).reset_index()

    # Round
    for col in numeric_cols:
        agg[col] = agg[col].round(6)

    return agg.to_dict("records")


def print_summary(agg_rows: list[dict]) -> None:
    """Print human-readable summary table focused on Q1 findings."""
    df = pd.DataFrame(agg_rows)
    print("\n" + "=" * 90)
    print("DA9 Exp 0: DiT-x0 vs MLP-residual gap by difficulty quartile")
    print("  Positive gap = DiT wins")
    print("=" * 90)

    for h in [1, 4]:
        h_df = df[df["horizon"] == h].copy()
        if h_df.empty:
            continue
        print(f"\n--- Horizon {h} ---")
        print(f"{'Encoder':<16} {'Quartile':<16} {'DiT-x0':>8} {'MLP-res':>8} "
              f"{'MLP-fair':>8} {'Copy':>8} {'DiT-MLPr':>10}")
        print("-" * 80)
        for _, row in h_df.sort_values(["encoder", "quartile"]).iterrows():
            print(f"{row['encoder']:<16} {row['quartile']:<16} "
                  f"{row['dit_x0_cossim']:>8.4f} {row['mlp_residual_cossim']:>8.4f} "
                  f"{row['mlp_fair_cossim']:>8.4f} {row['copy_cossim']:>8.4f} "
                  f"{row['dit_minus_mlp_residual']:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description="DA9 Exp 0: Dynamic subset analysis")
    parser.add_argument(
        "--encoders", nargs="+", default=None,
        help="Encoders to evaluate (default: all 6).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Seeds to evaluate (default: 0 1 2).",
    )
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    encoders = args.encoders if args.encoders else ENCODERS
    seeds = args.seeds

    print("DA9 Experiment 0: Dynamic Subset Analysis")
    print(f"  encoders: {encoders}")
    print(f"  seeds: {seeds}")
    print(f"  device: {device}")
    print(f"  horizon: {HORIZON}")

    # Run
    rows = run_exp0(encoders, seeds, device)

    if not rows:
        print("No results generated.")
        sys.exit(1)

    # Save per-seed CSV
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-seed CSV to {CSV_PATH} ({len(rows)} rows)")

    # Verify against DA8
    da8_ref = load_da8_reference()
    verify_aggregation(rows, da8_ref)

    # Seed-aggregated CSV
    agg_rows = aggregate_across_seeds(rows)
    agg_fieldnames = list(agg_rows[0].keys())
    with open(AGG_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fieldnames)
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"Saved aggregated CSV to {AGG_CSV_PATH} ({len(agg_rows)} rows)")

    # Summary
    print_summary(agg_rows)


if __name__ == "__main__":
    main()
