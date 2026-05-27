"""DA7: Fair DiT-vs-MLP evaluation and comparison table generation.

Evaluates MLP predictors (trained via ``scripts/train_mlp_fair.py``)
on the test set using the same adapter/normalization as DiT, then
produces a unified comparison table.

Output artifacts:
  - ``artifacts/full/mlp_rollout_results.json`` -- raw MLP results
  - ``artifacts/full/dit_vs_mlp_comparison.csv`` -- unified table
  - ``artifacts/full/dit_vs_mlp_table.tex`` -- LaTeX for report

Usage
-----
    python -m evaluation.dit_eval
    python -m evaluation.dit_eval --device cuda
    python -m evaluation.dit_eval --encoders vit_s16 clip_b32
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from evaluation.dit_utils import (
    DEFAULT_HORIZON,
    NATIVE_DIMS,
    ROLLOUT_RESULTS_PATH,
    TARGET_DIM,
    build_windows,
    load_embeddings,
)
from models.fourier_embed import FourierActionEmbedding
from models.latent_pred import LatentPredictor

MLP_CKPT_ROOT = Path("outputs/latent_predictors_fair")
MLP_RESULTS_PATH = Path("artifacts/full/mlp_rollout_results.json")
COMPARISON_CSV_PATH = Path("artifacts/full/dit_vs_mlp_comparison.csv")
LATEX_PATH = Path("artifacts/full/dit_vs_mlp_table.tex")

ENCODER_NAMES = sorted(NATIVE_DIMS.keys())
SEEDS = [0, 1, 2]
VARIANTS = ["conditioned", "unconditioned"]


def evaluate_mlp(
    encoder_name: str,
    variant: str,
    seed: int,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run MLP predictor on test set.

    1. Load embeddings, build test windows.
    2. Load MLP checkpoint (contains adapter weights, z_mean, z_std,
       predictor + fourier_embed state dicts).
    3. Reconstruct adapter from checkpoint weights (NOT re-init).
    4. Normalize test inputs.
    5. Forward pass in normalized space.
    6. Inverse transform to adapted space.
    7. Compute CosSim, MSE, copy baseline (same metric space as DiT).

    Returns dict with same schema as DiT rollout results.
    """
    ckpt_path = (
        MLP_CKPT_ROOT / encoder_name / variant / f"seed_{seed}" / "checkpoint.pt"
    )
    if not ckpt_path.exists():
        return {"error": f"Missing checkpoint: {ckpt_path}"}

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    z_mean = ckpt["z_mean"].to(device)
    z_std = ckpt["z_std"].to(device)

    # Reconstruct adapter from checkpoint
    native_dim = NATIVE_DIMS[encoder_name]
    needs_adapter = native_dim != TARGET_DIM
    if needs_adapter and ckpt["adapter_state_dict"]:
        adapter: nn.Module = nn.Linear(native_dim, TARGET_DIM, bias=False).to(device)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        for p in adapter.parameters():
            p.requires_grad_(False)
    else:
        adapter = nn.Identity().to(device)

    # Reconstruct predictor + fourier_embed
    predictor = LatentPredictor.from_canonical().to(device)
    predictor.load_state_dict(ckpt["predictor_state_dict"])
    predictor.eval()

    fourier_embed = FourierActionEmbedding.from_canonical().to(device)
    fourier_embed.load_state_dict(ckpt["fourier_embed_state_dict"])
    fourier_embed.eval()

    # Load test data
    data = load_embeddings(encoder_name)
    test_win = build_windows(data, "test", DEFAULT_HORIZON)
    if test_win is None:
        return {"error": f"No test windows for {encoder_name}"}

    z_t_test, act_test, zf_test = test_win
    n_test = len(z_t_test)
    horizon = zf_test.shape[1]

    t0 = time.time()

    # Evaluate
    cossim_sums = [0.0] * horizon
    mse_sums = [0.0] * horizon
    copy_cossim_sums = [0.0] * horizon
    total_samples = 0

    batch_size = 256
    from torch.utils.data import DataLoader, TensorDataset

    test_ds = TensorDataset(z_t_test, act_test, zf_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for z_t_b, act_b, zf_b in test_loader:
            z_t_b = z_t_b.to(device)
            act_b = act_b.to(device)
            zf_b = zf_b.to(device)
            B = z_t_b.shape[0]

            # Adapter projection + normalize
            B_f, H, _ = zf_b.shape
            z_t_adapted = adapter(z_t_b)
            zf_adapted = adapter(zf_b.reshape(B_f * H, -1)).reshape(B_f, H, TARGET_DIM)

            z_t_norm = (z_t_adapted - z_mean) / z_std
            zf_norm = (zf_adapted - z_mean) / z_std

            # Action embedding
            a_embed = fourier_embed(act_b)
            if variant == "unconditioned":
                a_embed = torch.zeros_like(a_embed)

            # Forward pass in normalized space
            z_hat_norm = predictor(z_t_norm, a_embed)  # (B, H, 384)

            # Inverse transform for metrics in adapted space
            z_hat = z_hat_norm * z_std + z_mean
            z_t_orig = z_t_adapted  # already in adapted unnormalized space
            zf_orig = zf_adapted

            # Per-horizon metrics
            for k in range(horizon):
                z_hat_k = z_hat[:, k]
                z_real_k = zf_orig[:, k]

                cs = F.cosine_similarity(z_hat_k, z_real_k, dim=-1)
                cossim_sums[k] += cs.sum().item()

                mse = ((z_hat_k - z_real_k) ** 2).mean(dim=-1)
                mse_sums[k] += mse.sum().item()

                copy_cs = F.cosine_similarity(z_t_orig, z_real_k, dim=-1)
                copy_cossim_sums[k] += copy_cs.sum().item()

            total_samples += B

    elapsed = time.time() - t0

    cossim_by_horizon = [s / total_samples for s in cossim_sums]
    mse_by_horizon = [s / total_samples for s in mse_sums]
    copy_baseline_cossim = [s / total_samples for s in copy_cossim_sums]

    return {
        "encoder": encoder_name,
        "variant": variant,
        "seed": seed,
        "n_test_windows": total_samples,
        "metrics": {
            "cossim_by_horizon": cossim_by_horizon,
            "mse_by_horizon": mse_by_horizon,
            "copy_baseline_cossim": copy_baseline_cossim,
        },
        "time_s": round(elapsed, 1),
    }


def load_dit_results() -> dict:
    """Load DiT results from rollout_results.json."""
    with ROLLOUT_RESULTS_PATH.open() as fh:
        return json.load(fh)


def _build_summary(all_results: list[dict]) -> dict:
    """Build per-encoder seed-averaged summary from raw results.

    Same convention as ``rollout_dit.py::_build_summary()``.
    """
    grouped: dict = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if "error" in r:
            continue
        grouped[r["encoder"]][r["variant"]].append(r["metrics"])

    summary: dict = {}
    for enc, variants in sorted(grouped.items()):
        summary[enc] = {}
        for var, metrics_list in sorted(variants.items()):
            n_seeds = len(metrics_list)
            horizon = len(metrics_list[0]["cossim_by_horizon"])

            cossim_means, cossim_stds = [], []
            mse_means, mse_stds = [], []
            copy_means = []

            for k in range(horizon):
                vals = [m["cossim_by_horizon"][k] for m in metrics_list]
                mean = sum(vals) / n_seeds
                cossim_means.append(mean)
                if n_seeds > 1:
                    var_val = sum((v - mean) ** 2 for v in vals) / (n_seeds - 1)
                    cossim_stds.append(var_val**0.5)
                else:
                    cossim_stds.append(0.0)

                vals = [m["mse_by_horizon"][k] for m in metrics_list]
                mean = sum(vals) / n_seeds
                mse_means.append(mean)
                if n_seeds > 1:
                    var_val = sum((v - mean) ** 2 for v in vals) / (n_seeds - 1)
                    mse_stds.append(var_val**0.5)
                else:
                    mse_stds.append(0.0)

                vals = [m["copy_baseline_cossim"][k] for m in metrics_list]
                copy_means.append(sum(vals) / n_seeds)

            summary[enc][var] = {
                "cossim_mean": cossim_means,
                "cossim_std": cossim_stds,
                "mse_mean": mse_means,
                "mse_std": mse_stds,
                "copy_baseline_cossim": copy_means,
                "n_seeds": n_seeds,
            }

    return summary


def build_comparison_table(
    dit_results: list[dict],
    mlp_results: list[dict],
) -> list[dict]:
    """Build unified comparison rows.

    Columns: model, encoder, variant, horizon, cossim_mean, cossim_std_seed,
             mse_mean, mse_std_seed, n_seeds, n_test_windows, source.

    Copy baseline gets variant="none" (single row per encoder, since it is
    action-independent).
    """
    dit_summary = _build_summary(dit_results)
    mlp_summary = _build_summary(mlp_results)

    rows: list[dict] = []

    for enc in sorted(set(list(dit_summary.keys()) + list(mlp_summary.keys()))):
        # DiT rows
        if enc in dit_summary:
            for var in sorted(dit_summary[enc].keys()):
                s = dit_summary[enc][var]
                horizon = len(s["cossim_mean"])
                for k in range(horizon):
                    rows.append({
                        "model": "dit",
                        "encoder": enc,
                        "variant": var,
                        "horizon": k + 1,
                        "cossim_mean": s["cossim_mean"][k],
                        "cossim_std_seed": s["cossim_std"][k],
                        "mse_mean": s["mse_mean"][k],
                        "mse_std_seed": s["mse_std"][k],
                        "n_seeds": s["n_seeds"],
                        "n_test_windows": dit_results[0]["n_test_windows"],
                        "source": "rollout_results.json",
                    })

        # MLP rows
        if enc in mlp_summary:
            for var in sorted(mlp_summary[enc].keys()):
                s = mlp_summary[enc][var]
                horizon = len(s["cossim_mean"])
                for k in range(horizon):
                    rows.append({
                        "model": "mlp",
                        "encoder": enc,
                        "variant": var,
                        "horizon": k + 1,
                        "cossim_mean": s["cossim_mean"][k],
                        "cossim_std_seed": s["cossim_std"][k],
                        "mse_mean": s["mse_mean"][k],
                        "mse_std_seed": s["mse_std"][k],
                        "n_seeds": s["n_seeds"],
                        "n_test_windows": mlp_results[0]["n_test_windows"],
                        "source": "mlp_rollout_results.json",
                    })

        # Copy baseline (single set of rows per encoder, variant="none")
        # Use DiT copy baseline as the validated reference
        if enc in dit_summary:
            any_var = list(dit_summary[enc].keys())[0]
            s = dit_summary[enc][any_var]
            horizon = len(s["copy_baseline_cossim"])
            for k in range(horizon):
                rows.append({
                    "model": "copy_baseline",
                    "encoder": enc,
                    "variant": "none",
                    "horizon": k + 1,
                    "cossim_mean": s["copy_baseline_cossim"][k],
                    "cossim_std_seed": 0.0,  # same across seeds by definition
                    "mse_mean": None,
                    "mse_std_seed": None,
                    "n_seeds": s["n_seeds"],
                    "n_test_windows": dit_results[0]["n_test_windows"],
                    "source": "rollout_results.json",
                })

    return rows


def export_csv(rows: list[dict], path: Path) -> None:
    """Write comparison table to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "encoder", "variant", "horizon",
        "cossim_mean", "cossim_std_seed",
        "mse_mean", "mse_std_seed",
        "n_seeds", "n_test_windows", "source",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_latex(rows: list[dict], path: Path) -> None:
    """Export LaTeX table grouped by encoder, columns by model x horizon.

    Format: encoder | variant | model | k=1 | k=2 | k=3 | k=4
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{DiT vs MLP cosine similarity (mean $\pm$ std across seeds)}",
        r"\label{tab:dit-vs-mlp}",
        r"\scriptsize",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Encoder & Variant & Model & $k{=}1$ & $k{=}2$ & $k{=}3$ & $k{=}4$ \\",
        r"\midrule",
    ]

    # Group by encoder
    from itertools import groupby

    sorted_rows = sorted(rows, key=lambda r: (r["encoder"], r["variant"], r["model"]))
    prev_enc = None

    for (enc, var, model), group in groupby(
        sorted_rows, key=lambda r: (r["encoder"], r["variant"], r["model"])
    ):
        if prev_enc is not None and enc != prev_enc:
            lines.append(r"\midrule")
        prev_enc = enc

        group_list = sorted(group, key=lambda r: r["horizon"])
        cells = []
        for g in group_list:
            if g["cossim_mean"] is not None:
                mean = g["cossim_mean"]
                std = g["cossim_std_seed"]
                if std > 0:
                    cells.append(f"{mean:.4f}$\\pm${std:.4f}")
                else:
                    cells.append(f"{mean:.4f}")
            else:
                cells.append("--")

        while len(cells) < 4:
            cells.append("--")

        enc_display = enc.replace("_", r"\_")
        lines.append(
            f"  {enc_display} & {var} & {model} & "
            + " & ".join(cells)
            + r" \\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with path.open("w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="DA7: DiT vs MLP evaluation and comparison."
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=ENCODER_NAMES,
        choices=ENCODER_NAMES,
        help="Encoders to evaluate (default: all).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seeds (default: 0 1 2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: auto-detect).",
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

    # Verify all expected checkpoints exist
    missing = []
    for enc in args.encoders:
        for var in VARIANTS:
            for seed in args.seeds:
                ckpt = (
                    MLP_CKPT_ROOT / enc / var / f"seed_{seed}" / "checkpoint.pt"
                )
                if not ckpt.exists():
                    missing.append(str(ckpt))

    if missing:
        print("[ERROR] Missing MLP checkpoints:")
        for m in missing:
            print(f"  - {m}")
        print("\nRun scripts/train_mlp_fair.py first.")
        sys.exit(1)

    # Load DiT results
    dit_data = load_dit_results()
    dit_results = dit_data["results"]

    # Run MLP evaluation
    print("=" * 60)
    print("DA7: MLP Rollout Evaluation")
    print(f"  encoders: {args.encoders}")
    print(f"  seeds: {args.seeds}")
    print(f"  device: {device}")
    print("=" * 60)

    mlp_results: list[dict] = []
    for enc in args.encoders:
        for var in VARIANTS:
            for seed in args.seeds:
                print(f"\n[eval] {enc}/{var}/seed={seed}")
                result = evaluate_mlp(enc, var, seed, device=device)
                if "error" in result:
                    print(f"  [ERROR] {result['error']}")
                    sys.exit(1)

                # Print per-horizon summary
                m = result["metrics"]
                print(f"  {'k':>3}  {'CosSim':>8}  {'MSE':>8}  {'CopyBL':>8}")
                for k in range(len(m["cossim_by_horizon"])):
                    print(
                        f"  k={k+1}:  {m['cossim_by_horizon'][k]:>8.4f}  "
                        f"{m['mse_by_horizon'][k]:>8.4f}  "
                        f"{m['copy_baseline_cossim'][k]:>8.4f}"
                    )
                mlp_results.append(result)

    # Save MLP results
    mlp_summary = _build_summary(mlp_results)
    mlp_output = {
        "results": mlp_results,
        "summary": mlp_summary,
    }
    MLP_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MLP_RESULTS_PATH.open("w") as fh:
        json.dump(mlp_output, fh, indent=2)
    print(f"\n[saved] {MLP_RESULTS_PATH}")

    # Build comparison table
    comparison_rows = build_comparison_table(dit_results, mlp_results)
    export_csv(comparison_rows, COMPARISON_CSV_PATH)
    print(f"[saved] {COMPARISON_CSV_PATH}")

    export_latex(comparison_rows, LATEX_PATH)
    print(f"[saved] {LATEX_PATH}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: DiT vs MLP CosSim (seed-averaged, k=1)")
    print("=" * 60)
    dit_summary = _build_summary(dit_results)
    print(f"  {'Encoder':<15} {'Variant':<14} {'DiT':>8} {'MLP':>8} {'CopyBL':>8} {'Winner':>8}")
    print(f"  {'-'*13:<15} {'-'*12:<14} {'-'*6:>8} {'-'*6:>8} {'-'*6:>8} {'-'*6:>8}")

    for enc in sorted(args.encoders):
        for var in VARIANTS:
            dit_cs = dit_summary.get(enc, {}).get(var, {}).get("cossim_mean", [None])[0]
            mlp_cs = mlp_summary.get(enc, {}).get(var, {}).get("cossim_mean", [None])[0]
            copy_cs = dit_summary.get(enc, {}).get(var, {}).get("copy_baseline_cossim", [None])[0]

            winner = ""
            if dit_cs is not None and mlp_cs is not None:
                winner = "DiT" if dit_cs > mlp_cs else "MLP"

            print(
                f"  {enc:<15} {var:<14} "
                f"{dit_cs:>8.4f} {mlp_cs:>8.4f} "
                f"{copy_cs:>8.4f} {winner:>8}"
            )


if __name__ == "__main__":
    main()
