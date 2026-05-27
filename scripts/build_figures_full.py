"""Generate publication figures from full-dataset results (RSK-09).

Reads ``outputs/analysis/encoder_summary_with_ci.csv`` (produced by
``analysis/paired_tests.py``) and generates a 300-dpi RMSE bar chart
with bootstrap CI error bars for all 6 encoders.

Usage
-----
    python scripts/build_figures_full.py
    python scripts/build_figures_full.py --out-dir artifacts/full/figures
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/CI environments
import matplotlib.pyplot as plt
import numpy as np

from config import ENCODER_DISPLAY

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "analysis" / "encoder_summary_with_ci.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "artifacts" / "full" / "figures"
DPI = 300


# Display names for encoders (sorted by steer RMSE in the chart)
ENCODER_DISPLAY = {
    "vjepa2_rep64": "V-JEPA2\n(fpc64)",
    "vjepa2_rep1": "V-JEPA2\n(fpc1)",
    "dino_vits14": "DINOv2\nViT-S/14",
    "clip_b32": "CLIP\nViT-B/32",
    "vit_s16": "ViT-S/16\n(supervised)",
    "vq_track": "VQ-VAE\nTracker",
}


def save_png_pdf(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / stem}.{{png,pdf}}")


def build_rmse_bar_chart(summary_path: Path, out_dir: Path) -> None:
    """Dual bar chart: steer + accel RMSE with bootstrap CI error bars."""
    with summary_path.open() as fh:
        rows = list(csv.DictReader(fh))

    # Sort by steer RMSE (ascending = best first)
    rows.sort(key=lambda r: float(r["steer_rmse_scene_mean"]))

    encoders = [r["encoder"] for r in rows]
    steer_mean = np.array([float(r["steer_rmse_scene_mean"]) for r in rows])
    steer_lo = np.array([float(r["steer_ci95_lo"]) for r in rows])
    steer_hi = np.array([float(r["steer_ci95_hi"]) for r in rows])
    accel_mean = np.array([float(r["accel_rmse_scene_mean"]) for r in rows])
    accel_lo = np.array([float(r["accel_ci95_lo"]) for r in rows])
    accel_hi = np.array([float(r["accel_ci95_hi"]) for r in rows])

    # Error bars are relative to the mean
    steer_err = np.array([steer_mean - steer_lo, steer_hi - steer_mean])
    accel_err = np.array([accel_mean - accel_lo, accel_hi - accel_mean])

    x = np.arange(len(encoders))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        x - w / 2, steer_mean, width=w,
        yerr=steer_err, capsize=3, color="#4878CF", alpha=0.85,
        label="Steering RMSE", error_kw={"linewidth": 1},
    )
    ax.bar(
        x + w / 2, accel_mean, width=w,
        yerr=accel_err, capsize=3, color="#D65F5F", alpha=0.85,
        label="Acceleration RMSE", error_kw={"linewidth": 1},
    )

    labels = [ENCODER_DISPLAY.get(e, e) for e in encoders]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("RMSE (normalized)", fontsize=10)
    ax.set_title(
        "Action Prediction RMSE by Encoder (850 scenes, full dataset)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(bottom=0)

    n_scenes = int(rows[0]["num_scenes"])
    ax.text(
        0.99, 0.02,
        f"n = {n_scenes} test scenes, 3 seeds, 95% bootstrap CI",
        transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
        color="gray",
    )

    save_png_pdf(fig, out_dir, "encoder_rmse_bars")


def print_ranking_table(summary_path: Path) -> None:
    """Print encoder ranking to stdout."""
    with summary_path.open() as fh:
        rows = list(csv.DictReader(fh))

    rows.sort(key=lambda r: float(r["steer_rmse_scene_mean"]))

    print("\n  Encoder Ranking (by steering RMSE, ascending):")
    print(f"  {'Rank':<5} {'Encoder':<15} {'Steer RMSE':<15} {'Accel RMSE':<15} {'Scenes'}")
    print(f"  {'----':<5} {'-------':<15} {'----------':<15} {'----------':<15} {'------'}")
    for i, r in enumerate(rows, 1):
        steer = float(r["steer_rmse_scene_mean"])
        accel = float(r["accel_rmse_scene_mean"])
        n = int(r["num_scenes"])
        ci_lo = float(r["steer_ci95_lo"])
        ci_hi = float(r["steer_ci95_hi"])
        print(f"  {i:<5} {r['encoder']:<15} {steer:.6f}       {accel:.6f}       {n}")
    print()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate publication figures from full-dataset results."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to encoder_summary_with_ci.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for figures.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[build_figures] Generating full-dataset figures...")
    build_rmse_bar_chart(args.summary, args.out_dir)
    print_ranking_table(args.summary)
    print("[build_figures] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
