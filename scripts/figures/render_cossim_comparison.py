#!/usr/bin/env python3
"""Render publication-ready CosSim comparison bar chart.

Generates a grouped bar chart comparing conditioned vs unconditioned
cosine similarity across prediction horizons k=1..4.

Data source:
- artifacts/cossim_eval/cossim_results.csv (single-seed format)
- artifacts/cossim_eval/cossim_results.json (metadata/provenance)

Chart design:
- Grouped bars at each horizon k
- Conditioned (blue) vs Unconditioned (orange)
- Auto-detects single-seed vs multi-seed CSV format
- Error bars displayed only for multi-seed data

Usage:
    python figures/render_cossim_comparison.py
    python figures/render_cossim_comparison.py --cossim-csv path/to/results.csv
    python figures/render_cossim_comparison.py --out-dir outputs/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import load_canonical

# Default paths
DEFAULT_CSV_PATH = Path("artifacts/cossim_eval/cossim_results.csv")
DEFAULT_JSON_PATH = Path("artifacts/cossim_eval/cossim_results.json")
DEFAULT_OUT_DIR = Path("outputs/figures")

DPI = 300

# Project color palette (from scripts/build_cossim_figures.py)
COLOR_CONDITIONED = "#4878CF"  # Blue
COLOR_UNCONDITIONED = "#EE854A"  # Orange

# CSV schema patterns (from evaluation/latent_eval.py and scripts/build_cossim_figures.py)
SINGLESEED_COLUMNS = ("k", "cossim_conditioned", "cossim_unconditioned", "delta_cossim")
MULTISEED_COLUMNS = (
    "k",
    "cond_mean",
    "cond_ci95_lo",
    "cond_ci95_hi",
    "uncond_mean",
    "uncond_ci95_lo",
    "uncond_ci95_hi",
    "delta_mean",
    "delta_ci95_lo",
    "delta_ci95_hi",
)


def load_cossim_csv(path: Path) -> tuple[pd.DataFrame, bool]:
    """Load and validate CosSim CSV (single-seed or multi-seed).

    Returns
    -------
    df : pd.DataFrame
        Normalized DataFrame with standardized column names:
        k, cond_mean, cond_ci95_lo, cond_ci95_hi, uncond_mean, uncond_ci95_lo,
        uncond_ci95_hi, delta_mean, delta_ci95_lo, delta_ci95_hi

        For single-seed CSVs, CI columns are set to mean (no error bars).

    has_ci : bool
        True if multi-seed CSV with CI columns, False if single-seed.

    Raises
    ------
    FileNotFoundError
        If CSV does not exist.
    ValueError
        If schema validation fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"CosSim CSV not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{path}: CosSim CSV has no rows")

    # Detect format
    is_multiseed = set(MULTISEED_COLUMNS).issubset(df.columns)
    is_singleseed = set(SINGLESEED_COLUMNS).issubset(df.columns)

    if is_multiseed:
        # Multi-seed format with CIs
        missing = [c for c in MULTISEED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path}: Multi-seed format detected but missing columns {missing!r}"
            )
        df = df[list(MULTISEED_COLUMNS)].copy()
        has_ci = True
        print("[render_cossim_comparison] Detected multi-seed CSV with 95% CI")

    elif is_singleseed:
        # Single-seed format — convert to multi-seed schema with no error bars
        df = df[list(SINGLESEED_COLUMNS)].copy()
        df = df.rename(
            columns={
                "cossim_conditioned": "cond_mean",
                "cossim_unconditioned": "uncond_mean",
                "delta_cossim": "delta_mean",
            }
        )
        # Set CI bounds = mean (no error bars)
        df["cond_ci95_lo"] = df["cond_mean"]
        df["cond_ci95_hi"] = df["cond_mean"]
        df["uncond_ci95_lo"] = df["uncond_mean"]
        df["uncond_ci95_hi"] = df["uncond_mean"]
        df["delta_ci95_lo"] = df["delta_mean"]
        df["delta_ci95_hi"] = df["delta_mean"]
        has_ci = False
        print("[render_cossim_comparison] Detected single-seed CSV (no error bars)")

    else:
        raise ValueError(
            f"{path}: Unrecognized CSV schema. Got columns: {list(df.columns)!r}\n"
            f"Expected either:\n"
            f"  - Single-seed: {list(SINGLESEED_COLUMNS)!r}\n"
            f"  - Multi-seed: {list(MULTISEED_COLUMNS)!r}"
        )

    # Validate k column is 1..N contiguous
    df["k"] = df["k"].astype(int)
    df = df.sort_values("k").reset_index(drop=True)
    expected_k = list(range(1, len(df) + 1))
    if list(df["k"]) != expected_k:
        raise ValueError(
            f"{path}: 'k' column must be 1..N contiguous; "
            f"got {list(df['k'])!r}, expected {expected_k!r}"
        )

    # Verify delta_mean == cond_mean - uncond_mean (catches corrupted CSVs)
    computed_delta = df["cond_mean"] - df["uncond_mean"]
    if not np.allclose(df["delta_mean"], computed_delta, atol=1e-6):
        raise ValueError(
            f"{path}: delta_mean inconsistent with cond_mean - uncond_mean. "
            f"Max deviation: {np.abs(df['delta_mean'] - computed_delta).max():.3e}"
        )

    # Validate CI bounds (lo <= mean <= hi) for multi-seed data
    if has_ci:
        for col_prefix in ["cond", "uncond", "delta"]:
            mean_col = f"{col_prefix}_mean"
            lo_col = f"{col_prefix}_ci95_lo"
            hi_col = f"{col_prefix}_ci95_hi"

            # Check lo <= mean
            if not (df[lo_col] <= df[mean_col]).all():
                bad_idx = (df[lo_col] > df[mean_col]).idxmax()
                raise ValueError(
                    f"{path}: {lo_col} > {mean_col} at row {bad_idx}: "
                    f"{df.loc[bad_idx, lo_col]:.6f} > {df.loc[bad_idx, mean_col]:.6f}"
                )

            # Check mean <= hi
            if not (df[mean_col] <= df[hi_col]).all():
                bad_idx = (df[mean_col] > df[hi_col]).idxmax()
                raise ValueError(
                    f"{path}: {mean_col} > {hi_col} at row {bad_idx}: "
                    f"{df.loc[bad_idx, mean_col]:.6f} > {df.loc[bad_idx, hi_col]:.6f}"
                )

    # Validate cosine similarity range [-1, 1] for cond and uncond
    for col in ["cond_mean", "uncond_mean"]:
        if not ((df[col] >= -1) & (df[col] <= 1)).all():
            bad_idx = ((df[col] < -1) | (df[col] > 1)).idxmax()
            raise ValueError(
                f"{path}: {col} contains value outside [-1, 1] at row {bad_idx}: "
                f"{df.loc[bad_idx, col]:.6f}"
            )

    return df, has_ci


def load_metadata(json_path: Path) -> dict:
    """Load metadata from cossim_results.json if available."""
    if not json_path.exists():
        return {}

    with json_path.open("r") as f:
        data = json.load(f)

    return data.get("metadata", {})


def build_caption(has_ci: bool) -> str:
    """Build canonical caption from config (trainval subset + seed info)."""
    cfg = load_canonical()
    counts = cfg.expected_split_counts
    train = counts["p0_train"]
    val = counts["p0_val"]
    test = counts["p0_test"]
    seed = cfg.global_seed

    base = f"trainval-mirror subset ({train}/{val}/{test}, seed {seed})"
    if has_ci:
        return f"{base}\nmulti-seed, 95% bootstrap CI"
    else:
        return f"{base}\nsingle-seed run (no CI)"


def save_png_pdf(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Save figure as both PNG and PDF at 300 DPI."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / stem}.{{png,pdf}}")


def render_cossim_comparison(
    cossim_df: pd.DataFrame,
    has_ci: bool,
    metadata: dict,
    out_dir: Path,
) -> None:
    """Render grouped bar chart comparing conditioned vs unconditioned CosSim."""
    k = cossim_df["k"].values
    cond_mean = cossim_df["cond_mean"].values
    cond_lo = cossim_df["cond_ci95_lo"].values
    cond_hi = cossim_df["cond_ci95_hi"].values
    uncond_mean = cossim_df["uncond_mean"].values
    uncond_lo = cossim_df["uncond_ci95_lo"].values
    uncond_hi = cossim_df["uncond_ci95_hi"].values

    # Error bars: [distance below mean, distance above mean]
    cond_err = np.array([cond_mean - cond_lo, cond_hi - cond_mean])
    uncond_err = np.array([uncond_mean - uncond_lo, uncond_hi - uncond_mean])

    # If single-seed, set yerr=None to suppress error bars
    cond_yerr = cond_err if has_ci else None
    uncond_yerr = uncond_err if has_ci else None

    # Bar positioning
    bar_width = 0.35
    x_cond = k - bar_width / 2
    x_uncond = k + bar_width / 2

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Conditioned bars
    ax.bar(
        x_cond,
        cond_mean,
        width=bar_width,
        yerr=cond_yerr,
        capsize=4 if has_ci else None,
        label="Conditioned",
        color=COLOR_CONDITIONED,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
    )

    # Unconditioned bars
    ax.bar(
        x_uncond,
        uncond_mean,
        width=bar_width,
        yerr=uncond_yerr,
        capsize=4 if has_ci else None,
        label="Unconditioned",
        color=COLOR_UNCONDITIONED,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
    )

    # Axis labels and title
    ax.set_xlabel("Prediction Horizon (k)", fontsize=11)
    ax.set_ylabel("Cosine Similarity", fontsize=11)

    # Title with encoder info if available
    title = "CosSim: Conditioned vs Unconditioned by Horizon"
    if "encoder" in metadata:
        title += f" ({metadata['encoder']})"
    if has_ci:
        title += " (multi-seed, 95% CI)"
    else:
        title += " (single seed)"
    ax.set_title(title, fontsize=12, pad=15)

    # Legend and grid
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(k)
    ax.set_xticklabels(k)

    # Caption annotation (bottom-right, gray)
    caption = build_caption(has_ci)
    ax.text(
        0.98,
        0.02,
        caption,
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        color="gray",
    )

    # Provenance note if available
    if metadata:
        prov_lines = []
        if "n_samples" in metadata:
            prov_lines.append(f"n={metadata['n_samples']} samples")
        if "horizon" in metadata:
            prov_lines.append(f"H={metadata['horizon']}")
        if prov_lines:
            prov_text = ", ".join(prov_lines)
            ax.text(
                0.02,
                0.98,
                prov_text,
                transform=ax.transAxes,
                fontsize=7,
                ha="left",
                va="top",
                color="gray",
            )

    # Save figure
    save_png_pdf(fig, out_dir, "cossim_comparison")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render publication-ready CosSim comparison bar chart."
    )
    parser.add_argument(
        "--cossim-csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Path to cossim_results.csv (single-seed or multi-seed). Default: {DEFAULT_CSV_PATH}",
    )
    parser.add_argument(
        "--cossim-json",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help=f"Path to cossim_results.json for metadata (optional). Default: {DEFAULT_JSON_PATH}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for figures. Default: {DEFAULT_OUT_DIR}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    print("[render_cossim_comparison] Loading CSV...")
    df, has_ci = load_cossim_csv(args.cossim_csv)

    print("[render_cossim_comparison] Loading metadata...")
    metadata = load_metadata(args.cossim_json)

    print("[render_cossim_comparison] Generating CosSim comparison chart...")
    render_cossim_comparison(df, has_ci, metadata, args.out_dir)

    print(f"[render_cossim_comparison] Done. Output: {args.out_dir}/cossim_comparison.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
