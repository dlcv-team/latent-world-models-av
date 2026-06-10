"""Generate CosSim figures (Fig 3 & Fig 4) from M3's evaluation results.

Reads ``artifacts/cossim_eval/cossim_results.csv`` (produced by M3's
latent predictor evaluation) and generates 300-dpi PNG+PDF figures:

* Fig 3: CosSim line plot (conditioned vs unconditioned, with error bars if available)
* Fig 4: DeltaCosSim bar chart (Δ = cond - uncond, with error bars if available)

Supports two CSV formats:
- **Single-seed** (current main): ``k,cossim_conditioned,cossim_unconditioned,delta_cossim``
  → plots without error bars
- **Multi-seed** (future): ``k,cond_mean,cond_ci95_lo,cond_ci95_hi,...``
  → plots with 95% CI error bars

Usage
-----
    python scripts/build_cossim_figures.py
    python scripts/build_cossim_figures.py --cossim-csv path/to/results.csv
    python scripts/build_cossim_figures.py --out-dir outputs/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/CI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_canonical
DEFAULT_CSV_PATH = PROJECT_ROOT / "artifacts" / "cossim_eval" / "cossim_results.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "artifacts" / "cossim_eval" / "figures"
DPI = 300

# Required columns for multi-seed CSV (wide format with pre-computed CIs)
MULTISEED_COLUMNS: tuple[str, ...] = (
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

# Required columns for single-seed CSV (legacy format from PR#21)
SINGLESEED_COLUMNS: tuple[str, ...] = (
    "k",
    "cossim_conditioned",
    "cossim_unconditioned",
    "delta_cossim",
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
        print("[build_cossim_figures] Detected multi-seed CSV with 95% CI")

    elif is_singleseed:
        # Single-seed format (PR#21) — convert to multi-seed schema with no error bars
        df = df[list(SINGLESEED_COLUMNS)].copy()
        df = df.rename(columns={
            "cossim_conditioned": "cond_mean",
            "cossim_unconditioned": "uncond_mean",
            "delta_cossim": "delta_mean",
        })
        # Set CI bounds = mean (no error bars)
        df["cond_ci95_lo"] = df["cond_mean"]
        df["cond_ci95_hi"] = df["cond_mean"]
        df["uncond_ci95_lo"] = df["uncond_mean"]
        df["uncond_ci95_hi"] = df["uncond_mean"]
        df["delta_ci95_lo"] = df["delta_mean"]
        df["delta_ci95_hi"] = df["delta_mean"]
        has_ci = False
        print("[build_cossim_figures] Detected single-seed CSV (no error bars)")

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

    return df, has_ci


def _build_caption(has_ci: bool) -> str:
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
    fig.savefig(out_dir / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / stem}.{{png,pdf}}")


def build_fig3_cossim_lines(cossim_df: pd.DataFrame, has_ci: bool, out_dir: Path) -> None:
    """Fig 3: CosSim line plot (conditioned vs unconditioned, with error bars if available)."""
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

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        k,
        cond_mean,
        yerr=cond_yerr,
        marker="o",
        markersize=6,
        capsize=4 if has_ci else 0,
        linewidth=2,
        label="Conditioned",
        color="#4878CF",
        alpha=0.9,
    )
    ax.errorbar(
        k,
        uncond_mean,
        yerr=uncond_yerr,
        marker="s",
        markersize=6,
        capsize=4 if has_ci else 0,
        linewidth=2,
        label="Unconditioned",
        color="#EE854A",
        alpha=0.9,
    )

    ax.set_xlabel("Prediction Horizon (k)", fontsize=11)
    ax.set_ylabel("Cosine Similarity", fontsize=11)

    # Title and caption adapt to data format
    if has_ci:
        ax.set_title("CosSim by Prediction Horizon (multi-seed, 95% CI)", fontsize=12)
    else:
        ax.set_title("CosSim by Prediction Horizon (single seed)", fontsize=12)

    caption = _build_caption(has_ci)

    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k)

    # Caption annotation (bottom-right, gray)
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

    save_png_pdf(fig, out_dir, "fig3_cossim_lines")


def build_fig4_delta_bars(cossim_df: pd.DataFrame, has_ci: bool, out_dir: Path) -> None:
    """Fig 4: DeltaCosSim bar chart (Δ = cond - uncond, with error bars if available)."""
    k = cossim_df["k"].values
    delta_mean = cossim_df["delta_mean"].values
    delta_lo = cossim_df["delta_ci95_lo"].values
    delta_hi = cossim_df["delta_ci95_hi"].values

    # Error bars: [distance below mean, distance above mean]
    delta_err = np.array([delta_mean - delta_lo, delta_hi - delta_mean])

    # Color bars by sign: green for positive, red for negative/zero
    colors = np.where(delta_mean > 0, "#6BAA75", "#D65F5F")

    fig, ax = plt.subplots(figsize=(7, 5))

    # If single-seed, set yerr=None to suppress error bars
    delta_yerr = delta_err if has_ci else None

    ax.bar(
        k,
        delta_mean,
        yerr=delta_yerr,
        capsize=4 if has_ci else 0,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
        error_kw={"linewidth": 1.5} if has_ci else {},
    )

    # Horizontal reference line at y=0
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)

    ax.set_xlabel("Prediction Horizon (k)", fontsize=11)
    ax.set_ylabel("Δ CosSim (conditioned - unconditioned)", fontsize=11)

    # Title and caption adapt to data format
    if has_ci:
        ax.set_title("DeltaCosSim by Prediction Horizon (multi-seed, 95% CI)", fontsize=12)
    else:
        ax.set_title("DeltaCosSim by Prediction Horizon (single seed)", fontsize=12)

    caption = _build_caption(has_ci)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(k)

    # Caption annotation (bottom-right, gray)
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

    save_png_pdf(fig, out_dir, "fig4_delta_bars")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate CosSim figures (Fig 3 & 4) from M3's evaluation results."
    )
    parser.add_argument(
        "--cossim-csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Path to cossim_results.csv (single-seed or multi-seed). Default: {DEFAULT_CSV_PATH}",
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
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[build_cossim_figures] Loading CSV...")
    df, has_ci = load_cossim_csv(args.cossim_csv)

    print("[build_cossim_figures] Generating Fig 3...")
    build_fig3_cossim_lines(df, has_ci, args.out_dir)

    print("[build_cossim_figures] Generating Fig 4...")
    build_fig4_delta_bars(df, has_ci, args.out_dir)

    print(f"[build_cossim_figures] Wrote 4 files to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
