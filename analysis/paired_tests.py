"""Paired-t-test analysis with Bonferroni correction + bootstrap CIs.

Reads per-scene RMSE from ``<probe_root>/<encoder>/per_scene_rmse.csv``
(the layout produced by both ``training/train_probe.py`` and
``scripts/adopt_pilot_artifacts.py``), runs
:func:`scipy.stats.ttest_rel` on every unordered encoder pair, applies
Bonferroni correction over the actual pair count, and emits three
artifacts under ``<output_root>/``:

* ``paired_tests.csv`` — one row per pair with t-stat, p-value,
  Bonferroni-corrected p, effect size (Cohen's d), and the
  ``n_comparisons`` count read by downstream figure scripts.
* ``encoder_summary_with_ci.csv`` — per-encoder scene-mean RMSE with
  95% bootstrap CIs.
* ``paired_tests.tex`` — a LaTeX ``tabular`` snippet whose footnote
  states the actual ``n_comparisons`` value, read from the CSV
  (no hardcoded counts in the figure layer).

Canonical hyperparameters (alpha, bootstrap settings) come from
``configs/canonical.yaml``; they are not CLI-overridable on purpose.

Usage
-----
    python -m analysis.paired_tests
    python -m analysis.paired_tests --probe-root outputs/probes
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from config import CanonicalConfig, load_canonical


SUPPORTED_METRICS = ("steer_rmse", "accel_rmse")
PAIRED_TESTS_COLUMNS = [
    "encoder_a",
    "encoder_b",
    "n_scenes",
    "t_stat",
    "p_value",
    "n_comparisons",
    "bonferroni_alpha",
    "p_bonferroni",
    "mean_diff_a_minus_b",
    "cohens_d",
]
SUMMARY_COLUMNS = [
    "encoder",
    "steer_rmse_scene_mean",
    "steer_ci95_lo",
    "steer_ci95_hi",
    "accel_rmse_scene_mean",
    "accel_ci95_lo",
    "accel_ci95_hi",
    "num_scenes",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_per_scene_rmse(
    probe_root: Path, metric: str = "steer_rmse"
) -> dict[str, pd.Series]:
    """Scan ``<probe_root>/*/per_scene_rmse.csv`` and return per-encoder series.

    Each returned series is indexed by ``scene_name`` so paired tests can
    align cleanly across encoders. The function asserts identical scene
    sets across all encoders; mismatches raise ``ValueError`` so silent
    misalignment can't sneak in.

    Encoder directories without a ``per_scene_rmse.csv`` are skipped
    silently (e.g., a stale dir from a prior partial run).
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"metric must be one of {SUPPORTED_METRICS}, got {metric!r}"
        )

    probe_root = Path(probe_root)
    if not probe_root.exists():
        raise FileNotFoundError(f"probe-root does not exist: {probe_root}")

    per_encoder: dict[str, pd.Series] = {}
    for enc_dir in sorted(p for p in probe_root.iterdir() if p.is_dir()):
        csv_path = enc_dir / "per_scene_rmse.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if metric not in df.columns or "scene_name" not in df.columns:
            raise ValueError(
                f"{csv_path}: missing required columns "
                f"({metric!r}, 'scene_name'); got {list(df.columns)!r}"
            )
        # If fold_id is present and >0 rows exist per scene, mean across
        # folds. Pilot data has a single fold so this is a no-op there.
        if "fold_id" in df.columns:
            df = df.groupby("scene_name", as_index=True)[metric].mean()
        else:
            df = df.set_index("scene_name")[metric]
        per_encoder[enc_dir.name] = df.sort_index()

    if not per_encoder:
        raise FileNotFoundError(
            f"no per_scene_rmse.csv files found under {probe_root}"
        )

    # Require identical scene sets across encoders.
    reference_scenes = next(iter(per_encoder.values())).index
    for enc, series in per_encoder.items():
        if not series.index.equals(reference_scenes):
            missing = set(reference_scenes) - set(series.index)
            extra = set(series.index) - set(reference_scenes)
            raise ValueError(
                f"encoder {enc!r} has a mismatched scene set; "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

    return per_encoder


# ---------------------------------------------------------------------------
# Paired tests
# ---------------------------------------------------------------------------


def compute_paired_tests(
    per_encoder: dict[str, pd.Series], alpha: float
) -> pd.DataFrame:
    """Run pairwise ``ttest_rel`` over per-scene RMSE.

    The pair count is computed from the data (not asserted), so the
    Bonferroni correction adapts when encoders are added or removed.

    Cohen's d for paired samples: ``mean(diff) / std(diff, ddof=1)``.
    """
    encoders = sorted(per_encoder)
    pairs = list(itertools.combinations(encoders, 2))
    n_comparisons = len(pairs)
    if n_comparisons == 0:
        raise ValueError(
            "compute_paired_tests requires at least 2 encoders; got "
            f"{encoders!r}"
        )
    bonferroni_alpha = alpha / n_comparisons

    rows = []
    for a, b in pairs:
        sa = per_encoder[a].values
        sb = per_encoder[b].values
        diff = sa - sb
        mean_diff = float(np.mean(diff))
        sd_diff = float(np.std(diff, ddof=1))
        cohens_d = mean_diff / sd_diff if sd_diff > 0 else float("inf")

        t_stat, p_value = stats.ttest_rel(sa, sb)
        p_bonferroni = min(float(p_value) * n_comparisons, 1.0)

        rows.append(
            {
                "encoder_a": a,
                "encoder_b": b,
                "n_scenes": int(len(diff)),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "n_comparisons": n_comparisons,
                "bonferroni_alpha": bonferroni_alpha,
                "p_bonferroni": p_bonferroni,
                "mean_diff_a_minus_b": mean_diff,
                "cohens_d": cohens_d,
            }
        )
    return pd.DataFrame(rows, columns=PAIRED_TESTS_COLUMNS)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_mean_ci(
    values: np.ndarray,
    n_resamples: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    """Return ``(mean, ci_lo, ci_hi)`` via nonparametric bootstrap of the mean."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    if n == 0:
        raise ValueError("bootstrap_mean_ci received zero-length input")
    boot_means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    alpha = 1.0 - confidence_level
    ci_lo = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(values.mean()), ci_lo, ci_hi


def compute_encoder_summary_with_ci(
    probe_root: Path,
    n_resamples: int,
    seed: int,
    confidence_level: float,
) -> pd.DataFrame:
    """Per-encoder scene-mean RMSE + bootstrap CI for steer and accel.

    Output schema matches the pilot's ``encoder_summary_with_ci_5enc.csv``
    so figure scripts can read either source.
    """
    steer = load_per_scene_rmse(probe_root, metric="steer_rmse")
    accel = load_per_scene_rmse(probe_root, metric="accel_rmse")

    rows = []
    # Use a single seed sequence keyed by encoder name to keep
    # per-encoder CIs reproducible without coupling them to each other.
    for idx, enc in enumerate(sorted(steer)):
        steer_mean, steer_lo, steer_hi = bootstrap_mean_ci(
            steer[enc].values, n_resamples, seed + idx, confidence_level
        )
        accel_mean, accel_lo, accel_hi = bootstrap_mean_ci(
            accel[enc].values, n_resamples, seed + idx, confidence_level
        )
        rows.append(
            {
                "encoder": enc,
                "steer_rmse_scene_mean": steer_mean,
                "steer_ci95_lo": steer_lo,
                "steer_ci95_hi": steer_hi,
                "accel_rmse_scene_mean": accel_mean,
                "accel_ci95_lo": accel_lo,
                "accel_ci95_hi": accel_hi,
                "num_scenes": int(steer[enc].shape[0]),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------


def render_latex_table(paired_tests_df: pd.DataFrame) -> str:
    """Render a LaTeX ``tabular`` snippet from the paired-tests DataFrame.

    The footnote text dynamically cites the ``n_comparisons`` value
    pulled from the DataFrame, so the figure layer never hardcodes
    pair counts.
    """
    if paired_tests_df.empty:
        raise ValueError("cannot render LaTeX for an empty paired-tests df")
    n_comparisons = int(paired_tests_df["n_comparisons"].iloc[0])

    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Encoder A & Encoder B & $n_{\text{scenes}}$ & $t$ & $p$ & "
        r"$p_{\text{bonf}}$ & $\Delta\overline{\text{RMSE}}$ & Cohen's $d$ \\",
        r"\midrule",
    ]
    for _, row in paired_tests_df.iterrows():
        lines.append(
            f"{_escape_latex(row['encoder_a'])} & {_escape_latex(row['encoder_b'])} & "
            f"{int(row['n_scenes'])} & "
            f"{row['t_stat']:.3f} & "
            f"{row['p_value']:.4f} & "
            f"{row['p_bonferroni']:.4f} & "
            f"{row['mean_diff_a_minus_b']:+.4f} & "
            f"{row['cohens_d']:+.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\\\[2pt] \\footnotesize Bonferroni correction over n={n_comparisons} unordered encoder pairs.",
        ]
    )
    return "\n".join(lines) + "\n"


def _escape_latex(s: str) -> str:
    """Minimal LaTeX escape for underscores in encoder names."""
    return str(s).replace("_", r"\_")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paired_tests",
        description=(
            "Run paired t-tests + Bonferroni + Cohen's d + bootstrap CIs "
            "over per-encoder per-scene RMSE."
        ),
    )
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=None,
        help=(
            "Directory containing <encoder>/per_scene_rmse.csv. "
            "Default: <repo>/outputs/probes/."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where to write CSVs and .tex. Default: <repo>/outputs/analysis/.",
    )
    parser.add_argument(
        "--metric",
        choices=SUPPORTED_METRICS,
        default="steer_rmse",
        help="Which RMSE column drives the paired test. Default: steer_rmse.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg: CanonicalConfig = load_canonical()

    probe_root: Path = (
        args.probe_root if args.probe_root is not None
        else (cfg.root / "outputs" / "probes")
    )
    output_root: Path = (
        args.output_root if args.output_root is not None
        else (cfg.root / "outputs" / "analysis")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    alpha = float(cfg.raw["evaluation"]["paired_tests"]["alpha"])
    boot_cfg = cfg.raw["evaluation"]["bootstrap"]
    n_resamples = int(boot_cfg["n_resamples"])
    seed = int(boot_cfg["seed"])
    confidence_level = float(boot_cfg["confidence_level"])

    per_encoder = load_per_scene_rmse(probe_root.resolve(), metric=args.metric)
    paired_df = compute_paired_tests(per_encoder, alpha=alpha)
    summary_df = compute_encoder_summary_with_ci(
        probe_root.resolve(),
        n_resamples=n_resamples,
        seed=seed,
        confidence_level=confidence_level,
    )
    latex = render_latex_table(paired_df)

    paired_df.to_csv(output_root / "paired_tests.csv", index=False)
    summary_df.to_csv(output_root / "encoder_summary_with_ci.csv", index=False)
    (output_root / "paired_tests.tex").write_text(latex)

    print(f"[paired_tests] encoders: {sorted(per_encoder)}")
    print(f"[paired_tests] n_comparisons = {paired_df['n_comparisons'].iloc[0]}")
    print(f"[paired_tests] wrote -> {output_root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
