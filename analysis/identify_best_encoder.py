"""Identify the best encoder from A12's analysis outputs.

Reads ``encoder_summary_with_ci.csv`` and ``paired_tests.csv`` (both
produced by :mod:`analysis.paired_tests`) and emits two artifacts
under the same analysis directory:

* ``best_encoder.json`` — machine-readable winner + the list of
  encoders it Bonferroni-beats, with effect sizes.
* ``best_encoder_summary.md`` — a one-paragraph human-readable
  writeup M3 can paste into the BC training kickoff message.

The winner is the encoder with the minimum
``steer_rmse_scene_mean``; ties are broken lexicographically (rare,
but flagged in the JSON output for honesty). "Significantly beats"
uses Bonferroni-corrected p-values from ``paired_tests.csv``,
filtering pairs where the winner is in fact better (positive
``mean_diff_a_minus_b`` if the winner is in column B; negative if
in column A).

Usage
-----
    python -m analysis.identify_best_encoder
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from config import load_canonical


METRIC = "steer_rmse_scene_mean"


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


def identify_best_encoder(
    summary_df: pd.DataFrame, paired_df: pd.DataFrame
) -> dict:
    """Pure function: pick the winner and list its Bonferroni-significant wins.

    ``summary_df`` must have columns
    ``[encoder, steer_rmse_scene_mean, steer_ci95_lo, steer_ci95_hi,
       num_scenes]``. ``paired_df`` must follow the
    :data:`analysis.paired_tests.PAIRED_TESTS_COLUMNS` schema.
    """
    if METRIC not in summary_df.columns:
        raise ValueError(
            f"summary_df missing {METRIC!r}; got {list(summary_df.columns)!r}"
        )
    if "encoder" not in summary_df.columns:
        raise ValueError("summary_df missing 'encoder' column")
    if paired_df.empty:
        raise ValueError("paired_df is empty; cannot identify significant pairs")

    # Sort ascending by RMSE; tie-break lex ascending on encoder name.
    sorted_summary = summary_df.sort_values(
        by=[METRIC, "encoder"], kind="stable"
    ).reset_index(drop=True)
    winner_row = sorted_summary.iloc[0]
    winner = str(winner_row["encoder"])
    tied = bool(
        len(sorted_summary) > 1
        and sorted_summary.iloc[1][METRIC] == winner_row[METRIC]
    )

    # Bonferroni metadata is uniform across rows; pull from row 0.
    n_comparisons = int(paired_df["n_comparisons"].iloc[0])
    bonferroni_alpha = float(paired_df["bonferroni_alpha"].iloc[0])
    alpha = bonferroni_alpha * n_comparisons  # invert the Bonferroni split

    # Walk paired_df: keep the rows where (winner is in the pair) AND
    # (winner's mean is lower, i.e. mean_diff_a_minus_b has the correct
    # sign) AND (p_bonferroni < alpha).
    beats: list[dict] = []
    for _, row in paired_df.iterrows():
        if winner not in (row["encoder_a"], row["encoder_b"]):
            continue
        # Determine the "other" encoder and the sign convention.
        if row["encoder_a"] == winner:
            other = row["encoder_b"]
            # winner has lower RMSE => mean_diff = a - b < 0
            winner_is_better = row["mean_diff_a_minus_b"] < 0
            mean_diff_winner_minus_other = float(row["mean_diff_a_minus_b"])
            cohens_d = float(row["cohens_d"])
        else:
            other = row["encoder_a"]
            # winner = b => winner-other = -(a-b) > 0 reversed: report
            # consistently as "winner - other" so all signs are negative
            # when winner is better.
            winner_is_better = row["mean_diff_a_minus_b"] > 0
            mean_diff_winner_minus_other = -float(row["mean_diff_a_minus_b"])
            cohens_d = -float(row["cohens_d"])

        if not winner_is_better:
            continue
        if float(row["p_bonferroni"]) >= alpha:
            continue

        beats.append(
            {
                "encoder": str(other),
                "p_bonferroni": float(row["p_bonferroni"]),
                "p_value": float(row["p_value"]),
                "mean_diff_winner_minus_other": mean_diff_winner_minus_other,
                "cohens_d": cohens_d,
            }
        )

    # Stable order by p_bonferroni ascending (most significant first).
    beats.sort(key=lambda b: (b["p_bonferroni"], b["encoder"]))

    return {
        "best_encoder": winner,
        "metric": METRIC,
        "value": float(winner_row[METRIC]),
        "ci_95": [
            float(winner_row["steer_ci95_lo"]),
            float(winner_row["steer_ci95_hi"]),
        ],
        "num_scenes": int(winner_row["num_scenes"]),
        "n_comparisons": n_comparisons,
        "bonferroni_alpha": bonferroni_alpha,
        "alpha": alpha,
        "significantly_beats": beats,
        "tied": tied,
        "fallback_caveat": None,
    }


def attach_fallback_caveat(result: dict, probe_root: Path) -> dict:
    """Mutate ``result`` to carry the winner's ``fallback_caveat`` string.

    Reads ``<probe_root>/<winner>/provenance.json`` if present. When the
    file or the field is missing, leaves ``fallback_caveat`` as None.
    Returns the same dict for chaining.
    """
    winner = result["best_encoder"]
    provenance_path = probe_root / winner / "provenance.json"
    if provenance_path.exists():
        try:
            payload = json.loads(provenance_path.read_text())
        except json.JSONDecodeError:
            return result
        caveat = payload.get("fallback_caveat")
        # Treat empty strings as None for downstream "is there a caveat" checks.
        if caveat:
            result["fallback_caveat"] = str(caveat)
    return result


def render_summary_markdown(result: dict) -> str:
    """One-paragraph human-readable summary M3 can paste into Slack."""
    winner = result["best_encoder"]
    value = result["value"]
    lo, hi = result["ci_95"]
    beats = result["significantly_beats"]
    n_comparisons = result["n_comparisons"]

    if beats:
        beat_phrases = ", ".join(
            f"{b['encoder']} (p_bonf = {b['p_bonferroni']:.4f}, "
            f"Cohen's d = {b['cohens_d']:+.2f})"
            for b in beats
        )
        significance_clause = (
            f" The win is Bonferroni-significant over {beat_phrases} "
            f"(correction over n={n_comparisons} unordered pairs)."
        )
    else:
        significance_clause = (
            f" No pair clears Bonferroni-corrected significance "
            f"(n={n_comparisons} comparisons)."
        )

    tie_clause = (
        " There is a tie at the top by scene-mean RMSE; the winner is "
        "reported lexicographically and a re-run on a perturbed seed "
        "would be informative."
        if result.get("tied")
        else ""
    )

    caveat_clause = (
        f" Note: {result['fallback_caveat']}"
        if result.get("fallback_caveat")
        else ""
    )

    return (
        f"**Best encoder: `{winner}`.** "
        f"Scene-mean steering RMSE on the 40 test scenes is "
        f"{value:.4f} with 95% bootstrap CI [{lo:.4f}, {hi:.4f}], "
        f"computed over {result['num_scenes']} scenes."
        f"{significance_clause}"
        f"{tie_clause}"
        f"{caveat_clause} "
        f"**Action item:** M3 can start BC training on top of the frozen "
        f"`{winner}` encoder.\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="identify_best_encoder",
        description=(
            "Read paired_tests.csv + encoder_summary_with_ci.csv "
            "produced by analysis.paired_tests; identify the winner; "
            "emit best_encoder.json + best_encoder_summary.md."
        ),
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=None,
        help="Where to read the A12 CSVs. Default: <repo>/outputs/analysis/.",
    )
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=None,
        help=(
            "Where to read the winner's provenance.json for the "
            "fallback caveat. Default: <repo>/outputs/probes/."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where to write the JSON + Markdown. Default: --analysis-root.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg = load_canonical()
    analysis_root: Path = (
        args.analysis_root if args.analysis_root is not None
        else (cfg.root / "outputs" / "analysis")
    )
    probe_root: Path = (
        args.probe_root if args.probe_root is not None
        else (cfg.root / "outputs" / "probes")
    )
    output_root: Path = (
        args.output_root if args.output_root is not None else analysis_root
    )

    summary_path = analysis_root / "encoder_summary_with_ci.csv"
    paired_path = analysis_root / "paired_tests.csv"
    for required in (summary_path, paired_path):
        if not required.exists():
            raise FileNotFoundError(
                f"missing analysis input: {required} "
                f"(did you run `python -m analysis.paired_tests` first?)"
            )

    summary_df = pd.read_csv(summary_path)
    paired_df = pd.read_csv(paired_path)
    result = identify_best_encoder(summary_df, paired_df)
    attach_fallback_caveat(result, probe_root)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "best_encoder.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (output_root / "best_encoder_summary.md").write_text(
        render_summary_markdown(result)
    )

    print(f"[identify_best_encoder] winner: {result['best_encoder']} "
          f"({result['metric']} = {result['value']:.4f})")
    print(f"[identify_best_encoder] significantly beats: "
          f"{[b['encoder'] for b in result['significantly_beats']] or '(none)'}")
    print(f"[identify_best_encoder] wrote -> {output_root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
