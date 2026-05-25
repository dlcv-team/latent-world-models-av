"""DeltaCosSim analysis + 3-paragraph results summary (C5).

Reads the per-horizon CosSim CSV produced by ``evaluation.latent_eval``
(C4) -- canonical location ``artifacts/cossim_eval/cossim_results.csv``
-- and emits two artifacts under ``outputs/analysis/``:

* ``delta_cossim_summary.json`` -- machine-readable analysis result.
* ``delta_cossim_summary.md`` -- three-paragraph human-readable summary
  covering (1) the CosSim values at k=1..4, (2) whether DeltaCosSim
  confirms or refutes action-conditioning utility, and (3) how the
  latent predictor's evidence stacks up against the BC baseline numbers
  for the same encoder (read from ``baselines.json``).

The CSV is also the **shared deliverable for M2's figure pipeline**;
the canonical copy lives at ``artifacts/cossim_eval/cossim_results.csv``
and is kept in sync by re-running ``evaluation.latent_eval``.

Design
------
Three pure functions do all of the work; the CLI just wires them up
and writes the artifacts.  This is the same shape as
``analysis.identify_best_encoder`` so reviewers don't have to learn a
new pattern.

CLI
---
    python -m analysis.delta_cossim_summary
    python -m analysis.delta_cossim_summary --cossim-csv path/to/cossim_results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config import CanonicalConfig, load_canonical

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Default encoder reported in the C4 run (``vjepa2_rep64``).
DEFAULT_ENCODER = "vjepa2_rep64"

#: Schema version of the JSON artifact this module emits.
SCHEMA_VERSION = "1.0"

#: Required columns in the input cossim CSV.  Matches
#: ``evaluation.latent_eval.CSV_COLUMNS`` exactly.
REQUIRED_CSV_COLUMNS: tuple[str, ...] = (
    "k",
    "cossim_conditioned",
    "cossim_unconditioned",
    "delta_cossim",
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_cossim_csv(path: Path) -> pd.DataFrame:
    """Load + validate a CosSim CSV emitted by ``evaluation.latent_eval``.

    Validates column presence, integer ``k``, no NaNs, and that
    ``delta_cossim == cossim_conditioned - cossim_unconditioned`` row by
    row (within fp32 tolerance) so a hand-edited CSV can't silently
    drift the headline numbers.
    """
    if not path.exists():
        raise FileNotFoundError(f"cossim CSV not found: {path}")
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: cossim CSV missing required columns {missing!r}; "
            f"got {list(df.columns)!r}"
        )
    if df.empty:
        raise ValueError(f"{path}: cossim CSV has no rows")
    if df[list(REQUIRED_CSV_COLUMNS)].isna().any().any():
        raise ValueError(f"{path}: cossim CSV contains NaN values")

    df = df.copy()
    df["k"] = df["k"].astype(int)
    df = df.sort_values("k").reset_index(drop=True)
    if list(df["k"]) != list(range(1, len(df) + 1)):
        raise ValueError(
            f"{path}: cossim CSV 'k' column must be 1..N contiguous; "
            f"got {list(df['k'])!r}"
        )

    derived_delta = df["cossim_conditioned"] - df["cossim_unconditioned"]
    if not ((derived_delta - df["delta_cossim"]).abs() < 1e-6).all():
        raise ValueError(
            f"{path}: delta_cossim column does not match "
            f"(cossim_conditioned - cossim_unconditioned) within 1e-6"
        )
    return df


def load_bc_baseline(
    baselines_json_path: Path, encoder: str
) -> dict[str, Any] | None:
    """Look up an encoder's BC baseline numbers from ``baselines.json``.

    Returns ``None`` if the file or the encoder entry is missing -- BC
    comparison is documented as optional in the summary paragraph,
    not an error condition, so downstream code can degrade gracefully.
    """
    if not baselines_json_path.exists():
        return None
    payload = json.loads(baselines_json_path.read_text())
    encoders = payload.get("encoders", {})
    if encoder not in encoders:
        return None
    entry = encoders[encoder]
    return {
        "encoder": encoder,
        "test_rmse_mean": float(entry["test_rmse_mean"]),
        "test_rmse_std": float(entry["test_rmse_std"]),
        "n_seeds": int(len(entry.get("seeds", []))) or None,
        "dataset": payload.get("dataset"),
        "n_test_samples": int(payload.get("split", {}).get("test", 0)) or None,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def summarize_delta_cossim(cossim_df: pd.DataFrame) -> dict[str, Any]:
    """Pure analysis: turn a per-horizon CosSim DataFrame into a summary dict.

    Surfaces the key questions the C5 spec asks:

    * Is ``DeltaCosSim > 0`` at any horizon?  (``delta_positive_at_any_horizon``)
    * Which horizons are positive / negative / zero?
    * Which horizon has the strongest / weakest action signal?
    * Is the per-horizon CosSim_conditioned series monotonically
      decaying with ``k`` (the expected pattern as prediction gets
      harder)?

    Returned dict is JSON-serializable as-is and stable across runs of
    the same input.
    """
    horizons = []
    positive_ks: list[int] = []
    negative_ks: list[int] = []
    zero_ks: list[int] = []

    for _, row in cossim_df.iterrows():
        k = int(row["k"])
        cond = float(row["cossim_conditioned"])
        uncond = float(row["cossim_unconditioned"])
        delta = float(row["delta_cossim"])
        horizons.append(
            {
                "k": k,
                "cossim_conditioned": cond,
                "cossim_unconditioned": uncond,
                "delta_cossim": delta,
                "delta_positive": delta > 0,
            }
        )
        if delta > 0:
            positive_ks.append(k)
        elif delta < 0:
            negative_ks.append(k)
        else:
            zero_ks.append(k)

    deltas = [h["delta_cossim"] for h in horizons]
    cond_series = [h["cossim_conditioned"] for h in horizons]
    uncond_series = [h["cossim_unconditioned"] for h in horizons]

    best_horizon = max(horizons, key=lambda h: h["delta_cossim"])
    worst_horizon = min(horizons, key=lambda h: h["delta_cossim"])

    # Monotonic non-increasing in k is the textbook expectation for a
    # working autoregressive predictor (further horizons = harder).
    cond_monotonic_nonincreasing = all(
        cond_series[i] >= cond_series[i + 1] - 1e-6
        for i in range(len(cond_series) - 1)
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "horizon": len(horizons),
        "per_horizon": horizons,
        "delta_positive_at_any_horizon": len(positive_ks) > 0,
        "delta_positive_horizons": positive_ks,
        "delta_negative_horizons": negative_ks,
        "delta_zero_horizons": zero_ks,
        "mean_delta": sum(deltas) / len(deltas),
        "mean_cossim_conditioned": sum(cond_series) / len(cond_series),
        "mean_cossim_unconditioned": sum(uncond_series) / len(uncond_series),
        "best_horizon": {
            "k": best_horizon["k"],
            "delta_cossim": best_horizon["delta_cossim"],
        },
        "worst_horizon": {
            "k": worst_horizon["k"],
            "delta_cossim": worst_horizon["delta_cossim"],
        },
        "cond_cossim_monotonic_nonincreasing": cond_monotonic_nonincreasing,
        "interpretation": _interpret_action_conditioning(
            len(positive_ks) > 0, sum(deltas) / len(deltas)
        ),
    }


def _interpret_action_conditioning(
    any_positive: bool, mean_delta: float
) -> dict[str, Any]:
    """Short structured interpretation used by ``render_summary_markdown``.

    The verdict text is the *conclusion* (what the numbers mean) -- the
    data observation itself ("Delta is positive/negative at k=...") is
    rendered separately by the caller, so the two are not stitched into
    the same sentence and don't repeat each other.
    """
    if any_positive and mean_delta > 0:
        verdict = (
            "Action conditioning helps: the predictor extracts measurable "
            "signal from `a_t` over the unconditional baseline."
        )
        action_conditioning_helps = True
    elif any_positive and mean_delta <= 0:
        verdict = (
            "Action conditioning is inconsistent: it helps on some "
            "horizons but the horizon-mean is non-positive, so any "
            "practical benefit is fragile."
        )
        action_conditioning_helps = False
    else:
        verdict = (
            "The current latent predictor pipeline does not surface a "
            "benefit from action conditioning -- on this evidence the "
            "action input is being ignored or actively hurting prediction."
        )
        action_conditioning_helps = False
    return {
        "action_conditioning_helps": action_conditioning_helps,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_summary_markdown(
    analysis: dict[str, Any],
    bc_baseline: dict[str, Any] | None,
    cossim_metadata: dict[str, Any] | None,
    encoder: str = DEFAULT_ENCODER,
) -> str:
    """Render the three-paragraph results summary as a single Markdown string.

    Paragraph 1: CosSim values at each horizon (cond / uncond / delta).
    Paragraph 2: DeltaCosSim verdict on action-conditioning utility.
    Paragraph 3: BC-baseline comparison and the "richer representation"
    question.  When the BC baseline is not available, paragraph 3 falls
    back to a self-contained statement rather than producing a hole.
    """
    encoder_label = (
        cossim_metadata.get("encoder") if cossim_metadata else None
    ) or encoder
    n_samples = cossim_metadata.get("n_samples") if cossim_metadata else None
    n_samples_clause = (
        f" on the {n_samples:,}-sequence test split" if n_samples else ""
    )

    paragraph_1 = _render_cossim_paragraph(
        analysis, encoder_label, n_samples_clause
    )
    paragraph_2 = _render_delta_paragraph(analysis)
    paragraph_3 = _render_bc_comparison_paragraph(
        analysis, bc_baseline, encoder_label
    )

    return (
        f"# DeltaCosSim results summary — `{encoder_label}` (C5)\n\n"
        f"{paragraph_1}\n\n"
        f"{paragraph_2}\n\n"
        f"{paragraph_3}\n"
    )


def _render_cossim_paragraph(
    analysis: dict[str, Any], encoder_label: str, n_samples_clause: str
) -> str:
    """Paragraph 1: raw CosSim numbers at each horizon."""
    horizon = analysis["horizon"]
    per_h = analysis["per_horizon"]
    cond_str = ", ".join(
        f"CosSim_cond(k={h['k']})={h['cossim_conditioned']:.4f}" for h in per_h
    )
    uncond_str = ", ".join(
        f"CosSim_uncond(k={h['k']})={h['cossim_unconditioned']:.4f}" for h in per_h
    )
    decay_clause = (
        "CosSim_conditioned is monotonically non-increasing in k, "
        "matching the expected pattern that prediction gets harder at "
        "longer horizons."
        if analysis["cond_cossim_monotonic_nonincreasing"]
        else "CosSim_conditioned is **not** monotonically decreasing in k, "
        "which is unusual for an autoregressive latent predictor and "
        "deserves a second look from M1."
    )
    return (
        f"**CosSim values at k=1..{horizon}.** Evaluated against "
        f"`{encoder_label}`{n_samples_clause}, the conditioned predictor "
        f"achieves {cond_str}, while the unconditional baseline (actions "
        f"zeroed out) achieves {uncond_str}. {decay_clause}"
    )


def _render_delta_paragraph(analysis: dict[str, Any]) -> str:
    """Paragraph 2: DeltaCosSim verdict on action conditioning."""
    horizon = analysis["horizon"]
    deltas_str = ", ".join(
        f"Δ(k={h['k']})={h['delta_cossim']:+.4f}"
        for h in analysis["per_horizon"]
    )
    best = analysis["best_horizon"]
    mean_delta = analysis["mean_delta"]
    verdict = analysis["interpretation"]["verdict"]

    if analysis["delta_positive_at_any_horizon"]:
        positive_ks = analysis["delta_positive_horizons"]
        ks_str = ", ".join(f"k={k}" for k in positive_ks)
        signal_clause = (
            f"DeltaCosSim is positive at {ks_str} "
            f"(best at k={best['k']}, "
            f"Δ={best['delta_cossim']:+.4f}); "
            f"horizon-mean Δ={mean_delta:+.4f}."
        )
    else:
        signal_clause = (
            f"DeltaCosSim is non-positive at every horizon "
            f"(best at k={best['k']}, "
            f"Δ={best['delta_cossim']:+.4f}; "
            f"horizon-mean Δ={mean_delta:+.4f})."
        )

    return (
        f"**Does action conditioning help?** Per-horizon deltas are "
        f"{deltas_str}. {signal_clause} {verdict}"
    )


def _render_bc_comparison_paragraph(
    analysis: dict[str, Any],
    bc_baseline: dict[str, Any] | None,
    encoder_label: str,
) -> str:
    """Paragraph 3: BC baseline comparison and "richer representation" verdict.

    BC RMSE and CosSim are **not commensurable** -- BC predicts
    ``(steer, accel)`` from ``z_t`` while the latent predictor predicts
    ``z_{t+k}`` from ``(z_t, a_t)``.  The paragraph is explicit about
    that and uses the DeltaCosSim sign as the primary evidence for the
    "richer representation" claim.
    """
    helps = analysis["interpretation"]["action_conditioning_helps"]
    mean_cond = analysis["mean_cossim_conditioned"]

    if bc_baseline is None:
        bc_clause = (
            f"BC baseline numbers for `{encoder_label}` are not available "
            f"in `baselines.json`; the comparison below is qualitative only."
        )
    else:
        bc_clause = (
            f"On the same encoder, the BC baseline reports "
            f"test RMSE = {bc_baseline['test_rmse_mean']:.4f} "
            f"(± {bc_baseline['test_rmse_std']:.4f} over "
            f"{bc_baseline['n_seeds'] or '?'} seeds) on the "
            f"{bc_baseline['n_test_samples'] or '?'}-sample test split — "
            f"i.e. a non-trivial linear-decoder mapping from `z_t` to "
            f"(steer, accel) does exist."
        )

    if helps:
        richer_clause = (
            f"Combined with the positive DeltaCosSim signal "
            f"(mean CosSim_cond ≈ {mean_cond:.4f}), this supports the "
            f"claim that the latent predictor captures action-driven "
            f"dynamics on top of what BC alone can recover from `z_t` — "
            f"i.e. the predictor surfaces *richer* representation than "
            f"the static BC mapping."
        )
    else:
        richer_clause = (
            f"However, with DeltaCosSim non-positive at every horizon and "
            f"CosSim_uncond essentially matching CosSim_cond "
            f"(mean ≈ {mean_cond:.4f}), the current evidence does **not** "
            f"support the claim that the latent predictor provides a "
            f"richer representation than BC: actions add no measurable "
            f"signal beyond what `z_t` already encodes, so the predictor "
            f"is effectively learning an identity / smoothing operator. "
            f"The honest read is that the *encoder* captures decodable "
            f"action-relevant features at time t (BC works), but the "
            f"*trained predictor* in this run does not exploit them to "
            f"model action-driven dynamics."
        )

    return (
        f"**Comparison to BC baseline.** {bc_clause} {richer_clause}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="delta_cossim_summary",
        description=(
            "Summarize the C4 cossim_results.csv into a three-paragraph "
            "Markdown writeup plus a machine-readable JSON. Reads the "
            "vendored CSV at artifacts/cossim_eval/ by default."
        ),
    )
    parser.add_argument(
        "--cossim-csv",
        type=Path,
        default=None,
        help=(
            "Path to cossim_results.csv. "
            "Default: <repo>/artifacts/cossim_eval/cossim_results.csv."
        ),
    )
    parser.add_argument(
        "--cossim-json",
        type=Path,
        default=None,
        help=(
            "Path to cossim_results.json (read for metadata only -- the "
            "numbers come from the CSV). "
            "Default: sibling of --cossim-csv."
        ),
    )
    parser.add_argument(
        "--baselines-json",
        type=Path,
        default=None,
        help=(
            "Path to baselines.json for the BC comparison paragraph. "
            "Default: <repo>/baselines.json. Pass a path that does not "
            "exist to skip the BC paragraph gracefully."
        ),
    )
    parser.add_argument(
        "--encoder",
        default=DEFAULT_ENCODER,
        help=(
            f"Encoder name to look up in baselines.json. "
            f"Default: {DEFAULT_ENCODER}. Ignored if the cossim JSON "
            f"metadata block already names an encoder."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where to write the JSON + Markdown. Default: <repo>/outputs/analysis/.",
    )
    return parser


def _default_paths(cfg: CanonicalConfig) -> tuple[Path, Path, Path, Path]:
    """Resolve the four default file paths from the canonical config root."""
    root = cfg.root
    return (
        root / "artifacts" / "cossim_eval" / "cossim_results.csv",
        root / "artifacts" / "cossim_eval" / "cossim_results.json",
        root / "baselines.json",
        root / "outputs" / "analysis",
    )


def _load_cossim_metadata(json_path: Path) -> dict[str, Any] | None:
    """Best-effort metadata read; missing JSON is non-fatal."""
    if not json_path.exists():
        return None
    payload = json.loads(json_path.read_text())
    md = payload.get("metadata")
    return dict(md) if md else None


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg = load_canonical()
    default_csv, default_json, default_bc, default_out = _default_paths(cfg)

    cossim_csv: Path = args.cossim_csv if args.cossim_csv is not None else default_csv
    if args.cossim_json is not None:
        cossim_json: Path = args.cossim_json
    else:
        cossim_json = cossim_csv.with_name("cossim_results.json")
        if not cossim_json.exists():
            cossim_json = default_json
    baselines_json: Path = (
        args.baselines_json if args.baselines_json is not None else default_bc
    )
    output_root: Path = (
        args.output_root if args.output_root is not None else default_out
    )

    cossim_df = load_cossim_csv(cossim_csv)
    cossim_metadata = _load_cossim_metadata(cossim_json)
    encoder = (
        cossim_metadata.get("encoder")
        if cossim_metadata and cossim_metadata.get("encoder")
        else args.encoder
    )
    bc_baseline = load_bc_baseline(baselines_json, encoder)

    analysis = summarize_delta_cossim(cossim_df)
    markdown = render_summary_markdown(
        analysis, bc_baseline, cossim_metadata, encoder=encoder
    )

    output_root.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "schema_version": SCHEMA_VERSION,
        "encoder": encoder,
        "analysis": analysis,
        "bc_baseline": bc_baseline,
        "cossim_metadata": cossim_metadata,
        "source_paths": {
            "cossim_csv": str(cossim_csv),
            "cossim_json": str(cossim_json) if cossim_json.exists() else None,
            "baselines_json": str(baselines_json) if baselines_json.exists() else None,
        },
    }
    (output_root / "delta_cossim_summary.json").write_text(
        json.dumps(json_payload, indent=2, sort_keys=True) + "\n"
    )
    (output_root / "delta_cossim_summary.md").write_text(markdown)

    print(
        f"[delta_cossim_summary] encoder={encoder}  "
        f"any_delta_positive={analysis['delta_positive_at_any_horizon']}  "
        f"mean_delta={analysis['mean_delta']:+.6f}"
    )
    print(f"[delta_cossim_summary] wrote -> {output_root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
