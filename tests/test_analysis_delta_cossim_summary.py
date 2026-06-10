"""Unit tests for :mod:`analysis.delta_cossim_summary` (C5).

Covers:

* CSV loader: schema, NaN, contiguous-``k``, derived-Delta consistency.
* ``summarize_delta_cossim``: positive / negative / mixed / zero deltas,
  best/worst horizon selection, monotonic-decay flag.
* ``load_bc_baseline``: present / missing encoder / missing file.
* ``render_summary_markdown``: 3 paragraphs, key claims present in
  each, BC-baseline-absent fallback, "richer representation" verdict
  flips with the sign of Delta.
* End-to-end CLI: writes JSON + Markdown with the expected schema.
* Smoke test against the vendored ``artifacts/cossim_eval/`` CSV so
  the script can't drift away from the canonical artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis import delta_cossim_summary as dcs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cossim_df(rows: list[tuple[int, float, float]]) -> pd.DataFrame:
    """Build a per-horizon cossim DataFrame from ``(k, cond, uncond)`` triples.

    ``delta_cossim`` is filled in derived (cond - uncond) so loader
    validation passes by default.
    """
    return pd.DataFrame(
        [
            {
                "k": k,
                "cossim_conditioned": cond,
                "cossim_unconditioned": uncond,
                "delta_cossim": cond - uncond,
            }
            for (k, cond, uncond) in rows
        ]
    )


def _write_baselines_json(path: Path, encoder: str, rmse: float = 0.077) -> Path:
    payload = {
        "version": "1.0",
        "dataset": "nuscenes_v1.0-trainval_full",
        "split": {"train": 24930, "val": 2603, "test": 6019},
        "encoders": {
            encoder: {
                "test_rmse_mean": rmse,
                "test_rmse_std": 0.001,
                "test_mse_mean": rmse * rmse,
                "test_mse_std": 0.0001,
                "seeds": [0, 1, 2],
                "per_seed_rmse": {"0": rmse, "1": rmse, "2": rmse},
            }
        },
    }
    path.write_text(json.dumps(payload))
    return path


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


def test_load_cossim_csv_roundtrip(tmp_path):
    df = _cossim_df([(1, 0.5, 0.4), (2, 0.45, 0.35), (3, 0.40, 0.30), (4, 0.35, 0.25)])
    csv_path = tmp_path / "cossim_results.csv"
    df.to_csv(csv_path, index=False)

    loaded = dcs.load_cossim_csv(csv_path)
    assert list(loaded["k"]) == [1, 2, 3, 4]
    assert (loaded["delta_cossim"] - 0.1).abs().max() < 1e-9


def test_load_cossim_csv_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="cossim CSV not found"):
        dcs.load_cossim_csv(tmp_path / "nope.csv")


def test_load_cossim_csv_missing_column_raises(tmp_path):
    bad = pd.DataFrame({"k": [1], "cossim_conditioned": [0.5]})
    p = tmp_path / "bad.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        dcs.load_cossim_csv(p)


def test_load_cossim_csv_empty_raises(tmp_path):
    empty = pd.DataFrame(
        columns=["k", "cossim_conditioned", "cossim_unconditioned", "delta_cossim"]
    )
    p = tmp_path / "empty.csv"
    empty.to_csv(p, index=False)
    with pytest.raises(ValueError, match="no rows"):
        dcs.load_cossim_csv(p)


def test_load_cossim_csv_nan_raises(tmp_path):
    df = _cossim_df([(1, 0.5, 0.4)])
    df.loc[0, "delta_cossim"] = float("nan")
    p = tmp_path / "nan.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="NaN"):
        dcs.load_cossim_csv(p)


def test_load_cossim_csv_non_contiguous_k_raises(tmp_path):
    df = _cossim_df([(1, 0.5, 0.4), (3, 0.4, 0.3)])  # k=2 missing
    p = tmp_path / "gap.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="must be 1..N contiguous"):
        dcs.load_cossim_csv(p)


def test_load_cossim_csv_inconsistent_delta_raises(tmp_path):
    df = _cossim_df([(1, 0.5, 0.4)])
    df.loc[0, "delta_cossim"] = 0.5  # cond - uncond is 0.1, not 0.5
    p = tmp_path / "inc.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="delta_cossim column does not match"):
        dcs.load_cossim_csv(p)


# ---------------------------------------------------------------------------
# summarize_delta_cossim
# ---------------------------------------------------------------------------


def test_summarize_all_positive_deltas():
    df = _cossim_df([(1, 0.8, 0.2), (2, 0.7, 0.3), (3, 0.6, 0.4), (4, 0.5, 0.45)])
    out = dcs.summarize_delta_cossim(df)

    assert out["delta_positive_at_any_horizon"] is True
    assert out["delta_positive_horizons"] == [1, 2, 3, 4]
    assert out["delta_negative_horizons"] == []
    assert out["best_horizon"]["k"] == 1
    assert out["worst_horizon"]["k"] == 4
    assert out["mean_delta"] > 0
    assert out["interpretation"]["action_conditioning_helps"] is True


def test_summarize_all_negative_deltas():
    df = _cossim_df([(1, 0.5, 0.6), (2, 0.5, 0.6), (3, 0.5, 0.6), (4, 0.5, 0.6)])
    out = dcs.summarize_delta_cossim(df)

    assert out["delta_positive_at_any_horizon"] is False
    assert out["delta_negative_horizons"] == [1, 2, 3, 4]
    assert out["delta_positive_horizons"] == []
    assert out["mean_delta"] < 0
    assert out["interpretation"]["action_conditioning_helps"] is False
    # The verdict text is the *conclusion*; the data observation
    # ("Delta is non-positive at every horizon") is rendered separately
    # by render_summary_markdown to avoid repetition (regression guard
    # for the doubled-sentence bug fixed in this PR).
    assert "does not surface a benefit" in out["interpretation"]["verdict"]
    assert "non-positive" not in out["interpretation"]["verdict"]


def test_summarize_mixed_deltas_picks_best_and_worst():
    df = _cossim_df([(1, 0.6, 0.5), (2, 0.4, 0.5), (3, 0.55, 0.5), (4, 0.30, 0.5)])
    out = dcs.summarize_delta_cossim(df)

    assert out["delta_positive_at_any_horizon"] is True
    assert out["delta_positive_horizons"] == [1, 3]
    assert out["delta_negative_horizons"] == [2, 4]
    assert out["best_horizon"]["k"] == 1
    assert out["worst_horizon"]["k"] == 4


def test_summarize_zero_delta_classified_as_neither_positive_nor_negative():
    df = _cossim_df([(1, 0.5, 0.5)])
    out = dcs.summarize_delta_cossim(df)
    assert out["delta_positive_horizons"] == []
    assert out["delta_negative_horizons"] == []
    assert out["delta_zero_horizons"] == [1]
    assert out["delta_positive_at_any_horizon"] is False


def test_summarize_spec_question_strict_greater_than_zero():
    """Spec asks: "determine if DeltaCosSim > 0 at any horizon" (strict).

    Exact-zero Delta does NOT count as positive; any strictly positive
    Delta on any horizon does. Guards the strict ``> 0`` semantics
    against a future drift to ``>= 0``.
    """
    # All zero -> False.
    out_zero = dcs.summarize_delta_cossim(_cossim_df([(1, 0.5, 0.5), (2, 0.5, 0.5)]))
    assert out_zero["delta_positive_at_any_horizon"] is False

    # One strictly positive -> True.
    out_one_positive = dcs.summarize_delta_cossim(
        _cossim_df([(1, 0.51, 0.5), (2, 0.5, 0.5)])
    )
    assert out_one_positive["delta_positive_at_any_horizon"] is True
    assert out_one_positive["delta_positive_horizons"] == [1]

    # One tiny epsilon-positive (above any practical float noise) -> True.
    out_eps = dcs.summarize_delta_cossim(_cossim_df([(1, 0.5 + 1e-9, 0.5)]))
    assert out_eps["delta_positive_at_any_horizon"] is True


def test_summarize_monotonic_flag_true_when_strictly_decreasing():
    df = _cossim_df([(1, 0.9, 0.1), (2, 0.8, 0.1), (3, 0.7, 0.1), (4, 0.6, 0.1)])
    out = dcs.summarize_delta_cossim(df)
    assert out["cond_cossim_monotonic_nonincreasing"] is True


def test_summarize_monotonic_flag_false_when_cond_increases():
    df = _cossim_df([(1, 0.7, 0.1), (2, 0.8, 0.1), (3, 0.6, 0.1), (4, 0.5, 0.1)])
    out = dcs.summarize_delta_cossim(df)
    assert out["cond_cossim_monotonic_nonincreasing"] is False


def test_summarize_mixed_positive_but_negative_mean_does_not_claim_helps():
    """Edge case: one tiny positive + big negatives -> verdict still 'does not help'."""
    df = _cossim_df(
        [(1, 0.51, 0.50), (2, 0.30, 0.50), (3, 0.30, 0.50), (4, 0.30, 0.50)]
    )
    out = dcs.summarize_delta_cossim(df)
    assert out["delta_positive_at_any_horizon"] is True
    assert out["mean_delta"] < 0
    assert out["interpretation"]["action_conditioning_helps"] is False
    assert "inconsistent" in out["interpretation"]["verdict"]


# ---------------------------------------------------------------------------
# load_bc_baseline
# ---------------------------------------------------------------------------


def test_load_bc_baseline_returns_canonical_fields(tmp_path):
    p = _write_baselines_json(tmp_path / "baselines.json", "vjepa2_rep64", rmse=0.077)
    bc = dcs.load_bc_baseline(p, "vjepa2_rep64")
    assert bc is not None
    assert bc["encoder"] == "vjepa2_rep64"
    assert bc["test_rmse_mean"] == pytest.approx(0.077)
    assert bc["n_seeds"] == 3
    assert bc["dataset"] == "nuscenes_v1.0-trainval_full"
    assert bc["n_test_samples"] == 6019


def test_load_bc_baseline_missing_encoder_returns_none(tmp_path):
    p = _write_baselines_json(tmp_path / "baselines.json", "vjepa2_rep64")
    assert dcs.load_bc_baseline(p, "not_an_encoder") is None


def test_load_bc_baseline_missing_file_returns_none(tmp_path):
    assert dcs.load_bc_baseline(tmp_path / "missing.json", "vjepa2_rep64") is None


# ---------------------------------------------------------------------------
# render_summary_markdown
# ---------------------------------------------------------------------------


def _positive_df() -> pd.DataFrame:
    return _cossim_df(
        [(1, 0.80, 0.10), (2, 0.70, 0.15), (3, 0.60, 0.20), (4, 0.50, 0.25)]
    )


def _negative_df() -> pd.DataFrame:
    """The current real-world shape (cond ~ uncond, Delta < 0)."""
    return _cossim_df(
        [
            (1, 0.9955, 0.9999),
            (2, 0.9955, 0.9999),
            (3, 0.9956, 0.9999),
            (4, 0.9955, 0.9999),
        ]
    )


def test_render_markdown_has_three_paragraphs():
    """The deliverable is explicitly a 3-paragraph summary."""
    analysis = dcs.summarize_delta_cossim(_positive_df())
    md = dcs.render_summary_markdown(analysis, None, None)
    body = md.split("\n\n", 1)[1]  # strip leading H1
    paragraphs = [p for p in body.strip().split("\n\n") if p.strip()]
    assert len(paragraphs) == 3, paragraphs


def test_render_markdown_paragraph_1_quotes_each_horizon():
    analysis = dcs.summarize_delta_cossim(_positive_df())
    md = dcs.render_summary_markdown(analysis, None, None)
    for k in (1, 2, 3, 4):
        assert f"CosSim_cond(k={k})" in md
        assert f"CosSim_uncond(k={k})" in md


def test_render_markdown_paragraph_2_states_positive_verdict():
    analysis = dcs.summarize_delta_cossim(_positive_df())
    md = dcs.render_summary_markdown(analysis, None, None)
    assert "Does action conditioning help?" in md
    assert "action conditioning helps" in md.lower()


def test_render_markdown_paragraph_2_states_negative_verdict():
    analysis = dcs.summarize_delta_cossim(_negative_df())
    md = dcs.render_summary_markdown(analysis, None, None)
    assert "Does action conditioning help?" in md
    assert "non-positive at every horizon" in md
    assert "does not surface a benefit" in md


def test_render_markdown_paragraph_2_does_not_repeat_non_positive_clause():
    """Regression guard: 'non-positive at every horizon' must appear once
    (in the data-observation sentence), not also in the verdict sentence,
    otherwise paragraph 2 reads as two near-identical sentences."""
    analysis = dcs.summarize_delta_cossim(_negative_df())
    md = dcs.render_summary_markdown(analysis, None, None)
    # Strip the H1 header + paragraph 1 + paragraph 3; isolate paragraph 2.
    body = md.split("\n\n", 1)[1]
    paragraphs = [p for p in body.strip().split("\n\n") if p.strip()]
    paragraph_2 = paragraphs[1]
    assert paragraph_2.count("non-positive") == 1, paragraph_2
    assert paragraph_2.count("DeltaCosSim is") == 1, paragraph_2


def test_render_markdown_paragraph_3_richer_claim_flips_with_delta_sign():
    md_pos = dcs.render_summary_markdown(
        dcs.summarize_delta_cossim(_positive_df()), None, None
    )
    md_neg = dcs.render_summary_markdown(
        dcs.summarize_delta_cossim(_negative_df()), None, None
    )
    # Positive Delta: paragraph 3 should make the "richer" claim.
    # ("*richer*" is italicized in the rendered text, so we check the
    # surrounding phrasing rather than an unbroken substring.)
    assert "richer" in md_pos
    assert "richer*" in md_pos and "representation" in md_pos
    # Negative Delta: paragraph 3 should explicitly refute the claim.
    assert "does **not**" in md_neg
    assert "richer representation than BC" in md_neg


def test_render_markdown_paragraph_3_uses_bc_baseline_when_present():
    analysis = dcs.summarize_delta_cossim(_negative_df())
    bc = {
        "encoder": "vjepa2_rep64",
        "test_rmse_mean": 0.0774,
        "test_rmse_std": 0.0009,
        "n_seeds": 3,
        "dataset": "nuscenes_v1.0-trainval_full",
        "n_test_samples": 6019,
    }
    md = dcs.render_summary_markdown(analysis, bc, None)
    assert "0.0774" in md
    assert "0.0009" in md
    assert "6019" in md


def test_render_markdown_paragraph_3_handles_missing_bc_baseline():
    analysis = dcs.summarize_delta_cossim(_negative_df())
    md = dcs.render_summary_markdown(analysis, None, None)
    assert "not available" in md


def test_render_markdown_uses_encoder_from_cossim_metadata():
    analysis = dcs.summarize_delta_cossim(_negative_df())
    md = dcs.render_summary_markdown(
        analysis,
        bc_baseline=None,
        cossim_metadata={"encoder": "vit_s16", "n_samples": 1234},
    )
    assert "`vit_s16`" in md
    assert "1,234-sequence" in md


def test_render_markdown_quotes_monotonicity_finding():
    md_monotonic = dcs.render_summary_markdown(
        dcs.summarize_delta_cossim(
            _cossim_df([(1, 0.9, 0.1), (2, 0.8, 0.1), (3, 0.7, 0.1), (4, 0.6, 0.1)])
        ),
        None,
        None,
    )
    md_not_monotonic = dcs.render_summary_markdown(
        dcs.summarize_delta_cossim(
            _cossim_df([(1, 0.7, 0.1), (2, 0.8, 0.1), (3, 0.7, 0.1), (4, 0.6, 0.1)])
        ),
        None,
        None,
    )
    assert "monotonically non-increasing" in md_monotonic
    assert "not** monotonically" in md_not_monotonic


def test_render_markdown_paragraph_3_mixed_delta_does_not_say_non_positive_everywhere():
    """Blocking review item: when some horizons are positive but mean is
    negative, paragraph 3 must NOT claim 'non-positive at every horizon'."""
    df = _cossim_df(
        [(1, 0.51, 0.50), (2, 0.30, 0.50), (3, 0.30, 0.50), (4, 0.30, 0.50)]
    )
    analysis = dcs.summarize_delta_cossim(df)
    assert analysis["delta_positive_at_any_horizon"] is True
    assert analysis["interpretation"]["action_conditioning_helps"] is False

    md = dcs.render_summary_markdown(analysis, None, None)
    body = md.split("\n\n", 1)[1]
    paragraphs = [p for p in body.strip().split("\n\n") if p.strip()]
    paragraph_3 = paragraphs[2]
    assert "non-positive at every horizon" not in paragraph_3
    assert "minority of horizons" in paragraph_3
    assert "does **not**" in paragraph_3


def test_render_markdown_title_uses_double_dash_not_em_dash():
    analysis = dcs.summarize_delta_cossim(_negative_df())
    md = dcs.render_summary_markdown(analysis, None, None)
    title = md.split("\n")[0]
    assert "—" not in title
    assert "--" in title


def test_load_cossim_csv_non_integer_k_raises(tmp_path):
    """k=1.5 must not silently truncate to k=1."""
    df = pd.DataFrame(
        {
            "k": [1.0, 1.5],
            "cossim_conditioned": [0.5, 0.4],
            "cossim_unconditioned": [0.4, 0.3],
            "delta_cossim": [0.1, 0.1],
        }
    )
    p = tmp_path / "float_k.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="non-integer"):
        dcs.load_cossim_csv(p)


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------


def _seed_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create vendored-style inputs for an end-to-end CLI run."""
    artifacts_dir = tmp_path / "artifacts" / "cossim_eval"
    artifacts_dir.mkdir(parents=True)
    df = _negative_df()
    csv_path = artifacts_dir / "cossim_results.csv"
    df.to_csv(csv_path, index=False)

    json_path = artifacts_dir / "cossim_results.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "metadata": {
                    "encoder": "vjepa2_rep64",
                    "n_samples": 5419,
                    "horizon": 4,
                    "z_dim": 384,
                    "generated_at_utc": "2026-05-25T00:00:00+00:00",
                },
            }
        )
    )

    baselines_path = _write_baselines_json(
        tmp_path / "baselines.json", "vjepa2_rep64", rmse=0.0774
    )
    output_root = tmp_path / "outputs" / "analysis"
    return csv_path, json_path, baselines_path, output_root


def test_cli_main_writes_json_and_markdown(tmp_path, capsys):
    csv_path, json_path, baselines_path, output_root = _seed_inputs(tmp_path)

    rc = dcs.main(
        [
            "--cossim-csv",
            str(csv_path),
            "--cossim-json",
            str(json_path),
            "--baselines-json",
            str(baselines_path),
            "--output-root",
            str(output_root),
        ]
    )
    assert rc == 0

    md_path = output_root / "delta_cossim_summary.md"
    json_out = output_root / "delta_cossim_summary.json"
    assert md_path.exists()
    assert json_out.exists()

    md = md_path.read_text()
    assert "# DeltaCosSim results summary" in md
    assert "vjepa2_rep64" in md
    assert "0.0774" in md
    assert "5,419-sequence" in md

    payload = json.loads(json_out.read_text())
    assert payload["schema_version"] == "1.0"
    assert payload["encoder"] == "vjepa2_rep64"
    assert payload["analysis"]["delta_positive_at_any_horizon"] is False
    assert payload["bc_baseline"]["test_rmse_mean"] == pytest.approx(0.0774)
    assert payload["source_paths"]["cossim_csv"] == str(csv_path)

    stdout = capsys.readouterr().out
    assert "any_delta_positive=False" in stdout
    assert "encoder=vjepa2_rep64" in stdout


def test_cli_main_works_without_baselines_json(tmp_path):
    csv_path, json_path, baselines_path, output_root = _seed_inputs(tmp_path)
    baselines_path.unlink()  # drop the BC source

    rc = dcs.main(
        [
            "--cossim-csv",
            str(csv_path),
            "--cossim-json",
            str(json_path),
            "--baselines-json",
            str(baselines_path),
            "--output-root",
            str(output_root),
        ]
    )
    assert rc == 0

    md = (output_root / "delta_cossim_summary.md").read_text()
    assert "not available" in md
    payload = json.loads((output_root / "delta_cossim_summary.json").read_text())
    assert payload["bc_baseline"] is None


def test_cli_main_works_without_cossim_json(tmp_path):
    csv_path, json_path, baselines_path, output_root = _seed_inputs(tmp_path)
    json_path.unlink()

    rc = dcs.main(
        [
            "--cossim-csv",
            str(csv_path),
            "--cossim-json",
            str(json_path),
            "--baselines-json",
            str(baselines_path),
            "--output-root",
            str(output_root),
            "--encoder",
            "vjepa2_rep64",
        ]
    )
    assert rc == 0
    payload = json.loads((output_root / "delta_cossim_summary.json").read_text())
    assert payload["cossim_metadata"] is None
    assert payload["encoder"] == "vjepa2_rep64"


# ---------------------------------------------------------------------------
# Smoke test against the vendored artifact
# ---------------------------------------------------------------------------


def test_smoke_vendored_artifact_parses_summarizes_and_renders():
    """The vendored CSV at artifacts/cossim_eval/ must remain self-consistent
    AND fully renderable through the markdown pipeline.

    Guards against:

    * Loader contract drift (column order, contiguous k, delta = cond - uncond).
    * Markdown renderer crashing on a real cossim_metadata block.
    * The committed snapshot at artifacts/cossim_eval/delta_cossim_summary.md
      drifting away from what the script would produce now (stale-artifact
      guard).
    """
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = repo_root / "artifacts" / "cossim_eval"
    csv_path = artifacts_dir / "cossim_results.csv"
    json_path = artifacts_dir / "cossim_results.json"
    md_path = artifacts_dir / "delta_cossim_summary.md"
    if not csv_path.exists():
        pytest.skip(f"vendored artifact not present at {csv_path}")

    df = dcs.load_cossim_csv(csv_path)
    analysis = dcs.summarize_delta_cossim(df)
    assert analysis["horizon"] >= 1
    assert "verdict" in analysis["interpretation"]
    assert isinstance(analysis["delta_positive_at_any_horizon"], bool)

    metadata = (
        json.loads(json_path.read_text()).get("metadata")
        if json_path.exists()
        else None
    )
    bc = dcs.load_bc_baseline(
        repo_root / "configs" / "baselines.json",
        (metadata or {}).get("encoder", dcs.DEFAULT_ENCODER),
    )
    rendered = dcs.render_summary_markdown(analysis, bc, metadata)

    # Sanity: rendered MD has the 3 paragraph anchors and is non-empty.
    assert "**CosSim values at k=" in rendered
    assert "**Does action conditioning help?**" in rendered
    assert "**Comparison to BC baseline.**" in rendered

    if md_path.exists():
        committed = md_path.read_text().strip()
        assert committed == rendered.strip(), (
            f"vendored {md_path.name} is stale -- regenerate with:\n"
            f"  python -m analysis.delta_cossim_summary && "
            f"cp outputs/analysis/delta_cossim_summary.md "
            f"artifacts/cossim_eval/"
        )
