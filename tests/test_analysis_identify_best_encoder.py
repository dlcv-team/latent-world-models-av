"""Tests for ``analysis.identify_best_encoder``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from analysis import identify_best_encoder as ibe


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _summary(encoder_to_rmse: dict[str, float]) -> pd.DataFrame:
    """Synthetic encoder_summary_with_ci.csv as a DataFrame."""
    rows = []
    for enc, rmse in encoder_to_rmse.items():
        rows.append(
            {
                "encoder": enc,
                "steer_rmse_scene_mean": rmse,
                "steer_ci95_lo": rmse - 0.02,
                "steer_ci95_hi": rmse + 0.02,
                "accel_rmse_scene_mean": rmse * 0.5,
                "accel_ci95_lo": rmse * 0.5 - 0.02,
                "accel_ci95_hi": rmse * 0.5 + 0.02,
                "num_scenes": 40,
            }
        )
    return pd.DataFrame(rows)


def _paired_row(
    a: str, b: str, p_bonferroni: float, mean_diff: float, cohens_d: float = 0.5,
    n_comparisons: int = 3, alpha: float = 0.05,
) -> dict:
    """Single paired_tests.csv row matching the canonical schema."""
    return {
        "encoder_a": a,
        "encoder_b": b,
        "n_scenes": 40,
        "t_stat": 2.0,
        "p_value": p_bonferroni / n_comparisons,
        "n_comparisons": n_comparisons,
        "bonferroni_alpha": alpha / n_comparisons,
        "p_bonferroni": p_bonferroni,
        "mean_diff_a_minus_b": mean_diff,
        "cohens_d": cohens_d,
    }


# ---------------------------------------------------------------------------
# identify_best_encoder
# ---------------------------------------------------------------------------


def test_identify_best_encoder_picks_min_rmse():
    summary = _summary({"enc_a": 0.10, "enc_b": 0.12, "enc_c": 0.08})
    # Bonferroni significant: c vs a (c better, mean_diff_a_minus_b > 0)
    paired = pd.DataFrame(
        [
            _paired_row("enc_a", "enc_b", p_bonferroni=1.0, mean_diff=-0.02),
            _paired_row("enc_a", "enc_c", p_bonferroni=0.003, mean_diff=0.02),
            _paired_row("enc_b", "enc_c", p_bonferroni=0.001, mean_diff=0.04),
        ]
    )
    result = ibe.identify_best_encoder(summary, paired)
    assert result["best_encoder"] == "enc_c"
    assert result["value"] == 0.08
    assert result["tied"] is False


def test_identify_best_encoder_includes_only_bonferroni_significant_pairs():
    summary = _summary({"enc_a": 0.10, "enc_b": 0.12, "enc_c": 0.08})
    paired = pd.DataFrame(
        [
            _paired_row("enc_a", "enc_b", p_bonferroni=1.0, mean_diff=-0.02),
            # enc_c beats enc_a, significant
            _paired_row("enc_a", "enc_c", p_bonferroni=0.003, mean_diff=0.02),
            # enc_c beats enc_b but NOT significant (p > alpha=0.05)
            _paired_row("enc_b", "enc_c", p_bonferroni=0.5, mean_diff=0.04),
        ]
    )
    result = ibe.identify_best_encoder(summary, paired)
    beats = [b["encoder"] for b in result["significantly_beats"]]
    assert beats == ["enc_a"]


def test_identify_best_encoder_orders_beats_by_p_bonferroni_ascending():
    summary = _summary({"enc_a": 0.20, "enc_b": 0.10, "enc_c": 0.30})
    # enc_b beats both with different p values
    paired = pd.DataFrame(
        [
            _paired_row("enc_a", "enc_b", p_bonferroni=0.01, mean_diff=0.10),
            _paired_row("enc_b", "enc_c", p_bonferroni=0.001, mean_diff=-0.20),
            _paired_row("enc_a", "enc_c", p_bonferroni=1.0, mean_diff=-0.10),
        ]
    )
    result = ibe.identify_best_encoder(summary, paired)
    assert result["best_encoder"] == "enc_b"
    beats_order = [b["encoder"] for b in result["significantly_beats"]]
    assert beats_order == ["enc_c", "enc_a"]  # 0.001 < 0.01


def test_identify_best_encoder_handles_tie():
    summary = _summary({"enc_a": 0.10, "enc_b": 0.10, "enc_c": 0.15})
    paired = pd.DataFrame(
        [
            _paired_row("enc_a", "enc_b", p_bonferroni=1.0, mean_diff=0.0),
            _paired_row("enc_a", "enc_c", p_bonferroni=0.5, mean_diff=-0.05),
            _paired_row("enc_b", "enc_c", p_bonferroni=0.5, mean_diff=-0.05),
        ]
    )
    result = ibe.identify_best_encoder(summary, paired)
    assert result["tied"] is True
    # Lex break: enc_a < enc_b alphabetically.
    assert result["best_encoder"] == "enc_a"


def test_identify_best_encoder_dict_schema():
    summary = _summary({"enc_a": 0.10, "enc_b": 0.12})
    paired = pd.DataFrame(
        [_paired_row("enc_a", "enc_b", p_bonferroni=0.001, mean_diff=-0.02)]
    )
    result = ibe.identify_best_encoder(summary, paired)
    expected_keys = {
        "best_encoder",
        "metric",
        "value",
        "ci_95",
        "num_scenes",
        "n_comparisons",
        "bonferroni_alpha",
        "alpha",
        "significantly_beats",
        "tied",
        "fallback_caveat",
    }
    assert set(result) == expected_keys


def test_identify_best_encoder_rejects_empty_paired():
    summary = _summary({"enc_a": 0.10, "enc_b": 0.12})
    with pytest.raises(ValueError, match="empty"):
        ibe.identify_best_encoder(summary, pd.DataFrame())


def test_identify_best_encoder_rejects_missing_metric_column():
    summary = pd.DataFrame([{"encoder": "x", "other_col": 0.1}])
    paired = pd.DataFrame(
        [_paired_row("x", "y", p_bonferroni=0.01, mean_diff=-0.01)]
    )
    with pytest.raises(ValueError, match="steer_rmse_scene_mean"):
        ibe.identify_best_encoder(summary, paired)


# ---------------------------------------------------------------------------
# attach_fallback_caveat
# ---------------------------------------------------------------------------


def test_attach_fallback_caveat_reads_provenance(tmp_path):
    winner = "vq_track"
    enc_dir = tmp_path / winner
    enc_dir.mkdir()
    (enc_dir / "provenance.json").write_text(
        json.dumps({"fallback_caveat": "VQ falls back to DINOv2 per FR-08."})
    )
    result = {"best_encoder": winner, "fallback_caveat": None}
    ibe.attach_fallback_caveat(result, tmp_path)
    assert "FR-08" in result["fallback_caveat"]


def test_attach_fallback_caveat_missing_provenance_is_null(tmp_path):
    result = {"best_encoder": "vit_s16", "fallback_caveat": None}
    ibe.attach_fallback_caveat(result, tmp_path)
    assert result["fallback_caveat"] is None


def test_attach_fallback_caveat_empty_string_stays_null(tmp_path):
    """Empty caveat strings are normalised to null (downstream-friendly)."""
    winner = "vit_s16"
    enc_dir = tmp_path / winner
    enc_dir.mkdir()
    (enc_dir / "provenance.json").write_text(
        json.dumps({"fallback_caveat": ""})
    )
    result = {"best_encoder": winner, "fallback_caveat": None}
    ibe.attach_fallback_caveat(result, tmp_path)
    assert result["fallback_caveat"] is None


# ---------------------------------------------------------------------------
# render_summary_markdown
# ---------------------------------------------------------------------------


def test_render_summary_markdown_contains_winner_and_action_item():
    result = {
        "best_encoder": "enc_x",
        "value": 0.090,
        "ci_95": [0.060, 0.120],
        "num_scenes": 40,
        "n_comparisons": 10,
        "significantly_beats": [],
        "tied": False,
        "fallback_caveat": None,
    }
    md = ibe.render_summary_markdown(result)
    assert "`enc_x`" in md
    assert "M3" in md
    assert "BC training" in md


def test_render_summary_markdown_lists_significant_pairs():
    result = {
        "best_encoder": "enc_x",
        "value": 0.090,
        "ci_95": [0.060, 0.120],
        "num_scenes": 40,
        "n_comparisons": 10,
        "significantly_beats": [
            {"encoder": "enc_y", "p_bonferroni": 0.001, "cohens_d": -0.5},
            {"encoder": "enc_z", "p_bonferroni": 0.003, "cohens_d": -0.6},
        ],
        "tied": False,
        "fallback_caveat": None,
    }
    md = ibe.render_summary_markdown(result)
    assert "enc_y" in md and "enc_z" in md
    assert "Bonferroni-significant" in md


def test_render_summary_markdown_handles_no_significant_pairs():
    result = {
        "best_encoder": "enc_x",
        "value": 0.090,
        "ci_95": [0.060, 0.120],
        "num_scenes": 40,
        "n_comparisons": 10,
        "significantly_beats": [],
        "tied": False,
        "fallback_caveat": None,
    }
    md = ibe.render_summary_markdown(result)
    assert "No pair clears Bonferroni" in md


def test_render_summary_markdown_includes_fallback_when_present():
    result = {
        "best_encoder": "vq_track",
        "value": 0.090,
        "ci_95": [0.060, 0.120],
        "num_scenes": 40,
        "n_comparisons": 10,
        "significantly_beats": [],
        "tied": False,
        "fallback_caveat": "VQ-VAE uses DINOv2 fallback.",
    }
    md = ibe.render_summary_markdown(result)
    assert "VQ-VAE uses DINOv2 fallback." in md


def test_render_summary_markdown_calls_out_tie():
    result = {
        "best_encoder": "enc_a",
        "value": 0.10,
        "ci_95": [0.08, 0.12],
        "num_scenes": 40,
        "n_comparisons": 3,
        "significantly_beats": [],
        "tied": True,
        "fallback_caveat": None,
    }
    md = ibe.render_summary_markdown(result)
    assert "tie" in md.lower()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def _write_synthetic_analysis(root: Path) -> None:
    """Write minimal encoder_summary_with_ci.csv + paired_tests.csv."""
    root.mkdir(parents=True, exist_ok=True)
    _summary({"enc_a": 0.10, "enc_b": 0.12, "enc_c": 0.08}).to_csv(
        root / "encoder_summary_with_ci.csv", index=False
    )
    pd.DataFrame(
        [
            _paired_row("enc_a", "enc_b", p_bonferroni=1.0, mean_diff=-0.02),
            _paired_row("enc_a", "enc_c", p_bonferroni=0.003, mean_diff=0.02),
            _paired_row("enc_b", "enc_c", p_bonferroni=0.001, mean_diff=0.04),
        ]
    ).to_csv(root / "paired_tests.csv", index=False)


def test_main_smoke_with_synthetic_inputs(tmp_path):
    analysis_root = tmp_path / "analysis"
    _write_synthetic_analysis(analysis_root)
    rc = ibe.main(
        [
            "--analysis-root", str(analysis_root),
            "--probe-root", str(tmp_path / "probes"),  # missing dir is fine
        ]
    )
    assert rc == 0
    payload = json.loads((analysis_root / "best_encoder.json").read_text())
    assert payload["best_encoder"] == "enc_c"
    md = (analysis_root / "best_encoder_summary.md").read_text()
    assert "enc_c" in md


def test_main_rejects_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing analysis input"):
        ibe.main(
            [
                "--analysis-root", str(tmp_path / "nope"),
                "--probe-root", str(tmp_path / "probes"),
            ]
        )


def test_main_writes_to_separate_output_root(tmp_path):
    analysis_root = tmp_path / "analysis"
    output_root = tmp_path / "out"
    _write_synthetic_analysis(analysis_root)
    rc = ibe.main(
        [
            "--analysis-root", str(analysis_root),
            "--probe-root", str(tmp_path / "probes"),
            "--output-root", str(output_root),
        ]
    )
    assert rc == 0
    assert (output_root / "best_encoder.json").exists()
    assert (output_root / "best_encoder_summary.md").exists()
    # Analysis root not touched (only inputs read).
    assert not (analysis_root / "best_encoder.json").exists()


# ---------------------------------------------------------------------------
# Integration: real in-repo pilot
# ---------------------------------------------------------------------------


def _adopt_in_repo_pilot(out_root: Path) -> None:
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    import adopt_pilot_artifacts as apa

    apa.adopt(
        artifact_root=apa.DEFAULT_ARTIFACT_ROOT,
        output_root=out_root,
        cfg_manifest_sha256="(test)",
        cfg_version="1.0.0",
        retry_report_root=apa.DEFAULT_RETRY_REPORT_ROOT,
    )


def _in_repo_pilot_present() -> bool:
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    import adopt_pilot_artifacts as apa

    return (apa.DEFAULT_ARTIFACT_ROOT / "per_scene" / "per_scene_rmse.csv").exists()


@pytest.mark.skipif(
    not _in_repo_pilot_present(),
    reason="in-repo pilot artifacts not present",
)
def test_against_in_repo_pilot(tmp_path):
    """End-to-end against the committed pilot.

    Asserts vjepa2_rep64 wins, beats clip_b32 + vit_s16 with Bonferroni
    significance, and the result is consistent with the canonical key
    pinned in tests/data/pilot_baselines.json::expected_best_encoder.
    """
    from analysis import paired_tests as pt_mod

    probe_root = tmp_path / "probes"
    analysis_root = tmp_path / "analysis"
    _adopt_in_repo_pilot(probe_root)
    assert pt_mod.main(
        ["--probe-root", str(probe_root), "--output-root", str(analysis_root)]
    ) == 0
    assert ibe.main(
        [
            "--analysis-root", str(analysis_root),
            "--probe-root", str(probe_root),
        ]
    ) == 0

    payload = json.loads((analysis_root / "best_encoder.json").read_text())
    assert payload["best_encoder"] == "vjepa2_rep64"
    beats = {b["encoder"] for b in payload["significantly_beats"]}
    assert {"clip_b32", "vit_s16"} <= beats
    for b in payload["significantly_beats"]:
        assert b["p_bonferroni"] < 0.05
    assert payload["fallback_caveat"] is None  # winner isn't VQ

    # Cross-reference the pinned fixture.
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    fixture = json.loads(
        (_REPO_ROOT / "tests" / "data" / "pilot_baselines.json").read_text()
    )
    # Fixture uses canonical key "vjepa2"; result uses pilot name "vjepa2_rep64".
    assert fixture["expected_best_encoder"] == "vjepa2"
    assert payload["best_encoder"].startswith("vjepa2")

    md = (analysis_root / "best_encoder_summary.md").read_text()
    assert "vjepa2_rep64" in md and "M3" in md
