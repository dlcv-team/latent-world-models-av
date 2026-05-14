"""Tests for ``analysis.paired_tests``."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import paired_tests as pt


# ---------------------------------------------------------------------------
# Synthetic probe-dir fixture
# ---------------------------------------------------------------------------


def _write_per_scene_rmse(
    root: Path,
    encoder: str,
    scenes: list[str],
    steer_values: list[float],
    accel_values: list[float] | None = None,
) -> Path:
    """Write a synthetic per_scene_rmse.csv under <root>/<encoder>/."""
    if accel_values is None:
        accel_values = [v * 0.5 for v in steer_values]
    enc_dir = root / encoder
    enc_dir.mkdir(parents=True, exist_ok=True)
    csv_path = enc_dir / "per_scene_rmse.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["encoder", "scene_name", "scenario", "fold_id", "steer_rmse", "accel_rmse", "n"]
        )
        for scene, s_v, a_v in zip(scenes, steer_values, accel_values):
            w.writerow([encoder, scene, "urban", 0, s_v, a_v, 15])
    return csv_path


@pytest.fixture
def three_encoder_probe_dir(tmp_path: Path) -> Path:
    """3 encoders × 5 scenes with deterministic, distinct means."""
    root = tmp_path / "probes"
    scenes = [f"scene-{i:04d}" for i in range(5)]
    # Encoder A: ~0.10 mean, Encoder B: ~0.12, Encoder C: ~0.08.
    _write_per_scene_rmse(root, "enc_a", scenes, [0.09, 0.10, 0.11, 0.10, 0.10])
    _write_per_scene_rmse(root, "enc_b", scenes, [0.11, 0.12, 0.13, 0.12, 0.12])
    _write_per_scene_rmse(root, "enc_c", scenes, [0.07, 0.08, 0.09, 0.08, 0.08])
    return root


# ---------------------------------------------------------------------------
# load_per_scene_rmse
# ---------------------------------------------------------------------------


def test_load_per_scene_rmse_three_encoders(three_encoder_probe_dir):
    out = pt.load_per_scene_rmse(three_encoder_probe_dir)
    assert set(out) == {"enc_a", "enc_b", "enc_c"}
    for series in out.values():
        assert len(series) == 5
        assert list(series.index) == [f"scene-{i:04d}" for i in range(5)]


def test_load_raises_on_mismatched_scene_sets(tmp_path):
    root = tmp_path / "probes"
    _write_per_scene_rmse(root, "enc_a", ["s-0", "s-1", "s-2"], [0.1, 0.1, 0.1])
    _write_per_scene_rmse(root, "enc_b", ["s-0", "s-1"], [0.1, 0.1])  # missing s-2
    with pytest.raises(ValueError, match="mismatched scene set"):
        pt.load_per_scene_rmse(root)


def test_load_raises_when_root_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="probe-root does not exist"):
        pt.load_per_scene_rmse(tmp_path / "nope")


def test_load_skips_dirs_without_per_scene_csv(tmp_path):
    root = tmp_path / "probes"
    _write_per_scene_rmse(root, "enc_a", ["s-0"], [0.1])
    (root / "stale_no_csv").mkdir()
    out = pt.load_per_scene_rmse(root)
    assert set(out) == {"enc_a"}


def test_load_rejects_unsupported_metric(three_encoder_probe_dir):
    with pytest.raises(ValueError, match="metric must be one of"):
        pt.load_per_scene_rmse(three_encoder_probe_dir, metric="bogus")


# ---------------------------------------------------------------------------
# compute_paired_tests
# ---------------------------------------------------------------------------


def test_compute_paired_tests_columns_and_shape(three_encoder_probe_dir):
    per_enc = pt.load_per_scene_rmse(three_encoder_probe_dir)
    df = pt.compute_paired_tests(per_enc, alpha=0.05)
    assert list(df.columns) == pt.PAIRED_TESTS_COLUMNS
    assert len(df) == 3  # C(3, 2)
    # n_comparisons must be filled in for every row.
    assert (df["n_comparisons"] == 3).all()
    # bonferroni_alpha = alpha / n_comparisons.
    assert np.allclose(df["bonferroni_alpha"], 0.05 / 3)
    # Pair ordering is deterministic (lexicographic over sorted encoders).
    assert list(zip(df["encoder_a"], df["encoder_b"])) == [
        ("enc_a", "enc_b"), ("enc_a", "enc_c"), ("enc_b", "enc_c"),
    ]


def test_compute_paired_tests_known_values():
    """Verify mean_diff_a_minus_b and Cohen's d against hand-computed values.

    Series A and B differ by a known per-scene delta with non-zero
    spread; the paired t-test is well-defined and we can recompute
    each cell.
    """
    scenes = [f"s-{i}" for i in range(20)]
    a_vals = np.linspace(0.10, 0.30, 20)
    b_vals = a_vals - np.linspace(0.02, 0.06, 20)  # diff varies, mean ≈ 0.04
    per_enc = {
        "a": pd.Series(a_vals, index=scenes),
        "b": pd.Series(b_vals, index=scenes),
    }
    df = pt.compute_paired_tests(per_enc, alpha=0.05)
    row = df.iloc[0]
    assert row["encoder_a"] == "a" and row["encoder_b"] == "b"
    assert row["n_scenes"] == 20

    diff = a_vals - b_vals
    expected_mean_diff = float(np.mean(diff))
    expected_d = float(np.mean(diff) / np.std(diff, ddof=1))
    assert math.isclose(row["mean_diff_a_minus_b"], expected_mean_diff, rel_tol=1e-9)
    assert math.isclose(row["cohens_d"], expected_d, rel_tol=1e-9)
    assert row["t_stat"] > 0  # a > b on average
    assert 0.0 < row["p_value"] <= 1.0


def test_cohens_d_uses_sd_of_paired_differences():
    """Verify formula against a hand-computed value."""
    scenes = [f"s-{i}" for i in range(4)]
    a = pd.Series([0.10, 0.20, 0.30, 0.40], index=scenes)
    b = pd.Series([0.08, 0.19, 0.27, 0.40], index=scenes)
    per_enc = {"a": a, "b": b}
    df = pt.compute_paired_tests(per_enc, alpha=0.05)
    diff = a.values - b.values
    expected_d = float(np.mean(diff) / np.std(diff, ddof=1))
    assert math.isclose(df.iloc[0]["cohens_d"], expected_d, rel_tol=1e-9)


def test_bonferroni_correction_uses_n_comparisons_from_data():
    """Same alpha, different encoder counts → different bonferroni_alpha."""
    scenes = [f"s-{i}" for i in range(10)]
    # Use distinct seeds per encoder so paired diffs are non-degenerate.
    per_3 = {
        n: pd.Series(np.random.RandomState(seed).rand(10), index=scenes)
        for seed, n in enumerate(("a", "b", "c"))
    }
    per_5 = {
        n: pd.Series(np.random.RandomState(seed).rand(10), index=scenes)
        for seed, n in enumerate(("a", "b", "c", "d", "e"))
    }
    df_3 = pt.compute_paired_tests(per_3, alpha=0.05)
    df_5 = pt.compute_paired_tests(per_5, alpha=0.05)
    assert (df_3["n_comparisons"] == 3).all()
    assert (df_5["n_comparisons"] == 10).all()
    assert np.allclose(df_3["bonferroni_alpha"], 0.05 / 3)
    assert np.allclose(df_5["bonferroni_alpha"], 0.05 / 10)
    # p_bonferroni clamps at 1.0.
    assert (df_3["p_bonferroni"] <= 1.0).all()
    assert (df_5["p_bonferroni"] <= 1.0).all()


def test_compute_paired_tests_rejects_single_encoder():
    per_enc = {"a": pd.Series([0.1, 0.2], index=["s-0", "s-1"])}
    with pytest.raises(ValueError, match="at least 2 encoders"):
        pt.compute_paired_tests(per_enc, alpha=0.05)


# ---------------------------------------------------------------------------
# Bootstrap + encoder summary
# ---------------------------------------------------------------------------


def test_bootstrap_mean_ci_brackets_mean():
    rng = np.random.RandomState(0)
    x = rng.normal(loc=0.1, scale=0.02, size=40)
    mean, lo, hi = pt.bootstrap_mean_ci(x, n_resamples=200, seed=42, confidence_level=0.95)
    assert lo <= mean <= hi
    # CI for n=40 sample should be tight.
    assert hi - lo < 0.02


def test_bootstrap_mean_ci_rejects_empty():
    with pytest.raises(ValueError):
        pt.bootstrap_mean_ci(np.array([]), n_resamples=10, seed=0, confidence_level=0.95)


def test_compute_encoder_summary_with_ci_schema(three_encoder_probe_dir):
    df = pt.compute_encoder_summary_with_ci(
        three_encoder_probe_dir, n_resamples=100, seed=42, confidence_level=0.95,
    )
    assert list(df.columns) == pt.SUMMARY_COLUMNS
    assert set(df["encoder"]) == {"enc_a", "enc_b", "enc_c"}
    for _, row in df.iterrows():
        assert row["steer_ci95_lo"] <= row["steer_rmse_scene_mean"] <= row["steer_ci95_hi"]
        assert row["accel_ci95_lo"] <= row["accel_rmse_scene_mean"] <= row["accel_ci95_hi"]
        assert row["num_scenes"] == 5


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------


def test_render_latex_table_has_footnote_with_n_comparisons(three_encoder_probe_dir):
    per_enc = pt.load_per_scene_rmse(three_encoder_probe_dir)
    df = pt.compute_paired_tests(per_enc, alpha=0.05)
    tex = pt.render_latex_table(df)
    assert r"\begin{tabular}" in tex
    assert r"\end{tabular}" in tex
    # Footnote dynamically cites n_comparisons from the data.
    assert "n=3" in tex
    # No hardcoded 10 (the typical 5-encoder pair count) sneaks in.
    assert "n=10" not in tex


def test_render_latex_table_rejects_empty_df():
    with pytest.raises(ValueError, match="empty"):
        pt.render_latex_table(pd.DataFrame(columns=pt.PAIRED_TESTS_COLUMNS))


def test_render_latex_table_escapes_underscores(three_encoder_probe_dir):
    per_enc = pt.load_per_scene_rmse(three_encoder_probe_dir)
    df = pt.compute_paired_tests(per_enc, alpha=0.05)
    tex = pt.render_latex_table(df)
    assert r"enc\_a" in tex  # encoder names with underscores must be escaped


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_smoke_synthetic_probe_dir(three_encoder_probe_dir, tmp_path):
    out_root = tmp_path / "analysis"
    rc = pt.main(
        [
            "--probe-root", str(three_encoder_probe_dir),
            "--output-root", str(out_root),
        ]
    )
    assert rc == 0
    assert (out_root / "paired_tests.csv").exists()
    assert (out_root / "encoder_summary_with_ci.csv").exists()
    assert (out_root / "paired_tests.tex").exists()
    paired = pd.read_csv(out_root / "paired_tests.csv")
    assert list(paired.columns) == pt.PAIRED_TESTS_COLUMNS
    assert len(paired) == 3


def test_main_rejects_missing_probe_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        pt.main(["--probe-root", str(tmp_path / "nope"), "--output-root", str(tmp_path / "out")])


# ---------------------------------------------------------------------------
# Integration: real in-repo pilot
# ---------------------------------------------------------------------------


def _adopt_in_repo_pilot(out_root: Path) -> None:
    """Run the adopt script's adopt() against the in-repo pilot."""
    # Import lazily because scripts/ isn't a regular package.
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

    Verifies 5 encoders, n_comparisons=10, vjepa2_rep64 wins on steer
    scene-mean RMSE, and the vjepa2_rep64 vs clip_b32 pair is
    Bonferroni-significant (matches pilot's headline claims).
    """
    probe_root = tmp_path / "probes"
    out_root = tmp_path / "analysis"
    _adopt_in_repo_pilot(probe_root)

    rc = pt.main(["--probe-root", str(probe_root), "--output-root", str(out_root)])
    assert rc == 0

    paired = pd.read_csv(out_root / "paired_tests.csv")
    assert (paired["n_comparisons"] == 10).all()
    assert len(paired) == 10

    summary = pd.read_csv(out_root / "encoder_summary_with_ci.csv")
    assert len(summary) == 5
    # vjepa2_rep64 has the lowest scene-mean steer RMSE (matches pilot).
    best = summary.sort_values("steer_rmse_scene_mean").iloc[0]["encoder"]
    assert best == "vjepa2_rep64", (
        f"expected vjepa2_rep64 to win on steer, got {best!r}"
    )

    # vjepa2_rep64 vs clip_b32 must be Bonferroni-significant.
    pair = paired[
        (paired["encoder_a"].isin(["clip_b32", "vjepa2_rep64"]))
        & (paired["encoder_b"].isin(["clip_b32", "vjepa2_rep64"]))
    ]
    assert len(pair) == 1
    assert pair.iloc[0]["p_bonferroni"] < 0.05, pair.iloc[0]

    tex = (out_root / "paired_tests.tex").read_text()
    assert "n=10" in tex
