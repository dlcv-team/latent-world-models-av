"""Tests for ``scripts.adopt_pilot_artifacts``.

The "real-artifact" test runs against the committed pilot directory at
``<repo>/artifacts/pilot/`` and so executes on every clone. Unit tests
use a tiny synthetic artifact tree built in ``tmp_path`` to exercise
the script's branching independently of the real data.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

# Removed sys.path hack because scripts is now a proper package.
from scripts.utils import adopt_pilot_artifacts as apa


# ---------------------------------------------------------------------------
# Synthetic-artifact fixture
# ---------------------------------------------------------------------------


def _write_pilot_layout(root: Path) -> None:
    """Build a minimal pilot artifact tree under ``root``."""
    closure = root / apa.CANONICAL_CLOSURE_SUBDIR
    closure.mkdir(parents=True)

    # 5-encoder summary CSVs (one row per encoder, matching pilot names).
    with (closure / "probe_rmse_summary_5enc.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["encoder", "test_steer_rmse_mean", "test_accel_rmse_mean"])
        for enc in ("vit_s16", "dino_vits14", "clip_b32", "vjepa2_rep64", "vq_track"):
            w.writerow([enc, 0.1, 0.07])

    with (closure / "encoder_summary_with_ci_5enc.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["encoder", "steer_rmse_scene_mean", "num_scenes"])
        for enc in ("vit_s16", "dino_vits14", "clip_b32", "vjepa2_rep64", "vq_track"):
            w.writerow([enc, 0.1, 40])

    # Paired tests file — script doesn't split it, just requires existence.
    (closure / "paired_tests_5enc_bonferroni.csv").write_text(
        "encoder_a,encoder_b,t_stat\n"
    )

    per_scene_dir = (root / apa.PER_SCENE_RMSE_SUBPATH).parent
    per_scene_dir.mkdir(parents=True)
    with (root / apa.PER_SCENE_RMSE_SUBPATH).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["encoder", "scene_name", "scenario", "fold_id", "steer_rmse", "accel_rmse", "n"]
        )
        for enc in ("vit_s16", "dino_vits14", "clip_b32", "vjepa2_rep64", "vq_track"):
            for scene_idx in range(3):
                w.writerow([enc, f"scene-{scene_idx:04d}", "urban", 0, 0.1, 0.07, 15])
        # Add a vjepa2_rep1 row that should be filtered out (not part of canon).
        w.writerow(["vjepa2_rep1", "scene-0000", "urban", 0, 0.1, 0.07, 15])


def _write_retry_reports(root: Path) -> None:
    """Drop synthetic VQ + V-JEPA retry reports under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "vq_retry_report.json").write_text(
        '{"success": false, "attempts": [], "next_action": "fallback_to_dinov2"}\n'
    )
    (root / "vjepa2_retry_report.json").write_text(
        '{"success": true, "selected_repo": "facebook/vjepa2-vitl-fpc64-256"}\n'
    )


@pytest.fixture
def synthetic_pilot(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    _write_pilot_layout(root)
    return root


@pytest.fixture
def synthetic_retry_root(tmp_path: Path) -> Path:
    """Separate dir for retry reports; mirrors the on-disk pilot layout."""
    root = tmp_path / "retry_reports"
    _write_retry_reports(root)
    return root


# ---------------------------------------------------------------------------
# Unit tests (synthetic pilot)
# ---------------------------------------------------------------------------


def test_adopt_writes_per_scene_rmse_for_all_five_encoders(synthetic_pilot, tmp_path):
    out_root = tmp_path / "outputs" / "probes"
    counts = apa.adopt(
        artifact_root=synthetic_pilot,
        output_root=out_root,
        cfg_manifest_sha256="0123",
        cfg_version="1.0.0",
    )
    expected = {"vit_s16", "dino_vits14", "clip_b32", "vjepa2_rep64", "vq_track"}
    assert set(counts) == expected
    for enc in expected:
        per_scene = out_root / enc / "per_scene_rmse.csv"
        assert per_scene.exists()
        with per_scene.open() as fh:
            rows = list(csv.reader(fh))
        assert rows[0] == [
            "encoder",
            "scene_name",
            "scenario",
            "fold_id",
            "steer_rmse",
            "accel_rmse",
            "n",
        ]
        assert len(rows) == 4  # header + 3 scenes per encoder
        assert all(r[0] == enc for r in rows[1:])


def test_adopt_drops_vjepa2_rep1_rows(synthetic_pilot, tmp_path):
    """The 1-frame ablation row in the pilot file is NOT in the 5-encoder canon."""
    out_root = tmp_path / "outputs" / "probes"
    counts = apa.adopt(
        artifact_root=synthetic_pilot,
        output_root=out_root,
        cfg_manifest_sha256="0123",
        cfg_version="1.0.0",
    )
    assert "vjepa2_rep1" not in counts
    assert not (out_root / "vjepa2_rep1").exists()


def test_adopt_writes_provenance_with_vq_caveat(synthetic_pilot, tmp_path):
    out_root = tmp_path / "outputs" / "probes"
    apa.adopt(
        artifact_root=synthetic_pilot,
        output_root=out_root,
        cfg_manifest_sha256="abcd",
        cfg_version="1.0.0",
    )

    # Non-VQ: caveat is empty.
    payload = json.loads((out_root / "vit_s16" / "provenance.json").read_text())
    assert payload["fallback_caveat"] == ""
    assert payload["source"] == apa.PILOT_SOURCE_TAG
    assert payload["manifest_sha256"] == "abcd"
    assert payload["action_labels_sha256"] == apa.PILOT_ACTION_LABELS_SHA256

    # VQ: caveat non-empty, mentions FR-08 fallback.
    payload = json.loads((out_root / "vq_track" / "provenance.json").read_text())
    assert payload["fallback_caveat"]
    assert "FR-08" in payload["fallback_caveat"]


def test_adopt_writes_per_encoder_summary_copies(synthetic_pilot, tmp_path):
    out_root = tmp_path / "outputs" / "probes"
    apa.adopt(
        artifact_root=synthetic_pilot,
        output_root=out_root,
        cfg_manifest_sha256="0123",
        cfg_version="1.0.0",
    )
    for enc in ("vit_s16", "dino_vits14", "clip_b32", "vjepa2_rep64", "vq_track"):
        summary = out_root / enc / "probe_rmse_summary.csv"
        ci_summary = out_root / enc / "encoder_summary_with_ci.csv"
        assert summary.exists()
        assert ci_summary.exists()


def test_adopt_is_idempotent(synthetic_pilot, tmp_path):
    out_root = tmp_path / "outputs" / "probes"
    first = apa.adopt(
        artifact_root=synthetic_pilot,
        output_root=out_root,
        cfg_manifest_sha256="0123",
        cfg_version="1.0.0",
    )
    second = apa.adopt(
        artifact_root=synthetic_pilot,
        output_root=out_root,
        cfg_manifest_sha256="0123",
        cfg_version="1.0.0",
    )
    assert first == second


def test_adopt_raises_when_artifacts_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        apa.adopt(
            artifact_root=tmp_path / "nonexistent",
            output_root=tmp_path / "out",
            cfg_manifest_sha256="0123",
            cfg_version="1.0.0",
        )


# ---------------------------------------------------------------------------
# Retry-report copying
# ---------------------------------------------------------------------------


def test_adopt_copies_retry_reports_when_present(
    synthetic_pilot, synthetic_retry_root, tmp_path
):
    """VQ + V-JEPA retry reports land in per-encoder dirs and are referenced."""
    out_root = tmp_path / "outputs" / "probes"
    apa.adopt(
        artifact_root=synthetic_pilot,
        output_root=out_root,
        cfg_manifest_sha256="0123",
        cfg_version="1.0.0",
        retry_report_root=synthetic_retry_root,
    )

    assert (out_root / "vq_track" / "vq_retry_report.json").exists()
    assert (out_root / "vjepa2_rep64" / "vjepa2_retry_report.json").exists()

    vq_payload = json.loads((out_root / "vq_track" / "provenance.json").read_text())
    vjepa_payload = json.loads(
        (out_root / "vjepa2_rep64" / "provenance.json").read_text()
    )
    assert vq_payload["retry_report_path"] == "vq_retry_report.json"
    assert vjepa_payload["retry_report_path"] == "vjepa2_retry_report.json"

    # Non-VQ/V-JEPA encoders: retry_report_path is null.
    for enc in ("vit_s16", "dino_vits14", "clip_b32"):
        payload = json.loads((out_root / enc / "provenance.json").read_text())
        assert payload["retry_report_path"] is None


def test_adopt_handles_missing_retry_reports_gracefully(
    synthetic_pilot, tmp_path
):
    """Adoption succeeds when retry reports are absent; provenance records null."""
    out_root = tmp_path / "outputs" / "probes"
    empty_root = tmp_path / "no_retry"
    empty_root.mkdir()

    apa.adopt(
        artifact_root=synthetic_pilot,
        output_root=out_root,
        cfg_manifest_sha256="0123",
        cfg_version="1.0.0",
        retry_report_root=empty_root,
    )

    for enc in ("vq_track", "vjepa2_rep64"):
        payload = json.loads((out_root / enc / "provenance.json").read_text())
        assert payload["retry_report_path"] is None, (
            f"{enc} should record null when retry report missing"
        )
        assert not (out_root / enc / "vq_retry_report.json").exists()
        assert not (out_root / enc / "vjepa2_retry_report.json").exists()


# ---------------------------------------------------------------------------
# Integration test: real pilot directory if present.
# ---------------------------------------------------------------------------


def _real_pilot_present() -> bool:
    sources = apa.resolve_sources(apa.DEFAULT_ARTIFACT_ROOT)
    try:
        sources.validate()
    except FileNotFoundError:
        return False
    return True


@pytest.mark.skipif(
    not _real_pilot_present(),
    reason=f"Pilot artifacts not present at {apa.DEFAULT_ARTIFACT_ROOT}",
)
def test_adopt_against_real_pilot(tmp_path):
    """End-to-end adoption against the in-repo pilot artifacts."""
    out_root = tmp_path / "outputs" / "probes"
    counts = apa.adopt(
        artifact_root=apa.DEFAULT_ARTIFACT_ROOT,
        output_root=out_root,
        cfg_manifest_sha256="(test)",
        cfg_version="1.0.0",
        retry_report_root=apa.DEFAULT_RETRY_REPORT_ROOT,
    )
    expected_canon = {
        "vit_s16",
        "dino_vits14",
        "clip_b32",
        "vjepa2_rep64",
        "vq_track",
    }
    assert expected_canon <= set(counts), (
        f"Expected all 5 canonical encoders in pilot; got {set(counts)}"
    )
    # Each canon encoder should have 40 test scenes (canonical p0_test).
    for enc in expected_canon:
        assert counts[enc] == 40, f"{enc} pilot row count = {counts[enc]}, expected 40"

    # vq_track provenance carries the FR-08 caveat.
    vq_payload = json.loads(
        (out_root / "vq_track" / "provenance.json").read_text()
    )
    assert "FR-08" in vq_payload["fallback_caveat"]

    # Retry reports: in-repo pilot dir guarantees these exist, so they
    # must have been copied + referenced.
    assert (out_root / "vq_track" / "vq_retry_report.json").exists()
    assert vq_payload["retry_report_path"] == "vq_retry_report.json"

    vjepa_payload = json.loads(
        (out_root / "vjepa2_rep64" / "provenance.json").read_text()
    )
    assert (out_root / "vjepa2_rep64" / "vjepa2_retry_report.json").exists()
    assert vjepa_payload["retry_report_path"] == "vjepa2_retry_report.json"
