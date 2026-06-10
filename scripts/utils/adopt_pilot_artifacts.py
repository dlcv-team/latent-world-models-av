#!/usr/bin/env python
"""Adopt pre-computed pilot artifacts as the canonical probe sidecars.

The GCP pilot run already produced the canonical-closure numbers for
all five encoders following the protocol pinned in
``configs/canonical.yaml`` (5 CV folds × 3 seeds × 40 test scenes).
This script reads those files (committed under ``artifacts/pilot/``)
and writes them into the ``outputs/probes/<pilot_name>/`` layout that
downstream tasks (A12 paired t-tests, A12.5 baselines pin, B7/B8
figures, B6.5 sidecars) expect — so the rest of the team consumes the
same in-repo paths regardless of whether the numbers came from this
CLI or a future re-run via ``training/train_probe.py``.

The per-scene aggregate file (``per_scene_rmse.csv``) is the input
schema A12 needs. ``provenance.json`` records that the data came from
the pilot run, including the pilot action-labels SHA256 (which differs
from the live ``canonical.yaml`` value because PR #3 rotated the CSV
byte-for-byte while preserving the scientific content).

Source artifact layout (rooted at ``<repo>/artifacts/pilot/``)::

    artifacts/pilot/
        canonical_closure/
            probe_rmse_summary_5enc.csv
            encoder_summary_with_ci_5enc.csv
            paired_tests_5enc_bonferroni.csv
            bc_baseline_row.csv
            perturbation_rmse_5enc.csv
        per_scene/
            per_scene_rmse.csv
        retry_reports/
            vq_retry_report.json
            vjepa2_retry_report.json

The script is idempotent — running it twice produces the same files.
``outputs/probes/`` is gitignored, so nothing committed by this script
ends up tracked.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Project root resolution mirrors scripts/check_canonical_contract.py.
_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from config import load_canonical  # noqa: E402  (after sys.path tweak)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Pilot artifact root: committed in-repo under ``artifacts/pilot/`` so
# adoption works on a fresh clone. Override with ``--artifact-root``.
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "pilot"
CANONICAL_CLOSURE_SUBDIR = Path("canonical_closure")
PER_SCENE_RMSE_SUBPATH = Path("per_scene") / "per_scene_rmse.csv"

# Retry reports document the FR-08 VQ fallback decision and the V-JEPA2
# HF-transformers load path. They live under the same in-repo pilot
# root. Override with --retry-report-root.
DEFAULT_RETRY_REPORT_ROOT = REPO_ROOT / "artifacts" / "pilot" / "retry_reports"
RETRY_REPORT_FILES: dict[str, str] = {
    # pilot encoder name -> filename under DEFAULT_RETRY_REPORT_ROOT
    "vq_track": "vq_retry_report.json",
    "vjepa2_rep64": "vjepa2_retry_report.json",
}

# Pilot ran with the original action-labels CSV (M2 has since rotated it
# byte-for-byte; scientific content is equivalent — see project-memory).
PILOT_ACTION_LABELS_SHA256 = (
    "ff70d20ffb4dd152ed06661fe63dd00942d4ed35b87d39f53728a3b22c292a94"
)
PILOT_SOURCE_TAG = "pilot_gcp"

# The five pilot encoder names that A12 reads.
PILOT_ENCODER_NAMES: tuple[str, ...] = (
    "vit_s16",
    "dino_vits14",
    "clip_b32",
    "vqvae",  # NOTE: pilot per_scene_rmse uses "vq_track"; we keep both
    "vjepa2_rep64",
)
# Map pilot-side per_scene_rmse name -> canonical pilot_name. The pilot
# called the VQ row ``vq_track`` (denoting the DINOv2 fallback "track").
PILOT_PER_SCENE_RENAME = {
    "vq_track": "vq_track",
    "vit_s16": "vit_s16",
    "dino_vits14": "dino_vits14",
    "clip_b32": "clip_b32",
    "vjepa2_rep64": "vjepa2_rep64",
    "vjepa2_rep1": None,  # the 1-frame ablation; not part of the 5-encoder canon
}

# Historical pilot caveat: VQ wrapper used DINOv2 fallback in the pilot run.
# Current VQ loads real VQGAN (canonical.yaml v1.0.2), but pilot artifacts
# predate that change.
VQ_FALLBACK_CAVEAT = (
    "VQ-VAE: VQGAN checkpoint was not available during pilot run; "
    "the wrapper substituted DINOv2-S/14 embeddings per FR-08. "
    "Results labelled `vq_track` reflect this historical fallback, not "
    "current VQ-VAE behavior (which uses real VQGAN by default)."
)


@dataclass(frozen=True)
class AdoptionSources:
    closure_dir: Path  # holds the 5-encoder summary CSVs
    per_scene_rmse_path: Path  # the canonical-closure per-scene file

    def validate(self) -> None:
        for name, path in (
            ("closure_dir", self.closure_dir),
            ("per_scene_rmse_path", self.per_scene_rmse_path),
        ):
            if not path.exists():
                raise FileNotFoundError(f"missing pilot artifact: {name}={path}")
        for expected in (
            "probe_rmse_summary_5enc.csv",
            "encoder_summary_with_ci_5enc.csv",
            "paired_tests_5enc_bonferroni.csv",
        ):
            if not (self.closure_dir / expected).exists():
                raise FileNotFoundError(
                    f"missing pilot artifact: {self.closure_dir / expected}"
                )


def resolve_sources(artifact_root: Path) -> AdoptionSources:
    return AdoptionSources(
        closure_dir=(artifact_root / CANONICAL_CLOSURE_SUBDIR).resolve(),
        per_scene_rmse_path=(artifact_root / PER_SCENE_RMSE_SUBPATH).resolve(),
    )


# ---------------------------------------------------------------------------
# Per-encoder writers
# ---------------------------------------------------------------------------


def split_per_scene_rmse(
    pilot_csv_path: Path, out_root: Path
) -> dict[str, int]:
    """Split the pilot per-scene CSV into per-encoder canonical files.

    Returns a mapping ``{pilot_name: rows_written}`` so the caller can
    verify all five encoders were populated.
    """
    rows_by_encoder: dict[str, list[list[str]]] = defaultdict(list)
    with pilot_csv_path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        expected_header = [
            "encoder",
            "scene_name",
            "scenario",
            "fold_id",
            "steer_rmse",
            "accel_rmse",
            "n",
        ]
        if header != expected_header:
            raise ValueError(
                f"pilot per_scene_rmse.csv has unexpected header: {header!r}; "
                f"expected {expected_header!r}"
            )
        for row in reader:
            pilot_encoder = row[0]
            mapped = PILOT_PER_SCENE_RENAME.get(pilot_encoder, pilot_encoder)
            if mapped is None:
                continue  # e.g. vjepa2_rep1 — outside the 5-encoder canon
            row[0] = mapped
            rows_by_encoder[mapped].append(row)

    counts: dict[str, int] = {}
    for encoder, rows in rows_by_encoder.items():
        out_path = out_root / encoder / "per_scene_rmse.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(expected_header)
            writer.writerows(rows)
        counts[encoder] = len(rows)
    return counts


def split_summary_csv(
    pilot_csv_path: Path,
    out_root: Path,
    out_filename: str,
    encoder_col: str = "encoder",
) -> dict[str, int]:
    """Split a 5-row summary CSV into per-encoder copies.

    Used for ``probe_rmse_summary_5enc.csv`` and
    ``encoder_summary_with_ci_5enc.csv``. The per-encoder copy is a tiny
    2-line file (header + one row); easier for figure scripts to read
    by encoder dir than to grep the 5-row source.
    """
    counts: dict[str, int] = {}
    with pilot_csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames
        if header is None or encoder_col not in header:
            raise ValueError(
                f"pilot summary {pilot_csv_path} missing column "
                f"{encoder_col!r}; header={header!r}"
            )
        for row in reader:
            enc = row[encoder_col]
            out_path = out_root / enc / out_filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", newline="") as out_fh:
                writer = csv.DictWriter(out_fh, fieldnames=header)
                writer.writeheader()
                writer.writerow(row)
            counts[enc] = counts.get(enc, 0) + 1
    return counts


def copy_retry_reports(
    retry_report_root: Path, output_root: Path
) -> dict[str, Optional[str]]:
    """Copy VQ + V-JEPA retry reports into per-encoder dirs.

    Returns ``{pilot_encoder_name: relative_path_or_None}`` for the two
    encoders that have retry reports. ``None`` when the source file is
    missing; adoption then records ``retry_report_path: null`` in
    ``provenance.json`` and reviewers can pull the report from
    ``artifacts/pilot/retry_reports/`` directly.
    """
    out: dict[str, Optional[str]] = {}
    for encoder_name, filename in RETRY_REPORT_FILES.items():
        source = (retry_report_root / filename).resolve()
        if not source.exists():
            out[encoder_name] = None
            continue
        dest = output_root / encoder_name / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Plain text copy (json); no need for shutil.copy2's metadata.
        dest.write_text(source.read_text())
        out[encoder_name] = filename
    return out


def write_provenance(
    encoder_name: str,
    out_root: Path,
    manifest_sha256: str,
    config_version: str,
    retry_report_path: Optional[str] = None,
) -> None:
    """Write a provenance.json for a single encoder dir.

    ``retry_report_path`` is the filename of the FR-08 VQ retry report
    (``vq_track``) or the V-JEPA HF-transformers load report
    (``vjepa2_rep64``), relative to the encoder dir. Recorded as-is so
    figure scripts can open the report without knowing the
    artifact-root path. ``None`` for encoders without a retry report,
    and for vq/vjepa when the report file was absent at adoption time.
    """
    caveat = VQ_FALLBACK_CAVEAT if encoder_name == "vq_track" else ""
    payload = {
        "encoder_name": encoder_name,
        "source": PILOT_SOURCE_TAG,
        "manifest_sha256": manifest_sha256,
        "action_labels_sha256": PILOT_ACTION_LABELS_SHA256,
        "action_labels_sha256_note": (
            "Pilot CSV; differs from configs/canonical.yaml's current "
            "sha256 (PR #3 rotated the CSV byte-for-byte while preserving "
            "scientific content)."
        ),
        "config_version": config_version,
        "fallback_caveat": caveat,
        "retry_report_path": retry_report_path,
    }
    out_path = out_root / encoder_name / "provenance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adopt_pilot_artifacts",
        description=(
            "Adopt pre-computed pilot artifacts as the canonical "
            "outputs/probes/<encoder>/ sidecars."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help=(
            "Root containing 'canonical_closure/' and 'per_scene/'. "
            "Default: <repo>/artifacts/pilot/."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output directory; defaults to <repo_root>/outputs/probes/. "
            "Each pilot encoder gets a subdir named after pilot_name."
        ),
    )
    parser.add_argument(
        "--retry-report-root",
        type=Path,
        default=DEFAULT_RETRY_REPORT_ROOT,
        help=(
            "Directory containing vq_retry_report.json + "
            "vjepa2_retry_report.json. Missing files are tolerated and "
            "noted as null in provenance.json. Default: "
            "<repo>/artifacts/pilot/retry_reports/."
        ),
    )
    return parser


def adopt(
    artifact_root: Path,
    output_root: Path,
    cfg_manifest_sha256: str,
    cfg_version: str,
    retry_report_root: Optional[Path] = None,
) -> dict[str, int]:
    """Run the adoption end-to-end. Returns per-encoder row counts."""
    sources = resolve_sources(artifact_root)
    sources.validate()

    output_root.mkdir(parents=True, exist_ok=True)

    counts = split_per_scene_rmse(sources.per_scene_rmse_path, output_root)
    split_summary_csv(
        pilot_csv_path=sources.closure_dir / "probe_rmse_summary_5enc.csv",
        out_root=output_root,
        out_filename="probe_rmse_summary.csv",
    )
    split_summary_csv(
        pilot_csv_path=sources.closure_dir / "encoder_summary_with_ci_5enc.csv",
        out_root=output_root,
        out_filename="encoder_summary_with_ci.csv",
    )

    retry_map: dict[str, Optional[str]] = {}
    if retry_report_root is not None:
        retry_map = copy_retry_reports(retry_report_root.resolve(), output_root)

    for encoder_name in sorted(counts):
        write_provenance(
            encoder_name=encoder_name,
            out_root=output_root,
            manifest_sha256=cfg_manifest_sha256,
            config_version=cfg_version,
            retry_report_path=retry_map.get(encoder_name),
        )
    return counts


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    cfg = load_canonical()
    output_root: Path = (
        args.output_root if args.output_root is not None
        else (cfg.root / "outputs" / "probes")
    )

    counts = adopt(
        artifact_root=args.artifact_root.resolve(),
        output_root=output_root.resolve(),
        cfg_manifest_sha256=cfg.manifest_sha256,
        cfg_version=cfg.version,
        retry_report_root=args.retry_report_root,
    )

    print("[adopt_pilot_artifacts] populated:")
    for enc in sorted(counts):
        print(f"  {enc}: {counts[enc]} per-scene rows -> {output_root / enc}/")
    print("[adopt_pilot_artifacts] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
