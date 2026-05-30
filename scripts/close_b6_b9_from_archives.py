#!/usr/bin/env python3
"""Close B6.5-B9 artifact gaps from available fullscope artifacts.

This is a fallback path for the final-report artifact closure when raw
nuScenes trainval images/metadata are not mounted locally or on Modal. It does
not retrain or re-evaluate models. It creates report-ready source artifacts
from:

* committed full-dataset probe per-scene RMSE files in ``outputs/probes``;
* archived scene descriptions from the fullscope run;
* archived attribution and figure artifacts from the GCP fullscope package.

The generated manifest records these provenance caveats explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from config import load_canonical
from evaluation.metrics import bootstrap_mean_ci, denormalize_rmse_dataframe


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE_ROOT = Path("external_artifacts/fullscope_archive")
SCENARIO_ORDER = ["highway", "urban", "intersection", "other"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_description(description: str) -> str:
    desc = description.lower()
    if "highway" in desc or "freeway" in desc:
        return "highway"
    if "intersection" in desc or "junction" in desc:
        return "intersection"
    if "urban" in desc or "city" in desc or "downtown" in desc:
        return "urban"
    return "other"


def load_scene_descriptions(archive_root: Path) -> dict[str, str]:
    path = archive_root / "fullscope-cpu-eval-latest/inputs/scene_desc_map.json"
    with path.open() as fh:
        return json.load(fh)


def generate_per_scenario_rmse(archive_root: Path, output_path: Path) -> dict[str, Any]:
    """Compute per-scenario RMSE from probe outputs and archived scene descriptions.

    Note: The P0 test split only contains intersection and other scenarios;
    highway and urban are 0-scene buckets and will be absent from the output CSV.
    This is a data limitation (nuScenes split composition), not a bug.
    """
    cfg = load_canonical()
    scene_desc = load_scene_descriptions(archive_root)
    scene_to_bucket = {scene: classify_description(desc) for scene, desc in scene_desc.items()}

    bootstrap_cfg = cfg.raw["evaluation"]["bootstrap"]
    probe_root = PROJECT_ROOT / "outputs/probes"
    encoders = sorted(path.name for path in probe_root.iterdir() if path.is_dir())

    rows = []
    for encoder in encoders:
        df = pd.read_csv(probe_root / encoder / "per_scene_rmse.csv")
        df = df.copy()
        df["scenario"] = df["scene_name"].map(scene_to_bucket).fillna("other")
        for scenario in SCENARIO_ORDER:
            scenario_df = df[df["scenario"] == scenario]
            if scenario_df.empty:
                continue
            for metric in ["steer_rmse", "accel_rmse"]:
                values = scenario_df[metric].to_numpy()
                mean, ci_lo, ci_hi = bootstrap_mean_ci(
                    values,
                    n_resamples=bootstrap_cfg["n_resamples"],
                    seed=bootstrap_cfg["seed"],
                    confidence_level=bootstrap_cfg["confidence_level"],
                )
                rows.append(
                    {
                        "encoder": encoder,
                        "scenario": scenario,
                        "metric": metric,
                        "n_scenes": int(len(values)),
                        "mean": mean,
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                    }
                )

    result = pd.DataFrame(rows)
    result = denormalize_rmse_dataframe(result, cfg)
    result = result.sort_values(["encoder", "scenario", "metric"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    return {
        "path": str(output_path.relative_to(PROJECT_ROOT)),
        "rows": int(len(result)),
        "encoders": sorted(result["encoder"].unique().tolist()),
        "scenarios": sorted(result["scenario"].unique().tolist()),
        "metrics": sorted(result["metric"].unique().tolist()),
    }


def generate_data_quality_report(archive_root: Path, output_path: Path) -> dict[str, Any]:
    cfg = load_canonical()
    handoff = archive_root / "fullscope-cpu-eval-latest/handoff_manifest.json"
    archived_manifest = archive_root / "fullscope-cpu-eval-latest/inputs/trainval_subset_manifest.json"
    scene_desc = load_scene_descriptions(archive_root)

    with handoff.open() as fh:
        handoff_payload = json.load(fh)

    actions_entry = next(
        item for item in handoff_payload["files"] if item["path"] == "inputs/camfront_keyframe_actions.csv"
    )
    manifest_entry = next(
        item for item in handoff_payload["files"] if item["path"] == "inputs/trainval_subset_manifest.json"
    )

    report = {
        "source": "archived_fullscope_artifacts",
        "source_handoff_manifest": "fullscope-cpu-eval-latest/handoff_manifest.json",
        "max_can_alignment_us": cfg.raw["dataset"]["can_bus"]["max_alignment_us"],
        "blacklisted_scenes_dropped": None,
        "blacklisted_scene_ids": [],
        "samples_dropped_for_tolerance": None,
        "sample_retention_pct": None,
        "manifest_sha256": cfg.manifest_sha256,
        "archived_manifest_sha256": manifest_entry["sha256"],
        "archived_manifest_path": "fullscope-cpu-eval-latest/inputs/trainval_subset_manifest.json",
        "camfront_keyframe_actions_sha256": actions_entry["sha256"],
        "total_keyframes": cfg.raw["dataset"]["full_dataset"]["n_samples"],
        "retained_samples": cfg.raw["dataset"]["full_dataset"]["n_samples"],
        "scene_description_count": len(scene_desc),
        "caveat": (
            "Raw nuScenes trainval was not available in the local checkout or Modal "
            "volume during closure. Fields requiring live dataset filtering "
            "(blacklist drops, tolerance drops, retention percentage) are left null "
            "rather than fabricated."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(report, fh, indent=2)

    return {"path": str(output_path.relative_to(PROJECT_ROOT)), "caveat": report["caveat"]}


def run_b8_figures() -> list[str]:
    subprocess.run(
        ["python", "figures/render_figures.py", "--data-dir", "outputs/analysis"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    out_dir = PROJECT_ROOT / "artifacts/full/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ["figure1_encoder_rmse.pdf", "figure2_scenario_heatmap.pdf"]:
        src = PROJECT_ROOT / "outputs/analysis" / name
        dst = out_dir / name
        shutil.copy2(src, dst)
        copied.append(str(dst.relative_to(PROJECT_ROOT)))
    return copied


def record_legacy_attribution(archive_root: Path) -> dict[str, Any]:
    src_dir = archive_root / "fullscope-gpu-latest/attribution"
    out_dir = PROJECT_ROOT / "artifacts/full/attribution"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_src = src_dir / "phase_attribution_report.json"
    with report_src.open() as fh:
        legacy_report = json.load(fh)

    large_files = []
    for path in sorted(src_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src_dir)
        large_files.append(
            {
                "source_archive_path": f"fullscope-gpu-latest/attribution/{rel}",
                "intended_hf_path": f"b6_b9/artifacts/full/attribution/legacy_fullscope/{rel}",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    method_report = {
        "source": "legacy_fullscope_attribution_archive",
        "source_archive_path": "fullscope-gpu-latest/attribution",
        "caveat": (
            "These are archived fullscope attribution composites, not freshly "
            "generated B7 per-frame overlays from evaluation/gradcam.py. Raw "
            "trainval images were unavailable during closure."
        ),
        "legacy_report": legacy_report,
        "methods": {
            "vit_s16": "legacy GradCAM/proxy composite",
            "clip_b32": "legacy GradCAM/proxy composite",
            "dino_vits14": "legacy attention/proxy composite",
            "vjepa2_rep64": "legacy temporal/proxy composite",
            "vq_track": "legacy VQ/proxy composite",
        },
    }
    method_path = out_dir / "figures_method_report.json"
    with method_path.open("w") as fh:
        json.dump(method_report, fh, indent=2)

    large_manifest_path = out_dir / "legacy_fullscope_manifest.json"
    with large_manifest_path.open("w") as fh:
        json.dump(
            {
                "source_archive_path": "fullscope-gpu-latest/attribution",
                "upload_status": "pending_hf_auth",
                "hf_repo": "surlac/lwm-av-embeddings",
                "files": large_files,
            },
            fh,
            indent=2,
        )

    return {
        "source_archive_path": "fullscope-gpu-latest/attribution",
        "method_report": str(method_path.relative_to(PROJECT_ROOT)),
        "large_manifest": str(large_manifest_path.relative_to(PROJECT_ROOT)),
        "file_count": len(large_files),
        "total_bytes": sum(item["bytes"] for item in large_files),
        "hf_upload_status": "pending_hf_auth",
        "caveat": method_report["caveat"],
    }


def build_legacy_attribution_grid(archive_root: Path) -> str:
    input_dir = archive_root / "fullscope-gpu-latest/attribution"
    pngs = sorted(input_dir.glob("attribution_*.png"))
    if not pngs:
        raise FileNotFoundError(f"No legacy attribution PNGs found in {input_dir}")

    fig, axes = plt.subplots(len(pngs), 1, figsize=(8, 2.1 * len(pngs)), dpi=300)
    if len(pngs) == 1:
        axes = [axes]
    for ax, path in zip(axes, pngs):
        image = plt.imread(path)
        encoder = path.stem.replace("attribution_", "")
        ax.imshow(image)
        ax.set_title(encoder, fontsize=10)
        ax.axis("off")

    fig.suptitle("Legacy Fullscope Attribution Composites", fontsize=12)
    fig.tight_layout()
    out = PROJECT_ROOT / "artifacts/full/figures/attribution_grid.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out.relative_to(PROJECT_ROOT))


def write_manifest(details: dict[str, Any]) -> None:
    root = PROJECT_ROOT / "artifacts/full"
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == "b6_b9_closure_manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    manifest = {
        "task": "B6.5-B9 artifact closure",
        "source_mode": "mixed_generated_and_archived_fallback",
        "details": details,
        "files": files,
        "large_file_policy": (
            "Attribution PDFs/PNGs are multiple MB each. Upload to HuggingFace "
            "when HF_TOKEN is available; do not commit large visual bundles if "
            "repository size is a concern."
        ),
    }
    path = root / "b6_b9_closure_manifest.json"
    with path.open("w") as fh:
        json.dump(manifest, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    args = parser.parse_args()

    details: dict[str, Any] = {}

    details["data_quality"] = generate_data_quality_report(
        args.archive_root,
        PROJECT_ROOT / "artifacts/full/data_quality_report.json",
    )
    details["per_scenario_rmse"] = generate_per_scenario_rmse(
        args.archive_root,
        PROJECT_ROOT / "outputs/analysis/per_scenario_rmse.csv",
    )
    (PROJECT_ROOT / "artifacts/full/analysis").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        PROJECT_ROOT / "outputs/analysis/per_scenario_rmse.csv",
        PROJECT_ROOT / "artifacts/full/analysis/per_scenario_rmse.csv",
    )
    details["b8_figures"] = run_b8_figures()
    details["legacy_attribution"] = record_legacy_attribution(args.archive_root)
    details["legacy_attribution_grid"] = build_legacy_attribution_grid(args.archive_root)
    write_manifest(details)

    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
