"""Generate B6.5-B9 closure artifacts on Modal.

This runner uses the existing B6.5-B9 entrypoints, but executes them against
the ``nuscenes-full`` Modal volume where the v1.0-trainval metadata, CAM_FRONT
images, and CAN bus files already live.

Outputs are copied to ``/vol/b6_b9_artifacts`` so they can be downloaded with:

    modal volume get nuscenes-full /b6_b9_artifacts artifacts/b6_b9_modal

Usage:
    modal run scripts/run_b6_b9_closure_modal.py::generate_small_artifacts
    modal run scripts/run_b6_b9_closure_modal.py::generate_attribution_artifacts
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import modal


app = modal.App("lwm-av-b6-b9-closure")
vol = modal.Volume.from_name("nuscenes-full")

VOL_PATH = "/vol"
DATA_ROOT = Path(f"{VOL_PATH}/nuscenes")
ARTIFACT_ROOT = Path(f"{VOL_PATH}/b6_b9_artifacts")
APP_ROOT = Path("/app")

_project_root = str(Path(__file__).resolve().parent.parent)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("unzip", "libgl1-mesa-glx", "libglib2.0-0", "poppler-utils")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "timm==1.0.12",
        "open-clip-torch==2.29.0",
        "transformers>=4.51.0",
        "pytorch-lightning",
        "numpy>=1.26",
        "pandas",
        "scipy",
        "matplotlib==3.10.9",
        "Pillow>=10.0",
        "tqdm",
        "pyyaml",
        "grad-cam",
        "huggingface_hub",
    )
    # nuscenes-devkit pins old matplotlib versions that do not have py3.11
    # wheels. Install its runtime deps above and then install the package
    # itself without re-solving dependencies.
    .run_commands("python -m pip install nuscenes-devkit==1.1.11 --no-deps")
    .add_local_dir(
        _project_root,
        remote_path=str(APP_ROOT),
        ignore=["artifacts/**", ".git/**", "**/*.npz", "**/__pycache__/**"],
    )
)


def _run(cmd: list[str], *, timeout: int | None = None) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=APP_ROOT, check=True, timeout=timeout)


def _ensure_data_links() -> None:
    """Expose volume data at /app/data without overwriting repo Python files."""
    required = [
        DATA_ROOT / "v1.0-trainval",
        DATA_ROOT / "samples" / "CAM_FRONT",
        DATA_ROOT / "can_bus",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Modal volume is missing extracted nuScenes data: "
            + ", ".join(missing)
            + ". Run scripts/embed_full.py::setup_volume first."
        )

    data_dir = APP_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    links = {
        "v1.0-trainval": DATA_ROOT / "v1.0-trainval",
        "samples": DATA_ROOT / "samples",
        "maps": DATA_ROOT / "maps",
        "can_bus": DATA_ROOT / "can_bus",
    }
    for name, target in links.items():
        link = data_dir / name
        if link.exists() or link.is_symlink():
            if link.is_symlink() and Path(os.readlink(link)) == target:
                continue
            if link.is_dir() and not link.is_symlink():
                # Do not remove repo source directory; only handle data links.
                if name not in {"v1.0-trainval", "samples", "maps", "can_bus"}:
                    continue
                shutil.rmtree(link)
            else:
                link.unlink()
        link.symlink_to(target)
        print(f"[data] {link} -> {target}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[copy] {src} -> {dst}")


def _collect_manifest(extra: dict[str, object] | None = None) -> dict[str, object]:
    files = []
    if ARTIFACT_ROOT.exists():
        for path in sorted(p for p in ARTIFACT_ROOT.rglob("*") if p.is_file()):
            rel = path.relative_to(ARTIFACT_ROOT).as_posix()
            files.append(
                {
                    "path": rel,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )

    manifest: dict[str, object] = {
        "task": "B6.5-B9 artifact closure",
        "generated_at_unix": int(time.time()),
        "modal_app": "lwm-av-b6-b9-closure",
        "modal_volume": "nuscenes-full",
        "artifact_root": str(ARTIFACT_ROOT),
        "files": files,
    }
    if extra:
        manifest.update(extra)
    return manifest


def _write_manifest(extra: dict[str, object] | None = None) -> None:
    manifest = _collect_manifest(extra)
    path = ARTIFACT_ROOT / "artifacts" / "full" / "b6_b9_closure_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[manifest] wrote {path}")


@app.function(
    volumes={VOL_PATH: vol},
    image=base_image,
    timeout=3600,
)
def generate_small_artifacts() -> dict[str, object]:
    """Generate B6.5 data sidecars and B8 RMSE/heatmap PDFs."""
    _ensure_data_links()
    os.environ["PYTHONPATH"] = str(APP_ROOT)

    _run(["python", "scripts/generate_data_quality_report.py"])
    _run(["python", "scripts/generate_per_scenario_from_probes.py"])
    _run(["python", "figures/render_figures.py", "--data-dir", "outputs/analysis"])

    out = ARTIFACT_ROOT / "artifacts" / "full"
    _copy_file(APP_ROOT / "outputs/data_quality_report.json", out / "data_quality_report.json")
    _copy_file(
        APP_ROOT / "outputs/analysis/per_scenario_rmse.csv",
        out / "analysis/per_scenario_rmse.csv",
    )
    _copy_file(
        APP_ROOT / "outputs/analysis/figure1_encoder_rmse.pdf",
        out / "figures/figure1_encoder_rmse.pdf",
    )
    _copy_file(
        APP_ROOT / "outputs/analysis/figure2_scenario_heatmap.pdf",
        out / "figures/figure2_scenario_heatmap.pdf",
    )

    _write_manifest(
        {
            "completed": ["B6.5", "B8"],
            "commands": [
                "python scripts/generate_data_quality_report.py",
                "python scripts/generate_per_scenario_from_probes.py",
                "python figures/render_figures.py --data-dir outputs/analysis",
            ],
        }
    )
    vol.commit()
    return _collect_manifest()


@app.function(
    volumes={VOL_PATH: vol},
    image=base_image,
    gpu="A10G",
    timeout=6 * 3600,
)
def generate_attribution_artifacts() -> dict[str, object]:
    """Generate B7 attribution overlays/PDFs and B9 grid PDF."""
    _ensure_data_links()
    os.environ["PYTHONPATH"] = str(APP_ROOT)
    os.environ.setdefault("HF_HOME", f"{VOL_PATH}/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{VOL_PATH}/hf_cache")

    _run(
        [
            "python",
            "scripts/generate_attribution.py",
            "--split",
            "p0_test",
            "--device",
            "cuda",
            "--output-dir",
            "outputs/attribution",
            "--n-per-scenario",
            "5",
            "--seed",
            "42",
        ]
    )

    grid_out = ARTIFACT_ROOT / "artifacts/full/figures/attribution_grid.pdf"
    grid_out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "python",
            "scripts/generate_attribution_grid.py",
            "--input-dir",
            "outputs/attribution",
            "--output",
            str(grid_out),
            "--dpi",
            "300",
        ]
    )

    out = ARTIFACT_ROOT / "artifacts" / "full"
    attribution_out = out / "attribution"
    if attribution_out.exists():
        shutil.rmtree(attribution_out)
    shutil.copytree(APP_ROOT / "outputs/attribution", attribution_out)
    print(f"[copytree] outputs/attribution -> {attribution_out}")

    _write_manifest(
        {
            "completed": ["B7", "B9"],
            "commands": [
                "python scripts/generate_attribution.py --split p0_test --device cuda --output-dir outputs/attribution --n-per-scenario 5 --seed 42",
                "python scripts/generate_attribution_grid.py --input-dir outputs/attribution --output artifacts/full/figures/attribution_grid.pdf --dpi 300",
            ],
        }
    )
    vol.commit()
    return _collect_manifest()


@app.local_entrypoint()
def main(run_attribution: bool = False):
    print("[modal] Generating B6.5/B8 small artifacts...")
    small = generate_small_artifacts.remote()
    print(json.dumps(small, indent=2))
    if run_attribution:
        print("[modal] Generating B7/B9 attribution artifacts...")
        attr = generate_attribution_artifacts.remote()
        print(json.dumps(attr, indent=2))
