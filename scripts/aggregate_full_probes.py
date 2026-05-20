"""Aggregate full-dataset per-scene RMSE by running probe inference on embeddings.

Loads trained probe checkpoints from ``artifacts/full/probes/<encoder>/seed_*/``
and runs inference on the test-split embeddings from ``artifacts/full/embeddings/``.
Computes per-scene steer/accel RMSE (averaged across seeds) and writes the CSV
layout that ``analysis/paired_tests.py`` expects.

Also updates ``baselines.json`` with separate steer/accel RMSE fields
(RSK-08: EDD Section 9.1 requires separate reporting).

Usage
-----
    python scripts/aggregate_full_probes.py
    python scripts/aggregate_full_probes.py --probe-root artifacts/full/probes
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config import load_canonical
from models.probe import ActionProbe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROBE_ROOT = PROJECT_ROOT / "artifacts" / "full" / "probes"
DEFAULT_EMBED_ROOT = PROJECT_ROOT / "artifacts" / "full" / "embeddings"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "probes"
DEFAULT_BASELINES_PATH = PROJECT_ROOT / "baselines.json"

SEEDS = [0, 1, 2]

# Native embedding dimensions per encoder (must match training config).
NATIVE_DIMS: dict[str, int] = {
    "vit_s16": 384,
    "dino_vits14": 384,
    "clip_b32": 512,
    "vq_track": 256,
    "vjepa2_rep64": 1024,
    "vjepa2_rep1": 1024,
}

PER_SCENE_HEADER = [
    "encoder",
    "scene_name",
    "scenario",
    "fold_id",
    "steer_rmse",
    "accel_rmse",
    "n",
]

SUMMARY_HEADER = [
    "encoder",
    "cv_folds",
    "seeds",
    "test_steer_rmse_mean",
    "test_steer_rmse_std",
    "test_accel_rmse_mean",
    "test_accel_rmse_std",
    "val_steer_rmse_mean",
    "val_accel_rmse_mean",
    "num_scene_test_observations",
]


def load_test_data(
    embed_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load test-split embeddings, actions, and scene names.

    Returns
    -------
    embeddings, steer_norms, accel_norms, scene_names
        Arrays filtered to test split only.
    """
    npz = np.load(embed_path, allow_pickle=True)
    test_mask = npz["splits"] == "test"
    return (
        npz["embeddings"][test_mask],
        npz["steer_norms"][test_mask],
        npz["accel_norms"][test_mask],
        npz["scene_names"][test_mask],
    )


@torch.no_grad()
def compute_per_scene_rmse_for_seed(
    probe: ActionProbe,
    adapter: nn.Module,
    embeddings: np.ndarray,
    steer_norms: np.ndarray,
    accel_norms: np.ndarray,
    scene_names: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Run probe inference and compute per-scene steer/accel RMSE.

    Returns
    -------
    dict
        ``{scene_name: {"steer_rmse": float, "accel_rmse": float, "n": int}}``
    """
    probe.eval()
    adapter.eval()

    emb_t = torch.from_numpy(embeddings).float()
    projected = adapter(emb_t)
    preds = probe(projected).numpy()

    steer_pred = preds[:, 0]
    accel_pred = preds[:, 1]
    steer_err_sq = (steer_pred - steer_norms) ** 2
    accel_err_sq = (accel_pred - accel_norms) ** 2

    # Group by scene
    scene_errors: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"steer_sq": [], "accel_sq": []}
    )
    for i, scene in enumerate(scene_names):
        scene_errors[scene]["steer_sq"].append(float(steer_err_sq[i]))
        scene_errors[scene]["accel_sq"].append(float(accel_err_sq[i]))

    result: dict[str, dict[str, float]] = {}
    for scene, errs in scene_errors.items():
        result[scene] = {
            "steer_rmse": float(np.sqrt(np.mean(errs["steer_sq"]))),
            "accel_rmse": float(np.sqrt(np.mean(errs["accel_sq"]))),
            "n": len(errs["steer_sq"]),
        }

    return result


def load_probe_and_adapter(
    ckpt_path: Path, target_dim: int
) -> tuple[ActionProbe, nn.Module]:
    """Load probe + adapter from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    probe = ActionProbe.from_canonical()
    state_dict = ckpt["probe_state_dict"]
    # Handle checkpoints saved with bare Sequential keys ("0.weight")
    # vs wrapped keys ("net.0.weight")
    if any(k.startswith("net.") for k in state_dict):
        probe.load_state_dict(state_dict)
    else:
        probe.net.load_state_dict(state_dict)

    if ckpt.get("adapter_state_dict") is not None:
        native_dim = list(ckpt["adapter_state_dict"].values())[0].shape[1]
        adapter: nn.Module = nn.Linear(native_dim, target_dim, bias=False)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
    else:
        adapter = nn.Identity()

    return probe, adapter


def process_encoder(
    encoder_name: str,
    probe_dir: Path,
    embed_path: Path,
    output_dir: Path,
    target_dim: int,
    seeds: list[int],
) -> dict[str, float]:
    """Process one encoder: run inference for all seeds, average, write CSVs.

    Returns summary stats for baselines.json update.
    """
    embeddings, steer_norms, accel_norms, scene_names = load_test_data(embed_path)

    # Collect per-scene RMSE for each seed
    per_seed_scenes: list[dict[str, dict[str, float]]] = []
    for seed in seeds:
        ckpt_path = probe_dir / f"seed_{seed}" / "checkpoint.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing: {ckpt_path}")
        probe, adapter = load_probe_and_adapter(ckpt_path, target_dim)
        scene_rmse = compute_per_scene_rmse_for_seed(
            probe, adapter, embeddings, steer_norms, accel_norms, scene_names
        )
        per_seed_scenes.append(scene_rmse)

    # Average across seeds
    all_scenes = sorted(per_seed_scenes[0].keys())
    rows: list[list[str]] = []
    steer_means: list[float] = []
    accel_means: list[float] = []

    for scene in all_scenes:
        steer_avg = float(np.mean([s[scene]["steer_rmse"] for s in per_seed_scenes]))
        accel_avg = float(np.mean([s[scene]["accel_rmse"] for s in per_seed_scenes]))
        n = per_seed_scenes[0][scene]["n"]
        steer_means.append(steer_avg)
        accel_means.append(accel_avg)

        rows.append([
            encoder_name,
            scene,
            "",       # scenario
            "0",      # fold_id
            f"{steer_avg}",
            f"{accel_avg}",
            str(n),
        ])

    # Write per_scene_rmse.csv
    enc_dir = output_dir / encoder_name
    enc_dir.mkdir(parents=True, exist_ok=True)
    csv_path = enc_dir / "per_scene_rmse.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(PER_SCENE_HEADER)
        writer.writerows(rows)

    steer_arr = np.array(steer_means)
    accel_arr = np.array(accel_means)

    summary = {
        "test_steer_rmse_mean": float(steer_arr.mean()),
        "test_steer_rmse_std": float(steer_arr.std(ddof=1)),
        "test_accel_rmse_mean": float(accel_arr.mean()),
        "test_accel_rmse_std": float(accel_arr.std(ddof=1)),
        "num_scenes": len(rows),
    }

    # Write probe_rmse_summary.csv
    summary_path = enc_dir / "probe_rmse_summary.csv"
    with summary_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SUMMARY_HEADER)
        writer.writerow([
            encoder_name,
            1,
            len(seeds),
            f"{summary['test_steer_rmse_mean']}",
            f"{summary['test_steer_rmse_std']}",
            f"{summary['test_accel_rmse_mean']}",
            f"{summary['test_accel_rmse_std']}",
            "",
            "",
            summary["num_scenes"],
        ])

    print(
        f"  {encoder_name}: {len(rows)} scenes, "
        f"steer={summary['test_steer_rmse_mean']:.6f}+/-{summary['test_steer_rmse_std']:.6f}, "
        f"accel={summary['test_accel_rmse_mean']:.6f}+/-{summary['test_accel_rmse_std']:.6f}"
    )

    return summary


def update_baselines_json(
    baselines_path: Path,
    encoder_summaries: dict[str, dict[str, float]],
) -> None:
    """Add separate steer/accel RMSE fields to baselines.json (RSK-08)."""
    with baselines_path.open() as fh:
        baselines = json.load(fh)

    for encoder_name, summary in encoder_summaries.items():
        if encoder_name not in baselines["encoders"]:
            print(f"  WARNING: {encoder_name} not in baselines.json, skipping")
            continue

        enc = baselines["encoders"][encoder_name]
        enc["test_steer_rmse_mean"] = summary["test_steer_rmse_mean"]
        enc["test_steer_rmse_std"] = summary["test_steer_rmse_std"]
        enc["test_accel_rmse_mean"] = summary["test_accel_rmse_mean"]
        enc["test_accel_rmse_std"] = summary["test_accel_rmse_std"]

    with baselines_path.open("w") as fh:
        json.dump(baselines, fh, indent=2)
        fh.write("\n")

    print(f"  Updated {baselines_path} with separate steer/accel RMSE fields")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate full-dataset probe results: compute per-scene RMSE from checkpoints."
    )
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=DEFAULT_PROBE_ROOT,
        help="Root containing <encoder>/seed_*/checkpoint.pt.",
    )
    parser.add_argument(
        "--embed-root",
        type=Path,
        default=DEFAULT_EMBED_ROOT,
        help="Root containing <encoder>.npz embedding files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for per-encoder CSV files.",
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=DEFAULT_BASELINES_PATH,
        help="Path to baselines.json to update with steer/accel RMSE.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg = load_canonical()
    target_dim = cfg.target_embedding_dim

    if not args.probe_root.exists():
        print(f"ERROR: probe root not found: {args.probe_root}")
        return 1

    # Discover encoders
    encoder_dirs = sorted(
        d for d in args.probe_root.iterdir()
        if d.is_dir() and (d / "seed_0" / "checkpoint.pt").exists()
    )
    if not encoder_dirs:
        print(f"ERROR: no encoder directories with checkpoints found in {args.probe_root}")
        return 1

    print(f"[aggregate] Found {len(encoder_dirs)} encoders: {[d.name for d in encoder_dirs]}")
    print(f"[aggregate] Output root: {args.output_root}")

    encoder_summaries: dict[str, dict[str, float]] = {}
    for enc_dir in encoder_dirs:
        encoder_name = enc_dir.name
        embed_path = args.embed_root / f"{encoder_name}.npz"
        if not embed_path.exists():
            print(f"  WARNING: embeddings not found at {embed_path}, skipping {encoder_name}")
            continue

        summary = process_encoder(
            encoder_name, enc_dir, embed_path, args.output_root, target_dim, SEEDS
        )
        encoder_summaries[encoder_name] = summary

    # Update baselines.json
    if args.baselines.exists():
        update_baselines_json(args.baselines, encoder_summaries)
    else:
        print(f"  WARNING: {args.baselines} not found, skipping baselines update")

    print(f"[aggregate] Done. Wrote CSVs for {len(encoder_summaries)} encoders.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
