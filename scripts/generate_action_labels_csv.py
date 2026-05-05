#!/usr/bin/env python3
"""Generate action labels CSV from nuScenes CAN bus data.

This script extracts steering and acceleration labels from CAN bus messages
and creates the CSV file required by NuScenesFrameDataset.

Output: data/raw/camfront_keyframe_actions.csv
Columns: sample_token, scene_token, scene_name, timestamp_us, steer_norm, accel_norm

Usage:
    python scripts/generate_action_labels_csv.py
    python scripts/generate_action_labels_csv.py --dataroot data --version v1.0-trainval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make repo modules importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from tqdm import tqdm

from config import load_canonical, sha256_file


def extract_action_labels(
    dataroot: Path,
    version: str,
    camera: str = "CAM_FRONT",
    max_can_delta_us: int = 50_000,
    steer_divisor: float = 6.0,
    accel_divisor: float = 10.0,
    clip_range: tuple[float, float] = (-1.0, 1.0),
) -> pd.DataFrame:
    """Extract action labels from CAN bus data.

    Parameters
    ----------
    dataroot
        Path to nuScenes data directory
    version
        nuScenes version (e.g., 'v1.0-trainval', 'v1.0-mini')
    camera
        Camera name (default: CAM_FRONT)
    max_can_delta_us
        Maximum CAN timestamp alignment tolerance in microseconds
    steer_divisor
        Divisor for steering normalization (radians → normalized)
    accel_divisor
        Divisor for acceleration normalization (m/s² → normalized)
    clip_range
        Clip range for normalized actions

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: sample_token, scene_token, scene_name,
        timestamp_us, steer_norm, accel_norm
    """
    print(f"Loading nuScenes {version} from {dataroot}")
    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=True)

    print(f"Loading CAN bus data from {dataroot}")
    nusc_can = NuScenesCanBus(dataroot=str(dataroot))

    records = []
    dropped_no_camera = 0
    dropped_blacklist = 0
    dropped_can_alignment = 0
    dropped_missing_can = 0

    print(f"\nExtracting action labels from {len(nusc.scene)} scenes...")

    for scene in tqdm(nusc.scene, desc="Processing scenes"):
        scene_name = scene["name"]
        scene_token = scene["token"]

        # Check if scene is blacklisted or missing CAN
        if scene_name in nusc_can.can_blacklist:
            dropped_blacklist += 1
            continue

        # Try to get CAN messages for this scene
        try:
            steer_msgs = nusc_can.get_messages(scene_name, "steeranglefeedback")
            pose_msgs = nusc_can.get_messages(scene_name, "pose")
        except KeyError:
            # Scene not in CAN bus data
            dropped_missing_can += 1
            continue

        if len(steer_msgs) == 0 or len(pose_msgs) == 0:
            dropped_missing_can += 1
            continue

        # Build timestamp lookup tables
        steer_times = [msg["utime"] for msg in steer_msgs]
        steer_values = [msg["value"] for msg in steer_msgs]
        pose_times = [msg["utime"] for msg in pose_msgs]
        pose_accels = [msg["accel"][0] for msg in pose_msgs]  # Longitudinal acceleration

        # Process all samples in this scene
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)

            # Check if camera exists
            if camera not in sample["data"]:
                dropped_no_camera += 1
                sample_token = sample["next"]
                continue

            sample_timestamp = sample["timestamp"]

            # Find nearest steering CAN message
            steer_deltas = [abs(t - sample_timestamp) for t in steer_times]
            steer_idx = steer_deltas.index(min(steer_deltas))
            steer_delta = steer_deltas[steer_idx]

            # Find nearest pose CAN message
            pose_deltas = [abs(t - sample_timestamp) for t in pose_times]
            pose_idx = pose_deltas.index(min(pose_deltas))
            pose_delta = pose_deltas[pose_idx]

            # Check alignment tolerance
            max_delta = max(steer_delta, pose_delta)
            if max_delta > max_can_delta_us:
                dropped_can_alignment += 1
                sample_token = sample["next"]
                continue

            # Extract and normalize actions
            steer_raw = steer_values[steer_idx]
            accel_raw = pose_accels[pose_idx]

            # Normalize: value / divisor, then clip
            steer_norm = max(clip_range[0], min(clip_range[1], steer_raw / steer_divisor))
            accel_norm = max(clip_range[0], min(clip_range[1], accel_raw / accel_divisor))

            # Use the CAN timestamp (average of steer and pose) for timestamp_us
            can_timestamp = int((steer_times[steer_idx] + pose_times[pose_idx]) / 2)

            records.append(
                {
                    "sample_token": sample_token,
                    "scene_token": scene_token,
                    "scene_name": scene_name,
                    "timestamp_us": can_timestamp,
                    "steer_norm": steer_norm,
                    "accel_norm": accel_norm,
                }
            )

            sample_token = sample["next"]

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"  Total samples processed: {len(records)}")
    print(f"  Dropped (no camera): {dropped_no_camera}")
    print(f"  Dropped (blacklist): {dropped_blacklist}")
    print(f"  Dropped (missing CAN): {dropped_missing_can}")
    print(f"  Dropped (CAN alignment): {dropped_can_alignment}")
    print(f"{'='*60}\n")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Generate action labels CSV from CAN bus data")
    parser.add_argument(
        "--dataroot",
        type=Path,
        default=Path("data"),
        help="Path to nuScenes data directory (default: data)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        help="nuScenes version (default: v1.0-trainval)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: data/raw/camfront_keyframe_actions.csv)",
    )
    args = parser.parse_args()

    # Load canonical config for normalization constants
    cfg = load_canonical()

    # Extract normalization constants from config
    steer_norm = cfg.normalization("steering")
    accel_norm = cfg.normalization("acceleration")
    can_config = cfg.raw["dataset"]["can_bus"]

    print("Using normalization constants from canonical config:")
    print(f"  Steering divisor: {steer_norm['divisor']}")
    print(f"  Acceleration divisor: {accel_norm['divisor']}")
    print(f"  Clip range: {steer_norm['clip_range']}")
    print(f"  Max CAN alignment: {can_config['max_alignment_us']} us\n")

    # Extract action labels
    df = extract_action_labels(
        dataroot=args.dataroot,
        version=args.version,
        camera=cfg.raw["dataset"]["camera"],
        max_can_delta_us=can_config["max_alignment_us"],
        steer_divisor=steer_norm["divisor"],
        accel_divisor=accel_norm["divisor"],
        clip_range=steer_norm["clip_range"],
    )

    # Determine output path
    if args.output is None:
        output_path = cfg.root / cfg.raw["dataset"]["action_labels"]["relative_path"]
    else:
        output_path = args.output

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    print(f"Writing CSV to {output_path}")
    df.to_csv(output_path, index=False)

    # Compute and report SHA256
    actual_sha = sha256_file(output_path)
    expected_sha = cfg.raw["dataset"]["action_labels"]["sha256"]

    print(f"\n{'='*60}")
    print(f"CSV written: {output_path}")
    print(f"  Rows: {len(df):,} (plus header)")
    print(f"  SHA256: {actual_sha}")
    print(f"  Expected SHA256: {expected_sha}")

    if actual_sha == expected_sha:
        print(f"  ✅ SHA256 matches canonical config!")
    else:
        print(f"  ⚠️  SHA256 mismatch!")
        print(f"  This is normal if:")
        print(f"    - Using different dataset version ({args.version})")
        print(f"    - Normalization constants changed")
        print(f"    - CAN alignment tolerance changed")
        print(f"\n  To update canonical config, edit configs/canonical.yaml:")
        print(f"    dataset.action_labels.sha256: \"{actual_sha}\"")
        print(f"    dataset.action_labels.expected_rows: {len(df) + 1}")

    print(f"{'='*60}\n")

    # Show sample rows
    print("Sample rows:")
    print(df.head(10))

    # Show statistics
    print(f"\nAction statistics:")
    print(df[["steer_norm", "accel_norm"]].describe())

    return 0


if __name__ == "__main__":
    sys.exit(main())
