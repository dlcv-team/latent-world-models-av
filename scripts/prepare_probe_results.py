#!/usr/bin/env python3
"""Convert artifacts/full/probes JSON results to outputs/probes CSV format.

Reads per_scene_rmse.json files from artifacts/full/probes/<encoder>/seed_<N>/,
aggregates across 3 seeds (mean), and writes per_scene_rmse.csv files to
outputs/probes/<encoder>/ in the format expected by downstream analysis scripts.

Output CSV columns are steer_rmse and accel_rmse with normalized values
in [-1, 1] space. Source JSON files contain normalized values which are preserved
during aggregation. Downstream scripts convert to physical units at output time.

Encoder names are passed through unchanged. ENCODER_DISPLAY in config.py
handles presentation name mapping for both M1 (dino_vits14, vq_track) and P0
(dinov2_s14, vqvae) naming conventions.

Usage:
    python scripts/prepare_probe_results.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ENCODER_DISPLAY


def main():
    # Source and destination directories
    artifacts_root = Path("artifacts/full/probes")
    output_root = Path("outputs/probes")
    output_root.mkdir(parents=True, exist_ok=True)

    print("Converting probe results from JSON to CSV format...")
    print(f"Source: {artifacts_root}")
    print(f"Destination: {output_root}\n")

    # Process each encoder recognized by ENCODER_DISPLAY
    processed_count = 0
    for encoder_name in sorted(ENCODER_DISPLAY.keys()):
        encoder_dir = artifacts_root / encoder_name
        if not encoder_dir.exists():
            continue

        print(f"Processing {encoder_name}")
        processed_count += 1

        # Aggregate across seeds (keep values in normalized space)
        aggregated_data = defaultdict(lambda: {"steer_rmse": [], "accel_rmse": []})

        for seed in range(3):
            json_path = encoder_dir / f"seed_{seed}" / "per_scene_rmse.json"
            if not json_path.exists():
                print(f"  Warning: {json_path} not found")
                continue

            with open(json_path) as f:
                data = json.load(f)

            print(f"  Loaded seed {seed}: {len(data)} scenes")

            for scene_name, metrics in data.items():
                # Keep values in normalized space (no conversion)
                steer_rmse = metrics["steer_rmse"]
                accel_rmse = metrics["accel_rmse"]
                aggregated_data[scene_name]["steer_rmse"].append(steer_rmse)
                aggregated_data[scene_name]["accel_rmse"].append(accel_rmse)

        # Compute mean across seeds
        rows = []
        for scene_name, metrics in sorted(aggregated_data.items()):
            if not metrics["steer_rmse"]:  # No data for this scene
                continue

            rows.append({
                "scene_name": scene_name,
                "steer_rmse": sum(metrics["steer_rmse"]) / len(metrics["steer_rmse"]),
                "accel_rmse": sum(metrics["accel_rmse"]) / len(metrics["accel_rmse"]),
            })

        # Write CSV
        output_dir = output_root / encoder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "per_scene_rmse.csv"

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        print(f"  ✓ Wrote {len(rows)} scenes to {output_path}")
        print(f"    Mean steer RMSE (normalized): {df['steer_rmse'].mean():.4f}")
        print(f"    Mean accel RMSE (normalized): {df['accel_rmse'].mean():.4f}\n")

    if processed_count == 0:
        print(f"WARNING: No encoder directories found in {artifacts_root}")
    else:
        print(f"Done! Converted probe results for {processed_count} encoders.")
        print(f"\nNext steps:")
        print(f"  1. Run A12 analysis: python -m analysis.paired_tests")
        print(f"  2. Generate B6.5 per-scenario breakdown")
        print(f"  3. Render B8 figures: python figures/render_figures.py")


if __name__ == "__main__":
    main()
