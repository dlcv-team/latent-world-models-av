#!/usr/bin/env python3
"""Generate data_quality_report.json for B6.5.

Loads the dataset to collect quality stats, then writes the report using
evaluation.sidecars.write_data_quality_report().

Usage:
    python scripts/generate_data_quality_report.py
"""

from pathlib import Path

from config import load_canonical
from data.dataset import NuScenesFrameDataset
from evaluation.sidecars import write_data_quality_report


def main():
    """Generate data quality report."""
    import os
    import sys

    # Load configuration
    cfg = load_canonical()

    # Get dataroot from environment or default
    dataroot = os.environ.get("NUSCENES_DATAROOT")
    if not dataroot:
        print("Error: NUSCENES_DATAROOT environment variable not set")
        print("Set it with: export NUSCENES_DATAROOT=/path/to/nuscenes")
        sys.exit(1)

    # Determine split from config
    split_name = "p0_train"  # Use training split for data quality reporting

    print(f"Loading dataset (split={split_name})...")
    dataset = NuScenesFrameDataset(
        dataroot=dataroot,
        version=cfg.raw["dataset"]["version"],
        split=split_name,
        max_timestamp_delta_us=cfg.raw["dataset"]["can_bus"]["max_alignment_us"],
    )

    print(f"✓ Loaded {len(dataset)} samples")
    print(f"  Total keyframes scanned: {dataset.data_quality_stats['total_keyframes']}")
    print(f"  Blacklisted scenes dropped: {dataset.data_quality_stats['dropped_blacklist']}")
    print(f"  CAN alignment drops: {dataset.data_quality_stats['dropped_can_alignment']}")
    print(f"  Retention: {dataset.data_quality_stats['retained_samples'] / dataset.data_quality_stats['total_keyframes'] * 100:.1f}%")

    # Write report
    output_path = Path("outputs/data_quality_report.json")
    output_path.parent.mkdir(exist_ok=True)

    print(f"\nWriting report to {output_path}...")
    write_data_quality_report(dataset, output_path, cfg=cfg)

    print("✓ Data quality report generated successfully")

    # Show blacklisted scenes if any
    if dataset.data_quality_stats['blacklisted_scene_ids']:
        print(f"\nBlacklisted scene IDs ({len(dataset.data_quality_stats['blacklisted_scene_ids'])}):")
        for scene_id in dataset.data_quality_stats['blacklisted_scene_ids']:
            print(f"  - {scene_id}")


if __name__ == "__main__":
    main()
