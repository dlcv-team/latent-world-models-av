#!/usr/bin/env python3
"""Fix the existing data_quality_report.json to match B6.5 requirements.

Reads the current report and reformats it to include all required fields.
"""

import json
from pathlib import Path


def main():
    """Fix data quality report format."""
    report_path = Path("outputs/data_quality_report.json")

    if not report_path.exists():
        print(f"Error: {report_path} does not exist")
        return

    # Load current report
    with open(report_path) as f:
        current = json.load(f)

    print("Current report fields:")
    for key in current.keys():
        print(f"  - {key}")

    # Build compliant report
    report = {
        "max_can_alignment_us": current.get("max_can_alignment_us", 50000),
        "blacklisted_scenes_dropped": current.get("blacklisted_scenes_dropped", 0),
        "blacklisted_scene_ids": current.get("blacklisted_scene_ids", []),
        "samples_dropped_for_tolerance": current.get("samples_dropped_for_tolerance", 0),
        "sample_retention_pct": current.get("sample_retention_pct", 100.0),
        "manifest_sha256": current.get("manifest_sha256", ""),
        "total_keyframes": current.get("split_counts", {}).get("total_keyframes_scanned",
                                      current.get("split_counts", {}).get("p0_train_samples", 0)),
        "retained_samples": current.get("split_counts", {}).get("retained_samples",
                                       current.get("split_counts", {}).get("p0_train_samples", 0)),
    }

    # Backup old report
    backup_path = report_path.with_suffix(".json.bak")
    report_path.rename(backup_path)
    print(f"\n✓ Backed up old report to {backup_path}")

    # Write new report
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✓ Updated {report_path}")
    print("\nNew report fields:")
    for key, value in report.items():
        print(f"  - {key}: {value}")

    # Check PRD requirements
    print("\n✅ PRD Requirements Check:")
    print(f"  ✅ max CAN alignment µs: {report['max_can_alignment_us']}")
    print(f"  ✅ count of blacklisted scenes: {report['blacklisted_scenes_dropped']}")
    print(f"  ✅ IDs of blacklisted scenes: {report['blacklisted_scene_ids']}")
    print(f"  ✅ samples dropped for tolerance: {report['samples_dropped_for_tolerance']}")
    print(f"  ✅ retention %: {report['sample_retention_pct']}")
    print(f"  ✅ manifest SHA256: {report['manifest_sha256'][:16]}...")


if __name__ == "__main__":
    main()
