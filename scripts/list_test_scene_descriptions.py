#!/usr/bin/env python3
"""List p0_test scene descriptions for quick review.

Prints scene names, descriptions, and metadata without loading images.

Usage:
    python scripts/list_test_scene_descriptions.py
"""

import re

from nuscenes.nuscenes import NuScenes

from config import load_canonical, manifest_split


def parse_logfile_timestamp(logfile: str) -> str:
    """Extract timestamp from logfile name.

    Args:
        logfile: e.g., "n015-2018-07-18-11-07-57+0800"

    Returns:
        Time string "HH:MM:SS" or "unknown"
    """
    match = re.search(r'-(\d{2})-(\d{2})-(\d{2})[+-]\d{4}$', logfile)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        second = int(match.group(3))
        return f"{hour:02d}:{minute:02d}:{second:02d}"
    return "unknown"


def main():
    print("Loading configuration...")
    cfg = load_canonical()

    # Load NuScenes
    nuscenes_root = cfg.root / "data"
    version = cfg.raw["dataset"]["version"]
    print(f"Loading NuScenes {version}...")
    nusc = NuScenes(version=version, dataroot=str(nuscenes_root), verbose=False)

    # Get p0_test scenes
    test_scenes = manifest_split(cfg, "p0_test")
    print(f"\nFound {len(test_scenes)} scenes in p0_test\n")
    print("="*100)

    # Print scene info
    for i, scene_name in enumerate(sorted(test_scenes), 1):
        scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
        if scene is None:
            print(f"{i:2d}. {scene_name}: NOT FOUND")
            continue

        log = nusc.get("log", scene["log_token"])
        location = log.get("location", "unknown")
        logfile = log.get("logfile", "")
        timestamp = parse_logfile_timestamp(logfile)
        description = scene.get("description", "")

        # Determine time of day
        time_of_day = ""
        if timestamp != "unknown":
            hour = int(timestamp.split(":")[0])
            if hour >= 18 or hour < 6:
                time_of_day = "🌙 NIGHT"
            else:
                time_of_day = "☀️  DAY"

        print(f"{i:2d}. {scene_name}")
        print(f"    Time:        {timestamp} {time_of_day}")
        print(f"    Location:    {location}")
        print(f"    Samples:     {scene['nbr_samples']}")
        print(f"    Description: {description}")
        print()

    print("="*100)
    print(f"\nTotal: {len(test_scenes)} scenes")
    print("\nLook for keywords:")
    print("  Night: 'night', 'dark', 'evening', 'dusk', 'tunnel'")
    print("  Rain: 'rain', 'wet', 'water', 'drizzle'")


if __name__ == "__main__":
    main()
