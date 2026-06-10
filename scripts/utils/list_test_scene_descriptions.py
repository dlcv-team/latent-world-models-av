#!/usr/bin/env python3
"""List p0_test scene descriptions for quick review.

Prints scene names, descriptions, and metadata without loading images.

Usage:
    python scripts/list_test_scene_descriptions.py
"""

from nuscenes.nuscenes import NuScenes

from config import load_canonical, manifest_split
from evaluation.metrics import parse_logfile_timestamp, is_night_from_timestamp


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
        if is_night_from_timestamp(timestamp):
            time_of_day = "🌙 NIGHT"
        elif timestamp != "unknown":
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
