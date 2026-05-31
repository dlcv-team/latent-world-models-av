#!/usr/bin/env python3
"""Visualize p0_test scenes for manual night/rain classification.

Displays scene descriptions and sample images from each of the 40 p0_test scenes
to help manually identify night and rain conditions.

Usage:
    python scripts/visualize_test_scenes.py [--output-dir OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from PIL import Image

from config import load_canonical, manifest_split


def visualize_scene(nusc: NuScenes, scene_name: str, output_dir: Path | None = None):
    """Visualize a single scene with description and sample images.

    Args:
        nusc: NuScenes instance
        scene_name: Scene name (e.g., "scene-0003")
        output_dir: Optional directory to save visualizations
    """
    # Find scene
    scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
    if scene is None:
        print(f"  ⚠️  Scene {scene_name} not found")
        return

    # Get scene metadata
    description = scene.get("description", "")
    log = nusc.get("log", scene["log_token"])
    location = log.get("location", "unknown")

    print(f"\n{'='*80}")
    print(f"Scene: {scene_name}")
    print(f"Description: {description}")
    print(f"Location: {location}")
    print(f"Samples: {scene['nbr_samples']}")

    # Get first, middle, and last sample images
    sample_token = scene["first_sample_token"]
    samples_to_show = []

    for i in range(scene["nbr_samples"]):
        if i == 0 or i == scene["nbr_samples"] // 2 or i == scene["nbr_samples"] - 1:
            sample = nusc.get("sample", sample_token)
            samples_to_show.append(sample)

        sample = nusc.get("sample", sample_token)
        sample_token = sample.get("next", "")
        if not sample_token:
            break

    # Display images
    n_imgs = len(samples_to_show)
    if n_imgs == 0:
        print("  No images to display")
        return

    fig, axes = plt.subplots(1, n_imgs, figsize=(6 * n_imgs, 6))
    if n_imgs == 1:
        axes = [axes]

    for idx, sample in enumerate(samples_to_show):
        # Get CAM_FRONT image
        if "CAM_FRONT" not in sample["data"]:
            continue

        cam_token = sample["data"]["CAM_FRONT"]
        cam_data = nusc.get("sample_data", cam_token)
        img_path = nusc.dataroot + "/" + cam_data["filename"]

        # Load and display image
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(f"Frame {idx * (scene['nbr_samples'] // max(n_imgs-1, 1))}")
        axes[idx].axis("off")

    fig.suptitle(f"{scene_name}: {description}", fontsize=12, wrap=True)
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{scene_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved to: {output_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize p0_test scenes")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Save visualizations to directory instead of displaying (default: display interactively)"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Visualize specific scene only (e.g., 'scene-0003')"
    )
    args = parser.parse_args()

    print("Loading configuration...")
    cfg = load_canonical()

    # Load NuScenes
    nuscenes_root = cfg.root / "data"
    version = cfg.raw["dataset"]["version"]
    print(f"Loading NuScenes {version} from {nuscenes_root}...")
    nusc = NuScenes(version=version, dataroot=str(nuscenes_root), verbose=False)

    # Get p0_test scenes
    if args.scene:
        test_scenes = [args.scene]
        print(f"\nVisualizing 1 scene: {args.scene}")
    else:
        test_scenes = manifest_split(cfg, "p0_test")
        print(f"\nFound {len(test_scenes)} scenes in p0_test")

    # Visualize each scene
    for i, scene_name in enumerate(test_scenes, 1):
        print(f"\n[{i}/{len(test_scenes)}] Processing {scene_name}...")
        visualize_scene(nusc, scene_name, args.output_dir)

    if args.output_dir:
        print(f"\n✓ All visualizations saved to {args.output_dir}")
        print(f"\nNext steps:")
        print(f"1. Review the images in {args.output_dir}")
        print(f"2. Identify night scenes (low light, dusk, night, tunnels)")
        print(f"3. Identify rain scenes (visible rain, wet roads)")
        print(f"4. Update configs/environment_scene_lists.yaml with scene names")
    else:
        print("\n✓ Done. Close matplotlib windows to continue.")


if __name__ == "__main__":
    main()
