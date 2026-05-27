"""nuScenes dataset loader for frozen-encoder probing.

Returns dictionaries with image(s), actions, and metadata:
  - mode="single_frame": {"image": (3,224,224), "actions": (2,), ...}
  - mode="clip": {"image": (16,3,224,224), "actions": (2,), ...}

All images are [0, 1] float tensors with shared geometric transform.
Encoder-specific normalization (ImageNet vs CLIP) is deferred to wrappers.

Owner: Member 2
"""

from __future__ import annotations

# Allow running directly for testing
if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pathlib import Path
from typing import Any

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from torch.utils.data import Dataset
from torchvision import transforms

from config import load_canonical, manifest_split
from data.splits import get_can_blacklist, get_split


class NuScenesFrameDataset(Dataset):
    """PyTorch Dataset for nuScenes keyframes with CAN-bus action labels.

    Returns
    -------
    Dictionary with keys:
        image : torch.Tensor
            Single frame (3, 224, 224) if mode="single_frame"
            Clip (16, 3, 224, 224) if mode="clip"
            Values in [0, 1]
        actions : torch.Tensor
            Normalized actions, shape (2,) → [steering_norm, accel_norm]
            steering: clip(steeranglefeedback.value / 6.0, -1, 1)
            accel: clip(pose.accel[0] / 10.0, -1, 1)
        sample_token : str
            nuScenes sample token
        scene_name : str
            Scene name (e.g., "scene-0001")
        timestamp_us : int
            Sample timestamp in microseconds

    Parameters
    ----------
    split
        One of: "smoke_train", "smoke_val", "smoke_test", "p0_train", "p0_val", "p0_test", "p1p2_scenes"
        Version auto-detected from split: smoke_* → v1.0-mini, others → v1.0-trainval
    mode
        Either "single_frame" or "clip"
    clip_frames
        Number of frames for clip mode (default: 16)
    config_path
        Override canonical config path (primarily for testing)

    Examples
    --------
    >>> ds = NuScenesFrameDataset(split="p0_train", mode="single_frame")
    >>> batch = ds[0]
    >>> batch["image"].shape, batch["actions"].shape
    (torch.Size([3, 224, 224]), torch.Size([2]))

    >>> ds_clip = NuScenesFrameDataset(split="p0_train", mode="clip")
    >>> batch = ds_clip[0]
    >>> batch["image"].shape
    torch.Size([16, 3, 224, 224])
    """

    def __init__(
        self,
        split: str,
        mode: str = "single_frame",
        clip_frames: int = 16,
        config_path: Path | str | None = None,
    ) -> None:
        if mode not in ("single_frame", "clip"):
            raise ValueError(f"mode must be 'single_frame' or 'clip', got {mode}")

        self.cfg = load_canonical(config_path)
        self.split = split
        self.mode = mode
        self.clip_frames = clip_frames

        # Load nuScenes dataset
        nuscenes_root = self.cfg.root / "data"

        # Auto-detect nuScenes version from split prefix
        if split.startswith("smoke_"):
            # smoke_* splits require v1.0-mini
            if (nuscenes_root / "v1.0-mini").exists():
                version = "v1.0-mini"
            else:
                raise FileNotFoundError(
                    f"Split '{split}' requires v1.0-mini, but not found at {nuscenes_root / 'v1.0-mini'}"
                )
        else:
            # p0_*, p1p2_* splits require v1.0-trainval
            if (nuscenes_root / "v1.0-trainval").exists():
                version = "v1.0-trainval"
            elif (nuscenes_root / "v1.0-mini").exists():
                version = "v1.0-mini"
            else:
                raise FileNotFoundError(
                    f"No nuScenes dataset found at {nuscenes_root}. "
                    "Expected v1.0-trainval or v1.0-mini directory."
                )

        # Load scene list for this split (use different methods for smoke vs benchmark splits)
        if split.startswith("smoke_"):
            self.scene_names = set(get_split(split, dataroot=nuscenes_root))
        else:
            self.scene_names = set(manifest_split(self.cfg, split))

        self.nusc = NuScenes(
            version=version,
            dataroot=str(nuscenes_root),
            verbose=False,
        )

        # Load CAN bus data
        self.nusc_can = NuScenesCanBus(dataroot=str(nuscenes_root))

        # Load normalization constants from config
        self.camera = self.cfg.raw["dataset"]["camera"]
        self.image_size = tuple(self.cfg.raw["dataset"]["image_size"])
        self.max_can_delta_us = self.cfg.raw["dataset"]["can_bus"]["max_alignment_us"]

        steer_config = self.cfg.normalization("steering")
        accel_config = self.cfg.normalization("acceleration")
        self.steer_divisor = steer_config["divisor"]
        self.accel_divisor = accel_config["divisor"]
        self.clip_range = steer_config["clip_range"]

        # Build sample index
        self.samples = self._build_sample_index()

        # Transform: resize to 224x224 and convert to [0,1] tensor
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),  # Converts PIL [0,255] to tensor [0,1]
        ])

    def _build_sample_index(self) -> list[dict[str, Any]]:
        """Build filtered sample index based on split, camera, CAN alignment."""
        samples = []
        stats = {
            "total_keyframes": 0,
            "dropped_scene_not_in_split": 0,
            "dropped_blacklist": 0,
            "dropped_missing_can": 0,
            "dropped_no_camera": 0,
            "dropped_can_alignment": 0,
            "blacklisted_scene_ids": [],  # Track actual scene names that were blacklisted
        }

        # Get drop policy from config
        drop_blacklisted = self.cfg.raw["dataset"]["can_bus"]["drop_blacklisted_scenes"]

        # Get formatted CAN blacklist (strings: ["scene-0161", ...])
        can_blacklist_scenes = get_can_blacklist() if drop_blacklisted else []

        scenes_by_name = {s["name"]: s for s in self.nusc.scene}
        for scene_name in self.scene_names:
            scene = scenes_by_name.get(scene_name)
            if scene is None: continue

            scene_token = scene["token"]

            # Check CAN blacklist (only if config flag is True)
            if scene_name in can_blacklist_scenes:
                stats["dropped_blacklist"] += 1
                stats["blacklisted_scene_ids"].append(scene_name)
                continue

            # Check if CAN data exists (required for action labels)
            try:
                steer_msgs = self.nusc_can.get_messages(scene_name, "steeranglefeedback")
                pose_msgs = self.nusc_can.get_messages(scene_name, "pose")
            except KeyError:
                stats["dropped_missing_can"] += 1
                continue

            if len(steer_msgs) == 0 or len(pose_msgs) == 0:
                stats["dropped_missing_can"] += 1
                continue

            # Build timestamp lookup tables for this scene
            steer_times = np.array([msg["utime"] for msg in steer_msgs])
            steer_values = np.array([msg["value"] for msg in steer_msgs])
            pose_times = np.array([msg["utime"] for msg in pose_msgs])
            pose_accels = np.array([msg["accel"][0] for msg in pose_msgs])

            # Process keyframes in this scene
            sample_token = scene["first_sample_token"]
            while sample_token:
                stats["total_keyframes"] += 1
                sample = self.nusc.get("sample", sample_token)

                # Check if camera exists
                if self.camera not in sample["data"]:
                    stats["dropped_no_camera"] += 1
                    sample_token = sample["next"]
                    continue

                cam_token = sample["data"][self.camera]
                sample_timestamp = sample["timestamp"]

                # Find nearest CAN messages
                steer_idx = np.abs(steer_times - sample_timestamp).argmin()
                pose_idx = np.abs(pose_times - sample_timestamp).argmin()

                steer_delta = abs(steer_times[steer_idx] - sample_timestamp)
                pose_delta = abs(pose_times[pose_idx] - sample_timestamp)

                # Check alignment tolerance
                max_delta = max(steer_delta, pose_delta)
                if max_delta > self.max_can_delta_us:
                    stats["dropped_can_alignment"] += 1
                    sample_token = sample["next"]
                    continue

                # Extract and normalize actions
                steer_raw = float(steer_values[steer_idx])
                accel_raw = float(pose_accels[pose_idx])

                steer_norm = np.clip(steer_raw / self.steer_divisor, *self.clip_range)
                accel_norm = np.clip(accel_raw / self.accel_divisor, *self.clip_range)

                samples.append({
                    "sample_token": sample_token,
                    "scene_token": scene_token,
                    "scene_name": scene_name,
                    "cam_token": cam_token,
                    "timestamp_us": int(sample_timestamp),
                    "steer_norm": float(steer_norm),
                    "accel_norm": float(accel_norm),
                })

                sample_token = sample["next"]

        self.data_quality_stats = stats
        self.data_quality_stats["retained_samples"] = len(samples)

        return samples

    def _load_frame(self, cam_token: str) -> torch.Tensor:
        """Load single keyframe image.

        Returns
        -------
        torch.Tensor
            Shape (3, 224, 224), values in [0, 1]
        """
        cam_data = self.nusc.get("sample_data", cam_token)
        img_path = Path(self.nusc.dataroot) / cam_data["filename"]

        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)

    def _load_clip(self, cam_token: str, clip_len: int = 16) -> torch.Tensor:
        """Load temporal clip by traversing sample_data['prev'] links.

        Collects up to clip_len frames (12 Hz sample_data), walking backwards
        from the target keyframe. If fewer than clip_len frames exist before
        scene start, duplicate the earliest frame at the front.

        Parameters
        ----------
        cam_token
            Target keyframe camera token
        clip_len
            Number of frames in clip (default: 16)

        Returns
        -------
        torch.Tensor
            Shape (clip_len, 3, 224, 224), values in [0, 1]
            Frames ordered oldest → newest (target frame is last)
        """
        frames = []
        current_token = cam_token

        # Walk backwards collecting frames
        while current_token and len(frames) < clip_len:
            frame = self._load_frame(current_token)
            frames.append(frame)

            # Move to previous frame
            cam_data = self.nusc.get("sample_data", current_token)
            current_token = cam_data["prev"]

        # Reverse so oldest frame is first
        frames = frames[::-1]

        # Pad with earliest frame if needed
        if len(frames) < clip_len:
            earliest_frame = frames[0]
            padding_needed = clip_len - len(frames)
            frames = [earliest_frame] * padding_needed + frames

        return torch.stack(frames, dim=0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load a sample.

        Returns
        -------
        Dictionary with keys:
            image : torch.Tensor
                (3, 224, 224) if mode="single_frame"
                (clip_frames, 3, 224, 224) if mode="clip"
            actions : torch.Tensor, shape (2,)
            sample_token : str
            scene_name : str
            timestamp_us : int
        """
        record = self.samples[idx]
        cam_token = record["cam_token"]

        # Load image based on mode
        if self.mode == "single_frame":
            image = self._load_frame(cam_token)
        else:  # mode == "clip"
            image = self._load_clip(cam_token, clip_len=self.clip_frames)

        # Actions (already normalized during indexing)
        actions = torch.tensor(
            [record["steer_norm"], record["accel_norm"]],
            dtype=torch.float32,
        )

        return {
            "image": image,
            "actions": actions,
            "sample_token": record["sample_token"],
            "scene_name": record["scene_name"],
            "timestamp_us": record["timestamp_us"],
        }


if __name__ == "__main__":
    # Test dataset
    print("Testing NuScenesFrameDataset...")

    # Test single_frame mode
    print("\n=== Single Frame Mode ===")
    ds_single = NuScenesFrameDataset(split="p0_train", mode="single_frame")
    print(f"✓ Created dataset with {len(ds_single)} samples")

    if len(ds_single) > 0:
        batch = ds_single[0]
        print(f"✓ Sample 0:")
        print(f"  image: {batch['image'].shape}, range [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
        print(f"  actions: {batch['actions']}")
        print(f"  scene: {batch['scene_name']}, token: {batch['sample_token'][:8]}...")

    # Test clip mode
    print("\n=== Clip Mode ===")
    ds_clip = NuScenesFrameDataset(split="p0_train", mode="clip", clip_frames=16)
    print(f"✓ Created dataset with {len(ds_clip)} samples")

    if len(ds_clip) > 0:
        batch = ds_clip[0]
        print(f"✓ Sample 0:")
        print(f"  image: {batch['image'].shape}, range [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
        print(f"  actions: {batch['actions']}")
        print(f"  scene: {batch['scene_name']}, token: {batch['sample_token'][:8]}...")
