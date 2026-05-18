"""NuScenes dataset for action prediction from front camera images."""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus


class NuScenesFrameDataset(Dataset):
    """
    NuScenes dataset that loads CAM_FRONT keyframes and extracts action labels from CAN bus.

    Supports two modes:
    - 'frame': Returns single keyframes (default, backward compatible)
    - 'clip': Returns 16-frame temporal sequences for video encoders

    Returns (frame mode):
        frame: PIL Image (224x224) with geometry transforms applied
        action: numpy array [steering, accel] normalized to [-1, 1]
        scene_token: str
        timestamp_us: int

    Returns (clip mode):
        clip: torch.Tensor (16, 3, 224, 224) with geometry transforms applied
        action: numpy array [steering, accel] normalized to [-1, 1]
        scene_token: str
        timestamp_us: int (timestamp of target/last frame)
    """

    def __init__(self, dataroot: str, version: str = 'v1.0-mini', split: str = None, max_timestamp_delta_us: int = 50_000, mode: str = 'frame'):
        """
        Args:
            dataroot: Path to nuScenes dataset
            version: Dataset version (e.g., 'v1.0-mini', 'v1.0-trainval')
            split: Optional split name (e.g., 'smoke_train', 'train', 'test'). If None, uses all scenes.
            max_timestamp_delta_us: Maximum allowed time delta between sample and CAN message (microseconds)
            mode: Dataset mode - 'frame' for single frames or 'clip' for 16-frame sequences
        """
        if mode not in ['frame', 'clip']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'frame' or 'clip'.")

        self.dataroot = Path(dataroot)
        self.max_timestamp_delta_us = max_timestamp_delta_us
        self.mode = mode
        self.clip_length = 16

        # Initialize nuScenes
        self.nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
        self.nusc_can = NuScenesCanBus(dataroot=str(dataroot))

        # Get split definitions if split is specified
        if split:
            from data.splits import create_action_splits
            splits = create_action_splits(version, self.nusc, self.nusc_can)
            if split not in splits:
                available_splits = ', '.join(splits.keys())
                raise ValueError(f"Split '{split}' not found. Available splits: {available_splits}")
            self.split_info = splits[split]
            self.allowed_scenes = set(self.split_info['scenes'])
        else:
            self.split_info = None
            self.allowed_scenes = None

        # Shared geometry transform: resize and center crop to 224x224
        self.geometry_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])

        # Build valid sample index
        self.samples = self._build_sample_index()

    def _build_sample_index(self):
        """Build index of valid samples with available CAN data."""
        valid_samples = []

        # Initialize data quality tracking
        total_keyframes = 0
        dropped_blacklist = 0
        dropped_can_alignment = 0
        blacklisted_scene_ids = []
        seen_blacklisted_scenes = set()

        for sample in self.nusc.sample:
            # Only use keyframes
            if sample['prev'] == '' or sample['next'] == '':
                continue

            # Get scene information
            scene = self.nusc.get('scene', sample['scene_token'])
            scene_name = scene['name']

            # Filter by split if specified
            if self.allowed_scenes and scene_name not in self.allowed_scenes:
                continue

            total_keyframes += 1

            # Skip blacklisted scenes
            if scene_name in self.nusc_can.can_blacklist:
                dropped_blacklist += 1
                if scene_name not in seen_blacklisted_scenes:
                    blacklisted_scene_ids.append(scene_name)
                    seen_blacklisted_scenes.add(scene_name)
                continue

            # Check if CAN data exists for this scene
            try:
                # Try to access CAN messages to verify availability
                self.nusc_can.get_messages(scene_name, 'steeranglefeedback')
                self.nusc_can.get_messages(scene_name, 'pose')
            except KeyError:
                # Scene has missing CAN data
                continue

            # Get CAM_FRONT sample_data token
            cam_front_token = sample['data']['CAM_FRONT']
            sample_data = self.nusc.get('sample_data', cam_front_token)

            valid_samples.append({
                'sample_token': sample['token'],
                'sample_data_token': cam_front_token,
                'scene_token': sample['scene_token'],
                'scene_name': scene_name,
                'timestamp': sample_data['timestamp'],
            })

        # Store data quality stats for B6.5 reporting
        self.data_quality_stats = {
            'total_keyframes': total_keyframes,
            'dropped_blacklist': dropped_blacklist,
            'dropped_can_alignment': dropped_can_alignment,
            'retained_samples': len(valid_samples),
            'blacklisted_scene_ids': sorted(blacklisted_scene_ids),
        }

        return valid_samples

    def _collect_clip(self, target_sample_data_token, scene_token):
        """
        Collect 16-frame clip ending at target frame.

        Traverses backward through sample_data chain to collect 15 previous frames
        plus the target frame. If fewer than 16 frames exist before scene boundary,
        duplicates the earliest frame at the front.

        Args:
            target_sample_data_token: Token of target keyframe (last in clip)
            scene_token: Scene token to detect boundary crossings

        Returns:
            List of 16 dicts with keys: {'token', 'timestamp', 'filename'}
            Ordered chronologically (oldest first, target last)
        """
        frames = []
        current_token = target_sample_data_token

        while current_token and len(frames) < self.clip_length:
            sample_data = self.nusc.get('sample_data', current_token)

            # Check scene boundary
            sd_sample = self.nusc.get('sample', sample_data['sample_token'])
            if sd_sample['scene_token'] != scene_token:
                break

            frames.append({
                'token': sample_data['token'],
                'timestamp': sample_data['timestamp'],
                'filename': sample_data['filename']
            })

            current_token = sample_data['prev']

        # Reverse to chronological order (oldest first)
        frames.reverse()

        # Pad if necessary by duplicating earliest frame
        while len(frames) < self.clip_length:
            frames.insert(0, frames[0].copy())

        return frames

    def _get_nearest_can_value(self, messages, timestamp_us, key):
        """
        Get value from CAN message nearest to the given timestamp.

        Args:
            messages: List of CAN messages with 'utime' and data fields
            timestamp_us: Target timestamp in microseconds
            key: Key to extract from message data

        Returns:
            value: Extracted value or None if no message within tolerance
            delta_us: Time difference in microseconds
        """
        if not messages:
            return None, None

        # Find nearest message by timestamp
        min_delta = float('inf')
        nearest_msg = None

        for msg in messages:
            delta = abs(msg['utime'] - timestamp_us)
            if delta < min_delta:
                min_delta = delta
                nearest_msg = msg

        if min_delta > self.max_timestamp_delta_us:
            return None, None

        # Extract value using key
        if key == 'value':
            # For steeranglefeedback
            value = nearest_msg[key]
        elif key == 'accel':
            # For pose - extract accel[0] (longitudinal)
            value = nearest_msg[key][0]
        else:
            raise ValueError(f"Unknown key: {key}")

        return value, min_delta

    def _get_action_label(self, scene_name, timestamp_us):
        """
        Extract action labels from CAN bus data.

        Args:
            scene_name: Scene identifier
            timestamp_us: Sample timestamp in microseconds

        Returns:
            action: numpy array [steering, accel] normalized to [-1, 1]
                    or None if CAN data not available within tolerance
        """
        # Get steering angle from steeranglefeedback
        steer_msgs = self.nusc_can.get_messages(scene_name, 'steeranglefeedback')
        steer_value, _ = self._get_nearest_can_value(steer_msgs, timestamp_us, 'value')

        if steer_value is None:
            return None

        # Get longitudinal acceleration from pose
        pose_msgs = self.nusc_can.get_messages(scene_name, 'pose')
        accel_value, _ = self._get_nearest_can_value(pose_msgs, timestamp_us, 'accel')

        if accel_value is None:
            return None

        # Normalize steering: clip(value / 6.0, -1, 1)
        steering_normalized = np.clip(steer_value / 6.0, -1.0, 1.0)

        # Normalize acceleration: clip(accel / 10.0, -1, 1)
        accel_normalized = np.clip(accel_value / 10.0, -1.0, 1.0)

        action = np.array([steering_normalized, accel_normalized], dtype=np.float32)
        return action

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get dataset item.

        Returns (frame mode):
            frame: PIL Image (224x224) with geometry transforms applied
            action: numpy array [steering, accel] normalized to [-1, 1]
            scene_token: str
            timestamp_us: int

        Returns (clip mode):
            clip: torch.Tensor (16, 3, 224, 224) with geometry transforms applied
            action: numpy array [steering, accel] normalized to [-1, 1]
            scene_token: str
            timestamp_us: int (timestamp of target/last frame)
        """
        sample_info = self.samples[idx]

        if self.mode == 'frame':
            # Load CAM_FRONT image
            sample_data = self.nusc.get('sample_data', sample_info['sample_data_token'])
            img_path = self.dataroot / sample_data['filename']
            frame = Image.open(img_path).convert('RGB')

            # Apply shared geometry transform (resize + crop to 224x224)
            frame = self.geometry_transform(frame)

            # Get action label from CAN bus
            action = self._get_action_label(sample_info['scene_name'], sample_info['timestamp'])

            if action is None:
                action = np.array([0.0, 0.0], dtype=np.float32)

            return frame, action, sample_info['scene_token'], sample_info['timestamp']

        elif self.mode == 'clip':
            # Collect 16-frame clip
            frames = self._collect_clip(
                sample_info['sample_data_token'],
                sample_info['scene_token']
            )

            # Load and transform all frames
            clip_frames = []
            for frame_info in frames:
                img_path = self.dataroot / frame_info['filename']
                img = Image.open(img_path).convert('RGB')
                img = self.geometry_transform(img)
                clip_frames.append(img)

            # Convert to tensor: (16, 3, 224, 224)
            to_tensor = transforms.ToTensor()
            clip_tensor = torch.stack([to_tensor(img) for img in clip_frames])

            # Get action for target frame (last frame)
            action = self._get_action_label(sample_info['scene_name'], sample_info['timestamp'])

            if action is None:
                action = np.array([0.0, 0.0], dtype=np.float32)

            return clip_tensor, action, sample_info['scene_token'], sample_info['timestamp']


# Encoder-specific normalization wrappers
class ClipNormalizedDataset(Dataset):
    """Wrapper that applies CLIP normalization to frames."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        frame, action, scene_token, timestamp_us = self.base_dataset[idx]
        # Convert to tensor and normalize
        frame = self.to_tensor(frame)
        frame = self.normalize(frame)
        return frame, action, scene_token, timestamp_us


class ImageNetNormalizedDataset(Dataset):
    """Wrapper that applies ImageNet normalization to frames."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        frame, action, scene_token, timestamp_us = self.base_dataset[idx]
        # Convert to tensor and normalize
        frame = self.to_tensor(frame)
        frame = self.normalize(frame)
        return frame, action, scene_token, timestamp_us


class VJEPANormalizedDataset(Dataset):
    """
    Wrapper for V-JEPA video encoder.

    Applies ImageNet normalization to clip frames and optionally
    passes through a V-JEPA encoder to produce embeddings.

    For clip mode:
        - Input: (16, 3, 224, 224) tensor
        - Normalizes each frame with ImageNet stats
        - If encoder provided: returns (384,) embeddings
        - If encoder=None: returns normalized clip

    For frame mode:
        - Behaves like ImageNetNormalizedDataset
    """

    def __init__(self, base_dataset, encoder=None):
        """
        Args:
            base_dataset: NuScenesFrameDataset instance
            encoder: Optional V-JEPA encoder model
                     Expected signature: encoder(clip) -> embeddings
                     Input shape: (B, 16, 3, 224, 224)
                     Output shape: (B, 384)
        """
        if not hasattr(base_dataset, 'mode') or base_dataset.mode not in ['frame', 'clip']:
            raise ValueError("Base dataset must have mode='frame' or mode='clip'")

        self.base_dataset = base_dataset
        self.encoder = encoder
        self.mode = base_dataset.mode

        # ImageNet normalization (V-JEPA typically uses ImageNet stats)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data, action, scene_token, timestamp = self.base_dataset[idx]

        if self.mode == 'frame':
            # Frame mode: apply normalization to single frame
            frame = self.to_tensor(data)
            frame = self.normalize(frame)
            return frame, action, scene_token, timestamp

        elif self.mode == 'clip':
            # Clip mode: data is already tensor (16, 3, 224, 224)
            clip = data

            # Normalize each frame
            normalized_frames = []
            for i in range(clip.shape[0]):
                normalized_frames.append(self.normalize(clip[i]))
            clip_normalized = torch.stack(normalized_frames)

            # If encoder provided, get embeddings
            if self.encoder is not None:
                # Add batch dimension: (1, 16, 3, 224, 224)
                clip_batch = clip_normalized.unsqueeze(0)

                with torch.no_grad():
                    embeddings = self.encoder(clip_batch)

                # Remove batch dimension: (384,)
                embeddings = embeddings.squeeze(0)

                return embeddings, action, scene_token, timestamp
            else:
                # No encoder: return normalized clip
                return clip_normalized, action, scene_token, timestamp


class MockVJEPAEncoder(nn.Module):
    """Mock V-JEPA encoder for testing."""

    def __init__(self, embedding_dim=384):
        super().__init__()
        # Simple conv + pool + fc architecture
        self.conv = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(16, embedding_dim)

    def forward(self, x):
        """
        Args:
            x: (B, 16, 3, 224, 224) clip tensor
        Returns:
            embeddings: (B, 384) tensor
        """
        # Rearrange to (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x
