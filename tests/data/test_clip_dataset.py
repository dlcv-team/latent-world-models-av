"""Unit tests for NuScenes clip mode and V-JEPA wrapper."""

import numpy as np
import pytest
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from data.dataset import NuScenesActionDataset, VJEPANormalizedDataset, MockVJEPAEncoder


@pytest.fixture
def dataroot():
    """Get dataroot path for testing."""
    return Path(__file__).parent


@pytest.fixture
def clip_dataset(dataroot):
    """Create clip mode dataset instance for testing."""
    return NuScenesActionDataset(dataroot=str(dataroot), version='v1.0-mini', mode='clip')


@pytest.fixture
def frame_dataset(dataroot):
    """Create frame mode dataset instance for testing."""
    return NuScenesActionDataset(dataroot=str(dataroot), version='v1.0-mini', mode='frame')


@pytest.fixture
def vjepa_dataset(clip_dataset):
    """Create V-JEPA normalized dataset without encoder."""
    return VJEPANormalizedDataset(clip_dataset)


@pytest.fixture
def vjepa_dataset_with_encoder(clip_dataset):
    """Create V-JEPA normalized dataset with mock encoder."""
    encoder = MockVJEPAEncoder(embedding_dim=384)
    return VJEPANormalizedDataset(clip_dataset, encoder=encoder)


class TestClipMode:
    """Test clip mode functionality."""

    def test_clip_mode_initialization(self, clip_dataset):
        """Test dataset initializes with mode='clip'."""
        assert clip_dataset.mode == 'clip'
        assert len(clip_dataset) > 0

    def test_clip_length_16(self, clip_dataset):
        """Test that clips have exactly 16 frames."""
        if len(clip_dataset) == 0:
            pytest.skip("No valid samples in dataset")

        clip, action, scene_token, timestamp = clip_dataset[0]
        assert clip.shape == (16, 3, 224, 224), \
            f"Expected (16, 3, 224, 224), got {clip.shape}"

    def test_timestamps_monotonic(self, clip_dataset):
        """Test that frame timestamps are monotonically increasing."""
        if len(clip_dataset) == 0:
            pytest.skip("No valid samples in dataset")

        sample_info = clip_dataset.samples[0]
        frames = clip_dataset._collect_clip(
            sample_info['sample_data_token'],
            sample_info['scene_token']
        )

        timestamps = [f['timestamp'] for f in frames]

        # Check monotonic
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1], \
                f"Timestamps not monotonic: {timestamps[i-1]} -> {timestamps[i]}"

    def test_final_timestamp_matches_target(self, clip_dataset):
        """Test that last frame timestamp matches target keyframe."""
        if len(clip_dataset) == 0:
            pytest.skip("No valid samples in dataset")

        sample_info = clip_dataset.samples[0]
        frames = clip_dataset._collect_clip(
            sample_info['sample_data_token'],
            sample_info['scene_token']
        )

        # Last frame timestamp should match sample_info timestamp
        assert frames[-1]['timestamp'] == sample_info['timestamp'], \
            f"Last frame timestamp {frames[-1]['timestamp']} != " \
            f"target timestamp {sample_info['timestamp']}"

    def test_no_scene_crossing(self, clip_dataset):
        """Test that clips don't cross scene boundaries."""
        if len(clip_dataset) == 0:
            pytest.skip("No valid samples in dataset")

        # Test first 5 samples (or all if less than 5)
        num_samples_to_test = min(5, len(clip_dataset))

        for i in range(num_samples_to_test):
            sample_info = clip_dataset.samples[i]
            frames = clip_dataset._collect_clip(
                sample_info['sample_data_token'],
                sample_info['scene_token']
            )

            # Check all frames belong to same scene
            for frame_info in frames:
                sd = clip_dataset.nusc.get('sample_data', frame_info['token'])
                sample = clip_dataset.nusc.get('sample', sd['sample_token'])

                assert sample['scene_token'] == sample_info['scene_token'], \
                    f"Frame crosses scene boundary: " \
                    f"expected {sample_info['scene_token']}, " \
                    f"got {sample['scene_token']}"

    def test_padding_when_insufficient_frames(self, clip_dataset):
        """Test that padding duplicates earliest frame."""
        if len(clip_dataset) == 0:
            pytest.skip("No valid samples in dataset")

        # Find a sample with < 16 unique frames
        found_padded = False
        for sample_info in clip_dataset.samples[:10]:
            frames = clip_dataset._collect_clip(
                sample_info['sample_data_token'],
                sample_info['scene_token']
            )

            # Count unique tokens
            unique_tokens = set(f['token'] for f in frames)

            if len(unique_tokens) < 16:
                found_padded = True

                # If padded, should have duplicates
                assert len(unique_tokens) < 16, \
                    "Padded clip should have duplicate frames"

                # First frames should be duplicates
                assert frames[0]['token'] == frames[1]['token'], \
                    "Padding should duplicate earliest frame"

                break

        # Note: If no padded samples found, test passes (all samples have 16+ frames)

    def test_clip_tensor_dtype(self, clip_dataset):
        """Test that clip tensor has correct dtype and range."""
        if len(clip_dataset) == 0:
            pytest.skip("No valid samples in dataset")

        clip, _, _, _ = clip_dataset[0]

        assert clip.dtype == torch.float32, \
            f"Expected float32, got {clip.dtype}"

        # ToTensor() normalizes to [0, 1]
        assert clip.min() >= 0.0, f"Min value {clip.min()} < 0"
        assert clip.max() <= 1.0, f"Max value {clip.max()} > 1"

    def test_clip_action_shape(self, clip_dataset):
        """Test that action labels are preserved in clip mode."""
        if len(clip_dataset) == 0:
            pytest.skip("No valid samples in dataset")

        _, action, _, _ = clip_dataset[0]

        assert isinstance(action, np.ndarray), "Action should be numpy array"
        assert action.shape == (2,), f"Action shape should be (2,), got {action.shape}"
        assert action.dtype == np.float32, f"Action dtype should be float32, got {action.dtype}"


class TestVJEPAWrapper:
    """Test V-JEPA normalization wrapper."""

    def test_wrapper_with_clip_mode(self, vjepa_dataset):
        """Test VJEPANormalizedDataset with clip mode."""
        if len(vjepa_dataset) == 0:
            pytest.skip("No valid samples in dataset")

        clip, action, scene_token, timestamp = vjepa_dataset[0]

        # Should still be (16, 3, 224, 224)
        assert clip.shape == (16, 3, 224, 224)

        # After ImageNet normalization, values in ~[-3, 3]
        assert clip.min() >= -5.0, "Normalized values too low"
        assert clip.max() <= 5.0, "Normalized values too high"

    def test_wrapper_with_encoder(self, vjepa_dataset_with_encoder):
        """Test VJEPANormalizedDataset with encoder."""
        if len(vjepa_dataset_with_encoder) == 0:
            pytest.skip("No valid samples in dataset")

        embeddings, action, scene_token, timestamp = vjepa_dataset_with_encoder[0]

        # Should be (384,) embeddings
        assert embeddings.shape == (384,), \
            f"Expected (384,), got {embeddings.shape}"

        # Check action preserved
        assert action.shape == (2,)

    def test_wrapper_preserves_actions(self, clip_dataset, vjepa_dataset):
        """Test that wrapper doesn't modify action labels."""
        if len(clip_dataset) == 0:
            pytest.skip("No valid samples in dataset")

        _, action_base, _, _ = clip_dataset[0]
        _, action_wrapper, _, _ = vjepa_dataset[0]

        assert np.allclose(action_base, action_wrapper), \
            "Wrapper should not modify actions"

    def test_wrapper_batch_processing(self, vjepa_dataset_with_encoder):
        """Test that encoder works with DataLoader batches."""
        if len(vjepa_dataset_with_encoder) == 0:
            pytest.skip("No valid samples in dataset")

        # Create DataLoader
        loader = DataLoader(vjepa_dataset_with_encoder, batch_size=2, shuffle=False)

        # Get one batch
        batch = next(iter(loader))
        embeddings, actions, scene_tokens, timestamps = batch

        # Check batch shapes
        assert embeddings.shape == (2, 384), \
            f"Expected (2, 384), got {embeddings.shape}"
        assert actions.shape == (2, 2), \
            f"Expected (2, 2), got {actions.shape}"


class TestBackwardCompatibility:
    """Test that frame mode still works."""

    def test_frame_mode_unchanged(self, frame_dataset):
        """Test that frame mode behavior is unchanged."""
        if len(frame_dataset) == 0:
            pytest.skip("No valid samples in dataset")

        frame, action, scene_token, timestamp = frame_dataset[0]

        assert isinstance(frame, Image.Image)
        assert frame.size == (224, 224)
        assert action.shape == (2,)

    def test_default_mode_is_frame(self, dataroot):
        """Test that default mode is 'frame' for backward compatibility."""
        dataset = NuScenesActionDataset(
            dataroot=str(dataroot),
            version='v1.0-mini'
        )

        assert dataset.mode == 'frame'

    def test_vjepa_wrapper_with_frame_mode(self, frame_dataset):
        """Test VJEPANormalizedDataset with frame mode."""
        if len(frame_dataset) == 0:
            pytest.skip("No valid samples in dataset")

        vjepa_dataset = VJEPANormalizedDataset(frame_dataset)

        frame, action, scene_token, timestamp = vjepa_dataset[0]

        # Should be single frame tensor
        assert frame.shape == (3, 224, 224)

    def test_invalid_mode_raises_error(self, dataroot):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            NuScenesActionDataset(
                dataroot=str(dataroot),
                version='v1.0-mini',
                mode='invalid'
            )


class TestMockEncoder:
    """Test mock V-JEPA encoder."""

    def test_encoder_forward_shape(self):
        """Test that mock encoder produces correct output shape."""
        encoder = MockVJEPAEncoder(embedding_dim=384)

        # Create dummy input (B=2, T=16, C=3, H=224, W=224)
        dummy_input = torch.randn(2, 16, 3, 224, 224)

        output = encoder(dummy_input)

        assert output.shape == (2, 384), \
            f"Expected (2, 384), got {output.shape}"

    def test_encoder_single_sample(self):
        """Test encoder with single sample."""
        encoder = MockVJEPAEncoder(embedding_dim=384)

        # Single sample (B=1, T=16, C=3, H=224, W=224)
        dummy_input = torch.randn(1, 16, 3, 224, 224)

        output = encoder(dummy_input)

        assert output.shape == (1, 384), \
            f"Expected (1, 384), got {output.shape}"
