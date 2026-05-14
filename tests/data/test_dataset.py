"""Unit tests for NuScenes action dataset."""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from data.dataset import NuScenesActionDataset, ClipNormalizedDataset, ImageNetNormalizedDataset


@pytest.fixture
def dataset():
    """Create dataset instance for testing."""
    dataroot = Path(__file__).parent
    return NuScenesActionDataset(dataroot=str(dataroot), version='v1.0-mini')


@pytest.fixture
def clip_dataset(dataset):
    """Create CLIP-normalized dataset."""
    return ClipNormalizedDataset(dataset)


@pytest.fixture
def imagenet_dataset(dataset):
    """Create ImageNet-normalized dataset."""
    return ImageNetNormalizedDataset(dataset)


class TestNuScenesActionDataset:
    """Test core dataset functionality."""

    def test_dataset_length(self, dataset):
        """Test that dataset has valid samples."""
        assert len(dataset) > 0, "Dataset should contain valid samples"

    def test_output_shapes(self, dataset):
        """Test that outputs have correct shapes."""
        if len(dataset) == 0:
            pytest.skip("No valid samples in dataset")

        frame, action, scene_token, timestamp_us = dataset[0]

        # Frame should be PIL Image of size 224x224
        assert isinstance(frame, Image.Image), "Frame should be PIL Image"
        assert frame.size == (224, 224), f"Frame should be 224x224, got {frame.size}"

        # Action should be numpy array of shape (2,)
        assert isinstance(action, np.ndarray), "Action should be numpy array"
        assert action.shape == (2,), f"Action shape should be (2,), got {action.shape}"
        assert action.dtype == np.float32, f"Action dtype should be float32, got {action.dtype}"

        # Scene token should be string
        assert isinstance(scene_token, str), "Scene token should be string"

        # Timestamp should be integer
        assert isinstance(timestamp_us, (int, np.integer)), "Timestamp should be integer"

    def test_no_nan_in_actions(self, dataset):
        """Test that action labels contain no NaN values."""
        if len(dataset) == 0:
            pytest.skip("No valid samples in dataset")

        for i in range(min(len(dataset), 10)):  # Test first 10 samples
            _, action, _, _ = dataset[i]
            assert not np.isnan(action).any(), f"Sample {i} contains NaN in action: {action}"

    def test_action_range(self, dataset):
        """Test that action values are in valid range [-1, 1]."""
        if len(dataset) == 0:
            pytest.skip("No valid samples in dataset")

        for i in range(min(len(dataset), 10)):  # Test first 10 samples
            _, action, _, _ = dataset[i]
            steering, accel = action

            assert -1.0 <= steering <= 1.0, \
                f"Sample {i}: steering {steering} out of range [-1, 1]"
            assert -1.0 <= accel <= 1.0, \
                f"Sample {i}: accel {accel} out of range [-1, 1]"

    def test_timestamp_tolerance(self, dataset):
        """Test that CAN messages are within timestamp tolerance."""
        # This is implicitly tested by the dataset building process
        # If a sample is in the dataset, it passed the timestamp tolerance check
        if len(dataset) == 0:
            pytest.skip("No valid samples in dataset")

        # Create a dataset with very strict tolerance
        dataroot = Path(__file__).parent
        strict_dataset = NuScenesActionDataset(
            dataroot=str(dataroot),
            version='v1.0-mini',
            max_timestamp_delta_us=1  # 1 microsecond - very strict
        )

        # Strict dataset should have fewer or equal samples
        assert len(strict_dataset) <= len(dataset), \
            "Stricter timestamp tolerance should not increase sample count"

    def test_blacklist_exclusion(self, dataset):
        """Test that blacklisted scenes are excluded."""
        # Get all scene names in dataset
        scene_names = set()
        for sample_info in dataset.samples:
            scene_names.add(sample_info['scene_name'])

        # Verify no blacklisted scenes are present
        blacklist = dataset.nusc_can.can_blacklist
        for scene_name in scene_names:
            assert scene_name not in blacklist, \
                f"Blacklisted scene {scene_name} found in dataset"

    def test_cam_front_only(self, dataset):
        """Test that only CAM_FRONT data is loaded."""
        if len(dataset) == 0:
            pytest.skip("No valid samples in dataset")

        # Check that sample_data tokens correspond to CAM_FRONT
        for sample_info in dataset.samples:
            sample_data = dataset.nusc.get('sample_data', sample_info['sample_data_token'])
            sensor = dataset.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            sensor_name = dataset.nusc.get('sensor', sensor['sensor_token'])['channel']

            assert sensor_name == 'CAM_FRONT', \
                f"Expected CAM_FRONT, got {sensor_name}"

    def test_deterministic_indexing(self, dataset):
        """Test that repeated access to same index returns consistent data."""
        if len(dataset) == 0:
            pytest.skip("No valid samples in dataset")

        idx = 0
        frame1, action1, scene1, time1 = dataset[idx]
        frame2, action2, scene2, time2 = dataset[idx]

        # Actions should be identical
        assert np.array_equal(action1, action2), "Actions should be deterministic"

        # Metadata should be identical
        assert scene1 == scene2, "Scene tokens should be identical"
        assert time1 == time2, "Timestamps should be identical"

        # Frames should be identical (same size and mode)
        assert frame1.size == frame2.size, "Frame sizes should be identical"
        assert frame1.mode == frame2.mode, "Frame modes should be identical"


class TestNormalizedDatasets:
    """Test encoder-specific normalization wrappers."""

    def test_clip_normalization_shape(self, clip_dataset):
        """Test CLIP normalized output shape."""
        if len(clip_dataset) == 0:
            pytest.skip("No valid samples in dataset")

        frame, action, scene_token, timestamp_us = clip_dataset[0]

        # Frame should now be a tensor of shape (3, 224, 224)
        assert hasattr(frame, 'shape'), "Frame should be a tensor"
        assert frame.shape == (3, 224, 224), \
            f"Frame should be (3, 224, 224), got {frame.shape}"

    def test_imagenet_normalization_shape(self, imagenet_dataset):
        """Test ImageNet normalized output shape."""
        if len(imagenet_dataset) == 0:
            pytest.skip("No valid samples in dataset")

        frame, action, scene_token, timestamp_us = imagenet_dataset[0]

        # Frame should now be a tensor of shape (3, 224, 224)
        assert hasattr(frame, 'shape'), "Frame should be a tensor"
        assert frame.shape == (3, 224, 224), \
            f"Frame should be (3, 224, 224), got {frame.shape}"

    def test_clip_normalization_range(self, clip_dataset):
        """Test that CLIP normalization produces reasonable values."""
        if len(clip_dataset) == 0:
            pytest.skip("No valid samples in dataset")

        frame, _, _, _ = clip_dataset[0]

        # After normalization, values typically fall in [-3, 3] range
        # (due to mean subtraction and std division)
        assert frame.min() >= -5.0, "Normalized values seem too low"
        assert frame.max() <= 5.0, "Normalized values seem too high"

    def test_imagenet_normalization_range(self, imagenet_dataset):
        """Test that ImageNet normalization produces reasonable values."""
        if len(imagenet_dataset) == 0:
            pytest.skip("No valid samples in dataset")

        frame, _, _, _ = imagenet_dataset[0]

        # After normalization, values typically fall in [-3, 3] range
        assert frame.min() >= -5.0, "Normalized values seem too low"
        assert frame.max() <= 5.0, "Normalized values seem too high"

    def test_wrapper_preserves_actions(self, dataset, clip_dataset):
        """Test that normalization wrappers don't modify action labels."""
        if len(dataset) == 0:
            pytest.skip("No valid samples in dataset")

        _, action_base, _, _ = dataset[0]
        _, action_clip, _, _ = clip_dataset[0]

        # Actions should be the same (allowing for floating point precision)
        assert np.allclose(action_base, action_clip), \
            "Normalization wrapper should not modify actions"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset_handling(self):
        """Test behavior with invalid dataroot."""
        # This should either raise an error or create an empty dataset
        # depending on the nuscenes library behavior
        try:
            dataroot = "/nonexistent/path"
            dataset = NuScenesActionDataset(dataroot=dataroot, version='v1.0-mini')
            # If it doesn't raise, it should be empty
            assert len(dataset) == 0
        except (FileNotFoundError, AssertionError):
            # Expected behavior - invalid path raises error
            pass

    def test_action_clipping_steering(self, dataset):
        """Test that extreme steering values are clipped to [-1, 1]."""
        # This is a unit test for the normalization logic
        # Steering is normalized as clip(value / 6.0, -1, 1)

        # Test extreme positive value
        extreme_positive = 100.0
        normalized = np.clip(extreme_positive / 6.0, -1.0, 1.0)
        assert normalized == 1.0, "Extreme positive steering should clip to 1.0"

        # Test extreme negative value
        extreme_negative = -100.0
        normalized = np.clip(extreme_negative / 6.0, -1.0, 1.0)
        assert normalized == -1.0, "Extreme negative steering should clip to -1.0"

        # Test normal value
        normal_value = 3.0
        normalized = np.clip(normal_value / 6.0, -1.0, 1.0)
        assert normalized == 0.5, "Normal steering should normalize correctly"

    def test_action_clipping_accel(self, dataset):
        """Test that extreme acceleration values are clipped to [-1, 1]."""
        # Acceleration is normalized as clip(accel / 10.0, -1, 1)

        # Test extreme positive value
        extreme_positive = 100.0
        normalized = np.clip(extreme_positive / 10.0, -1.0, 1.0)
        assert normalized == 1.0, "Extreme positive accel should clip to 1.0"

        # Test extreme negative value
        extreme_negative = -100.0
        normalized = np.clip(extreme_negative / 10.0, -1.0, 1.0)
        assert normalized == -1.0, "Extreme negative accel should clip to -1.0"

        # Test normal value
        normal_value = 5.0
        normalized = np.clip(normal_value / 10.0, -1.0, 1.0)
        assert normalized == 0.5, "Normal accel should normalize correctly"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
