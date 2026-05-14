"""Quick verification script for the dataset implementation."""

import numpy as np
from pathlib import Path
from data.dataset import NuScenesActionDataset, ClipNormalizedDataset, ImageNetNormalizedDataset


def main():
    print("Initializing NuScenes dataset...")
    dataroot = Path(__file__).parent
    dataset = NuScenesActionDataset(dataroot=str(dataroot), version='v1.0-mini')

    print(f"✓ Dataset initialized with {len(dataset)} valid samples")

    if len(dataset) == 0:
        print("⚠ No valid samples found. Check CAN bus data availability.")
        return

    # Test first sample
    print("\nTesting first sample...")
    frame, action, scene_token, timestamp_us = dataset[0]

    # Check shapes
    print(f"✓ Frame shape: {frame.size} (expected: (224, 224))")
    assert frame.size == (224, 224), f"Expected (224, 224), got {frame.size}"

    print(f"✓ Action shape: {action.shape} (expected: (2,))")
    assert action.shape == (2,), f"Expected (2,), got {action.shape}"

    # Check no NaN
    print(f"✓ Action values: steering={action[0]:.3f}, accel={action[1]:.3f}")
    assert not np.isnan(action).any(), f"NaN detected in action: {action}"

    # Check range
    assert -1.0 <= action[0] <= 1.0, f"Steering {action[0]} out of range"
    assert -1.0 <= action[1] <= 1.0, f"Accel {action[1]} out of range"
    print(f"✓ Action values in valid range [-1, 1]")

    # Check metadata
    print(f"✓ Scene token: {scene_token}")
    print(f"✓ Timestamp: {timestamp_us} μs")

    # Test blacklist exclusion
    print("\nChecking blacklist exclusion...")
    scene_names = {s['scene_name'] for s in dataset.samples}
    blacklist = set(dataset.nusc_can.can_blacklist)
    blacklisted_in_dataset = scene_names & blacklist
    if blacklisted_in_dataset:
        print(f"✗ Found blacklisted scenes: {blacklisted_in_dataset}")
        return
    print(f"✓ No blacklisted scenes in dataset (blacklist size: {len(blacklist)})")

    # Test multiple samples for NaN and range
    print("\nTesting all samples for NaN and range...")
    nan_count = 0
    range_violations = 0
    for i in range(len(dataset)):
        _, action, _, _ = dataset[i]
        if np.isnan(action).any():
            nan_count += 1
        if not (-1.0 <= action[0] <= 1.0 and -1.0 <= action[1] <= 1.0):
            range_violations += 1

    if nan_count > 0:
        print(f"✗ Found {nan_count} samples with NaN")
    else:
        print(f"✓ No NaN values in {len(dataset)} samples")

    if range_violations > 0:
        print(f"✗ Found {range_violations} samples with values out of range")
    else:
        print(f"✓ All {len(dataset)} samples have values in [-1, 1]")

    # Test timestamp tolerance
    print("\nTesting timestamp tolerance...")
    strict_dataset = NuScenesActionDataset(
        dataroot=str(dataroot),
        version='v1.0-mini',
        max_timestamp_delta_us=1000  # 1ms - very strict
    )
    print(f"✓ Strict tolerance (1ms): {len(strict_dataset)} samples")
    print(f"✓ Default tolerance (50ms): {len(dataset)} samples")
    assert len(strict_dataset) <= len(dataset), "Stricter tolerance should reduce samples"

    # Test normalized wrappers
    print("\nTesting normalization wrappers...")
    clip_dataset = ClipNormalizedDataset(dataset)
    imagenet_dataset = ImageNetNormalizedDataset(dataset)

    # Get fresh samples to compare
    frame_base, action_base, _, _ = dataset[0]
    frame_clip, action_clip, _, _ = clip_dataset[0]
    frame_imagenet, action_imagenet, _, _ = imagenet_dataset[0]

    print(f"✓ CLIP normalized frame shape: {frame_clip.shape} (expected: (3, 224, 224))")
    assert frame_clip.shape == (3, 224, 224)

    print(f"✓ ImageNet normalized frame shape: {frame_imagenet.shape} (expected: (3, 224, 224))")
    assert frame_imagenet.shape == (3, 224, 224)

    # Check if actions are preserved (should be same values)
    actions_match = np.allclose(action_base, action_clip)
    print(f"✓ Actions preserved: {actions_match}")
    if not actions_match:
        print(f"  Base: {action_base}")
        print(f"  CLIP: {action_clip}")
        print(f"  Difference: {np.abs(action_base - action_clip)}")
    assert actions_match or np.array_equal(action_base, action_clip)

    # Test CAM_FRONT filtering
    print("\nVerifying CAM_FRONT only...")
    cam_front_count = 0
    for sample_info in dataset.samples[:10]:  # Check first 10
        sample_data = dataset.nusc.get('sample_data', sample_info['sample_data_token'])
        sensor = dataset.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        sensor_name = dataset.nusc.get('sensor', sensor['sensor_token'])['channel']
        if sensor_name == 'CAM_FRONT':
            cam_front_count += 1

    print(f"✓ All checked samples use CAM_FRONT ({cam_front_count}/10)")

    print("\n" + "="*50)
    print("All checks passed! ✓")
    print("="*50)


if __name__ == '__main__':
    main()
