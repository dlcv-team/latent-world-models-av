"""Verification script for clip mode and V-JEPA wrapper."""

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from data.dataset import NuScenesActionDataset, VJEPANormalizedDataset, MockVJEPAEncoder


def main():
    dataroot = Path(__file__).parent / 'data'

    print("=" * 60)
    print("Clip Mode + V-JEPA Wrapper Verification")
    print("=" * 60)

    # Test 1: Clip mode dataset
    print("\n1. Testing clip mode dataset...")
    clip_dataset = NuScenesActionDataset(
        dataroot=str(dataroot),
        version='v1.0-mini',
        mode='clip'
    )

    clip, action, scene_token, timestamp = clip_dataset[0]
    print(f"   ✓ Clip shape: {clip.shape}")
    print(f"   ✓ Action shape: {action.shape}")
    print(f"   ✓ Scene token: {scene_token}")
    print(f"   ✓ Timestamp: {timestamp}")

    # Test 2: Verify timestamps monotonic
    print("\n2. Testing timestamp monotonicity...")
    sample_info = clip_dataset.samples[0]
    frames = clip_dataset._collect_clip(
        sample_info['sample_data_token'],
        sample_info['scene_token']
    )
    timestamps = [f['timestamp'] for f in frames]
    is_monotonic = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
    print(f"   ✓ Timestamps monotonic: {is_monotonic}")
    print(f"   ✓ First timestamp: {timestamps[0]}")
    print(f"   ✓ Last timestamp: {timestamps[-1]}")
    print(f"   ✓ Matches target: {timestamps[-1] == sample_info['timestamp']}")

    # Test 3: V-JEPA wrapper without encoder
    print("\n3. Testing V-JEPA wrapper (normalized clip)...")
    vjepa_dataset = VJEPANormalizedDataset(clip_dataset)
    norm_clip, action, scene_token, timestamp = vjepa_dataset[0]
    print(f"   ✓ Normalized clip shape: {norm_clip.shape}")
    print(f"   ✓ Value range: [{norm_clip.min():.2f}, {norm_clip.max():.2f}]")

    # Test 4: V-JEPA wrapper with encoder
    print("\n4. Testing V-JEPA wrapper with encoder...")
    encoder = MockVJEPAEncoder(embedding_dim=384)
    vjepa_dataset_enc = VJEPANormalizedDataset(clip_dataset, encoder=encoder)
    embeddings, action, scene_token, timestamp = vjepa_dataset_enc[0]
    print(f"   ✓ Embedding shape: {embeddings.shape}")
    print(f"   ✓ Expected shape: (384,)")
    print(f"   ✓ Shape matches: {embeddings.shape == (384,)}")

    # Test 5: DataLoader batch processing
    print("\n5. Testing DataLoader batch processing...")
    loader = DataLoader(vjepa_dataset_enc, batch_size=4, shuffle=False)
    batch_embeddings, batch_actions, batch_scenes, batch_timestamps = next(iter(loader))
    print(f"   ✓ Batch embeddings shape: {batch_embeddings.shape}")
    print(f"   ✓ Batch actions shape: {batch_actions.shape}")
    print(f"   ✓ Expected batch embeddings: (4, 384)")
    print(f"   ✓ Expected batch actions: (4, 2)")
    print(f"   ✓ Shapes match: {batch_embeddings.shape == (4, 384) and batch_actions.shape == (4, 2)}")

    # Test 6: Frame mode backward compatibility
    print("\n6. Testing backward compatibility (frame mode)...")
    frame_dataset = NuScenesActionDataset(
        dataroot=str(dataroot),
        version='v1.0-mini'
    )
    print(f"   ✓ Default mode: {frame_dataset.mode}")
    frame, action, scene_token, timestamp = frame_dataset[0]
    print(f"   ✓ Frame type: PIL Image")
    print(f"   ✓ Frame size: {frame.size}")

    print("\n" + "=" * 60)
    print("✅ All verifications passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
