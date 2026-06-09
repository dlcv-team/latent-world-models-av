"""Tests for NuScenes dataset splits."""

import pytest
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from data.splits import (
    generate_mini_splits,
    get_split,
    get_split_from_canonical,
    verify_no_overlap,
    count_samples_per_split,
)
from data.dataset import NuScenesFrameDataset


skip_if_no_mini = pytest.mark.skipif(
    not Path("data/v1.0-mini").exists(),
    reason="NuScenes v1.0-mini data not available",
)

skip_if_no_trainval = pytest.mark.skipif(
    not Path("data/v1.0-trainval").exists(),
    reason="NuScenes v1.0-trainval data not available",
)

# Default: require mini for all tests (smoke splits)
pytestmark = skip_if_no_mini


@pytest.fixture
def nusc_mini():
    """NuScenes mini instance for testing."""
    return NuScenes(version='v1.0-mini', dataroot='data/', verbose=False)


def test_generate_mini_splits(nusc_mini):
    """Test smoke split generation from v1.0-mini."""
    splits = generate_mini_splits(nusc_mini, seed=42)

    # Verify structure
    assert 'smoke_train' in splits
    assert 'smoke_val' in splits
    assert 'smoke_test' in splits

    # Verify scene counts
    assert len(splits['smoke_train']) == 8
    assert len(splits['smoke_val']) == 1
    assert len(splits['smoke_test']) == 1

    # Verify all are lists of strings
    for split_name, scenes in splits.items():
        assert isinstance(scenes, list)
        assert all(isinstance(s, str) for s in scenes)


def test_generate_mini_splits_deterministic(nusc_mini):
    """Verify splits are deterministic with same seed."""
    splits1 = generate_mini_splits(nusc_mini, seed=42)
    splits2 = generate_mini_splits(nusc_mini, seed=42)

    assert splits1['smoke_val'] == splits2['smoke_val']
    assert splits1['smoke_test'] == splits2['smoke_test']
    assert splits1['smoke_train'] == splits2['smoke_train']


def test_generate_mini_splits_no_overlap(nusc_mini):
    """Verify smoke splits have no scene overlap."""
    splits = generate_mini_splits(nusc_mini, seed=42)
    verify_no_overlap(splits)  # Should not raise

    # Also verify manually
    val_scenes = set(splits['smoke_val'])
    test_scenes = set(splits['smoke_test'])
    train_scenes = set(splits['smoke_train'])

    assert val_scenes.isdisjoint(test_scenes)
    assert val_scenes.isdisjoint(train_scenes)
    assert test_scenes.isdisjoint(train_scenes)


def test_verify_no_overlap_raises_on_duplicate():
    """Test that verify_no_overlap raises on duplicate scenes."""
    bad_splits = {
        'train': ['scene-0001', 'scene-0002'],
        'val': ['scene-0001', 'scene-0003'],  # scene-0001 duplicated
    }

    with pytest.raises(ValueError, match="Scene overlap detected"):
        verify_no_overlap(bad_splits)


def test_count_samples_per_split(nusc_mini):
    """Test sample counting per split."""
    splits = generate_mini_splits(nusc_mini, seed=42)
    counts = count_samples_per_split(nusc_mini, splits)

    # Verify structure
    for split_name in ['smoke_train', 'smoke_val', 'smoke_test']:
        assert split_name in counts
        assert 'scenes' in counts[split_name]
        assert 'samples' in counts[split_name]

        # Verify counts match
        assert counts[split_name]['scenes'] == len(splits[split_name])
        assert counts[split_name]['samples'] > 0


def test_get_split_smoke_train(nusc_mini):
    """Test get_split for smoke_train."""
    scenes = get_split('smoke_train', dataroot='data/', seed=42)

    assert isinstance(scenes, list)
    assert len(scenes) == 8
    assert all(isinstance(s, str) for s in scenes)
    assert all(s.startswith('scene-') for s in scenes)


def test_get_split_smoke_val(nusc_mini):
    """Test get_split for smoke_val."""
    scenes = get_split('smoke_val', dataroot='data/', seed=42)

    assert isinstance(scenes, list)
    assert len(scenes) == 1


def test_get_split_smoke_test(nusc_mini):
    """Test get_split for smoke_test."""
    scenes = get_split('smoke_test', dataroot='data/', seed=42)

    assert isinstance(scenes, list)
    assert len(scenes) == 1


def test_get_split_deterministic():
    """Test that get_split is deterministic."""
    scenes1 = get_split('smoke_train', dataroot='data/', seed=42)
    scenes2 = get_split('smoke_train', dataroot='data/', seed=42)

    assert scenes1 == scenes2


def test_integration_with_dataset_smoke_train():
    """Test NuScenesFrameDataset integration with smoke_train split."""
    ds = NuScenesFrameDataset(split='smoke_train', mode='single_frame')

    # Verify dataset has samples
    assert len(ds) > 0

    # Verify all samples come from smoke_train scenes
    smoke_train_scenes = set(get_split('smoke_train', dataroot='data/'))
    for sample_info in ds.samples:
        assert sample_info['scene_name'] in smoke_train_scenes


def test_integration_with_dataset_all_smoke_splits():
    """Test dataset creation for all smoke splits."""
    for split_name in ['smoke_train', 'smoke_val', 'smoke_test']:
        ds = NuScenesFrameDataset(split=split_name, mode='single_frame')
        assert len(ds) > 0


def test_invalid_split_raises_error():
    """Test that invalid split name raises ValueError."""
    with pytest.raises(KeyError):  # KeyError when split not found in manifest_split
        NuScenesFrameDataset(split='invalid_split', mode='single_frame')


def test_generate_mini_splits_wrong_version():
    """Test that generate_mini_splits rejects non-mini versions."""
    # Create a trainval instance (if available) or mock it
    with pytest.raises(ValueError, match="Expected v1.0-mini"):
        nusc_fake = type('obj', (object,), {'version': 'v1.0-trainval'})()
        generate_mini_splits(nusc_fake)


@skip_if_no_trainval
class TestCanonicalSplits:
    """Tests for canonical manifest splits (requires v1.0-trainval)."""

    def test_get_split_from_canonical_p0_train(self):
        """Test loading p0_train from canonical manifest."""
        scenes = get_split_from_canonical('p0_train')

        assert isinstance(scenes, list)
        assert len(scenes) == 180
        assert all(isinstance(s, str) for s in scenes)

    def test_get_split_from_canonical_p0_val(self):
        """Test loading p0_val from canonical manifest."""
        scenes = get_split_from_canonical('p0_val')

        assert isinstance(scenes, list)
        assert len(scenes) == 20

    def test_get_split_from_canonical_p0_test(self):
        """Test loading p0_test from canonical manifest."""
        scenes = get_split_from_canonical('p0_test')

        assert isinstance(scenes, list)
        assert len(scenes) == 40

    def test_get_split_from_canonical_no_overlap(self):
        """Verify canonical P0 splits have no overlap."""
        splits = {
            'p0_train': get_split_from_canonical('p0_train'),
            'p0_val': get_split_from_canonical('p0_val'),
            'p0_test': get_split_from_canonical('p0_test'),
        }

        verify_no_overlap(splits)

    def test_integration_with_dataset_p0_train(self):
        """Test dataset integration with canonical p0_train."""
        ds = NuScenesFrameDataset(split='p0_train', mode='single_frame')

        assert len(ds) > 0

        # Verify all samples from p0_train scenes
        p0_train_scenes = set(get_split_from_canonical('p0_train'))
        for sample_info in ds.samples:
            assert sample_info['scene_name'] in p0_train_scenes




# --- Merged from main-tier2 ---

