"""Tests for NuScenes dataset splits."""

import pytest
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from data.splits import create_action_splits
from data.dataset import NuScenesActionDataset


@pytest.fixture
def nusc_mini():
    """NuScenes mini instance for testing."""
    return NuScenes(version='v1.0-mini', dataroot='data/', verbose=False)


@pytest.fixture
def nusc_can_mini():
    """NuScenesCanBus instance for testing."""
    return NuScenesCanBus(dataroot='data/')


@pytest.fixture
def splits_mini(nusc_mini, nusc_can_mini):
    """Mini splits for testing."""
    return create_action_splits('v1.0-mini', nusc_mini, nusc_can_mini)


def test_mini_splits_structure(splits_mini):
    """Verify mini splits have correct structure and scene counts."""
    assert 'smoke_train' in splits_mini
    assert 'smoke_val' in splits_mini
    assert 'smoke_test' in splits_mini

    # Verify scene counts
    assert splits_mini['smoke_train']['num_scenes'] == 8
    assert splits_mini['smoke_val']['num_scenes'] == 1
    assert splits_mini['smoke_test']['num_scenes'] == 1

    # Verify each split has required fields
    for split_name, split_info in splits_mini.items():
        assert 'scenes' in split_info
        assert 'num_scenes' in split_info
        assert 'num_frames' in split_info
        assert isinstance(split_info['scenes'], list)
        assert isinstance(split_info['num_scenes'], int)
        assert isinstance(split_info['num_frames'], int)


def test_mini_splits_deterministic(nusc_mini, nusc_can_mini):
    """Verify splits are deterministic across multiple calls."""
    splits1 = create_action_splits('v1.0-mini', nusc_mini, nusc_can_mini)
    splits2 = create_action_splits('v1.0-mini', nusc_mini, nusc_can_mini)

    # Verify smoke_val and smoke_test are identical
    assert splits1['smoke_val']['scenes'] == splits2['smoke_val']['scenes']
    assert splits1['smoke_test']['scenes'] == splits2['smoke_test']['scenes']
    assert splits1['smoke_train']['scenes'] == splits2['smoke_train']['scenes']


def test_mini_splits_no_overlap(splits_mini):
    """Verify smoke_val and smoke_test don't share scenes."""
    val_scenes = set(splits_mini['smoke_val']['scenes'])
    test_scenes = set(splits_mini['smoke_test']['scenes'])
    train_scenes = set(splits_mini['smoke_train']['scenes'])

    # No overlap between any splits
    assert val_scenes.isdisjoint(test_scenes)
    assert val_scenes.isdisjoint(train_scenes)
    assert test_scenes.isdisjoint(train_scenes)


def test_cross_split_uniqueness(splits_mini):
    """Verify no scene appears in multiple splits."""
    all_scenes = {}

    for split_name, split_info in splits_mini.items():
        for scene_name in split_info['scenes']:
            assert scene_name not in all_scenes, \
                f"Scene {scene_name} in both {all_scenes.get(scene_name)} and {split_name}"
            all_scenes[scene_name] = split_name


def test_frame_counts_positive(splits_mini):
    """Verify all frame counts are positive."""
    for split_name, split_info in splits_mini.items():
        assert split_info['num_frames'] > 0, \
            f"Split {split_name} has {split_info['num_frames']} frames"


def test_integration_with_dataset_mini(nusc_mini, nusc_can_mini):
    """Test dataset.py integration with split parameter."""
    # Create dataset for smoke_train split
    train_ds = NuScenesActionDataset(
        dataroot='data/',
        version='v1.0-mini',
        split='smoke_train'
    )

    # Verify dataset length matches expected frame count
    splits = create_action_splits('v1.0-mini', nusc_mini, nusc_can_mini)
    assert len(train_ds) == splits['smoke_train']['num_frames']

    # Verify all samples come from allowed scenes
    allowed_scenes = set(splits['smoke_train']['scenes'])
    for sample_info in train_ds.samples:
        assert sample_info['scene_name'] in allowed_scenes


def test_integration_with_dataset_all_splits(nusc_mini, nusc_can_mini):
    """Test dataset creation for all mini splits."""
    splits = create_action_splits('v1.0-mini', nusc_mini, nusc_can_mini)

    for split_name in ['smoke_train', 'smoke_val', 'smoke_test']:
        ds = NuScenesActionDataset(
            dataroot='data/',
            version='v1.0-mini',
            split=split_name
        )
        assert len(ds) == splits[split_name]['num_frames']


def test_invalid_split_raises_error():
    """Test that invalid split name raises ValueError."""
    with pytest.raises(ValueError, match="Split 'invalid_split' not found"):
        NuScenesActionDataset(
            dataroot='data/',
            version='v1.0-mini',
            split='invalid_split'
        )


def test_invalid_version_raises_error(nusc_mini, nusc_can_mini):
    """Test that invalid version raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported version"):
        create_action_splits('v1.0-invalid', nusc_mini, nusc_can_mini)


@pytest.mark.skipif(
    not pytest.config.getoption("--trainval", default=False),
    reason="Trainval tests require --trainval flag and full dataset"
)
class TestTrainvalSplits:
    """Tests for trainval splits (requires full dataset)."""

    @pytest.fixture
    def nusc_trainval(self):
        """NuScenes trainval instance for testing."""
        return NuScenes(
            version='v1.0-trainval',
            dataroot='data/v1.0-trainval_metadata/',
            verbose=False
        )

    @pytest.fixture
    def nusc_can_trainval(self):
        """NuScenesCanBus instance for trainval."""
        return NuScenesCanBus(dataroot='data/')

    @pytest.fixture
    def splits_trainval(self, nusc_trainval, nusc_can_trainval):
        """Trainval splits for testing."""
        return create_action_splits('v1.0-trainval', nusc_trainval, nusc_can_trainval)

    def test_trainval_splits_structure(self, splits_trainval):
        """Verify trainval splits have correct structure."""
        assert 'train' in splits_trainval
        assert 'internal_val' in splits_trainval
        assert 'test' in splits_trainval

        # Verify approximate scene counts
        assert 610 <= splits_trainval['train']['num_scenes'] <= 620  # ~616
        assert 65 <= splits_trainval['internal_val']['num_scenes'] <= 75  # ~69
        assert splits_trainval['test']['num_scenes'] == 150

    def test_trainval_blacklist_removal(self, splits_trainval, nusc_can_trainval):
        """Verify no blacklisted scenes in any split."""
        blacklist = nusc_can_trainval.can_blacklist

        for split_info in splits_trainval.values():
            for scene_name in split_info['scenes']:
                scene_num = int(scene_name.split('-')[1])
                assert scene_num not in blacklist, \
                    f"Blacklisted scene {scene_name} found in split"

    def test_trainval_split_ratio(self, splits_trainval):
        """Verify ~90/10 split ratio for train/internal_val."""
        train_scenes = splits_trainval['train']['num_scenes']
        val_scenes = splits_trainval['internal_val']['num_scenes']
        total = train_scenes + val_scenes

        train_ratio = train_scenes / total
        assert 0.88 <= train_ratio <= 0.92, \
            f"Train ratio {train_ratio:.2f} not close to 0.90"

    def test_trainval_cross_split_uniqueness(self, splits_trainval):
        """Verify no scene in multiple splits."""
        all_scenes = {}

        for split_name, split_info in splits_trainval.items():
            for scene_name in split_info['scenes']:
                assert scene_name not in all_scenes, \
                    f"Scene {scene_name} in both {all_scenes.get(scene_name)} and {split_name}"
                all_scenes[scene_name] = split_name

    def test_trainval_deterministic(self, nusc_trainval, nusc_can_trainval):
        """Verify trainval splits are deterministic."""
        splits1 = create_action_splits('v1.0-trainval', nusc_trainval, nusc_can_trainval)
        splits2 = create_action_splits('v1.0-trainval', nusc_trainval, nusc_can_trainval)

        assert splits1['train']['scenes'] == splits2['train']['scenes']
        assert splits1['internal_val']['scenes'] == splits2['internal_val']['scenes']
        assert splits1['test']['scenes'] == splits2['test']['scenes']


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--trainval",
        action="store_true",
        default=False,
        help="Run trainval tests (requires full dataset)"
    )
