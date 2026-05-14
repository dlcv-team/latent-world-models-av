"""NuScenes action prediction dataset splits."""

from typing import Dict, List
import random
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes


def create_action_splits(
    version: str,
    nusc: NuScenes,
    nusc_can: NuScenesCanBus
) -> Dict[str, Dict[str, any]]:
    """
    Create dataset splits for action prediction with CAN blacklist filtering.

    Args:
        version: Dataset version ('v1.0-mini' or 'v1.0-trainval')
        nusc: NuScenes instance
        nusc_can: NuScenesCanBus instance

    Returns:
        Dictionary mapping split names to split info:
        {
            'split_name': {
                'scenes': List[str],  # Scene names
                'num_scenes': int,
                'num_frames': int
            }
        }

    Raises:
        ValueError: If version is not supported
    """
    if version not in ['v1.0-mini', 'v1.0-trainval']:
        raise ValueError(f"Unsupported version: {version}. Use 'v1.0-mini' or 'v1.0-trainval'")

    # Get official NuScenes splits
    official_splits = create_splits_scenes()

    # Get CAN blacklist scene names
    blacklist = _get_can_blacklist_scene_names(nusc_can)

    # Create version-specific splits
    if version == 'v1.0-mini':
        splits = _create_mini_splits(official_splits, blacklist, nusc)
    else:  # v1.0-trainval
        splits = _create_trainval_splits(official_splits, blacklist, nusc)

    # Validate splits
    _validate_splits(splits)

    return splits


def _get_can_blacklist_scene_names(nusc_can: NuScenesCanBus) -> List[str]:
    """
    Convert CAN blacklist scene numbers to scene names.

    Args:
        nusc_can: NuScenesCanBus instance

    Returns:
        List of blacklisted scene names (e.g., ['scene-0161', 'scene-0162', ...])
    """
    blacklist_scene_names = [f"scene-{num:04d}" for num in nusc_can.can_blacklist]
    return blacklist_scene_names


def _count_frames_in_scenes(nusc: NuScenes, scene_names: List[str]) -> int:
    """
    Count total frames (samples) across given scenes.

    Args:
        nusc: NuScenes instance
        scene_names: List of scene names

    Returns:
        Total number of samples/frames

    Raises:
        ValueError: If a scene is not found in the dataset
    """
    # Create scene name to scene object mapping
    scene_map = {scene['name']: scene for scene in nusc.scene}

    total_frames = 0
    for scene_name in scene_names:
        scene = scene_map.get(scene_name)
        if scene is None:
            raise ValueError(f"Scene {scene_name} not found in dataset")
        total_frames += scene['nbr_samples']

    return total_frames


def _validate_splits(splits: Dict[str, Dict[str, any]]) -> None:
    """
    Validate that no scene appears in multiple splits.

    Args:
        splits: Dictionary of split definitions

    Raises:
        ValueError: If validation fails (scene in multiple splits or invalid frame count)
    """
    all_scenes = {}

    for split_name, split_info in splits.items():
        for scene_name in split_info['scenes']:
            if scene_name in all_scenes:
                raise ValueError(
                    f"Scene {scene_name} appears in both "
                    f"{all_scenes[scene_name]} and {split_name}"
                )
            all_scenes[scene_name] = split_name

    # Verify frame counts are positive
    for split_name, split_info in splits.items():
        if split_info['num_frames'] <= 0:
            raise ValueError(
                f"Split {split_name} has {split_info['num_frames']} frames"
            )


def _create_mini_splits(
    official_splits: Dict[str, List[str]],
    blacklist: List[str],
    nusc: NuScenes
) -> Dict[str, Dict[str, any]]:
    """
    Create smoke test splits from v1.0-mini.

    Args:
        official_splits: Official NuScenes split definitions
        blacklist: List of blacklisted scene names
        nusc: NuScenes instance

    Returns:
        Dictionary with smoke_train, smoke_val, smoke_test splits
    """
    # smoke_train: use mini_train as-is (8 scenes)
    smoke_train_scenes = official_splits['mini_train']

    # smoke_val + smoke_test: split mini_val deterministically (2 scenes → 1 each)
    mini_val_scenes = sorted(official_splits['mini_val'])  # Deterministic ordering

    # Save and restore random state to avoid side effects
    random_state = random.getstate()
    random.seed(42)
    random.shuffle(mini_val_scenes)
    random.setstate(random_state)

    # With 2 scenes: [0] goes to val, [1] goes to test
    smoke_val_scenes = [mini_val_scenes[0]]
    smoke_test_scenes = [mini_val_scenes[1]]

    return {
        'smoke_train': {
            'scenes': smoke_train_scenes,
            'num_scenes': len(smoke_train_scenes),
            'num_frames': _count_frames_in_scenes(nusc, smoke_train_scenes)
        },
        'smoke_val': {
            'scenes': smoke_val_scenes,
            'num_scenes': len(smoke_val_scenes),
            'num_frames': _count_frames_in_scenes(nusc, smoke_val_scenes)
        },
        'smoke_test': {
            'scenes': smoke_test_scenes,
            'num_scenes': len(smoke_test_scenes),
            'num_frames': _count_frames_in_scenes(nusc, smoke_test_scenes)
        }
    }


def _create_trainval_splits(
    official_splits: Dict[str, List[str]],
    blacklist: List[str],
    nusc: NuScenes
) -> Dict[str, Dict[str, any]]:
    """
    Create benchmark splits from v1.0-trainval.

    Args:
        official_splits: Official NuScenes split definitions
        blacklist: List of blacklisted scene names
        nusc: NuScenes instance

    Returns:
        Dictionary with train, internal_val, test splits
    """
    blacklist_set = set(blacklist)

    # Filter train scenes (remove CAN blacklist)
    train_scenes_filtered = [
        s for s in official_splits['train']
        if s not in blacklist_set
    ]

    # Save and restore random state to avoid side effects
    random_state = random.getstate()
    random.seed(42)
    random.shuffle(train_scenes_filtered)
    random.setstate(random_state)

    # Split 90/10
    split_idx = int(0.9 * len(train_scenes_filtered))
    train_scenes = train_scenes_filtered[:split_idx]
    internal_val_scenes = train_scenes_filtered[split_idx:]

    # Test: use official val after CAN filtering
    test_scenes = [
        s for s in official_splits['val']
        if s not in blacklist_set
    ]

    return {
        'train': {
            'scenes': train_scenes,
            'num_scenes': len(train_scenes),
            'num_frames': _count_frames_in_scenes(nusc, train_scenes)
        },
        'internal_val': {
            'scenes': internal_val_scenes,
            'num_scenes': len(internal_val_scenes),
            'num_frames': _count_frames_in_scenes(nusc, internal_val_scenes)
        },
        'test': {
            'scenes': test_scenes,
            'num_scenes': len(test_scenes),
            'num_frames': _count_frames_in_scenes(nusc, test_scenes)
        }
    }
