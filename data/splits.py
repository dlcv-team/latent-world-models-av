"""Split generation for nuScenes dataset.

**For v1.0-mini (smoke tests):** Use `get_split()` for dynamic generation.
**For v1.0-trainval (benchmarks):** Use `get_split_from_canonical()` with canonical manifest.

All splits are deterministic and verified for no scene overlap.

Example usage:
    >>> from data.splits import get_split_from_canonical
    >>> # Benchmark splits (v1.0-trainval)
    >>> p0_train = get_split_from_canonical("p0_train")
    >>> len(p0_train)
    180
    >>>
    >>> # Smoke splits (v1.0-mini, internal dev only)
    >>> from data.splits import get_split
    >>> smoke_train_scenes = get_split("smoke_train", dataroot="data")
    >>> len(smoke_train_scenes)
    8
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

SplitName = Literal[
    "smoke_train",
    "smoke_val",
    "smoke_test",  # v1.0-mini only
]


def get_can_blacklist() -> list[str]:
    """Return list of CAN-blacklisted scene names.

    Returns
    -------
    list[str]
        Scene names (e.g., ``["scene-0161", "scene-0162", ...]``) for scenes
        without CAN bus data, pulled from the installed devkit version.

    Notes
    -----
    Reads from ``nuscenes.can_bus.can_bus_api.NuScenesCanBus.can_blacklist``.
    As of devkit 1.1.x, this includes scene IDs: 161-176 except 169, and 309-314.

    This function creates a minimal temporary directory structure to instantiate
    the devkit without requiring the full nuScenes dataset.
    """
    import shutil
    import tempfile

    # Create minimal structure to instantiate NuScenesCanBus
    tmpdir = Path(tempfile.mkdtemp())
    try:
        (tmpdir / "can_bus").mkdir()
        nusc_can = NuScenesCanBus(dataroot=str(tmpdir))
        return [f"scene-{scene_id:04d}" for scene_id in nusc_can.can_blacklist]
    finally:
        shutil.rmtree(tmpdir)


def filter_scenes_by_can(
    scene_names: list[str],
    nusc_can: NuScenesCanBus,
) -> tuple[list[str], list[str]]:
    """Filter scenes that have required CAN messages.

    Parameters
    ----------
    scene_names
        List of scene names to filter.
    nusc_can
        NuScenesCanBus instance for checking CAN message availability.

    Returns
    -------
    kept_scenes : list[str]
        Scenes with valid CAN data (steeranglefeedback and pose messages).
    dropped_scenes : list[str]
        Scenes without valid CAN data or in the blacklist.

    Notes
    -----
    A scene is kept if:
      1. Not in CAN blacklist
      2. Has "steeranglefeedback" messages
      3. Has "pose" messages
    """
    can_blacklist = get_can_blacklist()
    kept, dropped = [], []

    for scene_name in scene_names:
        if scene_name in can_blacklist:
            dropped.append(scene_name)
            continue

        try:
            # Check if required CAN messages exist
            nusc_can.get_messages(scene_name, "steeranglefeedback")
            nusc_can.get_messages(scene_name, "pose")
            kept.append(scene_name)
        except KeyError:
            # Scene doesn't have required CAN messages
            dropped.append(scene_name)

    return kept, dropped


def generate_mini_splits(
    nusc: NuScenes,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Generate smoke splits from v1.0-mini.

    Parameters
    ----------
    nusc
        NuScenes instance with ``version='v1.0-mini'``.
    seed
        Random seed for deterministic split of ``mini_val`` into val/test.

    Returns
    -------
    dict[str, list[str]]
        Keys: ``"smoke_train"``, ``"smoke_val"``, ``"smoke_test"``.
        Values: Lists of scene names.

    Notes
    -----
    - ``smoke_train``: 8 scenes from nuScenes official ``mini_train``
    - ``smoke_val``: 1 scene from nuScenes official ``mini_val`` (1st after shuffle)
    - ``smoke_test``: 1 scene from nuScenes official ``mini_val`` (2nd after shuffle)
    """
    if nusc.version != "v1.0-mini":
        raise ValueError(
            f"Expected v1.0-mini, got {nusc.version}. "
            "Use generate_trainval_splits() for v1.0-trainval."
        )

    official_splits = create_splits_scenes()
    smoke_train = official_splits["mini_train"]  # 8 scenes
    mini_val = official_splits["mini_val"]  # 2 scenes

    # Deterministically split mini_val into smoke_val (1 scene) and smoke_test (1 scene)
    rng = np.random.default_rng(seed)
    shuffled = list(mini_val)
    rng.shuffle(shuffled)

    smoke_val = shuffled[:1]
    smoke_test = shuffled[1:]

    return {
        "smoke_train": smoke_train,
        "smoke_val": smoke_val,
        "smoke_test": smoke_test,
    }




def verify_no_overlap(splits: dict[str, list[str]]) -> None:
    """Verify no scene appears in multiple splits.

    Parameters
    ----------
    splits
        Dictionary mapping split names to scene name lists.

    Raises
    ------
    ValueError
        If any scene appears in more than one split.

    Examples
    --------
    >>> splits = {"train": ["scene-0001"], "val": ["scene-0002"]}
    >>> verify_no_overlap(splits)  # OK, no overlap
    >>> splits = {"train": ["scene-0001"], "val": ["scene-0001"]}
    >>> verify_no_overlap(splits)  # Raises ValueError
    """
    all_scenes = []
    for split_name, scenes in splits.items():
        all_scenes.extend([(scene, split_name) for scene in scenes])

    scene_counts = {}
    for scene, split_name in all_scenes:
        if scene not in scene_counts:
            scene_counts[scene] = []
        scene_counts[scene].append(split_name)

    overlaps = {
        scene: split_list
        for scene, split_list in scene_counts.items()
        if len(split_list) > 1
    }
    if overlaps:
        raise ValueError(f"Scene overlap detected: {overlaps}")


def get_split_from_canonical(
    split_name: str,
    config_path: Path | str | None = None,
) -> list[str]:
    """Get scene names from canonical manifest (configs/trainval_subset_manifest.json).

    Parameters
    ----------
    split_name
        One of: ``"p0_train"``, ``"p0_val"``, ``"p0_test"``, ``"p1p2_scenes"``, ``"p0_all"``.
    config_path
        Optional path to canonical.yaml. Defaults to repo root config.

    Returns
    -------
    list[str]
        Scene names from the canonical manifest.

    Notes
    -----
    This uses the pre-computed, SHA256-verified manifest from the canonical config.
    Use this for reproducibility and alignment with the benchmark paper.

    For development/smoke testing with v1.0-mini, use :func:`get_split` instead.

    Examples
    --------
    >>> p0_train = get_split_from_canonical("p0_train")
    >>> len(p0_train)
    180
    """
    from config import load_canonical, manifest_split

    cfg = load_canonical(config_path)
    return manifest_split(cfg, split_name)


def get_split(
    split_name: SplitName,
    dataroot: str | Path,
    seed: int = 42,
) -> list[str]:
    """Get scene names for v1.0-mini smoke splits.

    For v1.0-trainval benchmark splits, use :func:`get_split_from_canonical` instead.

    Parameters
    ----------
    split_name
        One of: ``"smoke_train"``, ``"smoke_val"``, ``"smoke_test"`` (v1.0-mini only).
    dataroot
        Path to nuScenes dataset root (must contain ``v1.0-mini``).
    seed
        Random seed for deterministic splits (default 42).

    Returns
    -------
    list[str]
        Scene names for the requested split.

    Raises
    ------
    ValueError
        If split_name is invalid or scene overlap detected.
    FileNotFoundError
        If v1.0-mini not found at dataroot.

    Examples
    --------
    >>> smoke_train = get_split("smoke_train", dataroot="data")
    >>> len(smoke_train)
    8

    Notes
    -----
    Smoke splits are for rapid development iteration only. For benchmark results,
    use ``get_split_from_canonical("p0_train")`` which loads the canonical manifest.
    """
    dataroot = Path(dataroot)

    # v1.0-mini splits only
    if not (dataroot / "v1.0-mini").exists():
        raise FileNotFoundError(
            f"v1.0-mini not found at {dataroot}. "
            f"Smoke splits require v1.0-mini dataset."
        )

    nusc = NuScenes(version="v1.0-mini", dataroot=str(dataroot), verbose=False)
    splits = generate_mini_splits(nusc, seed)

    # Verify no overlap
    verify_no_overlap(splits)

    return splits[split_name]


def count_samples_per_split(
    nusc: NuScenes,
    splits: dict[str, list[str]],
) -> dict[str, dict[str, int]]:
    """Count total samples (keyframes) per split.

    Parameters
    ----------
    nusc
        NuScenes instance.
    splits
        Dictionary mapping split names to scene name lists.

    Returns
    -------
    dict[str, dict[str, int]]
        Nested dict: ``{split_name: {"scenes": count, "samples": count}}``.

    Examples
    --------
    >>> nusc = NuScenes(version="v1.0-mini", dataroot="data", verbose=False)
    >>> splits = generate_mini_splits(nusc, seed=42)
    >>> counts = count_samples_per_split(nusc, splits)
    >>> counts["smoke_train"]["scenes"]
    8
    >>> 300 < counts["smoke_train"]["samples"] < 350
    True
    """
    counts = {}
    for split_name, scene_names in splits.items():
        scene_count = len(scene_names)
        sample_count = 0

        for scene_name in scene_names:
            # Find scene record
            scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
            if scene is None:
                continue

            # Count samples (keyframes) in this scene
            sample_token = scene["first_sample_token"]
            while sample_token:
                sample_count += 1
                sample = nusc.get("sample", sample_token)
                sample_token = sample["next"]

        counts[split_name] = {"scenes": scene_count, "samples": sample_count}

    return counts


