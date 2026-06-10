"""Perturbation sensitivity analysis for encoder robustness evaluation.

Measures encoder sensitivity to spatial perturbations by comparing RMSE on
masked vs unmasked images. Three perturbation types:
  - mask_left_lane: Zero out leftmost ~33% of image (columns 0-74)
  - mask_right_lane: Zero out rightmost ~33% of image (columns 150-224)
  - mask_lead_vehicle: Mask out closest vehicle in front using nuScenes 2D boxes

Usage:
    python -m evaluation.perturbation
    python -m evaluation.perturbation --n-frames 100 --seed 123

Output:
    outputs/analysis/perturbation_sensitivity.csv with columns:
    encoder, perturbation, steering_drmse, steering_ci_lo, steering_ci_hi,
    accel_drmse, accel_ci_lo, accel_ci_hi

IMPORTANT - Checkpoint Compatibility:
    This module requires probe checkpoints with adapter_state_dict (added in
    commit 42f9632). Prior checkpoints have a bug: they trained the adapter
    but didn't save its weights, causing random adapter initialization on load.

    If using old checkpoints: Re-train all probes with the current train_probe.py.
    The code will warn if adapter_state_dict is missing but will proceed with
    random adapters, producing incorrect DRMSE values.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from tqdm import tqdm

from config import load_canonical
from evaluation.metrics import bootstrap_mean_ci, compute_rmse
from models.probe import ActionProbe
from scripts.train_probe import ENCODER_REGISTRY

logger = logging.getLogger(__name__)

# Image dimensions and perturbation regions
# (Ranges specified in PRD B10 spec, authoritative source for task B10)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
# Lane masks per PRD B10: left ~33% (cols 0-74), right ~33% (cols 150-224) of 224px width
LEFT_LANE_MASK_END = 75      # Columns 0-74 (33.5% from left edge)
RIGHT_LANE_MASK_START = 150  # Columns 150-224 (33.2% from right edge)

# Minimum valid frames for reliable bootstrap CIs
# (Below this threshold, a warning is logged)
MIN_FRAMES_FOR_RELIABLE_CI = 10


# ---------------------------------------------------------------------------
# Encoder and probe loading
# ---------------------------------------------------------------------------


def load_encoder_and_probe(
    encoder_name: str,
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Load trained encoder, adapter, and probe from checkpoint.

    Parameters
    ----------
    encoder_name
        Encoder name from ENCODER_REGISTRY (e.g., "vits16", "dinov2")
    checkpoint_dir
        Directory containing checkpoint.pt (e.g., outputs/probes/vit_s16/)
    device
        Torch device for model placement

    Returns
    -------
    tuple[nn.Module, nn.Module, nn.Module]
        (encoder, adapter, probe) tuple, all in eval mode on device

    Raises
    ------
    FileNotFoundError
        If checkpoint.pt not found at checkpoint_dir
    ValueError
        If encoder_name not in ENCODER_REGISTRY
    RuntimeError
        If checkpoint missing required keys

    Examples
    --------
    >>> encoder, adapter, probe = load_encoder_and_probe(
    ...     "vits16",
    ...     Path("outputs/probes/vit_s16"),
    ...     torch.device("cpu")
    ... )  # doctest: +SKIP
    """
    if encoder_name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder {encoder_name!r}; "
            f"choices are {sorted(ENCODER_REGISTRY)}"
        )

    spec = ENCODER_REGISTRY[encoder_name]
    checkpoint_path = checkpoint_dir / "checkpoint.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Train probe first with: python -m training.train_probe "
            f"--encoder {encoder_name}"
        )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "probe_state_dict" not in ckpt:
        raise RuntimeError(
            f"Checkpoint missing 'probe_state_dict': {checkpoint_path}"
        )

    # Build encoder (frozen backbone + adapter)
    module = importlib.import_module(spec.module_path)
    encoder_cls = getattr(module, spec.class_name)

    # Use same initialization pattern as train_probe.py
    if encoder_name == "clip":
        encoder = encoder_cls(pretrained="openai")
    else:
        encoder = encoder_cls(pretrained=True)

    encoder = encoder.to(device).eval()

    # Extract adapter from encoder
    # The adapter is encoder.adapter, but we need to handle it separately
    # for loading trained weights if they exist in checkpoint
    if ckpt.get("adapter_state_dict") is not None:
        # Adapter weights were saved - validate structure before loading
        adapter_state = ckpt["adapter_state_dict"]
        if not adapter_state:
            raise RuntimeError(
                f"adapter_state_dict is empty in checkpoint: {checkpoint_path}"
            )
        if "weight" not in adapter_state:
            raise RuntimeError(
                f"adapter_state_dict missing 'weight' key in checkpoint: {checkpoint_path}. "
                f"Found keys: {list(adapter_state.keys())}"
            )

        # Infer adapter dimensions from saved weights
        native_dim = adapter_state["weight"].shape[1]
        adapter = nn.Linear(native_dim, 384, bias=False)
        adapter.load_state_dict(adapter_state)
        adapter = adapter.to(device).eval()
        logger.info(
            f"Loaded trained adapter for {encoder_name} ({native_dim}→384)"
        )
    else:
        # No adapter weights in checkpoint - check if encoder requires one
        adapter = encoder.adapter

        # Dynamic check: if adapter is non-Identity and weights are missing, that's a problem
        if not isinstance(adapter, nn.Identity):
            raise RuntimeError(
                f"Checkpoint for {encoder_name} missing 'adapter_state_dict' "
                f"but encoder has non-Identity adapter ({type(adapter).__name__}). "
                f"This means the adapter was trained but not saved. "
                f"Retrain probe with: python -m training.train_probe --encoder {encoder_name}"
            )
        # OK: This encoder uses Identity adapter (no trained weights needed)
        # Ensure adapter is on correct device (encoder.adapter is already on device via encoder.to(device),
        # but we make it explicit for consistency with the checkpoint path above)
        adapter = adapter.to(device).eval()
        logger.info(f"Using Identity adapter for {encoder_name} (no projection needed)")

    # Build and load probe
    cfg = load_canonical()
    probe = ActionProbe.from_canonical()
    probe.load_state_dict(ckpt["probe_state_dict"])
    probe = probe.to(device).eval()

    logger.info(
        f"Loaded {encoder_name} (pilot={spec.pilot_name}): "
        f"encoder on {device}, probe with {sum(p.numel() for p in probe.parameters())} params"
    )

    return encoder, adapter, probe


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------


def select_validation_frames(
    dataset_length: int, n_frames: int, seed: int
) -> list[int]:
    """Select random validation frame indices for perturbation analysis.

    Uses uniform random sampling without stratification. For analyses requiring
    scenario-balanced sampling, use a different selection strategy.

    Parameters
    ----------
    dataset_length
        Total number of frames in validation dataset
    n_frames
        Number of frames to randomly sample
    seed
        Random seed for reproducibility

    Returns
    -------
    list[int]
        Sorted list of frame indices to use for evaluation

    Examples
    --------
    >>> select_validation_frames(805, 50, 42)  # doctest: +SKIP
    [3, 12, 15, 27, ...]
    """
    if n_frames > dataset_length:
        raise ValueError(
            f"Cannot select {n_frames} frames from dataset with "
            f"{dataset_length} samples"
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(dataset_length, size=n_frames, replace=False).tolist()
    return sorted(indices)


# ---------------------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------------------


def get_lead_vehicle_bbox_2d(
    nusc: NuScenes,
    sample_token: str,
    image_width: int = 224,
    image_height: int = 224,
) -> Optional[tuple[int, int, int, int]]:
    """Get 2D bounding box of the closest vehicle in front of ego.

    Parameters
    ----------
    nusc
        NuScenes instance
    sample_token
        Sample token identifying the frame
    image_width
        Target image width (for clipping bbox coords)
    image_height
        Target image height (for clipping bbox coords)

    Returns
    -------
    tuple[int, int, int, int] or None
        2D bbox as (x1, y1, x2, y2) clipped to [0, image_width/height].
        Returns None if no vehicle found in front.

    Examples
    --------
    >>> nusc = NuScenes(version='v1.0-mini', dataroot='/data/nuscenes')  # doctest: +SKIP
    >>> bbox = get_lead_vehicle_bbox_2d(nusc, sample_token="abc123...")  # doctest: +SKIP
    >>> if bbox:  # doctest: +SKIP
    ...     x1, y1, x2, y2 = bbox  # doctest: +SKIP
    """
    # Get sample and CAM_FRONT data
    sample = nusc.get("sample", sample_token)
    cam_token = sample["data"]["CAM_FRONT"]
    cam_data = nusc.get("sample_data", cam_token)

    # Get camera calibration
    calib_token = cam_data["calibrated_sensor_token"]
    calib = nusc.get("calibrated_sensor", calib_token)
    camera_intrinsic = np.array(calib["camera_intrinsic"])

    # Get ego pose
    ego_pose_token = cam_data["ego_pose_token"]
    ego_pose = nusc.get("ego_pose", ego_pose_token)

    # Find vehicles in front
    vehicles_in_front = []

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)

        # Filter to vehicles
        if "vehicle" not in ann["category_name"]:
            continue

        # Get 3D center in global frame
        center_global = np.array(ann["translation"])

        # Transform to ego vehicle frame (nuScenes convention: x-forward, y-left, z-up)
        ego_translation = np.array(ego_pose["translation"])
        ego_rotation = np.array(ego_pose["rotation"])  # quaternion

        # Inverse transform: global → ego
        q_ego = Quaternion(ego_rotation)
        center_ego = q_ego.inverse.rotate(center_global - ego_translation)

        # Check if in front (positive x in ego frame)
        # Note: Later projection to camera frame (line ~330) uses camera coordinates
        # where z-forward, but this ego-frame filtering is correct for x-forward
        if center_ego[0] <= 0:
            continue

        # Compute distance
        distance = np.linalg.norm(center_ego)

        vehicles_in_front.append(
            {"ann_token": ann_token, "distance": distance, "center_ego": center_ego}
        )

    if len(vehicles_in_front) == 0:
        return None

    # Find closest vehicle
    closest = min(vehicles_in_front, key=lambda v: v["distance"])
    ann = nusc.get("sample_annotation", closest["ann_token"])

    # Get 3D bounding box corners in global frame
    box = Box(
        center=ann["translation"],
        size=ann["size"],
        orientation=Quaternion(ann["rotation"]),
    )

    # Transform box to camera frame
    # First to ego frame
    q_ego = Quaternion(ego_pose["rotation"])
    box.translate(-np.array(ego_pose["translation"]))
    box.rotate(q_ego.inverse)

    # Then to camera frame
    q_cam = Quaternion(calib["rotation"])
    box.translate(-np.array(calib["translation"]))
    box.rotate(q_cam.inverse)

    # Get 8 corners of 3D box
    corners_3d = box.corners()  # (3, 8)

    # Project to 2D
    corners_2d = view_points(
        corners_3d, camera_intrinsic, normalize=True
    )  # (3, 8)

    # Extract x, y coordinates (first 2 rows)
    x_coords = corners_2d[0, :]
    y_coords = corners_2d[1, :]

    # Compute 2D bounding box
    x1 = int(np.floor(x_coords.min()))
    y1 = int(np.floor(y_coords.min()))
    x2 = int(np.ceil(x_coords.max()))
    y2 = int(np.ceil(y_coords.max()))

    # Clip to image bounds
    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    # Check if bbox is valid (non-zero area)
    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def apply_perturbation(
    image: torch.Tensor,
    perturbation_type: str,
    nusc: Optional[NuScenes] = None,
    sample_token: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """Apply spatial perturbation to image tensor.

    Parameters
    ----------
    image
        Image tensor of shape (3, 224, 224) for single frame or
        (16, 3, 224, 224) for V-JEPA2 clips. Values in [0, 1].
    perturbation_type
        One of: "mask_left_lane", "mask_right_lane", "mask_lead_vehicle"
    nusc
        NuScenes instance (required only for mask_lead_vehicle)
    sample_token
        Sample token (required only for mask_lead_vehicle)

    Returns
    -------
    torch.Tensor or None
        Perturbed image with same shape as input. Returns None if
        perturbation_type is "mask_lead_vehicle" and no lead vehicle found.

    Examples
    --------
    >>> img = torch.rand(3, 224, 224)
    >>> masked = apply_perturbation(img, "mask_left_lane")
    >>> masked.shape
    torch.Size([3, 224, 224])
    >>> masked[:, :, 0:75].sum()  # Left columns zeroed
    tensor(0.)
    """
    # Clone to avoid modifying original
    perturbed = image.clone()

    if perturbation_type == "mask_left_lane":
        # Zero out columns 0-74 per PRD B10 spec
        perturbed[..., :LEFT_LANE_MASK_END] = 0

    elif perturbation_type == "mask_right_lane":
        # Zero out columns 150-224 per PRD B10 spec
        perturbed[..., RIGHT_LANE_MASK_START:IMAGE_WIDTH] = 0

    elif perturbation_type == "mask_lead_vehicle":
        if nusc is None or sample_token is None:
            raise ValueError(
                "mask_lead_vehicle requires nusc and sample_token arguments"
            )

        # Get 2D bounding box of closest vehicle
        bbox = get_lead_vehicle_bbox_2d(nusc, sample_token)

        if bbox is None:
            # No lead vehicle found
            return None

        # Zero out bounding box region
        x1, y1, x2, y2 = bbox

        # Handle both single-frame and clip tensors
        if image.ndim == 3:
            # Single frame: (3, 224, 224)
            perturbed[:, y1:y2, x1:x2] = 0
        elif image.ndim == 4:
            # Clip: (16, 3, 224, 224)
            perturbed[:, :, y1:y2, x1:x2] = 0
        else:
            raise ValueError(
                f"Unexpected image shape: {image.shape}. "
                f"Expected (3, 224, 224) or (16, 3, 224, 224)"
            )

    else:
        raise ValueError(
            f"Unknown perturbation_type: {perturbation_type}. "
            f"Must be one of: mask_left_lane, mask_right_lane, "
            f"mask_lead_vehicle"
        )

    return perturbed


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------


def evaluate_frame(
    encoder: nn.Module,
    adapter: nn.Module,
    probe: nn.Module,
    image: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """Run inference on a single frame and compute RMSE vs ground truth.

    Parameters
    ----------
    encoder
        Frozen encoder model (already on device, in eval mode)
    adapter
        Projection adapter (Identity or Linear, already on device)
    probe
        Action prediction probe (already on device, in eval mode)
    image
        Image tensor (3, 224, 224) or (16, 3, 224, 224) for clips.
        Values in [0, 1].
    actions
        Ground truth actions, shape (2,) → [steering_norm, accel_norm]
    device
        Torch device for computation

    Returns
    -------
    tuple[float, float]
        (steer_rmse, accel_rmse) in normalized space

    Notes
    -----
    This function computes RMSE on N=1 sample, which reduces to absolute error:
    RMSE = sqrt(mean((pred - target)^2)) = |pred - target| when N=1.
    The "RMSE" terminology is kept for consistency with compute_rmse(), but
    note that per-frame values are absolute errors. Statistical aggregation
    happens downstream via bootstrap in compute_drmse_with_ci(), which computes
    the mean and CI of per-frame DRMSE values across many frames.

    Examples
    --------
    >>> encoder = ...  # doctest: +SKIP
    >>> adapter = nn.Identity()  # doctest: +SKIP
    >>> probe = ...  # doctest: +SKIP
    >>> image = torch.rand(3, 224, 224)  # doctest: +SKIP
    >>> actions = torch.tensor([0.1, -0.3])  # doctest: +SKIP
    >>> steer_rmse, accel_rmse = evaluate_frame(
    ...     encoder, adapter, probe, image, actions, torch.device("cpu")
    ... )  # doctest: +SKIP
    """
    with torch.no_grad():
        # Add batch dimension and move to device
        image_batch = image.unsqueeze(0).to(device)

        # Forward pass: encoder → adapter → probe
        embedding = encoder(image_batch)
        projected = adapter(embedding)
        pred = probe(projected).cpu().numpy()  # (1, 2)

    # Compute RMSE vs ground truth
    actions_np = actions.cpu().numpy().reshape(1, 2)
    steer_rmse, accel_rmse = compute_rmse(pred, actions_np)

    return steer_rmse, accel_rmse


def compute_frame_drmse(
    encoder: nn.Module,
    adapter: nn.Module,
    probe: nn.Module,
    image_unmasked: torch.Tensor,
    image_masked: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """Compute delta RMSE between masked and unmasked predictions.

    Parameters
    ----------
    encoder, adapter, probe
        Model components (already on device, in eval mode)
    image_unmasked
        Original unperturbed image, shape (3, 224, 224) or (16, 3, 224, 224)
    image_masked
        Perturbed image with same shape
    actions
        Ground truth actions, shape (2,)
    device
        Torch device

    Returns
    -------
    tuple[float, float]
        (steering_drmse, accel_drmse) = RMSE_masked - RMSE_unmasked

    Notes
    -----
    Per-frame DRMSE values are deltas of absolute errors (not true RMSE, since
    N=1 in evaluate_frame). The bootstrap aggregation in compute_drmse_with_ci()
    provides the statistical summary: mean DRMSE and 95% CI across all frames.
    Positive DRMSE → masking degrades predictions (region is important).
    Negative DRMSE → masking improves predictions (region is distracting).

    Examples
    --------
    >>> # Positive DRMSE means masking increases error (region is important)
    >>> drmse_steer, drmse_accel = compute_frame_drmse(...)  # doctest: +SKIP
    """
    # Evaluate unmasked
    steer_rmse_unmasked, accel_rmse_unmasked = evaluate_frame(
        encoder, adapter, probe, image_unmasked, actions, device
    )

    # Evaluate masked
    steer_rmse_masked, accel_rmse_masked = evaluate_frame(
        encoder, adapter, probe, image_masked, actions, device
    )

    # Compute delta RMSE
    steering_drmse = steer_rmse_masked - steer_rmse_unmasked
    accel_drmse = accel_rmse_masked - accel_rmse_unmasked

    return steering_drmse, accel_drmse


def compute_drmse_with_ci(
    drmse_values: list[float],
    cfg: Optional[dict] = None,
) -> tuple[float, float, float]:
    """Compute mean DRMSE and bootstrap confidence interval.

    Parameters
    ----------
    drmse_values
        List of per-frame DRMSE values (RMSE_masked - RMSE_unmasked)
    cfg
        Canonical config dict with bootstrap settings. If None, loads from
        configs/canonical.yaml.

    Returns
    -------
    tuple[float, float, float]
        (mean_drmse, ci_lo, ci_hi) using bootstrap resampling

    Raises
    ------
    ValueError
        If drmse_values is empty

    Examples
    --------
    >>> drmse_values = [0.01, 0.02, 0.015, 0.018]  # doctest: +SKIP
    >>> mean, ci_lo, ci_hi = compute_drmse_with_ci(drmse_values)  # doctest: +SKIP
    >>> print(f"DRMSE: {mean:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")  # doctest: +SKIP
    """
    if len(drmse_values) == 0:
        raise ValueError("Cannot compute CI on empty DRMSE list")

    # Load bootstrap settings from canonical config
    if cfg is None:
        cfg = load_canonical()

    if hasattr(cfg, "raw"):
        # Config object from load_canonical()
        bootstrap_cfg = cfg.raw["evaluation"]["bootstrap"]
    else:
        # Plain dict (for testing)
        bootstrap_cfg = cfg["evaluation"]["bootstrap"]

    n_resamples = bootstrap_cfg["n_resamples"]
    seed = bootstrap_cfg["seed"]
    confidence_level = bootstrap_cfg["confidence_level"]

    # Convert to numpy array
    drmse_array = np.array(drmse_values, dtype=float)

    # Handle edge case: all values identical (std ≈ 0)
    # Use tolerance to avoid exact float comparison
    if np.std(drmse_array) < 1e-12:
        mean_val = float(drmse_array[0])
        logger.warning(
            f"All DRMSE values identical ({mean_val:.6f}). "
            f"Setting CIs to mean value."
        )
        return mean_val, mean_val, mean_val

    # Bootstrap the mean
    mean_drmse, ci_lo, ci_hi = bootstrap_mean_ci(
        drmse_array, n_resamples, seed, confidence_level
    )

    return mean_drmse, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Main perturbation analysis orchestration
# ---------------------------------------------------------------------------


def run_perturbation_analysis(
    n_frames: int = 50,
    seed: Optional[int] = None,
    output_dir: Path = Path("outputs/analysis"),
    device: Optional[torch.device] = None,
    encoder_names: Optional[list[str]] = None,
    split: str = "p0_val",
) -> Path:
    """Run perturbation sensitivity analysis across all encoders and perturbations.

    Parameters
    ----------
    n_frames
        Number of random validation frames to evaluate
    seed
        Random seed for frame selection and bootstrap CIs. If None, uses
        canonical global_seed from configs/canonical.yaml.
    output_dir
        Directory for output CSV (will be created if doesn't exist)
    device
        Torch device. If None, auto-selects: cuda > mps > cpu
    encoder_names
        List of encoder names to evaluate (registry keys from ENCODER_REGISTRY).
        If None, evaluates all registered encoders.
    split
        Dataset split to use (e.g., "p0_val", "smoke_val").
        Default is "p0_val" for canonical benchmark.

    Returns
    -------
    Path
        Path to saved CSV file

    Examples
    --------
    >>> csv_path = run_perturbation_analysis(n_frames=50)  # Uses canonical seed  # doctest: +SKIP
    >>> import pandas as pd  # doctest: +SKIP
    >>> df = pd.read_csv(csv_path)  # doctest: +SKIP
    >>> df.shape  # doctest: +SKIP
    (15, 8)  # 5 encoders × 3 perturbations
    """
    import pandas as pd
    from data.dataset import NuScenesFrameDataset

    # Load canonical config
    cfg = load_canonical()

    # Use canonical seed if not provided
    if seed is None:
        seed = cfg.global_seed

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Validate encoder names early (before expensive dataset operations)
    if encoder_names is None:
        encoder_names = sorted(ENCODER_REGISTRY.keys())
    else:
        # Validate provided encoder names
        invalid = set(encoder_names) - set(ENCODER_REGISTRY.keys())
        if invalid:
            raise ValueError(
                f"Invalid encoder names: {invalid}. "
                f"Valid options: {sorted(ENCODER_REGISTRY.keys())}"
            )

    logger.info(
        f"Starting perturbation analysis: {n_frames} frames, seed={seed}, device={device}"
    )

    # Initialize nuScenes API for lead vehicle masking
    import os

    nuscenes_dataroot = Path(os.getenv("NUSCENES_DATAROOT", "data/nuscenes"))

    # Auto-detect nuScenes version from split prefix (same logic as NuScenesFrameDataset)
    if split.startswith("smoke_"):
        dataset_version = "v1.0-mini"
    else:
        dataset_version = cfg.raw["dataset"]["version"]

    # Validate version directory exists before initializing NuScenes API
    version_dir = nuscenes_dataroot / dataset_version
    if not version_dir.exists():
        raise FileNotFoundError(
            f"NuScenes version '{dataset_version}' not found at {nuscenes_dataroot}. "
            f"Expected directory: {version_dir}\n"
            f"Download from https://nuscenes.org or set NUSCENES_DATAROOT environment variable."
        )

    logger.info(f"Initializing NuScenes API: {dataset_version} @ {nuscenes_dataroot}")
    nusc = NuScenes(version=dataset_version, dataroot=str(nuscenes_dataroot), verbose=False)

    # Results accumulator
    results = []

    # Perturbation types
    perturbations = ["mask_left_lane", "mask_right_lane", "mask_lead_vehicle"]

    for encoder_name in encoder_names:
        logger.info(f"Evaluating encoder: {encoder_name}")

        # Load encoder and probe
        spec = ENCODER_REGISTRY[encoder_name]
        checkpoint_dir = Path("outputs/probes") / spec.pilot_name

        try:
            encoder, adapter, probe = load_encoder_and_probe(
                encoder_name, checkpoint_dir, device
            )
        except FileNotFoundError as e:
            logger.error(
                f"Skipping {encoder_name}: checkpoint not found. "
                f"Train probe first with: python -m training.train_probe --encoder {encoder_name}"
            )
            continue

        # Load dataset (handle V-JEPA2 multi-frame vs single-frame)
        if encoder_name == "vjepa2":
            dataset = NuScenesFrameDataset(
                split=split, mode="clip", clip_frames=16
            )
        else:
            dataset = NuScenesFrameDataset(split=split, mode="single_frame")

        # Select random validation frames
        frame_indices = select_validation_frames(len(dataset), n_frames, seed)
        logger.info(
            f"Selected {len(frame_indices)} frames from {len(dataset)} total"
        )

        # Evaluate each perturbation type
        for perturbation_type in perturbations:
            logger.info(f"  Perturbation: {perturbation_type}")

            # Accumulate per-frame DRMSE values
            steering_drmse_list = []
            accel_drmse_list = []

            skipped_frames = 0

            # Progress bar for frame processing (disabled if logging level > INFO)
            for idx in tqdm(
                frame_indices,
                desc=f"{spec.pilot_name}/{perturbation_type}",
                leave=False,
                disable=logger.level > logging.INFO,
            ):
                # Load frame
                sample = dataset[idx]
                image = sample["image"]
                actions = sample["actions"]
                sample_token = sample["sample_token"]

                # Apply perturbation
                image_masked = apply_perturbation(
                    image, perturbation_type, nusc=nusc, sample_token=sample_token
                )

                # Skip frame if perturbation returns None (no lead vehicle)
                if image_masked is None:
                    skipped_frames += 1
                    continue

                # Compute DRMSE for this frame
                steering_drmse, accel_drmse = compute_frame_drmse(
                    encoder, adapter, probe, image, image_masked, actions, device
                )

                steering_drmse_list.append(steering_drmse)
                accel_drmse_list.append(accel_drmse)

            # Log skipped frames
            if skipped_frames > 0:
                logger.warning(
                    f"  Skipped {skipped_frames}/{n_frames} frames "
                    f"(perturbation returned None)"
                )

            # Validate minimum frame count for reliable statistics
            n_valid = len(steering_drmse_list)

            if n_valid == 0:
                logger.error(
                    f"  No valid frames for {encoder_name} + {perturbation_type}. Skipping."
                )
                continue
            elif n_valid < MIN_FRAMES_FOR_CI:
                logger.warning(
                    f"  Only {n_valid}/{n_frames} valid frames for {encoder_name} + "
                    f"{perturbation_type}. CIs may be unreliable (recommend >={MIN_FRAMES_FOR_CI})."
                )

            steer_mean, steer_ci_lo, steer_ci_hi = compute_drmse_with_ci(
                steering_drmse_list, cfg
            )
            accel_mean, accel_ci_lo, accel_ci_hi = compute_drmse_with_ci(
                accel_drmse_list, cfg
            )

            logger.info(
                f"    Steering DRMSE: {steer_mean:.4f} [{steer_ci_lo:.4f}, {steer_ci_hi:.4f}]"
            )
            logger.info(
                f"    Accel DRMSE:    {accel_mean:.4f} [{accel_ci_lo:.4f}, {accel_ci_hi:.4f}]"
            )

            # Append to results
            results.append(
                {
                    "encoder": spec.pilot_name,  # Use pilot_name for CSV (e.g., "vit_s16")
                    "perturbation": perturbation_type,
                    "steering_drmse": steer_mean,
                    "steering_ci_lo": steer_ci_lo,
                    "steering_ci_hi": steer_ci_hi,
                    "accel_drmse": accel_mean,
                    "accel_ci_lo": accel_ci_lo,
                    "accel_ci_hi": accel_ci_hi,
                }
            )

        # Free encoder/probe memory
        del encoder, adapter, probe
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        logger.info(f"Freed memory for {encoder_name}")

    # Save results to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "perturbation_sensitivity.csv"

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    logger.info(f"Saved results to {csv_path}")
    logger.info(f"Total rows: {len(df)} (expected: {len(encoder_names) * len(perturbations)} if all encoders trained)")

    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for perturbation sensitivity analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Perturbation sensitivity analysis for encoder robustness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (50 frames, seed 42)
  python -m evaluation.perturbation

  # Use 100 frames with different seed
  python -m evaluation.perturbation --n-frames 100 --seed 123

  # Custom output directory
  python -m evaluation.perturbation --output-dir results/perturbation

Environment variables:
  NUSCENES_DATAROOT: Path to nuScenes dataset (default: data/nuscenes)

Output:
  CSV file at <output-dir>/perturbation_sensitivity.csv with columns:
    encoder, perturbation, steering_drmse, steering_ci_lo, steering_ci_hi,
    accel_drmse, accel_ci_lo, accel_ci_hi

  15 rows total (5 encoders × 3 perturbations)
        """,
    )

    parser.add_argument(
        "--n-frames",
        type=int,
        default=50,
        help="Number of random validation frames to evaluate (default: 50)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for frame selection and bootstrap CIs (default: canonical global_seed)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis"),
        help="Output directory for CSV file (default: outputs/analysis)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Torch device (default: cuda if available, else cpu)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse device
    device = None
    if args.device is not None:
        device = torch.device(args.device)

    # Run analysis
    logger.info("=" * 70)
    logger.info("Perturbation Sensitivity Analysis")
    logger.info("=" * 70)

    csv_path = run_perturbation_analysis(
        n_frames=args.n_frames,
        seed=args.seed,
        output_dir=args.output_dir,
        device=device,
    )

    logger.info("=" * 70)
    logger.info(f"✓ Analysis complete! Results saved to: {csv_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
