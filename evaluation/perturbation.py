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
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------


def select_validation_frames(
    dataset_length: int, n_frames: int, seed: int
) -> list[int]:
    """Select random validation frame indices for perturbation analysis.

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

    rng = random.Random(seed)
    indices = rng.sample(range(dataset_length), n_frames)
    return sorted(indices)


# ---------------------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------------------


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
        # Zero out columns 0-74 (leftmost ~33%)
        perturbed[..., 0:75] = 0

    elif perturbation_type == "mask_right_lane":
        # Zero out columns 150-224 (rightmost ~33%)
        perturbed[..., 150:224] = 0

    elif perturbation_type == "mask_lead_vehicle":
        # TODO: Implement in commit 6
        # For now, return None to indicate "skip this frame"
        logger.warning(
            "mask_lead_vehicle not yet implemented, skipping frame"
        )
        return None

    else:
        raise ValueError(
            f"Unknown perturbation_type: {perturbation_type}. "
            f"Must be one of: mask_left_lane, mask_right_lane, "
            f"mask_lead_vehicle"
        )

    return perturbed
