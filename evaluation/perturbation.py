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

import importlib
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from nuscenes.nuscenes import NuScenes

from config import load_canonical
from evaluation.metrics import compute_rmse
from models.probe import ActionProbe
from training.train_probe import ENCODER_REGISTRY

logger = logging.getLogger(__name__)


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

    if encoder_name == "clip":
        encoder = encoder_cls(pretrained="openai")
    else:
        encoder = encoder_cls(pretrained=True)

    encoder = encoder.to(device).eval()

    # Extract adapter from encoder
    # The adapter is encoder.adapter, but we need to handle it separately
    # for loading trained weights if they exist in checkpoint
    if ckpt.get("adapter_state_dict") is not None:
        # Adapter weights were saved (from train_probes_full.py or patched train_probe.py)
        native_dim = list(ckpt["adapter_state_dict"].values())[0].shape[1]
        adapter = nn.Linear(native_dim, 384, bias=False)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        adapter = adapter.to(device).eval()
        logger.info(
            f"Loaded trained adapter for {encoder_name} ({native_dim}→384)"
        )
    else:
        # No adapter weights in checkpoint - use encoder's built-in adapter
        # This happens with current training/train_probe.py
        adapter = encoder.adapter
        if not isinstance(adapter, nn.Identity):
            logger.warning(
                f"No adapter_state_dict in checkpoint for {encoder_name}. "
                f"Using randomly initialized adapter. This may reduce accuracy."
            )

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
