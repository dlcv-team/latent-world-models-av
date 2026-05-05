"""Image preprocessing utilities for nuScenes dataset.

All preprocessing follows the contract: images are loaded, resized to the
canonical size from ``configs/canonical.yaml``, and converted to ``[0, 1]``
float tensors. NO encoder-specific normalization (ImageNet mean/std, CLIP
mean/std) is applied here — each encoder wrapper handles that internally.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def load_and_preprocess_image(
    path: Path, target_size: tuple[int, int]
) -> torch.Tensor:
    """Load an image, resize it, and convert to [0, 1] float tensor.

    Parameters
    ----------
    path
        Path to the image file (JPEG or PNG).
    target_size
        (height, width) tuple for resizing. Typically (224, 224) from
        canonical config.

    Returns
    -------
    torch.Tensor
        Image tensor of shape ``(3, H, W)`` with values in ``[0, 1]``.

    Notes
    -----
    Uses ``torchvision.transforms.ToTensor()`` which automatically:
      1. Converts PIL Image (0-255) to float tensor (0.0-1.0)
      2. Permutes from (H, W, C) to (C, H, W)
    """
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
        ]
    )
    return transform(img)


def validate_tensor_range(tensor: torch.Tensor, name: str) -> None:
    """Assert that a tensor is in the [0, 1] range.

    Parameters
    ----------
    tensor
        Tensor to validate.
    name
        Name for error messages.

    Raises
    ------
    ValueError
        If tensor has values outside [0, 1].

    Notes
    -----
    Used for debugging and validation during development. Can be disabled
    in production by removing calls or wrapping in ``if __debug__:``.
    """
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    if min_val < 0.0 or max_val > 1.0:
        raise ValueError(
            f"{name} tensor range [{min_val:.3f}, {max_val:.3f}] outside [0, 1]"
        )
