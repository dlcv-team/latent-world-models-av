"""Transparent loader for z_hat and z_real tensors with HuggingFace fallback.

Member 3's C4 evaluation code can load tensors without knowing whether
they come from local disk or HuggingFace::

    from data.z_hat import load_z_hat, load_z_real
    z_hat_cond   = load_z_hat("conditioned")              # (N_test, 4, 384)
    z_hat_uncond = load_z_hat("unconditioned")
    z_real_cond  = load_z_real(variant="conditioned")      # (N_test, 4, 384)
    z_real_uncond = load_z_real(variant="unconditioned")

Each variant has its own adapter projection, so ``z_real`` must match
the variant being evaluated.  ``load_z_real`` requires an explicit
``variant`` argument to enforce this.

Download cascade (same pattern as ``data/embeddings.py``):
  1. Local ``outputs/z_hat/``
  2. HuggingFace Hub ``surlac/lwm-av-embeddings``
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DIR = _PROJECT_ROOT / "outputs" / "z_hat"


def get_default_dir() -> Path:
    """Return the default directory for z_hat / z_real tensor storage."""
    return _DEFAULT_DIR


_HF_REPO = "surlac/lwm-av-embeddings"

_VALID_VARIANTS = ("conditioned", "unconditioned")


def _download_from_hf(filename: str, cache_dir: Path) -> Path:
    """Download a single .pt file from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading z_hat tensors. "
            "Install with: pip install huggingface_hub"
        )

    local_path = hf_hub_download(
        repo_id=_HF_REPO,
        filename=filename,
        repo_type="dataset",
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
    )
    return Path(local_path)


def load_z_hat(
    variant: str = "conditioned",
    directory: Optional[Path] = None,
) -> torch.Tensor:
    """Load predicted future latents from a trained latent predictor.

    Parameters
    ----------
    variant
        ``"conditioned"`` or ``"unconditioned"``.
    directory
        Override local directory. Defaults to ``outputs/z_hat/``.

    Returns
    -------
    torch.Tensor
        ``(N_test, horizon, z_dim)`` predicted latent states.
    """
    if variant not in _VALID_VARIANTS:
        raise ValueError(f"variant must be one of {_VALID_VARIANTS}, got {variant!r}")

    filename = f"z_hat_{variant}.pt"
    d = Path(directory) if directory else _DEFAULT_DIR
    local_path = d / filename

    if not local_path.exists() and directory is None:
        print(f"[z_hat] {filename} not found locally, downloading from HF ...")
        try:
            local_path = _download_from_hf(filename, _DEFAULT_DIR)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load {filename}. "
                f"Not found at {local_path} and HF download failed: {e}\n"
                f"Generate with: python scripts/export_z_hat.py --encoder vjepa2_rep64\n"
                f"Or download from: https://huggingface.co/datasets/{_HF_REPO}"
            ) from e

    return torch.load(local_path, map_location="cpu", weights_only=True)


def load_z_real(
    variant: str = "conditioned",
    directory: Optional[Path] = None,
) -> torch.Tensor:
    """Load real future latents (ground truth) from the frozen encoder.

    Each variant (conditioned / unconditional) was trained with its own
    adapter projection, so the z_real subspace differs per variant.
    Always pass the matching variant to get correct CosSim results.

    Parameters
    ----------
    variant
        ``"conditioned"`` or ``"unconditioned"``.  Must match the
        variant whose ``z_hat`` you are evaluating against.
    directory
        Override local directory. Defaults to ``outputs/z_hat/``.

    Returns
    -------
    torch.Tensor
        ``(N_test, horizon, z_dim)`` real encoder embeddings.
    """
    if variant not in _VALID_VARIANTS:
        raise ValueError(f"variant must be one of {_VALID_VARIANTS}, got {variant!r}")

    filename = f"z_real_{variant}.pt"
    d = Path(directory) if directory else _DEFAULT_DIR
    local_path = d / filename

    if not local_path.exists() and directory is None:
        print(f"[z_hat] {filename} not found locally, downloading from HF ...")
        try:
            local_path = _download_from_hf(filename, _DEFAULT_DIR)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load {filename}. "
                f"Not found at {local_path} and HF download failed: {e}\n"
                f"Generate with: python scripts/export_z_hat.py --encoder vjepa2_rep64\n"
                f"Or download from: https://huggingface.co/datasets/{_HF_REPO}"
            ) from e

    return torch.load(local_path, map_location="cpu", weights_only=True)
