"""Shared helpers for DA7 fair DiT-vs-MLP evaluation.

Provides deterministic adapter reconstruction, normalization stats
computation, copy-baseline validation, and window-building utilities.
All functions mirror the exact conventions used in ``scripts/train_dit.py``
and ``scripts/rollout_dit.py`` so that MLP predictors trained via
``scripts/train_mlp_fair.py`` use the *same* frozen adapter and
normalization as DiT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

# Native embedding dimensions per encoder. Matches
# configs/canonical.yaml::encoders::*::output_dim_native and
# scripts/train_dit.py::NATIVE_DIMS.
NATIVE_DIMS: dict[str, int] = {
    "vit_s16": 384,
    "dino_vits14": 384,
    "clip_b32": 512,
    "vq_track": 256,
    "vjepa2_rep64": 1024,
    "vjepa2_rep1": 1024,
}

EMBED_DIR = Path("artifacts/full/embeddings")
ROLLOUT_RESULTS_PATH = Path("artifacts/full/rollout_results.json")
TARGET_DIM = 384
DEFAULT_HORIZON = 4


def load_embeddings(encoder_name: str) -> dict[str, np.ndarray]:
    """Load pre-computed embeddings from ``artifacts/full/embeddings/``.

    Returns dict with keys: embeddings, splits, steer_norms, accel_norms,
    scene_names.
    """
    path = EMBED_DIR / f"{encoder_name}.npz"
    with np.load(path, allow_pickle=True) as f:
        return {
            "embeddings": f["embeddings"],
            "splits": f["splits"],
            "steer_norms": f["steer_norms"],
            "accel_norms": f["accel_norms"],
            "scene_names": f["scene_names"],
        }


def build_windows(
    data: dict[str, np.ndarray],
    split: str,
    horizon: int = DEFAULT_HORIZON,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build temporal sliding windows for a given split.

    Groups frames by scene and creates (z_t, action, z_future) windows
    respecting scene boundaries. Mirrors the window logic in
    ``scripts/train_dit.py`` lines 299-326.

    Returns (z_t, actions, z_future) tensors, or None if the split is
    empty.
    """
    embeddings = data["embeddings"]
    splits = data["splits"]
    steer_norms = data["steer_norms"]
    accel_norms = data["accel_norms"]
    scene_names = data["scene_names"]

    mask = splits == split
    emb = embeddings[mask]
    steers = steer_norms[mask]
    accels = accel_norms[mask]
    scenes = scene_names[mask]

    z_t_list: list[np.ndarray] = []
    action_list: list[list[float]] = []
    z_future_list: list[np.ndarray] = []

    for scene in np.unique(scenes):
        scene_mask = scenes == scene
        idx = np.where(scene_mask)[0]
        n_scene = len(idx)
        for j in range(n_scene - horizon):
            t_idx = idx[j]
            future_idx = idx[j + 1 : j + 1 + horizon]
            z_t_list.append(emb[t_idx])
            action_list.append([steers[t_idx], accels[t_idx]])
            z_future_list.append(emb[future_idx])

    if not z_t_list:
        return None

    return (
        torch.tensor(np.array(z_t_list), dtype=torch.float32),
        torch.tensor(np.array(action_list), dtype=torch.float32),
        torch.tensor(np.array(z_future_list), dtype=torch.float32),
    )


def reconstruct_adapter_and_stats(
    encoder_name: str,
    seed: int,
    train_windows: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """Reconstruct frozen orthogonal adapter + z_mean/z_std.

    Mirrors ``scripts/train_dit.py`` lines 338-360 exactly:

    1. ``torch.manual_seed(seed)``
    2. ``nn.init.orthogonal_(adapter.weight)`` if needs_adapter
    3. Compute z_mean, z_std from ALL training tokens (z_t + z_future)
       after adapter projection.

    Returns (adapter, z_mean, z_std).
    """
    native_dim = NATIVE_DIMS[encoder_name]
    needs_adapter = native_dim != TARGET_DIM
    device = torch.device(device)

    torch.manual_seed(seed)
    # NOTE: np.random.seed is also called in train_dit.py at this point,
    # but it does not affect torch's RNG for orthogonal init.
    np.random.seed(seed)

    if needs_adapter:
        # Orthogonal init on CPU (MPS lacks linalg_qr), then move to device
        adapter: nn.Module = nn.Linear(native_dim, TARGET_DIM, bias=False)
        nn.init.orthogonal_(adapter.weight)
        adapter = adapter.to(device)
        for p in adapter.parameters():
            p.requires_grad_(False)
    else:
        adapter = nn.Identity().to(device)

    z_t_train, _, zf_train = train_windows

    with torch.no_grad():
        z_t_proj = adapter(z_t_train.to(device))  # (N, 384)
        B_tr, H_tr, _ = zf_train.shape
        zf_proj = adapter(
            zf_train.reshape(-1, zf_train.shape[-1]).to(device)
        )  # (N*H, 384)
        all_proj = torch.cat([z_t_proj, zf_proj], dim=0)
        z_mean = all_proj.mean(dim=0)  # (384,)
        z_std = all_proj.std(dim=0).clamp(min=1e-6)  # (384,)
        del z_t_proj, zf_proj, all_proj

    return adapter, z_mean, z_std


def validate_adapter_copy_baseline(
    encoder_name: str,
    seed: int,
    adapter: nn.Module,
    test_windows: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    atol: float = 5e-3,
    device: str | torch.device = "cpu",
) -> dict:
    """Abort gate: compare copy baseline against DiT rollout_results.json.

    Recomputes copy baseline CosSim(adapter(z_t), adapter(z_real_k)) in
    adapted *unnormalized* space on the test set, then checks that it
    matches the DiT values within ``atol``.

    Default atol is 5e-3 rather than 1e-5 because ``nn.init.orthogonal_``
    uses ``torch.linalg.qr`` whose output varies across torch versions
    and devices for wide matrices (out_features < in_features). DiT
    checkpoints were trained on Modal (torch 2.6, CUDA); local
    reconstruction may differ slightly for clip_b32 (512->384) and
    vjepa (1024->384). Identity-dim encoders and tall-matrix adapters
    (vq_track, 256->384) match within 1e-7.

    Also verifies that ``n_test_windows`` matches exactly.

    Returns structured dict for provenance logging.
    """
    import torch.nn.functional as F

    device = torch.device(device)

    # Load DiT reference results
    with ROLLOUT_RESULTS_PATH.open() as fh:
        dit_data = json.load(fh)

    # Find matching entries (both variants should have same copy baseline)
    dit_entries = [
        r for r in dit_data["results"]
        if r["encoder"] == encoder_name and r["seed"] == seed
    ]
    if not dit_entries:
        return {
            "passed": False,
            "error": f"No DiT results for {encoder_name}/seed={seed}",
        }

    z_t_test, _, zf_test = test_windows
    n_test = len(z_t_test)
    horizon = zf_test.shape[1]

    # Check n_test_windows
    dit_n_test = dit_entries[0]["n_test_windows"]
    if n_test != dit_n_test:
        return {
            "passed": False,
            "error": (
                f"n_test_windows mismatch: local={n_test}, "
                f"dit={dit_n_test}"
            ),
        }

    # Compute copy baseline in adapted unnormalized space
    with torch.no_grad():
        z_t_adapted = adapter(z_t_test.to(device))  # (N, 384)
        B, H, _ = zf_test.shape
        zf_adapted = adapter(
            zf_test.reshape(B * H, -1).to(device)
        ).reshape(B, H, TARGET_DIM)  # (N, H, 384)

        copy_cossim = []
        for k in range(horizon):
            cs = F.cosine_similarity(z_t_adapted, zf_adapted[:, k], dim=-1)
            copy_cossim.append(cs.mean().item())

    # Compare against both variant entries
    max_diff = 0.0
    per_horizon_diffs: list[dict] = []
    for entry in dit_entries:
        dit_copy = entry["metrics"]["copy_baseline_cossim"]
        for k in range(horizon):
            diff = abs(copy_cossim[k] - dit_copy[k])
            max_diff = max(max_diff, diff)
            per_horizon_diffs.append({
                "variant": entry["variant"],
                "k": k,
                "local": copy_cossim[k],
                "dit": dit_copy[k],
                "diff": diff,
            })

    passed = max_diff <= atol
    return {
        "passed": passed,
        "max_copy_baseline_diff": max_diff,
        "n_test_windows": n_test,
        "copy_cossim_local": copy_cossim,
        "atol": atol,
        "per_horizon_diffs": per_horizon_diffs,
    }


def diagnose_normalization(
    z_mean: torch.Tensor,
    z_std: torch.Tensor,
    adapter: nn.Module,
    train_windows: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: str | torch.device = "cpu",
) -> dict:
    """Warning gate: verify normalized train tokens have sane statistics.

    Normalizes all training tokens with adapter + z_mean/z_std, then
    reports mean/max of per-dimension mean and std-1 deviations.

    Small deviations may occur from floating-point precision and
    ``torch.std`` using Bessel's correction (``unbiased=True``, the
    default). Large deviations (>0.1) indicate a bug.

    Returns dict for provenance logging.
    """
    device = torch.device(device)
    z_t_train, _, zf_train = train_windows

    with torch.no_grad():
        z_t_proj = adapter(z_t_train.to(device))
        B, H, _ = zf_train.shape
        zf_proj = adapter(
            zf_train.reshape(-1, zf_train.shape[-1]).to(device)
        )
        all_proj = torch.cat([z_t_proj, zf_proj], dim=0)

        # Normalize
        all_norm = (all_proj - z_mean.to(device)) / z_std.to(device)

        per_dim_mean = all_norm.mean(dim=0)  # (384,)
        per_dim_std = all_norm.std(dim=0)  # (384,)

    mean_abs_mean = per_dim_mean.abs().mean().item()
    max_abs_mean = per_dim_mean.abs().max().item()
    mean_abs_std_dev = (per_dim_std - 1).abs().mean().item()
    max_abs_std_dev = (per_dim_std - 1).abs().max().item()

    has_nan = (
        torch.isnan(per_dim_mean).any().item()
        or torch.isnan(per_dim_std).any().item()
    )
    has_inf = (
        torch.isinf(per_dim_mean).any().item()
        or torch.isinf(per_dim_std).any().item()
    )

    # Abort on NaN/Inf or large deviation
    abort = has_nan or has_inf or max_abs_mean > 0.1 or max_abs_std_dev > 0.1

    return {
        "mean_abs_mean": mean_abs_mean,
        "max_abs_mean": max_abs_mean,
        "mean_abs_std_dev": mean_abs_std_dev,
        "max_abs_std_dev": max_abs_std_dev,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "abort": abort,
    }
