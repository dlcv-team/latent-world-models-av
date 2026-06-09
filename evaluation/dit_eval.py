"""DA7/DA7.5: Fair model comparison and ablation evaluation.

Evaluates MLP and DiT-direct predictors on the test set using the same
adapter/normalization as DiT-DDIM, then produces a unified comparison
table with up to 4 models: copy_baseline, dit_direct, mlp, dit (DDIM).

Output artifacts:
  - ``artifacts/full/mlp_rollout_results.json`` -- raw MLP results
  - ``artifacts/full/dit_direct_rollout_results.json`` -- raw DiT-direct
  - ``artifacts/full/dit_vs_mlp_comparison.csv`` -- unified table
  - ``artifacts/full/dit_vs_mlp_table.tex`` -- LaTeX for report
  - ``artifacts/full/mlp_by_difficulty.csv`` -- hard-subset analysis

Usage
-----
    python -m evaluation.dit_eval
    python -m evaluation.dit_eval --dit-direct-lr 0.001
    python -m evaluation.dit_eval --difficulty
    python -m evaluation.dit_eval --encoders vit_s16 clip_b32
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from evaluation.dit_utils import (
    DEFAULT_HORIZON,
    NATIVE_DIMS,
    ROLLOUT_RESULTS_PATH,
    TARGET_DIM,
    build_windows,
    load_embeddings,
)
from models.fourier_embed import FourierActionEmbedding
from models.latent_pred import LatentPredictor

MLP_CKPT_ROOT = Path("outputs/latent_predictors_fair")
DIT_DIRECT_CKPT_ROOT = Path("outputs/dit_direct")
MLP_RESULTS_PATH = Path("artifacts/full/mlp_rollout_results.json")
DIT_DIRECT_RESULTS_PATH = Path("artifacts/full/dit_direct_rollout_results.json")
COMPARISON_CSV_PATH = Path("artifacts/full/dit_vs_mlp_comparison.csv")
LATEX_PATH = Path("artifacts/full/dit_vs_mlp_table.tex")

ENCODER_NAMES = sorted(NATIVE_DIMS.keys())
SEEDS = [0, 1, 2]
VARIANTS = ["conditioned", "unconditioned"]


def evaluate_mlp(
    encoder_name: str,
    variant: str,
    seed: int,
    device: torch.device = torch.device("cpu"),
    per_window: bool = False,
) -> dict:
    """Run MLP predictor on test set.

    1. Load embeddings, build test windows.
    2. Load MLP checkpoint (contains adapter weights, z_mean, z_std,
       predictor + fourier_embed state dicts).
    3. Reconstruct adapter from checkpoint weights (NOT re-init).
    4. Normalize test inputs.
    5. Forward pass in normalized space.
    6. Inverse transform to adapted space.
    7. Compute CosSim, MSE, copy baseline (same metric space as DiT).

    Returns dict with same schema as DiT rollout results.
    If ``per_window=True``, adds ``per_window_cossim`` and
    ``per_window_copy_cossim`` keys: lists of (N,) arrays per horizon.
    """
    ckpt_path = (
        MLP_CKPT_ROOT / encoder_name / variant / f"seed_{seed}" / "checkpoint.pt"
    )
    if not ckpt_path.exists():
        return {"error": f"Missing checkpoint: {ckpt_path}"}

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    z_mean = ckpt["z_mean"].to(device)
    z_std = ckpt["z_std"].to(device)

    # Reconstruct adapter from checkpoint
    native_dim = NATIVE_DIMS[encoder_name]
    needs_adapter = native_dim != TARGET_DIM
    if needs_adapter and ckpt["adapter_state_dict"]:
        adapter: nn.Module = nn.Linear(native_dim, TARGET_DIM, bias=False).to(device)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        for p in adapter.parameters():
            p.requires_grad_(False)
    else:
        adapter = nn.Identity().to(device)

    # Reconstruct predictor + fourier_embed
    predictor = LatentPredictor.from_canonical().to(device)
    predictor.load_state_dict(ckpt["predictor_state_dict"])
    predictor.eval()

    fourier_embed = FourierActionEmbedding.from_canonical().to(device)
    fourier_embed.load_state_dict(ckpt["fourier_embed_state_dict"])
    fourier_embed.eval()

    # Load test data
    data = load_embeddings(encoder_name)
    test_win = build_windows(data, "test", DEFAULT_HORIZON)
    if test_win is None:
        return {"error": f"No test windows for {encoder_name}"}

    z_t_test, act_test, zf_test = test_win
    n_test = len(z_t_test)
    horizon = zf_test.shape[1]

    t0 = time.time()

    # Evaluate
    cossim_sums = [0.0] * horizon
    mse_sums = [0.0] * horizon
    copy_cossim_sums = [0.0] * horizon
    total_samples = 0

    # Per-window storage (only allocated if requested)
    if per_window:
        pw_cossim: list[list[torch.Tensor]] = [[] for _ in range(horizon)]
        pw_copy: list[list[torch.Tensor]] = [[] for _ in range(horizon)]

    batch_size = 256
    from torch.utils.data import DataLoader, TensorDataset

    test_ds = TensorDataset(z_t_test, act_test, zf_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for z_t_b, act_b, zf_b in test_loader:
            z_t_b = z_t_b.to(device)
            act_b = act_b.to(device)
            zf_b = zf_b.to(device)
            B = z_t_b.shape[0]

            # Adapter projection + normalize
            B_f, H, _ = zf_b.shape
            z_t_adapted = adapter(z_t_b)
            zf_adapted = adapter(zf_b.reshape(B_f * H, -1)).reshape(B_f, H, TARGET_DIM)

            z_t_norm = (z_t_adapted - z_mean) / z_std
            zf_norm = (zf_adapted - z_mean) / z_std

            # Action embedding
            a_embed = fourier_embed(act_b)
            if variant == "unconditioned":
                a_embed = torch.zeros_like(a_embed)

            # Forward pass in normalized space
            z_hat_norm = predictor(z_t_norm, a_embed)  # (B, H, 384)

            # Inverse transform for metrics in adapted space
            z_hat = z_hat_norm * z_std + z_mean
            z_t_orig = z_t_adapted  # already in adapted unnormalized space
            zf_orig = zf_adapted

            # Per-horizon metrics
            for k in range(horizon):
                z_hat_k = z_hat[:, k]
                z_real_k = zf_orig[:, k]

                cs = F.cosine_similarity(z_hat_k, z_real_k, dim=-1)
                cossim_sums[k] += cs.sum().item()

                mse = ((z_hat_k - z_real_k) ** 2).mean(dim=-1)
                mse_sums[k] += mse.sum().item()

                copy_cs = F.cosine_similarity(z_t_orig, z_real_k, dim=-1)
                copy_cossim_sums[k] += copy_cs.sum().item()

                if per_window:
                    pw_cossim[k].append(cs.cpu())
                    pw_copy[k].append(copy_cs.cpu())

            total_samples += B

    elapsed = time.time() - t0

    cossim_by_horizon = [s / total_samples for s in cossim_sums]
    mse_by_horizon = [s / total_samples for s in mse_sums]
    copy_baseline_cossim = [s / total_samples for s in copy_cossim_sums]

    result = {
        "encoder": encoder_name,
        "variant": variant,
        "seed": seed,
        "n_test_windows": total_samples,
        "metrics": {
            "cossim_by_horizon": cossim_by_horizon,
            "mse_by_horizon": mse_by_horizon,
            "copy_baseline_cossim": copy_baseline_cossim,
        },
        "time_s": round(elapsed, 1),
    }
    if per_window:
        result["per_window_cossim"] = [torch.cat(pw_cossim[k]).numpy()
                                       for k in range(horizon)]
        result["per_window_copy_cossim"] = [torch.cat(pw_copy[k]).numpy()
                                            for k in range(horizon)]
    return result


# ---------------------------------------------------------------
# DiT-direct model (inline, mirrors scripts/train_dit_direct.py)
# ---------------------------------------------------------------


def _modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


class _DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning."""

    def __init__(self, dim=384, cond_dim=384, n_heads=6,
                 mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )
        self.drop = (nn.Dropout(dropout) if dropout > 0.0
                     else nn.Identity())
        self.adaln_linear = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.adaln_linear.weight)
        nn.init.zeros_(self.adaln_linear.bias)

    def forward(self, x, cond):
        mod = self.adaln_linear(cond).unsqueeze(1)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            mod.chunk(6, dim=-1)
        )
        h = _modulate(self.norm_attn(x), shift_a, scale_a)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_a * self.drop(attn_out)
        h = _modulate(self.norm_mlp(x), shift_m, scale_m)
        x = x + gate_m * self.drop(self.mlp(h))
        return x


class LatentDiTDirect(nn.Module):
    """DiT for direct latent future prediction (no diffusion)."""

    def __init__(self, z_dim=384, cond_dim=384, n_blocks=4,
                 n_heads=6, horizon=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.z_dim = z_dim
        self.horizon = horizon
        self.input_proj = nn.Linear(z_dim, z_dim)
        self.pos_embed = nn.Embedding(horizon, z_dim)
        self.z_t_proj = nn.Linear(z_dim, cond_dim)
        self.blocks = nn.ModuleList([
            _DiTBlock(dim=z_dim, cond_dim=cond_dim, n_heads=n_heads,
                      mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(z_dim, elementwise_affine=False)
        self.final_adaln = nn.Linear(cond_dim, 3 * z_dim)
        nn.init.zeros_(self.final_adaln.weight)
        nn.init.zeros_(self.final_adaln.bias)
        self.final_linear = nn.Linear(z_dim, z_dim)

    def forward(self, z_t, a_embed):
        B = z_t.shape[0]
        cond = self.z_t_proj(z_t) + a_embed
        pos_ids = torch.arange(self.horizon, device=z_t.device)
        x = self.input_proj(z_t).unsqueeze(1).expand(
            B, self.horizon, -1
        )
        x = x + self.pos_embed(pos_ids).unsqueeze(0)
        for block in self.blocks:
            x = block(x, cond)
        mod = self.final_adaln(cond).unsqueeze(1)
        shift, scale, gate = mod.chunk(3, dim=-1)
        x = gate * self.final_linear(
            _modulate(self.final_norm(x), shift, scale)
        )
        return x


def evaluate_dit_direct(
    encoder_name: str,
    variant: str,
    seed: int,
    lr: float,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run DiT-direct predictor on test set.

    Same evaluation protocol as ``evaluate_mlp``: loads checkpoint,
    reconstructs adapter/normalization, forward pass, inverse transform,
    computes CosSim/MSE/copy baseline in adapted unnormalized space.

    Returns dict with same schema as MLP/DiT results.
    """
    lr_tag = f"_lr{lr:.0e}".replace("+", "").replace("-0", "-")
    ckpt_path = (
        DIT_DIRECT_CKPT_ROOT / encoder_name / variant
        / f"seed_{seed}{lr_tag}" / "checkpoint.pt"
    )
    if not ckpt_path.exists():
        return {"error": f"Missing checkpoint: {ckpt_path}"}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    z_mean = ckpt["z_mean"].to(device)
    z_std = ckpt["z_std"].to(device)

    # Reconstruct adapter
    native_dim = NATIVE_DIMS[encoder_name]
    needs_adapter = native_dim != TARGET_DIM
    if needs_adapter and ckpt["adapter_state_dict"]:
        adapter: nn.Module = nn.Linear(native_dim, TARGET_DIM, bias=False).to(device)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        for p in adapter.parameters():
            p.requires_grad_(False)
    else:
        adapter = nn.Identity().to(device)

    # Reconstruct DiT-direct + Fourier
    dit = LatentDiTDirect().to(device)
    dit.load_state_dict(ckpt["dit_direct_state_dict"])
    dit.eval()

    fourier_embed = FourierActionEmbedding.from_canonical().to(device)
    fourier_embed.load_state_dict(ckpt["fourier_embed_state_dict"])
    fourier_embed.eval()

    # Load test data
    data = load_embeddings(encoder_name)
    test_win = build_windows(data, "test", DEFAULT_HORIZON)
    if test_win is None:
        return {"error": f"No test windows for {encoder_name}"}

    z_t_test, act_test, zf_test = test_win
    n_test = len(z_t_test)
    horizon = zf_test.shape[1]

    t0 = time.time()

    cossim_sums = [0.0] * horizon
    mse_sums = [0.0] * horizon
    copy_cossim_sums = [0.0] * horizon
    total_samples = 0

    batch_size = 256
    from torch.utils.data import DataLoader, TensorDataset

    test_ds = TensorDataset(z_t_test, act_test, zf_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for z_t_b, act_b, zf_b in test_loader:
            z_t_b = z_t_b.to(device)
            act_b = act_b.to(device)
            zf_b = zf_b.to(device)
            B = z_t_b.shape[0]

            B_f, H, _ = zf_b.shape
            z_t_adapted = adapter(z_t_b)
            zf_adapted = adapter(
                zf_b.reshape(B_f * H, -1)
            ).reshape(B_f, H, TARGET_DIM)

            z_t_norm = (z_t_adapted - z_mean) / z_std

            a_embed = fourier_embed(act_b)
            if variant == "unconditioned":
                a_embed = torch.zeros_like(a_embed)

            # Single forward pass (no DDIM)
            z_hat_norm = dit(z_t_norm, a_embed)  # (B, H, 384)

            z_hat = z_hat_norm * z_std + z_mean
            z_t_orig = z_t_adapted
            zf_orig = zf_adapted

            for k in range(horizon):
                z_hat_k = z_hat[:, k]
                z_real_k = zf_orig[:, k]

                cs = F.cosine_similarity(z_hat_k, z_real_k, dim=-1)
                cossim_sums[k] += cs.sum().item()

                mse = ((z_hat_k - z_real_k) ** 2).mean(dim=-1)
                mse_sums[k] += mse.sum().item()

                copy_cs = F.cosine_similarity(z_t_orig, z_real_k, dim=-1)
                copy_cossim_sums[k] += copy_cs.sum().item()

            total_samples += B

    elapsed = time.time() - t0

    return {
        "encoder": encoder_name,
        "variant": variant,
        "seed": seed,
        "lr": lr,
        "n_test_windows": total_samples,
        "metrics": {
            "cossim_by_horizon": [s / total_samples for s in cossim_sums],
            "mse_by_horizon": [s / total_samples for s in mse_sums],
            "copy_baseline_cossim": [s / total_samples
                                     for s in copy_cossim_sums],
        },
        "time_s": round(elapsed, 1),
    }


def load_dit_results() -> dict:
    """Load DiT results from rollout_results.json."""
    with ROLLOUT_RESULTS_PATH.open() as fh:
        return json.load(fh)


def _build_summary(all_results: list[dict]) -> dict:
    """Build per-encoder seed-averaged summary from raw results.

    Same convention as ``rollout_dit.py::_build_summary()``.
    """
    grouped: dict = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if "error" in r:
            continue
        grouped[r["encoder"]][r["variant"]].append(r["metrics"])

    summary: dict = {}
    for enc, variants in sorted(grouped.items()):
        summary[enc] = {}
        for var, metrics_list in sorted(variants.items()):
            n_seeds = len(metrics_list)
            horizon = len(metrics_list[0]["cossim_by_horizon"])

            cossim_means, cossim_stds = [], []
            mse_means, mse_stds = [], []
            copy_means = []

            for k in range(horizon):
                vals = [m["cossim_by_horizon"][k] for m in metrics_list]
                mean = sum(vals) / n_seeds
                cossim_means.append(mean)
                if n_seeds > 1:
                    var_val = sum((v - mean) ** 2 for v in vals) / (n_seeds - 1)
                    cossim_stds.append(var_val**0.5)
                else:
                    cossim_stds.append(0.0)

                vals = [m["mse_by_horizon"][k] for m in metrics_list]
                mean = sum(vals) / n_seeds
                mse_means.append(mean)
                if n_seeds > 1:
                    var_val = sum((v - mean) ** 2 for v in vals) / (n_seeds - 1)
                    mse_stds.append(var_val**0.5)
                else:
                    mse_stds.append(0.0)

                vals = [m["copy_baseline_cossim"][k] for m in metrics_list]
                copy_means.append(sum(vals) / n_seeds)

            summary[enc][var] = {
                "cossim_mean": cossim_means,
                "cossim_std": cossim_stds,
                "mse_mean": mse_means,
                "mse_std": mse_stds,
                "copy_baseline_cossim": copy_means,
                "n_seeds": n_seeds,
            }

    return summary


DIFFICULTY_CSV_PATH = Path("artifacts/full/mlp_by_difficulty.csv")


def build_difficulty_table(
    encoders: list[str],
    seeds: list[int],
    device: torch.device = torch.device("cpu"),
) -> list[dict]:
    """Split MLP vs copy comparison by copy-baseline difficulty quartiles.

    For each encoder, aggregates per-window CosSim across seeds and both
    variants (conditioned + unconditioned). Uses ``pd.qcut`` on the h=1
    copy baseline to define equal-count quartiles (Q1 = hardest scenes
    where copy baseline is lowest, Q4 = easiest).

    Returns list of dicts suitable for CSV export, one row per
    (encoder, variant, quartile, horizon).
    """
    import pandas as pd

    rows: list[dict] = []

    for enc in encoders:
        for var in VARIANTS:
            # Collect per-window arrays across seeds
            all_cossim: list[list[np.ndarray]] = None  # [horizon][seeds]
            all_copy: list[list[np.ndarray]] = None
            n_windows = None

            for seed in seeds:
                result = evaluate_mlp(enc, var, seed, device=device,
                                      per_window=True)
                if "error" in result:
                    print(f"  [SKIP] {enc}/{var}/seed={seed}: {result['error']}")
                    continue

                horizon = len(result["per_window_cossim"])
                if all_cossim is None:
                    all_cossim = [[] for _ in range(horizon)]
                    all_copy = [[] for _ in range(horizon)]

                for k in range(horizon):
                    all_cossim[k].append(result["per_window_cossim"][k])
                    all_copy[k].append(result["per_window_copy_cossim"][k])

                if n_windows is None:
                    n_windows = result["n_test_windows"]

            if all_cossim is None:
                continue

            # Average across seeds (per-window arrays are aligned)
            horizon = len(all_cossim)
            cossim_avg = [np.mean(all_cossim[k], axis=0) for k in range(horizon)]
            copy_avg = [np.mean(all_copy[k], axis=0) for k in range(horizon)]

            # Define quartiles on h=1 copy baseline
            copy_h1 = copy_avg[0]
            quartile_labels = pd.qcut(
                copy_h1, q=4,
                labels=["Q1 (hardest)", "Q2", "Q3", "Q4 (easiest)"],
            )

            for k in range(horizon):
                df = pd.DataFrame({
                    "mlp_cossim": cossim_avg[k],
                    "copy_cossim": copy_avg[k],
                    "quartile": quartile_labels,
                })
                for q_label, grp in df.groupby("quartile", observed=False):
                    rows.append({
                        "encoder": enc,
                        "variant": var,
                        "quartile": str(q_label),
                        "horizon": k + 1,
                        "mlp_cossim_mean": round(grp["mlp_cossim"].mean(), 6),
                        "copy_cossim_mean": round(grp["copy_cossim"].mean(), 6),
                        "mlp_minus_copy": round(
                            (grp["mlp_cossim"] - grp["copy_cossim"]).mean(), 6
                        ),
                        "n_windows": len(grp),
                    })

    return rows


def export_difficulty_csv(rows: list[dict], path: Path) -> None:
    """Write difficulty table to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "encoder", "variant", "quartile", "horizon",
        "mlp_cossim_mean", "copy_cossim_mean", "mlp_minus_copy", "n_windows",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_comparison_table(
    dit_results: list[dict],
    mlp_results: list[dict],
    dit_direct_results: list[dict] | None = None,
) -> list[dict]:
    """Build unified comparison rows.

    Columns: model, encoder, variant, horizon, cossim_mean, cossim_std_seed,
             mse_mean, mse_std_seed, n_seeds, n_test_windows, source.

    Copy baseline gets variant="none" (single row per encoder, since it is
    action-independent).
    """
    dit_summary = _build_summary(dit_results)
    mlp_summary = _build_summary(mlp_results)
    dd_summary = (_build_summary(dit_direct_results)
                  if dit_direct_results else {})

    rows: list[dict] = []

    all_encs = sorted(set(
        list(dit_summary.keys()) + list(mlp_summary.keys())
        + list(dd_summary.keys())
    ))
    for enc in all_encs:
        # DiT rows
        if enc in dit_summary:
            for var in sorted(dit_summary[enc].keys()):
                s = dit_summary[enc][var]
                horizon = len(s["cossim_mean"])
                for k in range(horizon):
                    rows.append({
                        "model": "dit",
                        "encoder": enc,
                        "variant": var,
                        "horizon": k + 1,
                        "cossim_mean": s["cossim_mean"][k],
                        "cossim_std_seed": s["cossim_std"][k],
                        "mse_mean": s["mse_mean"][k],
                        "mse_std_seed": s["mse_std"][k],
                        "n_seeds": s["n_seeds"],
                        "n_test_windows": dit_results[0]["n_test_windows"],
                        "source": "rollout_results.json",
                    })

        # MLP rows
        if enc in mlp_summary:
            for var in sorted(mlp_summary[enc].keys()):
                s = mlp_summary[enc][var]
                horizon = len(s["cossim_mean"])
                for k in range(horizon):
                    rows.append({
                        "model": "mlp",
                        "encoder": enc,
                        "variant": var,
                        "horizon": k + 1,
                        "cossim_mean": s["cossim_mean"][k],
                        "cossim_std_seed": s["cossim_std"][k],
                        "mse_mean": s["mse_mean"][k],
                        "mse_std_seed": s["mse_std"][k],
                        "n_seeds": s["n_seeds"],
                        "n_test_windows": mlp_results[0]["n_test_windows"],
                        "source": "mlp_rollout_results.json",
                    })

        # DiT-direct rows
        if enc in dd_summary:
            for var in sorted(dd_summary[enc].keys()):
                s = dd_summary[enc][var]
                horizon = len(s["cossim_mean"])
                for k in range(horizon):
                    rows.append({
                        "model": "dit_direct",
                        "encoder": enc,
                        "variant": var,
                        "horizon": k + 1,
                        "cossim_mean": s["cossim_mean"][k],
                        "cossim_std_seed": s["cossim_std"][k],
                        "mse_mean": s["mse_mean"][k],
                        "mse_std_seed": s["mse_std"][k],
                        "n_seeds": s["n_seeds"],
                        "n_test_windows": dit_direct_results[0][
                            "n_test_windows"],
                        "source": "dit_direct_rollout_results.json",
                    })

        # Copy baseline (single set of rows per encoder, variant="none")
        # Use DiT copy baseline as the validated reference
        if enc in dit_summary:
            any_var = list(dit_summary[enc].keys())[0]
            s = dit_summary[enc][any_var]
            horizon = len(s["copy_baseline_cossim"])
            for k in range(horizon):
                rows.append({
                    "model": "copy_baseline",
                    "encoder": enc,
                    "variant": "none",
                    "horizon": k + 1,
                    "cossim_mean": s["copy_baseline_cossim"][k],
                    "cossim_std_seed": 0.0,  # same across seeds by definition
                    "mse_mean": None,
                    "mse_std_seed": None,
                    "n_seeds": s["n_seeds"],
                    "n_test_windows": dit_results[0]["n_test_windows"],
                    "source": "rollout_results.json",
                })

    return rows


def export_csv(rows: list[dict], path: Path) -> None:
    """Write comparison table to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "encoder", "variant", "horizon",
        "cossim_mean", "cossim_std_seed",
        "mse_mean", "mse_std_seed",
        "n_seeds", "n_test_windows", "source",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_latex(rows: list[dict], path: Path) -> None:
    """Export LaTeX table grouped by encoder, columns by model x horizon.

    Format: encoder | variant | model | k=1 | k=2 | k=3 | k=4
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{DiT vs MLP cosine similarity (mean $\pm$ std across seeds)}",
        r"\label{tab:dit-vs-mlp}",
        r"\scriptsize",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Encoder & Variant & Model & $k{=}1$ & $k{=}2$ & $k{=}3$ & $k{=}4$ \\",
        r"\midrule",
    ]

    # Group by encoder
    from itertools import groupby

    sorted_rows = sorted(rows, key=lambda r: (r["encoder"], r["variant"], r["model"]))
    prev_enc = None

    for (enc, var, model), group in groupby(
        sorted_rows, key=lambda r: (r["encoder"], r["variant"], r["model"])
    ):
        if prev_enc is not None and enc != prev_enc:
            lines.append(r"\midrule")
        prev_enc = enc

        group_list = sorted(group, key=lambda r: r["horizon"])
        cells = []
        for g in group_list:
            if g["cossim_mean"] is not None:
                mean = g["cossim_mean"]
                std = g["cossim_std_seed"]
                if std > 0:
                    cells.append(f"{mean:.4f}$\\pm${std:.4f}")
                else:
                    cells.append(f"{mean:.4f}")
            else:
                cells.append("--")

        while len(cells) < 4:
            cells.append("--")

        enc_display = enc.replace("_", r"\_")
        lines.append(
            f"  {enc_display} & {var} & {model} & "
            + " & ".join(cells)
            + r" \\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with path.open("w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="DA7: DiT vs MLP evaluation and comparison."
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=ENCODER_NAMES,
        choices=ENCODER_NAMES,
        help="Encoders to evaluate (default: all).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seeds (default: 0 1 2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: auto-detect).",
    )
    parser.add_argument(
        "--difficulty",
        action="store_true",
        help="Generate hard-subset difficulty table (DA7.5).",
    )
    parser.add_argument(
        "--dit-direct-lr",
        type=float,
        default=0.0,
        help="Evaluate DiT-direct checkpoints at this LR (DA7.5). "
             "0 = skip DiT-direct evaluation.",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device
        else (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )

    # Verify all expected checkpoints exist
    missing = []
    for enc in args.encoders:
        for var in VARIANTS:
            for seed in args.seeds:
                ckpt = (
                    MLP_CKPT_ROOT / enc / var / f"seed_{seed}" / "checkpoint.pt"
                )
                if not ckpt.exists():
                    missing.append(str(ckpt))

    if missing:
        print("[ERROR] Missing MLP checkpoints:")
        for m in missing:
            print(f"  - {m}")
        print("\nRun scripts/train_mlp_fair.py first.")
        sys.exit(1)

    # Load DiT results
    dit_data = load_dit_results()
    dit_results = dit_data["results"]

    # Run MLP evaluation
    print("=" * 60)
    print("DA7: MLP Rollout Evaluation")
    print(f"  encoders: {args.encoders}")
    print(f"  seeds: {args.seeds}")
    print(f"  device: {device}")
    print("=" * 60)

    mlp_results: list[dict] = []
    for enc in args.encoders:
        for var in VARIANTS:
            for seed in args.seeds:
                print(f"\n[eval] {enc}/{var}/seed={seed}")
                result = evaluate_mlp(enc, var, seed, device=device)
                if "error" in result:
                    print(f"  [ERROR] {result['error']}")
                    sys.exit(1)

                # Print per-horizon summary
                m = result["metrics"]
                print(f"  {'k':>3}  {'CosSim':>8}  {'MSE':>8}  {'CopyBL':>8}")
                for k in range(len(m["cossim_by_horizon"])):
                    print(
                        f"  k={k+1}:  {m['cossim_by_horizon'][k]:>8.4f}  "
                        f"{m['mse_by_horizon'][k]:>8.4f}  "
                        f"{m['copy_baseline_cossim'][k]:>8.4f}"
                    )
                mlp_results.append(result)

    # Save MLP results
    mlp_summary = _build_summary(mlp_results)
    mlp_output = {
        "results": mlp_results,
        "summary": mlp_summary,
    }
    MLP_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MLP_RESULTS_PATH.open("w") as fh:
        json.dump(mlp_output, fh, indent=2)
    print(f"\n[saved] {MLP_RESULTS_PATH}")

    # DiT-direct evaluation (DA7.5)
    dd_results: list[dict] | None = None
    if args.dit_direct_lr > 0:
        print("\n" + "=" * 60)
        print("DA7.5: DiT-Direct Evaluation")
        print(f"  lr: {args.dit_direct_lr}")
        print("=" * 60)

        dd_results = []
        dd_variants = ["conditioned"]  # conditioned-only by design
        for enc in args.encoders:
            for var in dd_variants:
                for seed in args.seeds:
                    print(f"\n[eval-dd] {enc}/{var}/seed={seed}")
                    result = evaluate_dit_direct(
                        enc, var, seed, args.dit_direct_lr,
                        device=device,
                    )
                    if "error" in result:
                        print(f"  [SKIP] {result['error']}")
                        continue
                    m = result["metrics"]
                    print(f"  {'k':>3}  {'CosSim':>8}  {'MSE':>8}  "
                          f"{'CopyBL':>8}")
                    for k in range(len(m["cossim_by_horizon"])):
                        print(
                            f"  k={k+1}:  "
                            f"{m['cossim_by_horizon'][k]:>8.4f}  "
                            f"{m['mse_by_horizon'][k]:>8.4f}  "
                            f"{m['copy_baseline_cossim'][k]:>8.4f}"
                        )
                    dd_results.append(result)

        if dd_results:
            dd_summary = _build_summary(dd_results)
            dd_output = {"results": dd_results, "summary": dd_summary}
            DIT_DIRECT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with DIT_DIRECT_RESULTS_PATH.open("w") as fh:
                json.dump(dd_output, fh, indent=2)
            print(f"\n[saved] {DIT_DIRECT_RESULTS_PATH}")
        else:
            dd_results = None

    # Build comparison table
    comparison_rows = build_comparison_table(
        dit_results, mlp_results, dd_results,
    )
    export_csv(comparison_rows, COMPARISON_CSV_PATH)
    print(f"[saved] {COMPARISON_CSV_PATH}")

    export_latex(comparison_rows, LATEX_PATH)
    print(f"[saved] {LATEX_PATH}")

    # Hard-subset difficulty table (DA7.5)
    if args.difficulty:
        print("\n" + "=" * 60)
        print("DA7.5: Hard-Subset Difficulty Analysis")
        print("=" * 60)
        diff_rows = build_difficulty_table(args.encoders, args.seeds, device)
        export_difficulty_csv(diff_rows, DIFFICULTY_CSV_PATH)
        print(f"[saved] {DIFFICULTY_CSV_PATH}")

        # Print summary for h=1
        print(f"\n  {'Encoder':<15} {'Variant':<14} {'Quartile':<16} "
              f"{'MLP':>8} {'Copy':>8} {'Gap':>8} {'N':>6}")
        for r in diff_rows:
            if r["horizon"] == 1:
                print(
                    f"  {r['encoder']:<15} {r['variant']:<14} "
                    f"{r['quartile']:<16} {r['mlp_cossim_mean']:>8.4f} "
                    f"{r['copy_cossim_mean']:>8.4f} "
                    f"{r['mlp_minus_copy']:>+8.4f} {r['n_windows']:>6}"
                )

    # Print summary
    print("\n" + "=" * 60)
    dit_summary = _build_summary(dit_results)
    dd_sum = _build_summary(dd_results) if dd_results else {}
    has_dd = bool(dd_sum)

    if has_dd:
        print("SUMMARY: 4-Model CosSim Comparison (seed-averaged, k=1)")
        print("=" * 60)
        print(f"  {'Encoder':<15} {'Variant':<14} {'Copy':>8} "
              f"{'DiT-dir':>8} {'MLP':>8} {'DiT-eps':>8}")
        print(f"  {'-'*13:<15} {'-'*12:<14} {'-'*6:>8} "
              f"{'-'*6:>8} {'-'*6:>8} {'-'*6:>8}")

        for enc in sorted(args.encoders):
            for var in VARIANTS:
                dit_cs = dit_summary.get(enc, {}).get(
                    var, {}).get("cossim_mean", [None])[0]
                mlp_cs = mlp_summary.get(enc, {}).get(
                    var, {}).get("cossim_mean", [None])[0]
                copy_cs = dit_summary.get(enc, {}).get(
                    var, {}).get("copy_baseline_cossim", [None])[0]
                dd_cs = dd_sum.get(enc, {}).get(
                    var, {}).get("cossim_mean", [None])
                dd_cs = dd_cs[0] if dd_cs else None

                dd_str = f"{dd_cs:>8.4f}" if dd_cs else f"{'--':>8}"
                print(
                    f"  {enc:<15} {var:<14} "
                    f"{copy_cs:>8.4f} {dd_str} "
                    f"{mlp_cs:>8.4f} {dit_cs:>8.4f}"
                )
    else:
        print("SUMMARY: DiT vs MLP CosSim (seed-averaged, k=1)")
        print("=" * 60)
        print(f"  {'Encoder':<15} {'Variant':<14} {'DiT':>8} "
              f"{'MLP':>8} {'CopyBL':>8} {'Winner':>8}")
        print(f"  {'-'*13:<15} {'-'*12:<14} {'-'*6:>8} "
              f"{'-'*6:>8} {'-'*6:>8} {'-'*6:>8}")

        for enc in sorted(args.encoders):
            for var in VARIANTS:
                dit_cs = dit_summary.get(enc, {}).get(
                    var, {}).get("cossim_mean", [None])[0]
                mlp_cs = mlp_summary.get(enc, {}).get(
                    var, {}).get("cossim_mean", [None])[0]
                copy_cs = dit_summary.get(enc, {}).get(
                    var, {}).get("copy_baseline_cossim", [None])[0]

                winner = ""
                if dit_cs is not None and mlp_cs is not None:
                    winner = "DiT" if dit_cs > mlp_cs else "MLP"

                print(
                    f"  {enc:<15} {var:<14} "
                    f"{dit_cs:>8.4f} {mlp_cs:>8.4f} "
                    f"{copy_cs:>8.4f} {winner:>8}"
                )


if __name__ == "__main__":
    main()
