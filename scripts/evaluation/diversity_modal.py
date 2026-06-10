"""DA9 Exp 1: Diversity diagnostic on Modal.

Tests whether DiT-x0 samples diverse futures by running K=10 DDIM
rollouts with different initial noise x_T (eta=0). If all K trajectories
converge to the same output, the model has learned a unimodal conditional
and functions as a point predictor.

All checkpoints and embeddings live on the Modal volume.
Results downloaded locally as CSV.

Usage::

    modal run scripts/diversity_modal.py
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("lwm-av-diversity")
    vol = modal.Volume.from_name("nuscenes-full")
else:
    app = None
    vol = None

VOL_PATH = "/vol"
EMBED_DIR = f"{VOL_PATH}/embeddings"
DIT_DIR = f"{VOL_PATH}/dits"
MLP_FAIR_DIR = f"{VOL_PATH}/outputs/latent_predictors_fair"
MLP_RESIDUAL_DIR = f"{VOL_PATH}/outputs/latent_predictors_residual"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_DIM = 384
HORIZON = 4  # DA8 checkpoints, h=4

NATIVE_DIMS = {
    "vit_s16": 384,
    "dino_vits14": 384,
    "clip_b32": 512,
    "vq_track": 256,
    "vjepa2_rep64": 1024,
    "vjepa2_rep1": 1024,
}

PILOT_ENCODERS = ["vit_s16", "clip_b32", "vjepa2_rep64"]
PILOT_SEEDS = [0]
K_SAMPLES = 10
EVAL_BATCH_SIZE = 512

DIT_CONFIG = {
    "n_blocks": 4,
    "n_heads": 6,
    "z_dim": 384,
    "cond_dim": 384,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
}

DIFFUSION_CONFIG = {"n_train_steps": 1000}

FOURIER_CONFIG = {
    "n_frequencies": 64,
    "base": 2.0,
    "out_dim": 384,
}

if modal is not None:
    base_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch==2.5.1", "numpy>=1.26", "pandas>=2.0")
    )
else:
    base_image = None


def _modal_function_decorator(fn):
    if app is not None:
        return app.function(
            volumes={VOL_PATH: vol},
            image=base_image,
            gpu="A10G",
            timeout=3600,
            memory=16384,
        )(fn)
    return fn


@_modal_function_decorator
def evaluate_diversity(
    encoder_name: str,
    seed: int,
    k_samples: int = K_SAMPLES,
):
    """Run K-sample DDIM diversity analysis for one encoder/seed.

    Returns per-quartile, per-horizon-step metrics.
    """
    import math

    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    # -------------------------------------------------------------------
    # Inline model definitions (same as eval_horizon_modal.py)
    # -------------------------------------------------------------------

    class CosineNoiseSchedule(nn.Module):
        def __init__(self, n_steps: int = 1000, s: float = 0.008):
            super().__init__()
            self.n_steps = n_steps
            steps = torch.arange(n_steps + 1, dtype=torch.float64)
            f_t = torch.cos(((steps / n_steps) + s) / (1.0 + s) * (torch.pi / 2.0)) ** 2
            alphas_cumprod = f_t / f_t[0]
            alphas_cumprod = alphas_cumprod[:n_steps].float()
            self.register_buffer("alphas_cumprod", alphas_cumprod)

    class TimestepEmbedding(nn.Module):
        def __init__(self, cond_dim: int = 384):
            super().__init__()
            self.cond_dim = cond_dim
            self.mlp = nn.Sequential(
                nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim),
            )

        def forward(self, timestep):
            half_dim = self.cond_dim // 2
            freqs = torch.exp(
                -math.log(10000.0)
                * torch.arange(half_dim, device=timestep.device, dtype=torch.float32)
                / half_dim
            )
            args = timestep.float().unsqueeze(-1) * freqs.unsqueeze(0)
            emb = torch.cat([args.sin(), args.cos()], dim=-1)
            return self.mlp(emb)

    def _modulate(x, shift, scale):
        return x * (1.0 + scale) + shift

    class DiTBlock(nn.Module):
        def __init__(self, dim=384, cond_dim=384, n_heads=6, mlp_ratio=4.0, dropout=0.0):
            super().__init__()
            self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False)
            self.attn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True,
            )
            self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
            mlp_hidden = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, dim),
            )
            self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            self.adaln_linear = nn.Linear(cond_dim, 6 * dim)

        def forward(self, x, cond):
            mod = self.adaln_linear(cond).unsqueeze(1)
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mod.chunk(6, dim=-1)
            h = _modulate(self.norm_attn(x), shift_a, scale_a)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + gate_a * self.drop(attn_out)
            h = _modulate(self.norm_mlp(x), shift_m, scale_m)
            x = x + gate_m * self.drop(self.mlp(h))
            return x

    class LatentDiT(nn.Module):
        def __init__(self, z_dim=384, cond_dim=384, n_blocks=4, n_heads=6,
                     horizon=4, mlp_ratio=4.0, dropout=0.0):
            super().__init__()
            self.z_dim = z_dim
            self.horizon = horizon
            self.input_proj = nn.Linear(z_dim, z_dim)
            self.timestep_embed = TimestepEmbedding(cond_dim)
            self.z_t_proj = nn.Linear(z_dim, cond_dim)
            self.blocks = nn.ModuleList([
                DiTBlock(dim=z_dim, cond_dim=cond_dim, n_heads=n_heads,
                         mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(n_blocks)
            ])
            self.final_norm = nn.LayerNorm(z_dim, elementwise_affine=False)
            self.final_adaln = nn.Linear(cond_dim, 3 * z_dim)
            self.final_linear = nn.Linear(z_dim, z_dim)

        def forward(self, x_noisy, z_t, a_embed, timestep):
            cond = self.timestep_embed(timestep) + self.z_t_proj(z_t) + a_embed
            x = self.input_proj(x_noisy)
            for block in self.blocks:
                x = block(x, cond)
            mod = self.final_adaln(cond).unsqueeze(1)
            shift, scale, gate = mod.chunk(3, dim=-1)
            x = gate * self.final_linear(_modulate(self.final_norm(x), shift, scale))
            return x

    class FourierActionEmbedding(nn.Module):
        def __init__(self, action_dim=2, n_frequencies=64, base=2.0, out_dim=384):
            super().__init__()
            freqs = base ** torch.arange(n_frequencies, dtype=torch.float32) * torch.pi
            self.register_buffer("freqs", freqs)
            fourier_dim = action_dim * 2 * n_frequencies
            self.proj = nn.Sequential(
                nn.Linear(fourier_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim),
            )

        def forward(self, action):
            x = action.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
            x = torch.cat([x.sin(), x.cos()], dim=-1)
            x = x.flatten(1)
            return self.proj(x)

    class LatentPredictor(nn.Module):
        def __init__(self, z_dim=384, a_dim=384, horizon=4, hidden=512):
            super().__init__()
            self.horizon = horizon
            self.net = nn.Sequential(
                nn.Linear(z_dim + a_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, z_dim * horizon),
            )

        def forward(self, z_t, a_embed):
            x = torch.cat([z_t, a_embed], dim=-1)
            return self.net(x).view(z_t.shape[0], self.horizon, -1)

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------

    native_dim = NATIVE_DIMS[encoder_name]
    needs_adapter = native_dim != TARGET_DIM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizon = HORIZON

    data = np.load(f"{EMBED_DIR}/{encoder_name}.npz", allow_pickle=True)
    embeddings = data["embeddings"]
    splits = data["splits"]
    steer_norms = data["steer_norms"]
    accel_norms = data["accel_norms"]
    scene_names = data["scene_names"]

    def build_windows(split_name, h):
        mask = splits == split_name
        emb = embeddings[mask]
        steers = steer_norms[mask]
        accels = accel_norms[mask]
        scenes = scene_names[mask]
        z_t_list, action_list, z_future_list = [], [], []
        for scene in np.unique(scenes):
            idx = np.where(scenes == scene)[0]
            for j in range(len(idx) - h):
                z_t_list.append(emb[idx[j]])
                action_list.append([steers[idx[j]], accels[idx[j]]])
                z_future_list.append(emb[idx[j + 1 : j + 1 + h]])
        return (
            torch.tensor(np.array(z_t_list), dtype=torch.float32),
            torch.tensor(np.array(action_list), dtype=torch.float32),
            torch.tensor(np.array(z_future_list), dtype=torch.float32),
        )

    z_t_test, act_test, zf_test = build_windows("test", horizon)
    n_windows = len(z_t_test)
    print(f"[diversity] {encoder_name}/seed={seed}: {n_windows} test windows, K={k_samples}")

    # -------------------------------------------------------------------
    # Load DiT-x0 checkpoint (DA8, h=4)
    # -------------------------------------------------------------------

    dit_ckpt_path = f"{DIT_DIR}/{encoder_name}/conditioned__x0/seed_{seed}/checkpoint.pt"
    if not os.path.exists(dit_ckpt_path):
        print(f"[diversity] MISSING DiT: {dit_ckpt_path}")
        return None

    dit_ckpt = torch.load(dit_ckpt_path, map_location=device, weights_only=False)
    z_mean = dit_ckpt["z_mean"].to(device)
    z_std = dit_ckpt["z_std"].to(device)

    if needs_adapter and dit_ckpt.get("adapter_state_dict"):
        adapter = nn.Linear(native_dim, TARGET_DIM, bias=False).to(device)
        adapter.load_state_dict(dit_ckpt["adapter_state_dict"])
        for p in adapter.parameters():
            p.requires_grad_(False)
    else:
        adapter = nn.Identity().to(device)

    fourier_dit = FourierActionEmbedding(action_dim=2, **FOURIER_CONFIG).to(device)
    fourier_dit.load_state_dict(dit_ckpt["fourier_embed_state_dict"])
    fourier_dit.eval()

    dit = LatentDiT(**{**DIT_CONFIG, "horizon": horizon}).to(device)
    dit.load_state_dict(dit_ckpt["dit_state_dict"])
    dit.eval()

    schedule = CosineNoiseSchedule(n_steps=DIFFUSION_CONFIG["n_train_steps"]).to(device)
    alphas_cumprod = schedule.alphas_cumprod.to(device)

    # DDIM setup
    n_ddim_steps = 50
    T = DIFFUSION_CONFIG["n_train_steps"]
    stride = T // n_ddim_steps
    timesteps = list(reversed(list(range(0, T, stride))[:n_ddim_steps]))

    # -------------------------------------------------------------------
    # Run K DDIM samples with different x_T
    # -------------------------------------------------------------------

    z_t_dev = z_t_test.to(device)
    act_dev = act_test.to(device)

    # Pre-compute adapted z_t and action embeddings (shared across K samples)
    with torch.no_grad():
        z_t_adapted_all = []
        a_embed_all = []
        for start in range(0, n_windows, EVAL_BATCH_SIZE):
            end = min(start + EVAL_BATCH_SIZE, n_windows)
            z_t_b = (adapter(z_t_dev[start:end]) - z_mean) / z_std
            a_b = fourier_dit(act_dev[start:end])
            z_t_adapted_all.append(z_t_b)
            a_embed_all.append(a_b)
        z_t_adapted_all = torch.cat(z_t_adapted_all, dim=0)
        a_embed_all = torch.cat(a_embed_all, dim=0)

    # Collect K samples: each (N, H, 384) in normalized space
    all_samples_norm = []
    t0 = time.time()

    for k in range(k_samples):
        noise_seed = seed * 1000 + k
        sample_parts = []

        for start in range(0, n_windows, EVAL_BATCH_SIZE):
            end = min(start + EVAL_BATCH_SIZE, n_windows)
            B = end - start
            z_t_b = z_t_adapted_all[start:end]
            a_b = a_embed_all[start:end]

            torch.manual_seed(noise_seed + start)
            x = torch.randn(B, horizon, TARGET_DIM, device=device)

            with torch.no_grad():
                for i, t_val in enumerate(timesteps):
                    t = torch.full((B,), t_val, device=device, dtype=torch.long)
                    pred_x0 = dit(x, z_t=z_t_b, a_embed=a_b, timestep=t)

                    alpha_bar_t = alphas_cumprod[t_val]
                    if i < len(timesteps) - 1:
                        alpha_bar_prev = alphas_cumprod[timesteps[i + 1]]
                    else:
                        alpha_bar_prev = torch.tensor(1.0, device=device)

                    noise_direction = (
                        x - torch.sqrt(alpha_bar_t) * pred_x0
                    ) / torch.sqrt(1.0 - alpha_bar_t + 1e-8)

                    x = (
                        torch.sqrt(alpha_bar_prev) * pred_x0
                        + torch.sqrt(1.0 - alpha_bar_prev) * noise_direction
                    )

            sample_parts.append(x.cpu())

        all_samples_norm.append(torch.cat(sample_parts, dim=0))
        elapsed = time.time() - t0
        print(f"  Sample {k+1}/{k_samples} done ({elapsed:.1f}s)")

    # Stack: (K, N, H, 384) in normalized space
    samples_norm = torch.stack(all_samples_norm, dim=0)

    del dit, fourier_dit
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # Unnormalize samples and compute ground truth
    # -------------------------------------------------------------------

    # Unnormalize: z_hat = x * z_std + z_mean
    z_mean_cpu = z_mean.cpu()
    z_std_cpu = z_std.cpu()
    samples = samples_norm * z_std_cpu + z_mean_cpu  # (K, N, H, 384)

    # Ground truth in adapted space
    with torch.no_grad():
        zf_adapted_parts = []
        z_t_adapted_parts = []
        for start in range(0, n_windows, EVAL_BATCH_SIZE):
            end = min(start + EVAL_BATCH_SIZE, n_windows)
            B = end - start
            zf_b = zf_test[start:end].to(device)  # (B, H, native_dim)
            B_f, H, _ = zf_b.shape
            zf_a = adapter(zf_b.reshape(B_f * H, -1)).reshape(B_f, H, TARGET_DIM).cpu()
            z_t_a = adapter(z_t_dev[start:end]).cpu()
            zf_adapted_parts.append(zf_a)
            z_t_adapted_parts.append(z_t_a)
        zf_adapted = torch.cat(zf_adapted_parts, dim=0)  # (N, H, 384)
        z_t_adapted_cpu = torch.cat(z_t_adapted_parts, dim=0)  # (N, 384)

    del adapter
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # Load MLP baselines (DA8, h=4)
    # -------------------------------------------------------------------

    # MLP-fair
    mlp_fair_path = f"{MLP_FAIR_DIR}/{encoder_name}/conditioned/seed_{seed}/checkpoint.pt"
    mlp_fair_cossim_by_h = None
    if os.path.exists(mlp_fair_path):
        mlp_ckpt = torch.load(mlp_fair_path, map_location=device, weights_only=False)
        mlp_z_mean = mlp_ckpt["z_mean"].to(device)
        mlp_z_std = mlp_ckpt["z_std"].to(device)

        if needs_adapter and mlp_ckpt.get("adapter_state_dict"):
            mlp_adapter = nn.Linear(native_dim, TARGET_DIM, bias=False).to(device)
            mlp_adapter.load_state_dict(mlp_ckpt["adapter_state_dict"])
            for p in mlp_adapter.parameters():
                p.requires_grad_(False)
        else:
            mlp_adapter = nn.Identity().to(device)

        mlp_fourier = FourierActionEmbedding(action_dim=2, **FOURIER_CONFIG).to(device)
        mlp_fourier.load_state_dict(mlp_ckpt["fourier_embed_state_dict"])
        mlp_fourier.eval()

        mlp_fair = LatentPredictor(
            z_dim=TARGET_DIM, a_dim=TARGET_DIM, horizon=horizon,
        ).to(device)
        mlp_fair.load_state_dict(mlp_ckpt["predictor_state_dict"])
        mlp_fair.eval()

        # Run MLP-fair inference
        mlp_fair_preds = []
        with torch.no_grad():
            for start in range(0, n_windows, EVAL_BATCH_SIZE):
                end = min(start + EVAL_BATCH_SIZE, n_windows)
                z_t_b = (mlp_adapter(z_t_dev[start:end]) - mlp_z_mean) / mlp_z_std
                a_b = mlp_fourier(act_dev[start:end])
                pred = mlp_fair(z_t_b, a_b)
                pred_unnorm = pred * mlp_z_std + mlp_z_mean
                mlp_fair_preds.append(pred_unnorm.cpu())
        mlp_fair_preds = torch.cat(mlp_fair_preds, dim=0)  # (N, H, 384)

        # Compute ground truth in MLP-fair's adapted space
        mlp_fair_zf = []
        with torch.no_grad():
            for start in range(0, n_windows, EVAL_BATCH_SIZE):
                end = min(start + EVAL_BATCH_SIZE, n_windows)
                B = end - start
                zf_b = zf_test[start:end].to(device)
                B_f, H, _ = zf_b.shape
                zf_a = mlp_adapter(zf_b.reshape(B_f * H, -1)).reshape(B_f, H, TARGET_DIM).cpu()
                mlp_fair_zf.append(zf_a)
        mlp_fair_zf = torch.cat(mlp_fair_zf, dim=0)

        mlp_fair_cossim_by_h = []
        for h_k in range(horizon):
            cs = F.cosine_similarity(mlp_fair_preds[:, h_k], mlp_fair_zf[:, h_k], dim=-1)
            mlp_fair_cossim_by_h.append(cs)  # (N,) per-window

        del mlp_fair, mlp_fourier, mlp_adapter
        torch.cuda.empty_cache()

    # MLP-residual
    mlp_res_path = f"{MLP_RESIDUAL_DIR}/{encoder_name}/conditioned/seed_{seed}/checkpoint.pt"
    mlp_res_cossim_by_h = None
    if os.path.exists(mlp_res_path):
        res_ckpt = torch.load(mlp_res_path, map_location=device, weights_only=False)
        res_z_mean = res_ckpt["z_mean"].to(device)
        res_z_std = res_ckpt["z_std"].to(device)

        if needs_adapter and res_ckpt.get("adapter_state_dict"):
            res_adapter = nn.Linear(native_dim, TARGET_DIM, bias=False).to(device)
            res_adapter.load_state_dict(res_ckpt["adapter_state_dict"])
            for p in res_adapter.parameters():
                p.requires_grad_(False)
        else:
            res_adapter = nn.Identity().to(device)

        res_fourier = FourierActionEmbedding(action_dim=2, **FOURIER_CONFIG).to(device)
        res_fourier.load_state_dict(res_ckpt["fourier_embed_state_dict"])
        res_fourier.eval()

        res_mlp = LatentPredictor(
            z_dim=TARGET_DIM, a_dim=TARGET_DIM, horizon=horizon,
        ).to(device)
        res_mlp.load_state_dict(res_ckpt["predictor_state_dict"])
        res_mlp.eval()

        # Run MLP-residual inference
        mlp_res_preds = []
        with torch.no_grad():
            for start in range(0, n_windows, EVAL_BATCH_SIZE):
                end = min(start + EVAL_BATCH_SIZE, n_windows)
                z_t_b = (res_adapter(z_t_dev[start:end]) - res_z_mean) / res_z_std
                a_b = res_fourier(act_dev[start:end])
                pred_delta = res_mlp(z_t_b, a_b)
                pred = pred_delta + z_t_b.unsqueeze(1).expand(-1, horizon, -1)
                pred_unnorm = pred * res_z_std + res_z_mean
                mlp_res_preds.append(pred_unnorm.cpu())
        mlp_res_preds = torch.cat(mlp_res_preds, dim=0)

        # Compute ground truth in MLP-residual's adapted space
        mlp_res_zf = []
        with torch.no_grad():
            for start in range(0, n_windows, EVAL_BATCH_SIZE):
                end = min(start + EVAL_BATCH_SIZE, n_windows)
                B = end - start
                zf_b = zf_test[start:end].to(device)
                B_f, H, _ = zf_b.shape
                zf_a = res_adapter(zf_b.reshape(B_f * H, -1)).reshape(B_f, H, TARGET_DIM).cpu()
                mlp_res_zf.append(zf_a)
        mlp_res_zf = torch.cat(mlp_res_zf, dim=0)

        mlp_res_cossim_by_h = []
        for h_k in range(horizon):
            cs = F.cosine_similarity(mlp_res_preds[:, h_k], mlp_res_zf[:, h_k], dim=-1)
            mlp_res_cossim_by_h.append(cs)

        del res_mlp, res_fourier, res_adapter
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # Compute metrics per horizon step + difficulty quartile
    # -------------------------------------------------------------------

    rows = []

    for h_k in range(horizon):
        h = h_k + 1
        zf_k = zf_adapted[:, h_k]  # (N, 384)

        # Per-sample CosSim with ground truth: (K, N)
        per_sample_cossim = torch.stack([
            F.cosine_similarity(samples[k, :, h_k], zf_k, dim=-1)
            for k in range(k_samples)
        ], dim=0)

        # Best-of-K: max across K
        best_of_k_cs, best_idx = per_sample_cossim.max(dim=0)  # (N,)

        # Gather best sample for norm ratio + MSE
        best_samples = torch.stack([
            samples[best_idx[i], i, h_k] for i in range(n_windows)
        ])  # (N, 384)

        # Mean-of-K CosSim per window
        mean_of_k_cs = per_sample_cossim.mean(dim=0)  # (N,)

        # Pairwise distance: 1 - mean pairwise CosSim among K samples
        pairwise_cs_sum = 0.0
        n_pairs = 0
        for i_s in range(k_samples):
            for j_s in range(i_s + 1, k_samples):
                pw_cs = F.cosine_similarity(
                    samples[i_s, :, h_k], samples[j_s, :, h_k], dim=-1
                ).mean().item()
                pairwise_cs_sum += pw_cs
                n_pairs += 1
        pairwise_distance = 1.0 - (pairwise_cs_sum / n_pairs) if n_pairs > 0 else 0.0

        # Spread: std of per-sample CosSim across K
        spread = per_sample_cossim.std(dim=0).mean().item()

        # Copy baseline per window
        copy_per_window = F.cosine_similarity(z_t_adapted_cpu, zf_k, dim=-1)  # (N,)

        # Difficulty quartiles (per this horizon step)
        try:
            quartile_labels = pd.qcut(
                copy_per_window.numpy(), q=4,
                labels=["Q1 (hardest)", "Q2", "Q3", "Q4 (easiest)"],
            )
        except ValueError:
            quartile_labels = pd.qcut(
                pd.Series(copy_per_window.numpy()).rank(method="first"),
                q=4,
                labels=["Q1 (hardest)", "Q2", "Q3", "Q4 (easiest)"],
            )

        for q_label in ["Q1 (hardest)", "Q2", "Q3", "Q4 (easiest)"]:
            q_mask = np.array(quartile_labels) == q_label
            q_mask_t = torch.tensor(q_mask)
            n_q = int(q_mask.sum())

            # MLP per-window CosSim for this quartile
            mlp_fair_q = None
            if mlp_fair_cossim_by_h is not None:
                mlp_fair_q = round(mlp_fair_cossim_by_h[h_k][q_mask_t].mean().item(), 6)

            mlp_res_q = None
            if mlp_res_cossim_by_h is not None:
                mlp_res_q = round(mlp_res_cossim_by_h[h_k][q_mask_t].mean().item(), 6)

            row = {
                "encoder": encoder_name,
                "seed": seed,
                "eta": 0.0,
                "quartile": q_label,
                "horizon_step": h,
                "best_of_k_cossim": round(best_of_k_cs[q_mask_t].mean().item(), 6),
                "best_of_k_mse": round(
                    ((best_samples[q_mask_t] - zf_k[q_mask_t]) ** 2)
                    .mean(dim=-1).mean().item(), 6
                ),
                "best_of_k_norm_ratio": round(
                    (best_samples[q_mask_t].norm(dim=-1) /
                     (zf_k[q_mask_t].norm(dim=-1) + 1e-8)).mean().item(), 6
                ),
                "mean_of_k_cossim": round(mean_of_k_cs[q_mask_t].mean().item(), 6),
                "mlp_fair_cossim": mlp_fair_q,
                "mlp_residual_cossim": mlp_res_q,
                "copy_baseline": round(copy_per_window[q_mask_t].mean().item(), 6),
                "pairwise_distance": round(pairwise_distance, 6),
                "spread": round(spread, 6),
                "n_windows": n_q,
            }
            rows.append(row)

    elapsed = time.time() - t0
    print(f"[diversity] {encoder_name}/seed={seed}: done in {elapsed:.1f}s")
    print(f"  pairwise_distance (h=1): {rows[0]['pairwise_distance']:.6f}")
    print(f"  spread (h=1): {rows[0]['spread']:.6f}")

    return rows


# ===================================================================
# Entrypoint
# ===================================================================


def _modal_entrypoint_decorator(fn):
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_modal_entrypoint_decorator
def main():
    """DA9 Exp 1: Diversity diagnostic on Modal."""
    import csv as csv_mod

    t_start = time.time()

    encoders = PILOT_ENCODERS
    seeds = PILOT_SEEDS

    n_jobs = len(encoders) * len(seeds)
    print(f"\n{'='*70}")
    print(f"DA9 Exp 1: Diversity Diagnostic ({n_jobs} jobs)")
    print(f"  encoders: {encoders}")
    print(f"  seeds: {seeds}")
    print(f"  K: {K_SAMPLES}")
    print(f"  eta: 0.0 (deterministic DDIM, varying x_T)")
    print(f"  horizon: {HORIZON}")
    print(f"{'='*70}")

    # Launch all jobs
    futures = []
    for enc in encoders:
        for seed in seeds:
            futures.append((enc, seed, evaluate_diversity.spawn(enc, seed, K_SAMPLES)))

    # Collect results
    all_rows = []
    for enc, seed, future in futures:
        result = future.get()
        if result is not None:
            all_rows.extend(result)
            print(f"  {enc}/seed={seed}: {len(result)} rows")
        else:
            print(f"  {enc}/seed={seed}: MISSING checkpoint")

    wall_time = time.time() - t_start
    print(f"\nDone: {len(all_rows)} rows in {wall_time:.0f}s")

    if not all_rows:
        print("No results generated.")
        return

    # Save CSV locally
    csv_path = Path("artifacts/full/da9_diversity.csv")
    if not csv_path.parent.exists():
        csv_path = Path("code/latent-world-models-av/artifacts/full/da9_diversity.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved {len(all_rows)} rows to {csv_path}")

    # Print summary
    print(f"\n{'='*90}")
    print("Diversity Summary")
    print(f"{'='*90}")
    print(f"{'Encoder':<16} {'H':>3} {'PairDist':>10} {'Spread':>10} "
          f"{'Best-K':>10} {'Mean-K':>10} {'MLP-res':>10} {'Copy':>10}")
    print("-" * 90)

    for enc in encoders:
        enc_rows = [r for r in all_rows if r["encoder"] == enc]
        for h in [1, 4]:
            h_rows = [r for r in enc_rows if r["horizon_step"] == h]
            if not h_rows:
                continue
            # Average across quartiles (weighted by n_windows)
            total_w = sum(r["n_windows"] for r in h_rows)
            pw_dist = h_rows[0]["pairwise_distance"]  # same for all quartiles
            spread = h_rows[0]["spread"]
            best_k = sum(r["best_of_k_cossim"] * r["n_windows"] for r in h_rows) / total_w
            mean_k = sum(r["mean_of_k_cossim"] * r["n_windows"] for r in h_rows) / total_w
            mlp_res = sum(
                (r["mlp_residual_cossim"] or 0) * r["n_windows"] for r in h_rows
            ) / total_w
            copy_b = sum(r["copy_baseline"] * r["n_windows"] for r in h_rows) / total_w
            print(
                f"{enc:<16} {h:>3} {pw_dist:>10.6f} {spread:>10.6f} "
                f"{best_k:>10.4f} {mean_k:>10.4f} {mlp_res:>10.4f} {copy_b:>10.4f}"
            )

            if pw_dist < 0.001:
                print(f"  -> FINDING: Unimodal posterior (pairwise_dist ~ 0)")

    # Q1 detail
    print(f"\n{'='*90}")
    print("Q1 (hardest) Detail")
    print(f"{'='*90}")
    for enc in encoders:
        q1_rows = [
            r for r in all_rows
            if r["encoder"] == enc and r["quartile"] == "Q1 (hardest)"
        ]
        for r in q1_rows:
            mlp_r = r["mlp_residual_cossim"] or 0
            best_k = r["best_of_k_cossim"]
            mean_k = r["mean_of_k_cossim"]
            print(
                f"  {enc} h={r['horizon_step']}: "
                f"Best-K={best_k:.4f} Mean-K={mean_k:.4f} "
                f"MLP-res={mlp_r:.4f} Copy={r['copy_baseline']:.4f} "
                f"PairDist={r['pairwise_distance']:.6f}"
            )
