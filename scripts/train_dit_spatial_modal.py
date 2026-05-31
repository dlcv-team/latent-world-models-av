"""Train Spatial DiT + Spatial MLP on spatial patch-token embeddings.

Trains both models and saves checkpoints to Modal volume.
Evaluates with pool-then-compare CosSim against ground truth.

Usage::

    modal run scripts/train_dit_spatial_modal.py --encoder dino_vits14 --seed 0
    modal run scripts/train_dit_spatial_modal.py --encoder vit_s16 --seed 0
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("lwm-av-spatial-dit")
    vol = modal.Volume.from_name("nuscenes-full")
else:
    app = None
    vol = None

VOL_PATH = "/vol"
SPATIAL_DIR = f"{VOL_PATH}/embeddings/spatial"
CKPT_DIR = f"{VOL_PATH}/dits/spatial"

TARGET_DIM = 384

SPATIAL_TOKENS = {
    "vit_s16": 49,      # 7x7
    "dino_vits14": 64,   # 8x8
}

DIT_CONFIG = {
    "z_dim": 384,
    "cond_dim": 384,
    "n_blocks": 4,
    "n_heads": 6,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
}

MLP_CONFIG = {
    "z_dim": 384,
    "a_dim": 384,
    "hidden": 512,
    "dropout": 0.1,
}

FOURIER_CONFIG = {"n_frequencies": 64, "base": 2.0, "out_dim": 384}
DIFFUSION_STEPS = 1000
TRAIN_EPOCHS = 100
TRAIN_LR = 1e-4
TRAIN_BATCH = 64   # Smaller batch due to H*S tokens
EMA_DECAY = 0.999
N_DDIM_STEPS = 50

if modal is not None:
    base_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch==2.5.1", "numpy>=1.26")
    )
else:
    base_image = None


def _modal_function_decorator(fn):
    if app is not None:
        return app.function(
            volumes={VOL_PATH: vol},
            image=base_image,
            gpu="A100",  # A100 for attention-heavy spatial tokens (3-4x faster than A10G)
            timeout=14400,  # 4 hours
            memory=32768,
        )(fn)
    return fn


@_modal_function_decorator
def train_and_eval(
    encoder_name: str,
    seed: int,
    horizon: int = 16,
    model_type: str = "both",  # "dit", "mlp", or "both"
):
    """Train spatial DiT and/or MLP, then evaluate with pool-then-compare."""
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from copy import deepcopy

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_spatial = SPATIAL_TOKENS[encoder_name]

    print(f"[spatial-train] {encoder_name}/h{horizon}/s{seed}: S={n_spatial} tokens, device={device}")

    # ---- Inline model definitions ----
    # (Same as in latent_dit_spatial.py and latent_pred_spatial.py but inlined for Modal)

    class CosineNoiseSchedule(nn.Module):
        def __init__(self, n_steps=1000, s=0.008):
            super().__init__()
            steps = torch.arange(n_steps + 1, dtype=torch.float64)
            f_t = torch.cos(((steps / n_steps) + s) / (1 + s) * (torch.pi / 2)) ** 2
            alphas_cumprod = f_t / f_t[0]
            self.register_buffer("alphas_cumprod", alphas_cumprod[:n_steps].float())

    class TimestepEmbedding(nn.Module):
        def __init__(self, cond_dim=384):
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
            self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
            self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
            mlp_hidden = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, dim))
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.adaln_linear = nn.Linear(cond_dim, 6 * dim)

        def forward(self, x, cond):
            mod = self.adaln_linear(cond)
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mod.chunk(6, dim=-1)
            h = _modulate(self.norm_attn(x), shift_a, scale_a)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + gate_a * self.drop(attn_out)
            h = _modulate(self.norm_mlp(x), shift_m, scale_m)
            x = x + gate_m * self.drop(self.mlp(h))
            return x

    class SpatialTemporalDiT(nn.Module):
        def __init__(self, z_dim=384, cond_dim=384, n_blocks=4, n_heads=6,
                     horizon=16, n_spatial=49, mlp_ratio=4.0, dropout=0.0):
            super().__init__()
            self.z_dim = z_dim
            self.horizon = horizon
            self.n_spatial = n_spatial
            self.input_proj = nn.Linear(z_dim, z_dim)
            self.spatial_pos = nn.Parameter(torch.randn(1, n_spatial, z_dim) * 0.02)
            self.temporal_pos = nn.Parameter(torch.randn(1, horizon, z_dim) * 0.02)
            self.timestep_embed = TimestepEmbedding(cond_dim)
            self.z_t_proj = nn.Linear(z_dim, cond_dim)
            self.blocks = nn.ModuleList([
                DiTBlock(z_dim, cond_dim, n_heads, mlp_ratio, dropout)
                for _ in range(n_blocks)
            ])
            self.final_norm = nn.LayerNorm(z_dim, elementwise_affine=False)
            self.final_adaln = nn.Linear(cond_dim, 3 * z_dim)
            self.final_linear = nn.Linear(z_dim, z_dim)
            nn.init.zeros_(self.final_linear.weight)
            nn.init.zeros_(self.final_linear.bias)

        def forward(self, x_noisy, z_t_spatial, a_embed, timestep):
            B = x_noisy.shape[0]
            H, S, D = self.horizon, self.n_spatial, self.z_dim
            sp = self.spatial_pos.unsqueeze(1).expand(-1, H, -1, -1).reshape(1, H * S, D)
            tp = self.temporal_pos.unsqueeze(2).expand(-1, -1, S, -1).reshape(1, H * S, D)
            x = self.input_proj(x_noisy) + sp + tp
            z_t_pooled = z_t_spatial.mean(dim=1)
            cond_global = self.timestep_embed(timestep) + self.z_t_proj(z_t_pooled)
            a_broadcast = a_embed.unsqueeze(2).expand(-1, -1, S, -1).reshape(B, H * S, D)
            cond = cond_global.unsqueeze(1).expand(-1, H * S, -1) + a_broadcast
            for block in self.blocks:
                x = block(x, cond)
            mod = self.final_adaln(cond)
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
            if action.dim() == 2:
                action = action.unsqueeze(1)
            x = action.unsqueeze(-1) * self.freqs
            x = torch.cat([x.sin(), x.cos()], dim=-1)
            x = x.flatten(-2)
            return self.proj(x)

    class SpatialMLPPredictor(nn.Module):
        def __init__(self, z_dim=384, a_dim=384, horizon=16, n_spatial=49, hidden=512, dropout=0.1):
            super().__init__()
            self.horizon = horizon
            self.n_spatial = n_spatial
            input_dim = z_dim + z_dim + a_dim
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, z_dim),
            )

        def forward(self, z_t_spatial, a_embed):
            B, S, D = z_t_spatial.shape
            H = self.horizon
            z_t_pool = z_t_spatial.mean(dim=1)
            outputs = []
            for h in range(H):
                a_h = a_embed[:, h, :].unsqueeze(1).expand(-1, S, -1)
                z_pool_exp = z_t_pool.unsqueeze(1).expand(-1, S, -1)
                x = torch.cat([z_t_spatial, z_pool_exp, a_h], dim=-1)
                delta = self.net(x)
                outputs.append(z_t_spatial + delta)
            return torch.stack(outputs, dim=1).reshape(B, H * S, D)

    # ---- Load spatial embeddings ----

    spatial_path = f"{SPATIAL_DIR}/{encoder_name}_spatial.npz"
    if not os.path.exists(spatial_path):
        print(f"[spatial-train] ERROR: {spatial_path} not found!")
        return None

    data = np.load(spatial_path, allow_pickle=True)
    spatial_emb = data["spatial_embeddings"]  # (N, S, D)
    splits = data["splits"]
    steer_norms = data["steer_norms"]
    accel_norms = data["accel_norms"]
    scene_names = data["scene_names"]

    N_total, S_actual, D_actual = spatial_emb.shape
    assert S_actual == n_spatial, f"Expected S={n_spatial}, got {S_actual}"
    assert D_actual == TARGET_DIM, f"Expected D={TARGET_DIM}, got {D_actual}"
    print(f"[spatial-train] Loaded {N_total} frames, shape {spatial_emb.shape}")

    # ---- Build train/test windows ----

    def build_windows(split_name):
        mask = splits == split_name
        emb = spatial_emb[mask]
        steers = steer_norms[mask]
        accels = accel_norms[mask]
        scenes = scene_names[mask]

        z_t_list, action_seq_list, z_future_list = [], [], []
        for scene in np.unique(scenes):
            idx = np.where(scenes == scene)[0]
            for j in range(len(idx) - horizon):
                z_t_list.append(emb[idx[j]])           # (S, D)
                action_seq = np.stack([
                    np.array([steers[idx[j + k]], accels[idx[j + k]]])
                    for k in range(horizon)
                ])
                action_seq_list.append(action_seq)      # (H, 2)
                z_future_list.append(emb[idx[j + 1: j + 1 + horizon]])  # (H, S, D)

        return (
            torch.tensor(np.array(z_t_list), dtype=torch.float32),
            torch.tensor(np.array(action_seq_list), dtype=torch.float32),
            torch.tensor(np.array(z_future_list), dtype=torch.float32),
        )

    z_t_train, act_train, zf_train = build_windows("train")
    z_t_test, act_test, zf_test = build_windows("test")
    n_train, n_test = len(z_t_train), len(z_t_test)
    print(f"[spatial-train] Train: {n_train} windows, Test: {n_test} windows")

    # ---- Compute normalization stats (per spatial token, per dim) ----
    # Flatten train spatial embeddings to compute mean/std
    all_train_flat = z_t_train.reshape(-1, TARGET_DIM)  # (n_train*S, D)
    z_mean = all_train_flat.mean(dim=0).to(device)       # (D,)
    z_std = all_train_flat.std(dim=0).clamp(min=1e-6).to(device)  # (D,)

    def normalize(x):
        return (x - z_mean) / z_std

    def denormalize(x):
        return x * z_std + z_mean

    # ---- Helper: DDIM sampling ----
    def ddim_sample(model, fourier_embed, z_t_spatial_norm, act_seq, schedule):
        B = z_t_spatial_norm.shape[0]
        H, S, D = horizon, n_spatial, TARGET_DIM
        T = DIFFUSION_STEPS
        stride = T // N_DDIM_STEPS
        timesteps = list(reversed(list(range(0, T, stride))[:N_DDIM_STEPS]))
        alphas = schedule.alphas_cumprod

        a_embed = fourier_embed(act_seq)  # (B, H, D)
        x = torch.randn(B, H * S, D, device=device)

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            pred_x0 = model(x, z_t_spatial_norm, a_embed, t)
            alpha_t = alphas[t_val]
            alpha_prev = alphas[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)
            noise_dir = (x - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(1 - alpha_t + 1e-8)
            x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_dir

        return x  # (B, H*S, D)

    # ---- Helper: Evaluate with pool-then-compare ----
    def evaluate_model(predict_fn, z_t_dev, act_dev, zf_dev, label=""):
        """Evaluate spatial model using per-token CosSim (averaged across S).

        Computes CosSim independently for each spatial position, then averages.
        This preserves spatial discrimination that pool-then-compare washes out.
        """
        cossim_sums = [0.0] * horizon
        copy_sums = [0.0] * horizon
        total = 0
        eval_batch = 32

        with torch.no_grad():
            for start in range(0, len(z_t_dev), eval_batch):
                end = min(start + eval_batch, len(z_t_dev))
                B = end - start

                z_t_b = normalize(z_t_dev[start:end].to(device))   # (B, S, D)
                act_b = act_dev[start:end].to(device)               # (B, H, 2)
                zf_b = zf_dev[start:end].to(device)                 # (B, H, S, D)

                z_hat_norm = predict_fn(z_t_b, act_b)  # (B, H*S, D)
                z_hat = denormalize(z_hat_norm).reshape(B, horizon, n_spatial, TARGET_DIM)

                z_t_raw = z_t_dev[start:end].to(device)  # (B, S, D)

                for k in range(horizon):
                    # Per-token CosSim: compare each spatial position independently
                    # z_hat[:, k]: (B, S, D), zf_b[:, k]: (B, S, D)
                    # CosSim per token: (B, S), then mean over S
                    cs_per_token = F.cosine_similarity(
                        z_hat[:, k], zf_b[:, k], dim=-1  # (B, S)
                    )
                    cs_mean = cs_per_token.mean(dim=-1)  # (B,) -- mean over spatial
                    cossim_sums[k] += cs_mean.sum().item()

                    # Copy baseline: compare z_t[s] vs zf[k][s] per token
                    copy_per_token = F.cosine_similarity(
                        z_t_raw, zf_b[:, k], dim=-1  # (B, S)
                    )
                    copy_mean = copy_per_token.mean(dim=-1)  # (B,)
                    copy_sums[k] += copy_mean.sum().item()

                total += B

        result = {
            "cossim_by_step": [round(s / total, 6) for s in cossim_sums],
            "copy_by_step": [round(s / total, 6) for s in copy_sums],
            "mean_cossim": round(sum(cossim_sums) / (total * horizon), 6),
            "n_test": total,
        }
        print(f"  [{label}] mean CosSim: {result['mean_cossim']:.4f}, "
              f"step1: {result['cossim_by_step'][0]:.4f}, "
              f"stepN: {result['cossim_by_step'][-1]:.4f}")
        return result

    results = {}

    # ==================================================================
    # Train Spatial DiT
    # ==================================================================
    if model_type in ("dit", "both"):
        print(f"\n{'='*60}")
        print(f"Training Spatial DiT: {encoder_name}/h{horizon}/s{seed}")
        print(f"  Tokens per sample: {horizon * n_spatial} = {horizon}x{n_spatial}")
        print(f"{'='*60}")

        dit = SpatialTemporalDiT(
            **DIT_CONFIG, horizon=horizon, n_spatial=n_spatial,
        ).to(device)
        fourier = FourierActionEmbedding(**FOURIER_CONFIG).to(device)
        schedule = CosineNoiseSchedule(n_steps=DIFFUSION_STEPS).to(device)

        n_params = sum(p.numel() for p in dit.parameters()) + sum(p.numel() for p in fourier.parameters())
        print(f"  DiT params: {n_params:,}")

        optimizer = torch.optim.Adam(
            list(dit.parameters()) + list(fourier.parameters()), lr=TRAIN_LR,
        )

        # EMA
        ema_params = {}
        for name, param in list(dit.named_parameters()) + list(fourier.named_parameters()):
            ema_params[name] = param.data.clone()

        t_start = time.time()
        for epoch in range(TRAIN_EPOCHS):
            dit.train()
            fourier.train()
            perm = torch.randperm(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, TRAIN_BATCH):
                end = min(start + TRAIN_BATCH, n_train)
                idx = perm[start:end]
                B = len(idx)

                z_t_b = normalize(z_t_train[idx].to(device))   # (B, S, D)
                act_b = act_train[idx].to(device)               # (B, H, 2)
                zf_b = normalize(zf_train[idx].to(device).reshape(B, horizon * n_spatial, TARGET_DIM))

                # Diffusion: add noise to future spatial tokens
                t = torch.randint(0, DIFFUSION_STEPS, (B,), device=device)
                alpha_bar = schedule.alphas_cumprod[t].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
                noise = torch.randn_like(zf_b)
                x_noisy = torch.sqrt(alpha_bar) * zf_b + torch.sqrt(1 - alpha_bar) * noise

                a_embed = fourier(act_b)  # (B, H, D)
                pred_x0 = dit(x_noisy, z_t_b, a_embed, t)

                loss = F.mse_loss(pred_x0, zf_b)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
                optimizer.step()

                # EMA update
                with torch.no_grad():
                    for name, param in list(dit.named_parameters()) + list(fourier.named_parameters()):
                        ema_params[name].mul_(EMA_DECAY).add_(param.data, alpha=1 - EMA_DECAY)

                epoch_loss += loss.item()
                n_batches += 1

            if epoch % 20 == 0 or epoch == TRAIN_EPOCHS - 1:
                print(f"  epoch {epoch}: loss={epoch_loss/n_batches:.6f}")

        train_time = time.time() - t_start
        print(f"  Training done in {train_time:.0f}s")

        # Load EMA weights for evaluation
        dit_ema = deepcopy(dit)
        fourier_ema = deepcopy(fourier)
        with torch.no_grad():
            for name, param in dit_ema.named_parameters():
                if name in ema_params:
                    param.copy_(ema_params[name])
            for name, param in fourier_ema.named_parameters():
                if name in ema_params:
                    param.copy_(ema_params[name])
        dit_ema.eval()
        fourier_ema.eval()

        def dit_predict(z_t_b, act_b):
            return ddim_sample(dit_ema, fourier_ema, z_t_b, act_b, schedule)

        dit_result = evaluate_model(dit_predict, z_t_test, act_test, zf_test, "DiT-spatial")
        dit_result["model"] = "dit_spatial"
        dit_result["train_time_s"] = round(train_time, 1)
        dit_result["n_params"] = n_params
        results["dit_spatial"] = dit_result

        # Save checkpoint
        ckpt_dir = f"{CKPT_DIR}/{encoder_name}/h{horizon}/seed_{seed}"
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({
            "dit_state_dict": dit.state_dict(),
            "fourier_state_dict": fourier.state_dict(),
            "ema_params": ema_params,
            "z_mean": z_mean.cpu(),
            "z_std": z_std.cpu(),
            "config": {**DIT_CONFIG, "horizon": horizon, "n_spatial": n_spatial},
        }, f"{ckpt_dir}/dit_checkpoint.pt")
        print(f"  Saved DiT checkpoint to {ckpt_dir}")

    # ==================================================================
    # Train Spatial MLP
    # ==================================================================
    if model_type in ("mlp", "both"):
        print(f"\n{'='*60}")
        print(f"Training Spatial MLP: {encoder_name}/h{horizon}/s{seed}")
        print(f"{'='*60}")

        mlp = SpatialMLPPredictor(
            **MLP_CONFIG, horizon=horizon, n_spatial=n_spatial,
        ).to(device)
        fourier_mlp = FourierActionEmbedding(**FOURIER_CONFIG).to(device)

        n_params_mlp = sum(p.numel() for p in mlp.parameters()) + sum(p.numel() for p in fourier_mlp.parameters())
        print(f"  MLP params: {n_params_mlp:,}")

        optimizer_mlp = torch.optim.Adam(
            list(mlp.parameters()) + list(fourier_mlp.parameters()), lr=1e-3,
        )

        t_start = time.time()
        for epoch in range(50):  # MLP trains faster
            mlp.train()
            fourier_mlp.train()
            perm = torch.randperm(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, TRAIN_BATCH * 2):  # Larger batch OK for MLP
                end = min(start + TRAIN_BATCH * 2, n_train)
                idx = perm[start:end]
                B = len(idx)

                z_t_b = normalize(z_t_train[idx].to(device))
                act_b = act_train[idx].to(device)
                zf_b = normalize(zf_train[idx].to(device).reshape(B, horizon * n_spatial, TARGET_DIM))

                a_embed = fourier_mlp(act_b)
                z_hat = mlp(z_t_b, a_embed)

                loss = F.mse_loss(z_hat, zf_b)
                optimizer_mlp.zero_grad()
                loss.backward()
                optimizer_mlp.step()

                epoch_loss += loss.item()
                n_batches += 1

            if epoch % 10 == 0 or epoch == 49:
                print(f"  epoch {epoch}: loss={epoch_loss/n_batches:.6f}")

        mlp_train_time = time.time() - t_start
        print(f"  Training done in {mlp_train_time:.0f}s")

        mlp.eval()
        fourier_mlp.eval()

        def mlp_predict(z_t_b, act_b):
            a_embed = fourier_mlp(act_b)
            return mlp(z_t_b, a_embed)

        mlp_result = evaluate_model(mlp_predict, z_t_test, act_test, zf_test, "MLP-spatial")
        mlp_result["model"] = "mlp_spatial"
        mlp_result["train_time_s"] = round(mlp_train_time, 1)
        mlp_result["n_params"] = n_params_mlp
        results["mlp_spatial"] = mlp_result

        # Save checkpoint
        ckpt_dir_mlp = f"{CKPT_DIR}/{encoder_name}/h{horizon}/seed_{seed}"
        os.makedirs(ckpt_dir_mlp, exist_ok=True)
        torch.save({
            "mlp_state_dict": mlp.state_dict(),
            "fourier_state_dict": fourier_mlp.state_dict(),
            "z_mean": z_mean.cpu(),
            "z_std": z_std.cpu(),
            "config": {**MLP_CONFIG, "horizon": horizon, "n_spatial": n_spatial},
        }, f"{ckpt_dir_mlp}/mlp_checkpoint.pt")
        print(f"  Saved MLP checkpoint to {ckpt_dir_mlp}")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"RESULTS: {encoder_name}/h{horizon}/s{seed}")
    print(f"{'='*60}")
    for model_name, r in results.items():
        print(f"  {model_name}: mean_cossim={r['mean_cossim']:.4f}")

    if "dit_spatial" in results and "mlp_spatial" in results:
        dit_mean = results["dit_spatial"]["mean_cossim"]
        mlp_mean = results["mlp_spatial"]["mean_cossim"]
        gap = dit_mean - mlp_mean
        print(f"\n  DiT - MLP gap: {gap:+.4f}")
        print(f"  >>> {'DiT WINS!' if gap > 0 else 'MLP wins'} <<<")
        results["gap"] = round(gap, 6)
        results["dit_wins"] = gap > 0

    vol.commit()
    return results


# ===================================================================
# Entrypoint
# ===================================================================

def _modal_entrypoint_decorator(fn):
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_modal_entrypoint_decorator
def main(
    encoder: str = "dino_vits14",
    seed: int = 0,
    horizon: int = 16,
):
    """Train + evaluate spatial DiT and MLP."""
    if encoder not in SPATIAL_TOKENS:
        print(f"ERROR: encoder must be one of {list(SPATIAL_TOKENS.keys())}")
        return

    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"Spatial DiT Training Pipeline")
    print(f"  encoder: {encoder}, S={SPATIAL_TOKENS[encoder]}")
    print(f"  horizon: {horizon}, seed: {seed}")
    print(f"{'='*60}")

    result = train_and_eval.remote(encoder, seed, horizon)
    wall = time.time() - t_start
    print(f"\nDone in {wall:.0f}s")
    print(json.dumps(result, indent=2))

    # Save result locally
    out_path = Path(f"artifacts/full/spatial_dit_result_{encoder}_h{horizon}_s{seed}.json")
    if not out_path.parent.exists():
        out_path = Path(f"code/latent-world-models-av/artifacts/full/spatial_dit_result_{encoder}_h{horizon}_s{seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")
