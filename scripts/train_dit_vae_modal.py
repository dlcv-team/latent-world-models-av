"""VAE-latent direct-anchored DiT training (production pipeline bridge).

Encodes driving scenes as SD VAE 32x32x4 grids, patchifies to 8x8=64 tokens of 64-d,
trains direct-anchored DiT (no diffusion at 2Hz) vs MLP baseline (note: DiT ~5.4M
params vs MLP ~1.6M at hidden=1024, a 3.3x capacity advantage -- not param-matched).

Usage::

    modal run scripts/train_dit_vae_modal.py --smoke
    modal run --detach scripts/train_dit_vae_modal.py
    modal run scripts/train_dit_vae_modal.py --mode diffusion --smoke
    modal run --detach scripts/train_dit_vae_modal.py --mode diffusion --n-samples 16
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
    app = modal.App("lwm-av-vae-dit")
    vol = modal.Volume.from_name("nuscenes-full")
else:
    app = None
    vol = None

VOL_PATH = "/vol"
SPATIAL_DIR = f"{VOL_PATH}/embeddings/spatial"
CKPT_DIR = f"{VOL_PATH}/dits/vae_latent"
VAE_NPZ = f"{SPATIAL_DIR}/sd_vae_latents.npz"

PATCH_SIZE = 4
GRID_H = GRID_W = 32
N_SPATIAL = (GRID_H // PATCH_SIZE) * (GRID_W // PATCH_SIZE)  # 64
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 4  # 64
MODEL_DIM = 256
HORIZON = 16

DIT_CONFIG = {
    "z_dim": MODEL_DIM,
    "cond_dim": MODEL_DIM,
    "n_blocks": 4,
    "n_heads": 4,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
}
FOURIER_CONFIG = {"n_frequencies": 64, "base": 2.0, "out_dim": MODEL_DIM}
TRAIN_LR = 1e-4
TRAIN_BATCH = 64
EMA_DECAY = 0.999
DIFFUSION_STEPS = 1000
N_DDIM_STEPS = 50
N_DDIM_STEPS_SMOKE = 10
DIRECT_BASELINE_MEAN = 0.527
MLP_BASELINE_MEAN = 0.484
G6_WEAK_BEST_OF_K = 0.547
G6_DIVERSITY_MIN = 0.05
G7_BEST_OF_K_MIN = 0.03
G7_DIVERSITY_STRONG = 0.10
G7_DIVERSITY_PARTIAL = 0.05
G7_COND_DIRECT_REF = DIRECT_BASELINE_MEAN

if modal is not None:
    base_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.5.1", "numpy>=1.26", "Pillow>=10.0",
            "diffusers>=0.27", "matplotlib>=3.8", "accelerate", "transformers>=4.50",
        )
    )
else:
    base_image = None


def patchify(latents, patch_size=PATCH_SIZE):
    """(B, 4, 32, 32) -> (B, 64, 64)"""
    B, C, H, W = latents.shape
    p = patch_size
    x = latents.reshape(B, C, H // p, p, W // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // p) * (W // p), C * p * p)
    return x


def unpatchify(tokens, patch_size=PATCH_SIZE, channels=4, grid_h=GRID_H, grid_w=GRID_W):
    """(B, 64, 64) -> (B, 4, 32, 32)"""
    B = tokens.shape[0]
    p = patch_size
    gh, gw = grid_h // p, grid_w // p
    x = tokens.reshape(B, gh, gw, channels, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, channels, grid_h, grid_w)
    return x


def _modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


def _define_noise_schedule():
    import torch.nn as nn

    class CosineNoiseSchedule(nn.Module):
        def __init__(self, n_steps=1000, s=0.008):
            super().__init__()
            steps = torch.arange(n_steps + 1, dtype=torch.float64)
            f_t = torch.cos(((steps / n_steps) + s) / (1 + s) * (math.pi / 2)) ** 2
            alphas_cumprod = f_t / f_t[0]
            self.register_buffer("alphas_cumprod", alphas_cumprod[:n_steps].float())

        def forward(self):
            return self.alphas_cumprod

    return CosineNoiseSchedule


def _define_vae_models():
    import torch
    import torch.nn as nn

    class TimestepEmbedding(nn.Module):
        def __init__(self, cond_dim=MODEL_DIM):
            super().__init__()
            self.cond_dim = cond_dim
            self.mlp = nn.Sequential(
                nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim),
            )

        def forward(self, timestep):
            half = self.cond_dim // 2
            freqs = torch.exp(
                -math.log(10000.0) * torch.arange(half, device=timestep.device, dtype=torch.float32) / half
            )
            args = timestep.float().unsqueeze(-1) * freqs.unsqueeze(0)
            emb = torch.cat([args.sin(), args.cos()], dim=-1)
            return self.mlp(emb)

    class DiTBlock(nn.Module):
        def __init__(self, dim=MODEL_DIM, cond_dim=MODEL_DIM, n_heads=4, mlp_ratio=4.0, dropout=0.0):
            super().__init__()
            self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False)
            self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
            self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
            hidden = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
            self.adaln_linear = nn.Linear(cond_dim, 6 * dim)

        def forward(self, x, cond):
            mod = self.adaln_linear(cond)
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mod.chunk(6, dim=-1)
            h = _modulate(self.norm_attn(x), shift_a, scale_a)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + gate_a * attn_out
            h = _modulate(self.norm_mlp(x), shift_m, scale_m)
            x = x + gate_m * self.mlp(h)
            return x

    class AnchoredVAEDiT(nn.Module):
        def __init__(self, horizon=16, n_spatial=64, patch_dim=64, model_dim=256, **dit_kw):
            super().__init__()
            self.horizon = horizon
            self.n_spatial = n_spatial
            self.patch_dim = patch_dim
            self.model_dim = model_dim
            self.latent_up = nn.Linear(patch_dim, model_dim)
            self.latent_down = nn.Linear(model_dim, patch_dim)
            self.input_proj = nn.Linear(model_dim, model_dim)
            self.spatial_pos = nn.Parameter(torch.randn(1, n_spatial, model_dim) * 0.02)
            self.temporal_pos = nn.Parameter(torch.randn(1, horizon, model_dim) * 0.02)
            self.timestep_embed = TimestepEmbedding(model_dim)
            self.z_t_proj = nn.Linear(patch_dim, model_dim)
            n_heads = dit_kw.get("n_heads", 4)
            n_blocks = dit_kw.get("n_blocks", 4)
            mlp_ratio = dit_kw.get("mlp_ratio", 4.0)
            self.blocks = nn.ModuleList([
                DiTBlock(model_dim, model_dim, n_heads, mlp_ratio, 0.0) for _ in range(n_blocks)
            ])
            self.final_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
            self.final_adaln = nn.Linear(model_dim, 3 * model_dim)
            self.final_linear = nn.Linear(model_dim, model_dim)
            nn.init.zeros_(self.final_linear.weight)
            nn.init.zeros_(self.final_linear.bias)

        def forward(self, x_input_patch, z_t_patch, a_embed, timestep):
            B = x_input_patch.shape[0]
            H, S, Pd = self.horizon, self.n_spatial, self.patch_dim
            Md = self.model_dim
            x = self.latent_up(x_input_patch)
            sp = self.spatial_pos.unsqueeze(1).expand(-1, H, -1, -1).reshape(1, H * S, Md)
            tp = self.temporal_pos.unsqueeze(2).expand(-1, -1, S, -1).reshape(1, H * S, Md)
            x = self.input_proj(x) + sp + tp
            z_t_pooled = z_t_patch.mean(dim=1)
            cond_global = self.timestep_embed(timestep) + self.z_t_proj(z_t_pooled)
            a_broadcast = a_embed.unsqueeze(2).expand(-1, -1, S, -1).reshape(B, H * S, Md)
            cond = cond_global.unsqueeze(1).expand(-1, H * S, -1) + a_broadcast
            for block in self.blocks:
                x = block(x, cond)
            mod = self.final_adaln(cond)
            shift, scale, gate = mod.chunk(3, dim=-1)
            delta_md = gate * self.final_linear(_modulate(self.final_norm(x), shift, scale))
            delta_patch = self.latent_down(delta_md)
            z_t_rep = z_t_patch.unsqueeze(1).expand(-1, H, -1, -1).reshape(B, H * S, Pd)
            return z_t_rep + delta_patch

    class FourierActionEmbedding(nn.Module):
        def __init__(self, action_dim=2, n_frequencies=64, base=2.0, out_dim=MODEL_DIM):
            super().__init__()
            freqs = base ** torch.arange(n_frequencies, dtype=torch.float32) * torch.pi
            self.register_buffer("freqs", freqs)
            fdim = action_dim * 2 * n_frequencies
            self.proj = nn.Sequential(
                nn.Linear(fdim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim),
            )

        def forward(self, action):
            import torch
            if action.dim() == 2:
                action = action.unsqueeze(1)
            x = action.unsqueeze(-1) * self.freqs
            x = torch.cat([x.sin(), x.cos()], dim=-1).flatten(-2)
            return self.proj(x)

    class PatchMLPPredictor(nn.Module):
        def __init__(self, patch_dim=64, a_dim=MODEL_DIM, horizon=16, n_spatial=64, hidden=1024, dropout=0.1):
            super().__init__()
            self.horizon = horizon
            self.n_spatial = n_spatial
            input_dim = patch_dim * 2 + a_dim
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, patch_dim),
            )

        def forward(self, z_t_patch, a_embed):
            import torch
            B, S, Pd = z_t_patch.shape
            H = self.horizon
            z_pool = z_t_patch.mean(dim=1)
            outs = []
            for h in range(H):
                a_h = a_embed[:, h, :].unsqueeze(1).expand(-1, S, -1)
                z_pool_e = z_pool.unsqueeze(1).expand(-1, S, -1)
                x = torch.cat([z_t_patch, z_pool_e, a_h], dim=-1)
                outs.append(z_t_patch + self.net(x))
            return torch.stack(outs, dim=1).reshape(B, H * S, Pd)

    return TimestepEmbedding, DiTBlock, AnchoredVAEDiT, FourierActionEmbedding, PatchMLPPredictor


try:
    import torch
    (
        TimestepEmbedding,
        DiTBlock,
        AnchoredVAEDiT,
        FourierActionEmbedding,
        PatchMLPPredictor,
    ) = _define_vae_models()
    CosineNoiseSchedule = _define_noise_schedule()
except ImportError:
    TimestepEmbedding = DiTBlock = AnchoredVAEDiT = None
    FourierActionEmbedding = PatchMLPPredictor = None
    CosineNoiseSchedule = None


def evaluate_g7(result: dict) -> dict:
    """Pre-registered G7 gate for action-marginalized multi-future (Exp B)."""
    dist = result.get("dit_distributional") or {}
    uncond_direct = float(result.get("uncond_direct", {}).get("mean_cossim", 0))
    best = float(dist.get("best_of_k_mean_cossim", 0))
    div = float(dist.get("sample_diversity_l2", 0))
    cond_direct_ref = float(result.get("conditioned_direct_mean", G7_COND_DIRECT_REF))
    gap_best = best - uncond_direct
    cond_ok = abs(float(result.get("dit", {}).get("mean_cossim", 0)) - cond_direct_ref) <= 0.03
    if gap_best >= G7_BEST_OF_K_MIN and div > G7_DIVERSITY_STRONG and cond_ok:
        outcome = "strong_pass"
    elif gap_best >= 0.01 and div > G7_DIVERSITY_PARTIAL:
        outcome = "partial"
    else:
        outcome = "fail"
    return {
        "outcome": outcome,
        "best_of_k_mean_cossim": best,
        "uncond_direct_mean": uncond_direct,
        "best_minus_uncond_direct": round(gap_best, 6),
        "sample_diversity_l2": div,
        "conditioned_direct_mean": cond_direct_ref,
        "eval_subset": result.get("eval_subset", "high_action_variance"),
        "K": dist.get("n_samples"),
    }


def evaluate_g6(result: dict) -> dict:
    """Autonomous G6 gate from diffusion JSON vs fixed direct baseline."""
    mean = float(result.get("dit", {}).get("mean_cossim", 0))
    dist = result.get("dit_distributional") or {}
    best = float(dist.get("best_of_k_mean_cossim", 0))
    div = float(dist.get("sample_diversity_l2", 0))
    sane = mean == mean and best == best  # NaN check
    if sane and mean >= DIRECT_BASELINE_MEAN and mean >= MLP_BASELINE_MEAN:
        outcome = "strong_pass"
    elif mean < DIRECT_BASELINE_MEAN and best >= G6_WEAK_BEST_OF_K and div > G6_DIVERSITY_MIN:
        outcome = "weak_pass"
    else:
        outcome = "fail"
    return {
        "outcome": outcome,
        "mean_cossim": mean,
        "best_of_k_mean_cossim": best,
        "sample_diversity_l2": div,
        "direct_baseline_mean": DIRECT_BASELINE_MEAN,
        "mlp_baseline_mean": MLP_BASELINE_MEAN,
    }


def _modal_function_decorator(fn):
    if app is not None:
        return app.function(
            volumes={VOL_PATH: vol},
            image=base_image,
            gpu="A100",
            timeout=14400,
            memory=32768,
        )(fn)
    return fn


@_modal_function_decorator
def train_and_eval(
    seed: int = 0,
    horizon: int = 16,
    epochs: int = 100,
    smoke: bool = False,
    mlp_hidden: int = 1024,
    mode: str = "direct",
    n_samples: int = 1,
    action_dropout: float = 0.0,
    cfg_dropout: float = 0.0,
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from copy import deepcopy

    max_train = max_test = None
    if smoke:
        epochs = 4
        max_train = 2000
        max_test = 256

    mlp_epochs = max(epochs // 2, 10) if not smoke else 4

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_spatial = N_SPATIAL
    token_dim = PATCH_DIM

    assert mode in ("direct", "diffusion"), f"bad mode {mode}"
    is_diffusion = mode == "diffusion"
    n_ddim_steps = N_DDIM_STEPS_SMOKE if smoke else N_DDIM_STEPS
    print(f"[vae-dit] h{horizon}/s{seed} mode={mode} smoke={smoke} "
          f"mlp_hidden={mlp_hidden} n_samples={n_samples} action_dropout={action_dropout} device={device}")

    # ---- Load VAE latents ----
    if not os.path.exists(VAE_NPZ):
        print(f"[vae-dit] ERROR: {VAE_NPZ} not found. Run embed_vae_modal.py first.")
        return None

    data = np.load(VAE_NPZ, allow_pickle=True)
    vae_latents = data["vae_latents"]  # (N, 4, 32, 32)
    scene_names = data["scene_names"]
    splits = data["splits"]
    steer_norms = data["steer_norms"]
    accel_norms = data["accel_norms"]

    N_total = len(scene_names)
    print(f"[vae-dit] Loaded {N_total} VAE latents, shape {vae_latents.shape}")

    def build_windows(split_name):
        mask = splits == split_name
        lat = vae_latents[mask]
        steers = steer_norms[mask]
        accels = accel_norms[mask]
        scenes = scene_names[mask]
        z_t_list, act_list, zf_list = [], [], []
        for scene in np.unique(scenes):
            idx = np.where(scenes == scene)[0]
            for j in range(len(idx) - horizon):
                z_t_list.append(lat[idx[j]])
                act_list.append(np.stack([
                    np.array([steers[idx[j + k]], accels[idx[j + k]]]) for k in range(horizon)
                ]))
                zf_list.append(lat[idx[j + 1: j + 1 + horizon]])
        return (
            torch.tensor(np.array(z_t_list), dtype=torch.float32),
            torch.tensor(np.array(act_list), dtype=torch.float32),
            torch.tensor(np.array(zf_list), dtype=torch.float32),
        )

    z_t_train, act_train, zf_train = build_windows("train")
    z_t_test, act_test, zf_test = build_windows("test")
    if max_train and len(z_t_train) > max_train:
        z_t_train, act_train, zf_train = z_t_train[:max_train], act_train[:max_train], zf_train[:max_train]
    if max_test and len(z_t_test) > max_test:
        z_t_test, act_test, zf_test = z_t_test[:max_test], act_test[:max_test], zf_test[:max_test]

    n_train, n_test = len(z_t_train), len(z_t_test)
    print(f"[vae-dit] Train {n_train}, Test {n_test} windows")

    # Normalize patch tokens
    z_t_patch_train = patchify(z_t_train)
    flat = z_t_patch_train.reshape(-1, PATCH_DIM)
    z_mean = flat.mean(dim=0).to(device)
    z_std = flat.std(dim=0).clamp(min=1e-6).to(device)

    def norm_patches(grid_b):
        p = patchify(grid_b)
        return (p - z_mean) / z_std

    def denorm_tokens(tok):
        return tok * z_std + z_mean

    def embed_actions(fourier_embed, act_seq, dropout_p=0.0, force_zero=False):
        a_embed = fourier_embed(act_seq)
        if force_zero:
            return torch.zeros_like(a_embed)
        if dropout_p > 0 and fourier_embed.training:
            mask = (torch.rand(act_seq.shape[0], device=device) >= dropout_p).float()
            a_embed = a_embed * mask.view(-1, 1, 1)
        return a_embed

    def ddim_sample(model, fourier_embed, z_t_patch_norm, act_seq, schedule, n_steps, uncond=False):
        B = z_t_patch_norm.shape[0]
        H, S, D = horizon, n_spatial, PATCH_DIM
        T = DIFFUSION_STEPS
        stride = max(T // n_steps, 1)
        timesteps = list(reversed(list(range(0, T, stride))[:n_steps]))
        alphas = schedule.alphas_cumprod
        a_embed = embed_actions(fourier_embed, act_seq, force_zero=uncond)
        x = torch.randn(B, H * S, D, device=device)
        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            pred_x0 = model(x, z_t_patch_norm, a_embed, t)
            alpha_t = alphas[t_val]
            alpha_prev = alphas[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)
            noise_dir = (x - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(1 - alpha_t + 1e-8)
            x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_dir
        return x

    def ddim_sample_multi(model, fourier_embed, z_t_b, act_b, schedule, k, n_steps, uncond=False):
        return torch.stack([
            ddim_sample(model, fourier_embed, z_t_b, act_b, schedule, n_steps, uncond=uncond)
            for _ in range(k)
        ], dim=0)

    # High action-variance subset (top quartile steer std within horizon)
    steer_std_test = act_test[:, :, 0].std(dim=1)
    hv_threshold = float(torch.quantile(steer_std_test, 0.75).item())
    hv_mask = steer_std_test >= hv_threshold
    hv_indices = torch.where(hv_mask)[0]
    print(f"[vae-dit] high-variance windows: {len(hv_indices)}/{n_test} (steer_std>={hv_threshold:.4f})")

    def evaluate(predict_fn, label="", indices=None):
        cs_sums = [0.0] * horizon
        copy_sums = [0.0] * horizon
        total = 0
        idx_list = indices.tolist() if indices is not None else list(range(n_test))
        with torch.no_grad():
            for start in range(0, len(idx_list), 32):
                end = min(start + 32, len(idx_list))
                batch_idx = idx_list[start:end]
                B = len(batch_idx)
                z_t_b = norm_patches(z_t_test[batch_idx].to(device))
                act_b = act_test[batch_idx].to(device)
                zf_b = zf_test[batch_idx].to(device)
                z_hat = predict_fn(z_t_b, act_b)
                z_hat = denorm_tokens(z_hat).reshape(B, horizon, n_spatial, PATCH_DIM)
                z_t_raw = patchify(z_t_test[batch_idx].to(device))
                for k in range(horizon):
                    gt = patchify(zf_b[:, k])
                    cs_sums[k] += F.cosine_similarity(z_hat[:, k], gt, dim=-1).mean(dim=-1).sum().item()
                    copy_sums[k] += F.cosine_similarity(z_t_raw, gt, dim=-1).mean(dim=-1).sum().item()
                total += B
        res = {
            "cossim_by_step": [round(s / total, 6) for s in cs_sums],
            "copy_by_step": [round(s / total, 6) for s in copy_sums],
            "mean_cossim": round(sum(cs_sums) / (total * horizon), 6),
            "n_test": total,
        }
        print(f"  [{label}] mean={res['mean_cossim']:.4f} step1={res['cossim_by_step'][0]:.4f}")
        return res

    def evaluate_distributional(model, fourier_embed, schedule, k, n_steps, uncond=False, indices=None):
        mean_sums = [0.0] * horizon
        best_sums = [0.0] * horizon
        div_sums = 0.0
        total = 0
        idx_list = indices.tolist() if indices is not None else list(range(n_test))
        eval_batch = 8
        with torch.no_grad():
            for start in range(0, len(idx_list), eval_batch):
                end = min(start + eval_batch, len(idx_list))
                batch_idx = idx_list[start:end]
                B = len(batch_idx)
                z_t_b = norm_patches(z_t_test[batch_idx].to(device))
                act_b = act_test[batch_idx].to(device)
                zf_b = zf_test[batch_idx].to(device)
                samples = ddim_sample_multi(
                    model, fourier_embed, z_t_b, act_b, schedule, k, n_steps, uncond=uncond,
                )
                samples = denorm_tokens(samples).reshape(k, B, horizon, n_spatial, PATCH_DIM)
                for b in range(B):
                    for step in range(horizon):
                        gt = patchify(zf_b[:, step])
                        cs_list = [
                            F.cosine_similarity(samples[ki, b, step], gt[b], dim=-1).mean().item()
                            for ki in range(k)
                        ]
                        mean_sums[step] += sum(cs_list) / k
                        best_sums[step] += max(cs_list)
                    if k >= 2:
                        flat = samples[:, b].reshape(k, -1)
                        pw = torch.cdist(flat, flat, p=2)
                        div_sums += pw[torch.triu(torch.ones(k, k, device=device), diagonal=1) == 1].mean().item()
                total += B
        return {
            "n_samples": k,
            "mean_cossim": round(sum(mean_sums) / (total * horizon), 6),
            "best_of_k_mean_cossim": round(sum(best_sums) / (total * horizon), 6),
            "sample_diversity_l2": round(div_sums / total, 6) if k >= 2 else 0.0,
            "n_test": total,
            "uncond": uncond,
        }

    results = {
        "smoke": smoke,
        "mlp_hidden": mlp_hidden,
        "mode": mode,
        "action_dropout": action_dropout,
        "direct_baseline_mean": DIRECT_BASELINE_MEAN,
        "mlp_baseline_mean": MLP_BASELINE_MEAN,
        "conditioned_direct_mean": DIRECT_BASELINE_MEAN,
        "eval_subset": "high_action_variance_top_quartile_steer_std",
        "hv_threshold_steer_std": round(hv_threshold, 6),
        "n_hv_test": int(len(hv_indices)),
    }

    if is_diffusion:
        schedule = CosineNoiseSchedule(n_steps=DIFFUSION_STEPS).to(device)
        print(f"\n{'='*60}\nTraining VAE DiT-diffusion (DDIM steps={n_ddim_steps})\n{'='*60}")
        dit = AnchoredVAEDiT(horizon=horizon, n_spatial=n_spatial, **DIT_CONFIG).to(device)
        fourier = FourierActionEmbedding(**FOURIER_CONFIG).to(device)
        n_dit = sum(p.numel() for p in dit.parameters()) + sum(p.numel() for p in fourier.parameters())
        print(f"  DiT params: {n_dit:,}")
        opt = torch.optim.Adam(list(dit.parameters()) + list(fourier.parameters()), lr=TRAIN_LR)
        ema = {n: p.data.clone() for n, p in list(dit.named_parameters()) + list(fourier.named_parameters())}
        epoch_losses = []
        for epoch in range(epochs):
            dit.train()
            fourier.train()
            perm = torch.randperm(n_train)
            loss_sum, nb = 0.0, 0
            for start in range(0, n_train, TRAIN_BATCH):
                end = min(start + TRAIN_BATCH, n_train)
                idx = perm[start:end]
                B = len(idx)
                z_t_b = norm_patches(z_t_train[idx].to(device))
                act_b = act_train[idx].to(device)
                zf_b = norm_patches(zf_train[idx].to(device).reshape(B * horizon, 4, GRID_H, GRID_W))
                zf_b = zf_b.reshape(B, horizon * n_spatial, PATCH_DIM)
                a_emb = embed_actions(fourier, act_b, dropout_p=max(action_dropout, cfg_dropout))
                t = torch.randint(0, DIFFUSION_STEPS, (B,), device=device)
                alpha_bar = schedule.alphas_cumprod[t].unsqueeze(1).unsqueeze(2)
                noise = torch.randn_like(zf_b)
                x_noisy = torch.sqrt(alpha_bar) * zf_b + torch.sqrt(1 - alpha_bar) * noise
                pred = dit(x_noisy, z_t_b, a_emb, t)
                loss = F.mse_loss(pred, zf_b)
                if not torch.isfinite(loss):
                    print(f"  ERROR: non-finite loss at epoch {epoch}")
                    results["smoke_fail"] = "non_finite_loss"
                    vol.commit()
                    return results
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    for n, p in list(dit.named_parameters()) + list(fourier.named_parameters()):
                        ema[n].mul_(EMA_DECAY).add_(p.data, alpha=1 - EMA_DECAY)
                loss_sum += loss.item()
                nb += 1
            el = loss_sum / max(nb, 1)
            epoch_losses.append(round(el, 6))
            if epoch % max(epochs // 5, 1) == 0 or epoch == epochs - 1:
                print(f"  epoch {epoch}/{epochs}: loss={el:.6f}")
        results["epoch_losses"] = epoch_losses
        dit_e = deepcopy(dit)
        f_e = deepcopy(fourier)
        with torch.no_grad():
            for n, p in dit_e.named_parameters():
                if n in ema:
                    p.copy_(ema[n])
            for n, p in f_e.named_parameters():
                if n in ema:
                    p.copy_(ema[n])
        dit_e.eval()
        f_e.eval()

        eval_idx = hv_indices if action_dropout > 0 else None

        if action_dropout > 0:
            k = max(n_samples, 16)
            print(f"\n  Exp B eval on high-variance subset (K={k}, uncond sampling)")
            dist = evaluate_distributional(
                dit_e, f_e, schedule, k, n_ddim_steps, uncond=True, indices=eval_idx,
            )
            results["dit_distributional"] = dist
            results["dit"] = {"mean_cossim": dist["mean_cossim"], "n_params": n_dit, "n_ddim_steps": n_ddim_steps}
            print(f"    uncond diffusion mean={dist['mean_cossim']:.4f} "
                  f"best-of-K={dist['best_of_k_mean_cossim']:.4f} div={dist['sample_diversity_l2']:.4f}")

            # Unconditioned direct baseline (actions always zeroed)
            print(f"\n{'='*60}\nTraining unconditioned direct baseline\n{'='*60}")
            dit_u = AnchoredVAEDiT(horizon=horizon, n_spatial=n_spatial, **DIT_CONFIG).to(device)
            f_u = FourierActionEmbedding(**FOURIER_CONFIG).to(device)
            opt_u = torch.optim.Adam(list(dit_u.parameters()) + list(f_u.parameters()), lr=TRAIN_LR)
            u_epochs = epochs if not smoke else epochs
            for epoch in range(u_epochs):
                dit_u.train()
                f_u.train()
                perm = torch.randperm(n_train)
                for start in range(0, n_train, TRAIN_BATCH):
                    end = min(start + TRAIN_BATCH, n_train)
                    idx = perm[start:end]
                    B = len(idx)
                    z_t_b = norm_patches(z_t_train[idx].to(device))
                    act_b = act_train[idx].to(device)
                    zf_b = norm_patches(zf_train[idx].to(device).reshape(B * horizon, 4, GRID_H, GRID_W))
                    zf_b = zf_b.reshape(B, horizon * n_spatial, PATCH_DIM)
                    a_emb = embed_actions(f_u, act_b, force_zero=True)
                    z_rep = z_t_b.unsqueeze(1).expand(-1, horizon, -1, -1).reshape(B, horizon * n_spatial, PATCH_DIM)
                    t0 = torch.zeros(B, dtype=torch.long, device=device)
                    pred = dit_u(z_rep, z_t_b, a_emb, t0)
                    loss = F.mse_loss(pred, zf_b)
                    opt_u.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(dit_u.parameters(), 1.0)
                    opt_u.step()
                if epoch % max(u_epochs // 5, 1) == 0 or epoch == u_epochs - 1:
                    print(f"  uncond-direct epoch {epoch}/{u_epochs}")

            dit_u.eval()
            f_u.eval()

            def uncond_direct_pred(z_t_b, act_b):
                B = z_t_b.shape[0]
                z_rep = z_t_b.unsqueeze(1).expand(-1, horizon, -1, -1).reshape(B, horizon * n_spatial, PATCH_DIM)
                t0 = torch.zeros(B, dtype=torch.long, device=device)
                return dit_u(z_rep, z_t_b, embed_actions(f_u, act_b, force_zero=True), t0)

            results["uncond_direct"] = evaluate(uncond_direct_pred, "uncond-direct", indices=eval_idx)
            def copy_pred(z_t_b, act_b):
                B = z_t_b.shape[0]
                return z_t_b.unsqueeze(1).expand(-1, horizon, -1, -1).reshape(
                    B, horizon * n_spatial, PATCH_DIM
                )

            results["copy_baseline"] = evaluate(copy_pred, "copy", indices=eval_idx)
            results["g7"] = evaluate_g7(results)
            print(f"\n  G7 outcome: {results['g7']['outcome']}")
            if not smoke:
                ckpt_dir = f"{CKPT_DIR}/diffusion_ad{action_dropout}/h{horizon}/seed_{seed}"
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    "dit": dit.state_dict(),
                    "fourier": fourier.state_dict(),
                    "ema": ema,
                    "z_mean": z_mean.cpu(),
                    "z_std": z_std.cpu(),
                    "mode": "diffusion",
                    "action_dropout": action_dropout,
                    "uncond_direct": dit_u.state_dict(),
                    "uncond_fourier": f_u.state_dict(),
                }, f"{ckpt_dir}/dit.pt")
        else:
            def dit_predict(z_t_b, act_b):
                return ddim_sample(dit_e, f_e, z_t_b, act_b, schedule, n_ddim_steps)

            dit_res = evaluate(dit_predict, "DiT-vae-diffusion")
            dit_res["n_params"] = n_dit
            dit_res["n_ddim_steps"] = n_ddim_steps
            results["dit"] = dit_res

            if n_samples > 1:
                k = n_samples
                print(f"\n  Distributional eval: K={k}")
                dist = evaluate_distributional(dit_e, f_e, schedule, k, n_ddim_steps, uncond=False)
                results["dit_distributional"] = dist
                print(f"    best-of-K={dist['best_of_k_mean_cossim']:.4f} "
                      f"diversity={dist['sample_diversity_l2']:.4f}")

            if not smoke:
                ckpt_dir = f"{CKPT_DIR}/diffusion/h{horizon}/seed_{seed}"
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    "dit": dit.state_dict(),
                    "fourier": fourier.state_dict(),
                    "ema": ema,
                    "z_mean": z_mean.cpu(),
                    "z_std": z_std.cpu(),
                    "mode": "diffusion",
                    "cfg_dropout": cfg_dropout,
                }, f"{ckpt_dir}/dit.pt")
            results["g6"] = evaluate_g6(results)
            print(f"\n  G6 outcome: {results['g6']['outcome']}")

        vol.commit()
        return results

    # ---- Train DiT (direct) ----
    print(f"\n{'='*60}\nTraining VAE DiT-direct\n{'='*60}")
    dit = AnchoredVAEDiT(horizon=horizon, n_spatial=n_spatial, **DIT_CONFIG).to(device)
    fourier = FourierActionEmbedding(**FOURIER_CONFIG).to(device)
    n_dit = sum(p.numel() for p in dit.parameters()) + sum(p.numel() for p in fourier.parameters())
    print(f"  DiT params: {n_dit:,}")

    opt = torch.optim.Adam(list(dit.parameters()) + list(fourier.parameters()), lr=TRAIN_LR)
    ema = {n: p.data.clone() for n, p in list(dit.named_parameters()) + list(fourier.named_parameters())}

    for epoch in range(epochs):
        dit.train()
        fourier.train()
        perm = torch.randperm(n_train)
        loss_sum, nb = 0.0, 0
        for start in range(0, n_train, TRAIN_BATCH):
            end = min(start + TRAIN_BATCH, n_train)
            idx = perm[start:end]
            B = len(idx)
            z_t_b = norm_patches(z_t_train[idx].to(device))
            act_b = act_train[idx].to(device)
            zf_b = norm_patches(zf_train[idx].to(device).reshape(B * horizon, 4, GRID_H, GRID_W))
            zf_b = zf_b.reshape(B, horizon * n_spatial, PATCH_DIM)
            a_emb = fourier(act_b)
            z_rep = z_t_b.unsqueeze(1).expand(-1, horizon, -1, -1).reshape(B, horizon * n_spatial, PATCH_DIM)
            t0 = torch.zeros(B, dtype=torch.long, device=device)
            pred = dit(z_rep, z_t_b, a_emb, t0)
            loss = F.mse_loss(pred, zf_b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for n, p in list(dit.named_parameters()) + list(fourier.named_parameters()):
                    ema[n].mul_(EMA_DECAY).add_(p.data, alpha=1 - EMA_DECAY)
            loss_sum += loss.item()
            nb += 1
        if epoch % max(epochs // 5, 1) == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch}/{epochs}: loss={loss_sum/nb:.6f}")

    dit_e = deepcopy(dit)
    f_e = deepcopy(fourier)
    with torch.no_grad():
        for n, p in dit_e.named_parameters():
            if n in ema:
                p.copy_(ema[n])
        for n, p in f_e.named_parameters():
            if n in ema:
                p.copy_(ema[n])
    dit_e.eval()
    f_e.eval()

    def dit_pred(z_t_b, act_b):
        B = z_t_b.shape[0]
        z_rep = z_t_b.unsqueeze(1).expand(-1, horizon, -1, -1).reshape(B, horizon * n_spatial, PATCH_DIM)
        t0 = torch.zeros(B, dtype=torch.long, device=device)
        return dit_e(z_rep, z_t_b, f_e(act_b), t0)

    dit_res = evaluate(dit_pred, "DiT-vae-direct")
    dit_res["n_params"] = n_dit
    results["dit"] = dit_res

    if not smoke:
        os.makedirs(f"{CKPT_DIR}/h{horizon}/seed_{seed}", exist_ok=True)
        torch.save({
            "dit": dit.state_dict(),
            "fourier": fourier.state_dict(),
            "ema": ema,
            "z_mean": z_mean.cpu(),
            "z_std": z_std.cpu(),
        }, f"{CKPT_DIR}/h{horizon}/seed_{seed}/dit.pt")

    # ---- MLP ----
    print(f"\n{'='*60}\nTraining VAE MLP (hidden={mlp_hidden})\n{'='*60}")
    mlp = PatchMLPPredictor(hidden=mlp_hidden, horizon=horizon, n_spatial=n_spatial).to(device)
    f_mlp = FourierActionEmbedding(**FOURIER_CONFIG).to(device)
    n_mlp = sum(p.numel() for p in mlp.parameters()) + sum(p.numel() for p in f_mlp.parameters())
    print(f"  MLP params: {n_mlp:,}")
    opt_m = torch.optim.Adam(list(mlp.parameters()) + list(f_mlp.parameters()), lr=1e-3)
    for epoch in range(mlp_epochs):
        mlp.train()
        f_mlp.train()
        perm = torch.randperm(n_train)
        for start in range(0, n_train, TRAIN_BATCH * 2):
            end = min(start + TRAIN_BATCH * 2, n_train)
            idx = perm[start:end]
            B = len(idx)
            z_t_b = norm_patches(z_t_train[idx].to(device))
            zf_b = norm_patches(zf_train[idx].to(device).reshape(B * horizon, 4, GRID_H, GRID_W))
            zf_b = zf_b.reshape(B, horizon * n_spatial, PATCH_DIM)
            pred = mlp(z_t_b, f_mlp(act_train[idx].to(device)))
            loss = F.mse_loss(pred, zf_b)
            opt_m.zero_grad()
            loss.backward()
            opt_m.step()

    mlp.eval()
    f_mlp.eval()

    def mlp_pred(z_t_b, act_b):
        return mlp(z_t_b, f_mlp(act_b))

    mlp_res = evaluate(mlp_pred, "MLP-vae")
    mlp_res["n_params"] = n_mlp
    results["mlp"] = mlp_res
    results["gap"] = round(dit_res["mean_cossim"] - mlp_res["mean_cossim"], 6)
    results["dit_wins"] = results["gap"] > 0
    print(f"\n  DiT - MLP gap: {results['gap']:+.4f}")

    # Quick visual demo (3 test windows) when full run completes
    if not smoke and dit_res["mean_cossim"] > 0:
        try:
            from diffusers import AutoencoderKL
            import matplotlib.pyplot as plt
            vae_dec = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
            out_dir = f"{SPATIAL_DIR}/vae_figure"
            os.makedirs(out_dir, exist_ok=True)
            test_idx = np.where(splits == "test")[0]
            picks = [int(test_idx[i]) for i in range(min(3, len(test_idx)))]
            steps = [0, 4, 8, 12, 15]
            with torch.no_grad():
                for wi, frame_i in enumerate(picks):
                    z_t = torch.tensor(vae_latents[frame_i:frame_i + 1], device=device)
                    act = torch.stack([
                        torch.tensor([steer_norms[frame_i + k], accel_norms[frame_i + k]], device=device)
                        for k in range(horizon)
                    ]).unsqueeze(0)
                    zf_gt = torch.tensor(vae_latents[frame_i + 1: frame_i + 1 + horizon], device=device).unsqueeze(0)
                    z_t_n = norm_patches(z_t)
                    pred_n = dit_pred(z_t_n, act)
                    pred_lat = unpatchify(denorm_tokens(pred_n)).clamp(-3, 3)
                    fig, axes = plt.subplots(2, len(steps), figsize=(12, 5))
                    for col, k in enumerate(steps):
                        for row, grid in enumerate([zf_gt[:, k], pred_lat[:, k]]):
                            img = vae_dec.decode(grid / 0.18215).sample
                            im = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
                            axes[row, col].imshow(im)
                            axes[row, col].axis("off")
                            axes[row, col].set_title("GT" if row == 0 else "DiT")
                    fig.savefig(f"{out_dir}/demo_{wi}.png", dpi=100, bbox_inches="tight")
                    plt.close(fig)
            print(f"[vae-dit] Saved demo figures to {out_dir}/")
        except Exception as e:
            print(f"[vae-dit] Visual demo skipped: {e}")

    vol.commit()
    return results


def _modal_entrypoint_decorator(fn):
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_modal_entrypoint_decorator
def main(
    seed: int = 0,
    horizon: int = 16,
    epochs: int = 100,
    smoke: bool = False,
    mlp_hidden: int = 1024,
    mode: str = "direct",
    n_samples: int = 1,
    action_dropout: float = 0.0,
    cfg_dropout: float = 0.0,
):
    t0 = time.time()
    print(f"VAE DiT training seed={seed} mode={mode} smoke={smoke} "
          f"n_samples={n_samples} action_dropout={action_dropout} cfg_dropout={cfg_dropout}")
    result = train_and_eval.remote(
        seed, horizon, epochs, smoke, mlp_hidden, mode, n_samples, action_dropout, cfg_dropout,
    )
    print(json.dumps(result, indent=2))
    if mode == "diffusion":
        ad_tag = f"_ad{action_dropout}" if action_dropout > 0 else (f"_cfg{cfg_dropout}" if cfg_dropout > 0 else "")
        tag = "smoke" if smoke else "result"
        if action_dropout > 0 and not smoke:
            out = Path(f"artifacts/full/vae_diffusion_multifuture_result_h{horizon}_s{seed}.json")
        else:
            out = Path(f"artifacts/full/vae_dit_diffusion{ad_tag}_{tag}_h{horizon}_s{seed}.json")
        if smoke and result:
            losses = result.get("epoch_losses", [])
            loss_ok = len(losses) >= 2 and losses[-1] < losses[0]
            finite = "smoke_fail" not in result
            smoke_pass = loss_ok and finite
            result["smoke_gate"] = {
                "pass": smoke_pass,
                "loss_decreased": loss_ok,
                "finite": finite,
            }
            print(f"SMOKE GATE (B): {'PASS' if smoke_pass else 'FAIL'}")
            if not smoke_pass:
                print("Stopping: do not launch full Exp B run.")
    else:
        tag = "smoke" if smoke else "result"
        suffix = f"_mlpH{mlp_hidden}" if mlp_hidden != 1024 else ""
        out = Path(f"artifacts/full/vae_dit_direct_{tag}_h{horizon}_s{seed}{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out} ({time.time()-t0:.0f}s)")

