"""Anchored Spatial DiT debug experiments.
Tests hypothesis that removing autoregressive generation fixes temporal coherence.

Two modes, both giving the spatial DiT the SAME residual anchor (z_t) that the
spatial MLP baseline already has -- the missing anchor is the prime suspect for
the pooled-pilot failure (DiT step-1 CosSim << copy << MLP).

  --mode direct       Job D: DiT-direct-spatial-ANCHORED. No diffusion/DDIM.
                      pred = z_t + Transformer(z_t, action_seq, pos), single
                      forward pass. Fair architecture test vs residual MLP.

  --mode anchored_x0  Job A: anchored-x0 diffusion. Diffusion stays on the
                      absolute normalized future (SNR-safe), but the x0 head is
                      parameterized as pred_x0 = z_t + f(x_tau, z_t, a).

Both train an anchored spatial DiT and the (already-anchored) spatial MLP and
evaluate with per-token CosSim + copy baseline, matching
train_dit_spatial_modal.py exactly so numbers are commensurable.

Usage::

    modal run scripts/train_dit_spatial_anchored_modal.py --encoder vit_s16 --mode direct
    modal run scripts/train_dit_spatial_anchored_modal.py --encoder vit_s16 --mode anchored_x0
    modal run scripts/train_dit_spatial_anchored_modal.py --encoder vit_s16 --mode direct --smoke
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
    app = modal.App("lwm-av-spatial-anchored")
    vol = modal.Volume.from_name("nuscenes-full")
else:
    app = None
    vol = None

VOL_PATH = "/vol"
SPATIAL_DIR = f"{VOL_PATH}/embeddings/spatial"
CKPT_DIR = f"{VOL_PATH}/dits/spatial_anchored"

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
TRAIN_LR = 1e-4
TRAIN_BATCH = 64
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
            gpu="A100",
            timeout=14400,
            memory=32768,
        )(fn)
    return fn


@_modal_function_decorator
def train_and_eval(
    encoder_name: str,
    seed: int,
    mode: str = "direct",            # "direct" or "anchored_x0"
    horizon: int = 16,
    epochs: int = 100,
    smoke: bool = False,
    mlp_hidden: int = 512,           # MLP hidden dim (default 512 = ~1.3M; 2816 = ~12M param-matched)
    objective: str = "x0",           # "x0" or "vpred" (diffusion only)
    n_samples: int = 1,              # DDIM samples for distributional eval (K>1 for best-of-K)
    action_mode: str = "exact",      # "exact" | "coarse" (3-way mean-steer intent)
    window_stride: int = 1,        # 1 = 2 Hz; 2 ≈ 1 Hz (one-shot fallback)
    coarse_theta: float = 0.1,       # |mean future steer| threshold for straight
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from copy import deepcopy

    assert mode in ("direct", "anchored_x0"), f"bad mode {mode}"
    assert objective in ("x0", "vpred"), f"bad objective {objective}"
    assert action_mode in ("exact", "coarse"), f"bad action_mode {action_mode}"
    is_diffusion = mode == "anchored_x0"
    use_vpred = is_diffusion and objective == "vpred"

    # Smoke config: small + fast, just to validate the pipeline end-to-end.
    n_ddim_steps = N_DDIM_STEPS
    max_train = None
    max_test = None
    if smoke:
        epochs = 4
        n_ddim_steps = 10
        max_train = 2000
        max_test = 256

    dit_epochs = epochs
    mlp_epochs = max(epochs // 2, 10) if not smoke else 4

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_spatial = SPATIAL_TOKENS[encoder_name]

    print(f"[anchored] {encoder_name}/h{horizon}/s{seed} mode={mode} smoke={smoke}: "
          f"S={n_spatial}, device={device}")

    # ---- Inline model definitions ----

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

    class AnchoredSpatialDiT(nn.Module):
        """Spatial-temporal DiT with a residual anchor on the output.

        Output is parameterized as z_t (broadcast over horizon) + delta, so the
        network only has to learn the change from the current frame -- matching
        the MLP baseline's residual structure. In diffusion mode the input is the
        noised future and the (anchored) output is the x0 prediction. In direct
        mode the input is z_t broadcast (no noise) and t is fixed to 0.
        """
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

        def forward(self, x_input, z_t_spatial, a_embed, timestep, anchor_output=True):
            B = x_input.shape[0]
            H, S, D = self.horizon, self.n_spatial, self.z_dim
            sp = self.spatial_pos.unsqueeze(1).expand(-1, H, -1, -1).reshape(1, H * S, D)
            tp = self.temporal_pos.unsqueeze(2).expand(-1, -1, S, -1).reshape(1, H * S, D)
            x = self.input_proj(x_input) + sp + tp
            z_t_pooled = z_t_spatial.mean(dim=1)
            cond_global = self.timestep_embed(timestep) + self.z_t_proj(z_t_pooled)
            a_broadcast = a_embed.unsqueeze(2).expand(-1, -1, S, -1).reshape(B, H * S, D)
            cond = cond_global.unsqueeze(1).expand(-1, H * S, -1) + a_broadcast
            for block in self.blocks:
                x = block(x, cond)
            mod = self.final_adaln(cond)
            shift, scale, gate = mod.chunk(3, dim=-1)
            delta = gate * self.final_linear(_modulate(self.final_norm(x), shift, scale))
            if not anchor_output:
                return delta
            z_t_rep = z_t_spatial.unsqueeze(1).expand(-1, H, -1, -1).reshape(B, H * S, D)
            return z_t_rep + delta

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
        print(f"[anchored] ERROR: {spatial_path} not found!")
        return None

    data = np.load(spatial_path, allow_pickle=True)
    spatial_emb = data["spatial_embeddings"]
    splits = data["splits"]
    steer_norms = data["steer_norms"]
    accel_norms = data["accel_norms"]
    scene_names = data["scene_names"]

    N_total, S_actual, D_actual = spatial_emb.shape
    assert S_actual == n_spatial, f"Expected S={n_spatial}, got {S_actual}"
    assert D_actual == TARGET_DIM, f"Expected D={TARGET_DIM}, got {D_actual}"
    print(f"[anchored] Loaded {N_total} frames, shape {spatial_emb.shape}")

    def build_windows(split_name):
        mask = splits == split_name
        emb = spatial_emb[mask]
        steers = steer_norms[mask]
        accels = accel_norms[mask]
        scenes = scene_names[mask]

        z_t_list, action_seq_list, z_future_list = [], [], []
        for scene in np.unique(scenes):
            idx = np.where(scenes == scene)[0]
            for j in range(0, len(idx) - horizon, window_stride):
                z_t_list.append(emb[idx[j]])
                action_seq = np.stack([
                    np.array([steers[idx[j + k]], accels[idx[j + k]]])
                    for k in range(horizon)
                ])
                action_seq_list.append(action_seq)
                z_future_list.append(emb[idx[j + 1: j + 1 + horizon]])

        return (
            torch.tensor(np.array(z_t_list), dtype=torch.float32),
            torch.tensor(np.array(action_seq_list), dtype=torch.float32),
            torch.tensor(np.array(z_future_list), dtype=torch.float32),
        )

    z_t_train, act_train_exact, zf_train = build_windows("train")
    z_t_test, act_test_exact, zf_test = build_windows("test")

    if max_train is not None and len(z_t_train) > max_train:
        z_t_train, act_train_exact, zf_train = (
            z_t_train[:max_train], act_train_exact[:max_train], zf_train[:max_train],
        )
    if max_test is not None and len(z_t_test) > max_test:
        z_t_test, act_test_exact, zf_test = (
            z_t_test[:max_test], act_test_exact[:max_test], zf_test[:max_test],
        )

    # HV mask from exact logged actions (after any smoke subsample).
    steer_std_test = act_test_exact[:, :, 0].std(dim=1)
    hv_threshold = float(torch.quantile(steer_std_test, 0.75).item())
    hv_mask = steer_std_test >= hv_threshold
    hv_indices = torch.where(hv_mask)[0]
    print(f"[anchored] HV windows: {len(hv_indices)}/{len(act_test_exact)} "
          f"(steer_std>={hv_threshold:.4f})")

    coarse_meta = {"action_mode": action_mode, "coarse_theta": coarse_theta}
    if action_mode == "coarse":
        mean_steer_train = act_train_exact[:, :, 0].mean(dim=1)
        left_center = float(torch.quantile(mean_steer_train, 0.10).item())
        right_center = float(torch.quantile(mean_steer_train, 0.90).item())
        theta = coarse_theta

        def to_coarse(act_exact):
            out = act_exact.clone()
            mean_s = act_exact[:, :, 0].mean(dim=1)
            for i in range(len(act_exact)):
                m = mean_s[i].item()
                if m < -theta:
                    steer_val = left_center
                elif m > theta:
                    steer_val = right_center
                else:
                    steer_val = 0.0
                out[i, :, 0] = steer_val
                out[i, :, 1] = 0.0
            return out

        act_train = to_coarse(act_train_exact)
        act_test = to_coarse(act_test_exact)
        mean_test = act_test_exact[:, :, 0].mean(dim=1)
        n_test_w = len(act_test)
        n_left = int((mean_test < -theta).sum().item())
        n_right = int((mean_test > theta).sum().item())
        n_str = n_test_w - n_left - n_right
        coarse_meta.update({
            "left_center": round(left_center, 6),
            "right_center": round(right_center, 6),
            "coarse_balance_test": {
                "left": n_left, "straight": n_str, "right": n_right,
                "frac_left": round(n_left / n_test_w, 4),
                "frac_straight": round(n_str / n_test_w, 4),
                "frac_right": round(n_right / n_test_w, 4),
            },
        })
        print(f"[anchored] coarse θ={theta} centers L/R={left_center:.3f}/{right_center:.3f} "
              f"balance {coarse_meta['coarse_balance_test']}")
    else:
        act_train = act_train_exact
        act_test = act_test_exact

    n_train, n_test = len(z_t_train), len(z_t_test)
    print(f"[anchored] Train: {n_train} windows, Test: {n_test} windows")

    all_train_flat = z_t_train.reshape(-1, TARGET_DIM)
    z_mean = all_train_flat.mean(dim=0).to(device)
    z_std = all_train_flat.std(dim=0).clamp(min=1e-6).to(device)

    def normalize(x):
        return (x - z_mean) / z_std

    def denormalize(x):
        return x * z_std + z_mean

    def ddim_sample(model, fourier_embed, z_t_spatial_norm, act_seq, schedule, vpred=False, anchor_output=True):
        B = z_t_spatial_norm.shape[0]
        H, S, D = horizon, n_spatial, TARGET_DIM
        T = DIFFUSION_STEPS
        stride = T // n_ddim_steps
        timesteps = list(reversed(list(range(0, T, stride))[:n_ddim_steps]))
        alphas = schedule.alphas_cumprod

        a_embed = fourier_embed(act_seq)
        x = torch.randn(B, H * S, D, device=device)

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            out = model(x, z_t_spatial_norm, a_embed, t, anchor_output=anchor_output)
            alpha_t = alphas[t_val]
            if vpred:
                pred_x0 = torch.sqrt(alpha_t) * x - torch.sqrt(1 - alpha_t + 1e-8) * out
            else:
                pred_x0 = out
            alpha_prev = alphas[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0, device=device)
            noise_dir = (x - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(1 - alpha_t + 1e-8)
            x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_dir

        return x

    def ddim_sample_multi(model, fourier_embed, z_t_b, act_b, schedule, k, vpred=False, anchor_output=True):
        """Sample K futures; return (K, B, H*S, D) tensor."""
        samples = []
        for _ in range(k):
            samples.append(ddim_sample(model, fourier_embed, z_t_b, act_b, schedule, vpred=vpred, anchor_output=anchor_output))
        return torch.stack(samples, dim=0)

    def evaluate_model(predict_fn, z_t_dev, act_dev, zf_dev, label="", indices=None):
        cossim_sums = [0.0] * horizon
        copy_sums = [0.0] * horizon
        total = 0
        eval_batch = 32
        idx_list = indices.tolist() if indices is not None else list(range(len(z_t_dev)))

        with torch.no_grad():
            for start in range(0, len(idx_list), eval_batch):
                end = min(start + eval_batch, len(idx_list))
                batch_idx = idx_list[start:end]
                B = len(batch_idx)

                z_t_b = normalize(z_t_dev[batch_idx].to(device))
                act_b = act_dev[batch_idx].to(device)
                zf_b = zf_dev[batch_idx].to(device)

                z_hat_norm = predict_fn(z_t_b, act_b)
                z_hat = denormalize(z_hat_norm).reshape(B, horizon, n_spatial, TARGET_DIM)

                z_t_raw = z_t_dev[batch_idx].to(device)

                for k in range(horizon):
                    cs_per_token = F.cosine_similarity(z_hat[:, k], zf_b[:, k], dim=-1)
                    cossim_sums[k] += cs_per_token.mean(dim=-1).sum().item()
                    copy_per_token = F.cosine_similarity(z_t_raw, zf_b[:, k], dim=-1)
                    copy_sums[k] += copy_per_token.mean(dim=-1).sum().item()

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

    results = {
        "mode": mode,
        "smoke": smoke,
        "objective": objective,
        "action_mode": action_mode,
        "window_stride": window_stride,
        "hv_threshold_steer_std": round(hv_threshold, 6),
        "n_hv_test": int(len(hv_indices)),
        **coarse_meta,
    }

    # ==================================================================
    # Train anchored spatial DiT (direct OR anchored-x0 diffusion)
    # ==================================================================
    print(f"\n{'='*60}\nTraining Anchored DiT ({mode}): {encoder_name}/h{horizon}/s{seed}\n{'='*60}")

    dit = AnchoredSpatialDiT(**DIT_CONFIG, horizon=horizon, n_spatial=n_spatial).to(device)
    fourier = FourierActionEmbedding(**FOURIER_CONFIG).to(device)
    schedule = CosineNoiseSchedule(n_steps=DIFFUSION_STEPS).to(device)

    n_params = sum(p.numel() for p in dit.parameters()) + sum(p.numel() for p in fourier.parameters())
    print(f"  DiT params: {n_params:,}")

    optimizer = torch.optim.Adam(list(dit.parameters()) + list(fourier.parameters()), lr=TRAIN_LR)

    ema_params = {}
    for name, param in list(dit.named_parameters()) + list(fourier.named_parameters()):
        ema_params[name] = param.data.clone()

    t_start = time.time()
    for epoch in range(dit_epochs):
        dit.train(); fourier.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, TRAIN_BATCH):
            end = min(start + TRAIN_BATCH, n_train)
            idx = perm[start:end]
            B = len(idx)

            z_t_b = normalize(z_t_train[idx].to(device))
            act_b = act_train[idx].to(device)
            zf_b = normalize(zf_train[idx].to(device).reshape(B, horizon * n_spatial, TARGET_DIM))
            a_embed = fourier(act_b)

            if is_diffusion:
                t = torch.randint(0, DIFFUSION_STEPS, (B,), device=device)
                alpha_bar = schedule.alphas_cumprod[t].unsqueeze(1).unsqueeze(2)
                noise = torch.randn_like(zf_b)
                x_noisy = torch.sqrt(alpha_bar) * zf_b + torch.sqrt(1 - alpha_bar) * noise
                if use_vpred:
                    v_target = torch.sqrt(alpha_bar) * noise - torch.sqrt(1 - alpha_bar) * zf_b
                    pred = dit(x_noisy, z_t_b, a_embed, t, anchor_output=False)
                    loss = F.mse_loss(pred, v_target)
                else:
                    pred = dit(x_noisy, z_t_b, a_embed, t, anchor_output=True)
                    loss = F.mse_loss(pred, zf_b)
            else:
                # Direct: input is z_t broadcast over horizon, timestep fixed to 0.
                z_t_rep = z_t_b.unsqueeze(1).expand(-1, horizon, -1, -1).reshape(B, horizon * n_spatial, TARGET_DIM)
                t0 = torch.zeros((B,), device=device, dtype=torch.long)
                pred = dit(z_t_rep, z_t_b, a_embed, t0)
                loss = F.mse_loss(pred, zf_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                for name, param in list(dit.named_parameters()) + list(fourier.named_parameters()):
                    ema_params[name].mul_(EMA_DECAY).add_(param.data, alpha=1 - EMA_DECAY)

            epoch_loss += loss.item()
            n_batches += 1

        log_interval = max(dit_epochs // 5, 1)
        if epoch % log_interval == 0 or epoch == dit_epochs - 1:
            print(f"  epoch {epoch}/{dit_epochs}: loss={epoch_loss/n_batches:.6f}")

    train_time = time.time() - t_start
    print(f"  Training done in {train_time:.0f}s")

    dit_ema = deepcopy(dit)
    fourier_ema = deepcopy(fourier)
    with torch.no_grad():
        for name, param in dit_ema.named_parameters():
            if name in ema_params:
                param.copy_(ema_params[name])
        for name, param in fourier_ema.named_parameters():
            if name in ema_params:
                param.copy_(ema_params[name])
    dit_ema.eval(); fourier_ema.eval()

    anchor_out = not use_vpred
    if is_diffusion:
        def dit_predict(z_t_b, act_b):
            return ddim_sample(
                dit_ema, fourier_ema, z_t_b, act_b, schedule,
                vpred=use_vpred, anchor_output=anchor_out,
            )
    else:
        def dit_predict(z_t_b, act_b):
            B = z_t_b.shape[0]
            z_t_rep = z_t_b.unsqueeze(1).expand(-1, horizon, -1, -1).reshape(B, horizon * n_spatial, TARGET_DIM)
            a_embed = fourier_ema(act_b)
            t0 = torch.zeros((B,), device=device, dtype=torch.long)
            return dit_ema(z_t_rep, z_t_b, a_embed, t0)

    obj_tag = objective if is_diffusion else "direct"
    dit_label = f"DiT-{mode}-{obj_tag}"
    dit_result = evaluate_model(dit_predict, z_t_test, act_test, zf_test, dit_label)
    dit_result["model"] = f"dit_{mode}_{objective}" if is_diffusion else f"dit_{mode}"
    dit_result["train_time_s"] = round(train_time, 1)
    dit_result["n_params"] = n_params
    dit_result["objective"] = objective
    results["dit"] = dit_result

    def evaluate_distributional(indices=None, label="all"):
        """DDIM K-sample metrics; optional index subset (e.g. HV)."""
        k = n_samples
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
                z_t_b = normalize(z_t_test[batch_idx].to(device))
                act_b = act_test[batch_idx].to(device)
                zf_b = zf_test[batch_idx].to(device)
                samples = ddim_sample_multi(
                    dit_ema, fourier_ema, z_t_b, act_b, schedule, k,
                    vpred=use_vpred, anchor_output=anchor_out,
                )
                samples = denormalize(samples).reshape(k, B, horizon, n_spatial, TARGET_DIM)
                for b in range(B):
                    for step in range(horizon):
                        gt = zf_b[b, step]
                        cs_list = []
                        for ki in range(k):
                            pred = samples[ki, b, step]
                            cs_list.append(
                                F.cosine_similarity(pred, gt, dim=-1).mean().item()
                            )
                        mean_sums[step] += sum(cs_list) / k
                        best_sums[step] += max(cs_list)
                    if k >= 2:
                        flat = samples[:, b].reshape(k, -1)
                        pw = torch.cdist(flat, flat, p=2)
                        div_sums += pw[
                            torch.triu(torch.ones(k, k, device=device), diagonal=1) == 1
                        ].mean().item()
                total += B
        if total == 0:
            return {}
        return {
            "n_samples": k,
            "subset": label,
            "n_test": total,
            "mean_cossim_by_step": [round(s / total, 6) for s in mean_sums],
            "best_of_k_cossim_by_step": [round(s / total, 6) for s in best_sums],
            "mean_cossim": round(sum(mean_sums) / (total * horizon), 6),
            "best_of_k_mean_cossim": round(sum(best_sums) / (total * horizon), 6),
            "sample_diversity_l2": round(div_sums / total, 6) if k >= 2 else 0.0,
        }

    def _load_direct_predictor():
        """Load coarse/exact direct DiT checkpoint for matched-noise baseline."""
        ckpt_path = (
            f"{CKPT_DIR}/{encoder_name}/direct/h{horizon}/seed_{seed}"
            f"/am_{action_mode}/dit_checkpoint.pt"
        )
        if not os.path.exists(ckpt_path):
            print(f"  [matched-noise] no direct ckpt at {ckpt_path}")
            return None
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        d_direct = AnchoredSpatialDiT(
            **ckpt["config"], horizon=horizon, n_spatial=n_spatial
        ).to(device)
        f_direct = FourierActionEmbedding(**FOURIER_CONFIG).to(device)
        d_direct.load_state_dict(ckpt["dit_state_dict"])
        f_direct.load_state_dict(ckpt["fourier_state_dict"])
        ema_d = ckpt.get("ema_params", {})
        with torch.no_grad():
            for name, param in d_direct.named_parameters():
                if name in ema_d:
                    param.copy_(ema_d[name])
            for name, param in f_direct.named_parameters():
                if name in ema_d:
                    param.copy_(ema_d[name])
        d_direct.eval()
        f_direct.eval()

        def direct_fn(z_t_b, act_b):
            B = z_t_b.shape[0]
            z_t_rep = z_t_b.unsqueeze(1).expand(-1, horizon, -1, -1).reshape(
                B, horizon * n_spatial, TARGET_DIM
            )
            a_embed = f_direct(act_b)
            t0 = torch.zeros((B,), device=device, dtype=torch.long)
            return d_direct(z_t_rep, z_t_b, a_embed, t0)

        return direct_fn

    def evaluate_matched_noise_distributional(indices=None, direct_fn=None):
        """Direct pred + K Gaussian perturbations with σ matched to diffusion spread."""
        if direct_fn is None:
            direct_fn = _load_direct_predictor()
        if direct_fn is None:
            return {}
        k = n_samples
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
                z_t_b = normalize(z_t_test[batch_idx].to(device))
                act_b = act_test[batch_idx].to(device)
                zf_b = zf_test[batch_idx].to(device)
                z_hat_norm = direct_fn(z_t_b, act_b)
                z_hat = denormalize(z_hat_norm).reshape(B, horizon, n_spatial, TARGET_DIM)
                # Diffusion samples to estimate per-window σ
                diff_s = ddim_sample_multi(
                    dit_ema, fourier_ema, z_t_b, act_b, schedule, k,
                    vpred=use_vpred, anchor_output=anchor_out,
                )
                diff_s = denormalize(diff_s).reshape(k, B, horizon, n_spatial, TARGET_DIM)
                for b in range(B):
                    sigma = diff_s[:, b].reshape(k, -1).std(dim=0, unbiased=False).clamp(min=1e-6)
                    flat_direct = z_hat[b].reshape(-1)
                    matched = flat_direct.unsqueeze(0) + torch.randn(
                        k, flat_direct.numel(), device=device
                    ) * sigma.unsqueeze(0)
                    matched = matched.reshape(k, horizon, n_spatial, TARGET_DIM)
                    for step in range(horizon):
                        gt = zf_b[b, step]
                        cs_list = [
                            F.cosine_similarity(matched[ki, step], gt, dim=-1).mean().item()
                            for ki in range(k)
                        ]
                        mean_sums[step] += sum(cs_list) / k
                        best_sums[step] += max(cs_list)
                    if k >= 2:
                        pw = torch.cdist(matched.reshape(k, -1), matched.reshape(k, -1), p=2)
                        div_sums += pw[
                            torch.triu(torch.ones(k, k, device=device), diagonal=1) == 1
                        ].mean().item()
                total += B
        if total == 0:
            return {}
        return {
            "n_samples": k,
            "subset": "hv" if indices is not None else "all",
            "n_test": total,
            "mean_cossim": round(sum(mean_sums) / (total * horizon), 6),
            "best_of_k_mean_cossim": round(sum(best_sums) / (total * horizon), 6),
            "sample_diversity_l2": round(div_sums / total, 6) if k >= 2 else 0.0,
        }

    # Distributional eval (diffusion, K>1 samples)
    if is_diffusion and n_samples > 1:
        print(f"\n  Distributional eval: K={n_samples} (all + HV)")
        dist = evaluate_distributional(label="all")
        print(f"    [all] mean={dist['mean_cossim']:.4f} best-of-K={dist['best_of_k_mean_cossim']:.4f} "
              f"div={dist['sample_diversity_l2']:.4f}")
        results["dit_distributional"] = dist
        dist_hv = evaluate_distributional(indices=hv_indices, label="hv")
        print(f"    [HV]  mean={dist_hv['mean_cossim']:.4f} best-of-K={dist_hv['best_of_k_mean_cossim']:.4f} "
              f"div={dist_hv['sample_diversity_l2']:.4f}")
        results["dit_distributional_hv"] = dist_hv
        print("  Matched-noise control (σ from diffusion samples on HV)...")
        mn_hv = evaluate_matched_noise_distributional(indices=hv_indices)
        if mn_hv:
            print(f"    [matched-noise HV] best-of-K={mn_hv['best_of_k_mean_cossim']:.4f} "
                  f"mean={mn_hv['mean_cossim']:.4f}")
        results["matched_noise_distributional_hv"] = mn_hv

    if not smoke:
        ckpt_dir = (
            f"{CKPT_DIR}/{encoder_name}/{mode}/h{horizon}/seed_{seed}/am_{action_mode}"
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({
            "dit_state_dict": dit.state_dict(),
            "fourier_state_dict": fourier.state_dict(),
            "ema_params": ema_params,
            "z_mean": z_mean.cpu(), "z_std": z_std.cpu(),
            "config": {**DIT_CONFIG, "horizon": horizon, "n_spatial": n_spatial, "mode": mode},
        }, f"{ckpt_dir}/dit_checkpoint.pt")
        print(f"  Saved DiT checkpoint to {ckpt_dir}")

    # ==================================================================
    # Train spatial MLP (anchored residual baseline -- identical to pilot)
    # ==================================================================
    print(f"\n{'='*60}\nTraining Spatial MLP: {encoder_name}/h{horizon}/s{seed}\n{'='*60}")

    mlp_cfg = {**MLP_CONFIG, "hidden": mlp_hidden}
    mlp = SpatialMLPPredictor(**mlp_cfg, horizon=horizon, n_spatial=n_spatial).to(device)
    fourier_mlp = FourierActionEmbedding(**FOURIER_CONFIG).to(device)
    n_params_mlp = sum(p.numel() for p in mlp.parameters()) + sum(p.numel() for p in fourier_mlp.parameters())
    print(f"  MLP params: {n_params_mlp:,}")

    optimizer_mlp = torch.optim.Adam(list(mlp.parameters()) + list(fourier_mlp.parameters()), lr=1e-3)

    t_start = time.time()
    for epoch in range(mlp_epochs):
        mlp.train(); fourier_mlp.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, TRAIN_BATCH * 2):
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
        log_interval = max(mlp_epochs // 5, 1)
        if epoch % log_interval == 0 or epoch == mlp_epochs - 1:
            print(f"  epoch {epoch}/{mlp_epochs}: loss={epoch_loss/n_batches:.6f}")

    mlp_train_time = time.time() - t_start
    print(f"  Training done in {mlp_train_time:.0f}s")
    mlp.eval(); fourier_mlp.eval()

    def mlp_predict(z_t_b, act_b):
        a_embed = fourier_mlp(act_b)
        return mlp(z_t_b, a_embed)

    mlp_result = evaluate_model(mlp_predict, z_t_test, act_test, zf_test, "MLP-spatial")
    mlp_result["model"] = "mlp_spatial"
    mlp_result["train_time_s"] = round(mlp_train_time, 1)
    mlp_result["n_params"] = n_params_mlp
    results["mlp"] = mlp_result
    mlp_hv = evaluate_model(
        mlp_predict, z_t_test, act_test, zf_test, "MLP-spatial-HV", indices=hv_indices,
    )
    results["mlp_hv"] = mlp_hv

    if not is_diffusion:
        dit_hv = evaluate_model(
            dit_predict, z_t_test, act_test, zf_test, f"{dit_label}-HV", indices=hv_indices,
        )
        results["dit_hv"] = dit_hv

    # ==================================================================
    # Summary
    # ==================================================================
    dit_mean = results["dit"]["mean_cossim"]
    mlp_mean = results["mlp"]["mean_cossim"]
    copy_step1 = results["dit"]["copy_by_step"][0]
    gap = dit_mean - mlp_mean
    print(f"\n{'='*60}\nRESULTS: {encoder_name}/h{horizon}/s{seed} mode={mode}\n{'='*60}")
    print(f"  dit ({mode}): mean={dit_mean:.4f}, step1={results['dit']['cossim_by_step'][0]:.4f}")
    print(f"  mlp:          mean={mlp_mean:.4f}, step1={results['mlp']['cossim_by_step'][0]:.4f}")
    print(f"  copy step1:   {copy_step1:.4f}")
    print(f"  DiT - MLP gap: {gap:+.4f}  >>> {'DiT WINS!' if gap > 0 else 'MLP wins'} <<<")
    print(f"  DiT step1 - copy step1: {results['dit']['cossim_by_step'][0] - copy_step1:+.4f}")
    results["gap"] = round(gap, 6)
    results["dit_wins"] = gap > 0
    results["dit_step1_beats_copy"] = results["dit"]["cossim_by_step"][0] >= copy_step1
    results["mlp_hidden"] = mlp_hidden

    vol.commit()
    return results


def _modal_entrypoint_decorator(fn):
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_modal_entrypoint_decorator
def main(
    encoder: str = "vit_s16",
    seed: int = 0,
    mode: str = "direct",
    horizon: int = 16,
    epochs: int = 100,
    smoke: bool = False,
    mlp_hidden: int = 512,
    objective: str = "x0",
    n_samples: int = 1,
    action_mode: str = "exact",
    window_stride: int = 1,
    coarse_theta: float = 0.1,
):
    if encoder not in SPATIAL_TOKENS:
        print(f"ERROR: encoder must be one of {list(SPATIAL_TOKENS.keys())}")
        return
    if mode not in ("direct", "anchored_x0"):
        print("ERROR: mode must be 'direct' or 'anchored_x0'")
        return

    t_start = time.time()
    print(f"\n{'='*60}\nAnchored Spatial DiT: {encoder} mode={mode} h{horizon} s{seed} "
          f"epochs={epochs} smoke={smoke} mlp_hidden={mlp_hidden} objective={objective} "
          f"n_samples={n_samples} action_mode={action_mode} stride={window_stride}\n{'='*60}")

    result = train_and_eval.remote(
        encoder, seed, mode, horizon, epochs, smoke,
        mlp_hidden=mlp_hidden, objective=objective, n_samples=n_samples,
        action_mode=action_mode, window_stride=window_stride, coarse_theta=coarse_theta,
    )
    wall = time.time() - t_start
    print(f"\nDone in {wall:.0f}s")
    print(json.dumps(result, indent=2))

    tag = "smoke" if smoke else "result"
    if mode == "direct":
        name = "spatial_dit_direct_residual"
    else:
        name = f"spatial_dit_anchored_{objective}"
    mlp_suffix = f"_mlpH{mlp_hidden}" if mlp_hidden != 512 else ""
    am_suffix = "_coarse" if action_mode == "coarse" else ""
    stride_suffix = f"_stride{window_stride}" if window_stride != 1 else ""
    out_path = Path(
        f"artifacts/full/{name}_{tag}_{encoder}_h{horizon}_s{seed}"
        f"{am_suffix}{stride_suffix}{mlp_suffix}.json"
    )
    if not out_path.parent.exists():
        out_path = Path(f"code/latent-world-models-av/artifacts/full/{name}_{tag}_{encoder}_h{horizon}_s{seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")
