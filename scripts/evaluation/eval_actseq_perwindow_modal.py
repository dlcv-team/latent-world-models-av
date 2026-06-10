"""T0: Per-window DA11 evaluation for hard-subset analysis.

Same as eval_actseq_modal.py but saves per-window CosSim with scene names
and action statistics for hard-subset slicing.

Usage::

    FULL=1 modal run scripts/eval_actseq_perwindow_modal.py
"""

from __future__ import annotations

import csv
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
    app = modal.App("lwm-av-t0-perwindow")
    vol = modal.Volume.from_name("nuscenes-full")
else:
    app = None
    vol = None

VOL_PATH = "/vol"
EMBED_DIR = f"{VOL_PATH}/embeddings"
DIT_DIR = f"{VOL_PATH}/dits"
MLP_DIR = f"{VOL_PATH}/outputs"

TARGET_DIM = 384

NATIVE_DIMS = {
    "vit_s16": 384,
    "dino_vits14": 384,
    "clip_b32": 512,
    "vq_track": 256,
    "vjepa2_rep64": 1024,
    "vjepa2_rep1": 1024,
}

T0_ENCODERS = ["vit_s16", "clip_b32", "dino_vits14"]
T0_HORIZON = 16  # T0 focuses on h=16 only
T0_MODELS = ["dit_x0_actseq", "mlp_flat_actseq"]  # Minimal: only the two we compare

DIT_CONFIG = {
    "n_blocks": 4, "n_heads": 6, "z_dim": 384,
    "cond_dim": 384, "mlp_ratio": 4.0, "dropout": 0.0,
}
DIFFUSION_CONFIG = {"n_train_steps": 1000}
FOURIER_CONFIG = {"n_frequencies": 64, "base": 2.0, "out_dim": 384}
MLP_HIDDEN = 1024
EVAL_BATCH_SIZE = 512

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
            gpu="A10G",
            timeout=3600,
            memory=16384,
        )(fn)
    return fn


@_modal_function_decorator
def evaluate_perwindow(
    encoder_name: str,
    seed: int,
    model_type: str,
):
    """Evaluate one model at h=16, returning PER-WINDOW CosSim + metadata."""
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F

    horizon = T0_HORIZON

    # ---- Inline model definitions (same as eval_actseq_modal.py) ----

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
            if cond.dim() == 2:
                mod = self.adaln_linear(cond).unsqueeze(1)
            else:
                mod = self.adaln_linear(cond)
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mod.chunk(6, dim=-1)
            h = _modulate(self.norm_attn(x), shift_a, scale_a)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + gate_a * self.drop(attn_out)
            h = _modulate(self.norm_mlp(x), shift_m, scale_m)
            x = x + gate_m * self.drop(self.mlp(h))
            return x

    class LatentDiT(nn.Module):
        def __init__(self, z_dim=384, cond_dim=384, n_blocks=4, n_heads=6,
                     horizon=4, mlp_ratio=4.0, dropout=0.0, actseq=False):
            super().__init__()
            self.z_dim = z_dim
            self.horizon = horizon
            self.actseq = actseq
            self.input_proj = nn.Linear(z_dim, z_dim)
            if actseq:
                self.pos_embed = nn.Parameter(torch.zeros(1, horizon, z_dim))
            self.timestep_embed = TimestepEmbedding(cond_dim)
            self.z_t_proj = nn.Linear(z_dim, cond_dim)
            self.blocks = nn.ModuleList([
                DiTBlock(z_dim, cond_dim, n_heads, mlp_ratio, dropout)
                for _ in range(n_blocks)
            ])
            self.final_norm = nn.LayerNorm(z_dim, elementwise_affine=False)
            self.final_adaln = nn.Linear(cond_dim, 3 * z_dim)
            self.final_linear = nn.Linear(z_dim, z_dim)

        def forward(self, x_noisy, z_t, a_embed, timestep):
            if self.actseq:
                cond_global = self.timestep_embed(timestep) + self.z_t_proj(z_t)
                cond = cond_global.unsqueeze(1) + a_embed
                x = self.input_proj(x_noisy) + self.pos_embed[:, :x_noisy.shape[1], :]
            else:
                cond = self.timestep_embed(timestep) + self.z_t_proj(z_t) + a_embed
                x = self.input_proj(x_noisy)
            for block in self.blocks:
                x = block(x, cond)
            if cond.dim() == 2:
                mod = self.final_adaln(cond).unsqueeze(1)
            else:
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
            out = self.proj(x)
            if out.shape[1] == 1:
                out = out.squeeze(1)
            return out

    class LatentPredictorActSeq(nn.Module):
        def __init__(self, z_dim=384, a_dim=384, horizon=4, hidden=1024):
            super().__init__()
            self.horizon = horizon
            input_dim = z_dim + a_dim * horizon
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, z_dim * horizon),
            )

        def forward(self, z_t, a_embed_flat):
            x = torch.cat([z_t, a_embed_flat], dim=-1)
            return self.net(x).view(z_t.shape[0], self.horizon, -1)

    # ---- Load data ----
    native_dim = NATIVE_DIMS[encoder_name]
    needs_adapter = native_dim != TARGET_DIM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(f"{EMBED_DIR}/{encoder_name}.npz", allow_pickle=True)
    embeddings = data["embeddings"]
    splits = data["splits"]
    steer_norms = data["steer_norms"]
    accel_norms = data["accel_norms"]
    scene_names_arr = data["scene_names"]

    mask = splits == "test"
    emb = embeddings[mask]
    steers = steer_norms[mask]
    accels = accel_norms[mask]
    scenes = scene_names_arr[mask]

    # Build windows, tracking scene name and action sequences
    z_t_list, action_seq_list, z_future_list = [], [], []
    window_scene_names = []
    window_indices = []
    window_steer_seqs = []
    window_accel_seqs = []

    for scene in np.unique(scenes):
        idx = np.where(scenes == scene)[0]
        for j in range(len(idx) - horizon):
            z_t_list.append(emb[idx[j]])
            action_seq = np.stack([
                np.array([steers[idx[j + k]], accels[idx[j + k]]])
                for k in range(horizon)
            ])
            action_seq_list.append(action_seq)
            z_future_list.append(emb[idx[j + 1: j + 1 + horizon]])
            window_scene_names.append(scene)
            window_indices.append(j)
            # Save per-step steer/accel for action-variance analysis
            window_steer_seqs.append([steers[idx[j + k]] for k in range(horizon)])
            window_accel_seqs.append([accels[idx[j + k]] for k in range(horizon)])

    z_t_test = torch.tensor(np.array(z_t_list), dtype=torch.float32)
    act_seq_test = torch.tensor(np.array(action_seq_list), dtype=torch.float32)
    zf_test = torch.tensor(np.array(z_future_list), dtype=torch.float32)
    n_test = len(z_t_test)

    print(f"[T0] {model_type}/{encoder_name}/h{horizon}/s{seed}: {n_test} windows")

    # ---- Load checkpoint ----
    is_dit = model_type.startswith("dit")

    if model_type == "dit_x0_actseq":
        ckpt_path = f"{DIT_DIR}/{encoder_name}/conditioned__x0__actseq__h{horizon}/seed_{seed}/checkpoint.pt"
    elif model_type == "mlp_flat_actseq":
        ckpt_path = f"{MLP_DIR}/latent_predictors_residual_actseq_h{horizon}/{encoder_name}/conditioned/seed_{seed}/checkpoint.pt"
    else:
        print(f"[T0] Unknown model: {model_type}")
        return None

    if not os.path.exists(ckpt_path):
        print(f"[T0] MISSING: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    z_mean = ckpt["z_mean"].to(device)
    z_std = ckpt["z_std"].to(device)

    if needs_adapter and ckpt.get("adapter_state_dict"):
        adapter = nn.Linear(native_dim, TARGET_DIM, bias=False).to(device)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        for p in adapter.parameters():
            p.requires_grad_(False)
    else:
        adapter = nn.Identity().to(device)

    fourier_embed = FourierActionEmbedding(action_dim=2, **FOURIER_CONFIG).to(device)

    if is_dit:
        dit = LatentDiT(**{**DIT_CONFIG, "horizon": horizon, "actseq": True}).to(device)
        if "ema_state_dict" in ckpt:
            ema_sd = ckpt["ema_state_dict"]
            dit_ema = {k[4:]: v for k, v in ema_sd.items() if k.startswith("dit.")}
            fourier_ema = {k[8:]: v for k, v in ema_sd.items() if k.startswith("fourier.")}
            if dit_ema:
                dit.load_state_dict(dit_ema)
                if fourier_ema:
                    fourier_embed.load_state_dict(fourier_ema, strict=False)
                print(f"  [EMA weights loaded]")
            else:
                dit.load_state_dict(ckpt["dit_state_dict"])
                fourier_embed.load_state_dict(ckpt["fourier_embed_state_dict"])
        else:
            dit.load_state_dict(ckpt["dit_state_dict"])
            fourier_embed.load_state_dict(ckpt["fourier_embed_state_dict"])
        dit.eval()
        fourier_embed.eval()
        schedule = CosineNoiseSchedule(n_steps=DIFFUSION_CONFIG["n_train_steps"]).to(device)
        alphas_cumprod = schedule.alphas_cumprod
    else:
        fourier_embed.load_state_dict(ckpt["fourier_embed_state_dict"])
        fourier_embed.eval()
        predictor = LatentPredictorActSeq(
            z_dim=TARGET_DIM, a_dim=TARGET_DIM,
            horizon=horizon, hidden=ckpt.get("hidden", MLP_HIDDEN),
        ).to(device)
        predictor.load_state_dict(ckpt["predictor_state_dict"])
        predictor.eval()

    # ---- Evaluate: PER-WINDOW CosSim ----

    z_t_dev = z_t_test.to(device)
    act_seq_dev = act_seq_test.to(device)
    zf_dev = zf_test.to(device)

    if is_dit:
        n_ddim_steps = 50
        T = DIFFUSION_CONFIG["n_train_steps"]
        stride = T // n_ddim_steps
        timesteps = list(reversed(list(range(0, T, stride))[:n_ddim_steps]))

    # Per-window, per-step CosSim arrays
    all_cossim = np.zeros((n_test, horizon), dtype=np.float32)
    all_copy_cossim = np.zeros((n_test, horizon), dtype=np.float32)

    torch.manual_seed(seed)

    with torch.no_grad():
        for start in range(0, n_test, EVAL_BATCH_SIZE):
            end = min(start + EVAL_BATCH_SIZE, n_test)
            B = end - start
            z_t_b = z_t_dev[start:end]
            zf_b = zf_dev[start:end]
            H = horizon

            z_t_adapted = (adapter(z_t_b) - z_mean) / z_std
            zf_adapted = (
                adapter(zf_b.reshape(B * H, -1)).reshape(B, H, TARGET_DIM)
                - z_mean
            ) / z_std

            act_b = act_seq_dev[start:end]  # (B, H, 2)

            if is_dit:
                a_embed = fourier_embed(act_b)
                x = torch.randn(B, horizon, TARGET_DIM, device=device)
                for i, t_val in enumerate(timesteps):
                    t = torch.full((B,), t_val, device=device, dtype=torch.long)
                    pred_x0 = dit(x, z_t=z_t_adapted, a_embed=a_embed, timestep=t)
                    alpha_bar_t = alphas_cumprod[t_val]
                    if i < len(timesteps) - 1:
                        alpha_bar_prev = alphas_cumprod[timesteps[i + 1]]
                    else:
                        alpha_bar_prev = torch.tensor(1.0, device=device)
                    noise_dir = (x - torch.sqrt(alpha_bar_t) * pred_x0) / torch.sqrt(1 - alpha_bar_t + 1e-8)
                    x = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1 - alpha_bar_prev) * noise_dir
                z_hat_norm = x
            else:
                a_embed = fourier_embed(act_b)
                a_embed_flat = a_embed.reshape(B, -1)
                z_hat_delta = predictor(z_t_adapted, a_embed_flat)
                z_hat_norm = z_hat_delta + z_t_adapted.unsqueeze(1).expand(-1, H, -1)

            z_hat = z_hat_norm * z_std + z_mean
            zf_orig = zf_adapted * z_std + z_mean
            z_t_unnorm = z_t_adapted * z_std + z_mean

            for k in range(horizon):
                cs = F.cosine_similarity(z_hat[:, k], zf_orig[:, k], dim=-1)
                copy_cs = F.cosine_similarity(z_t_unnorm, zf_orig[:, k], dim=-1)
                all_cossim[start:end, k] = cs.cpu().numpy()
                all_copy_cossim[start:end, k] = copy_cs.cpu().numpy()

    # Compute per-window action variance (steer + accel std over horizon)
    steer_vars = [float(np.std(s)) for s in window_steer_seqs]
    accel_vars = [float(np.std(s)) for s in window_accel_seqs]

    # Build per-window results
    rows = []
    for w in range(n_test):
        for k in range(horizon):
            rows.append({
                "encoder": encoder_name,
                "seed": seed,
                "model": model_type,
                "scene_name": window_scene_names[w],
                "window_idx": window_indices[w],
                "horizon_step": k + 1,
                "cossim": round(float(all_cossim[w, k]), 6),
                "copy_cossim": round(float(all_copy_cossim[w, k]), 6),
                "steer_variance": round(steer_vars[w], 6),
                "accel_variance": round(accel_vars[w], 6),
            })

    print(
        f"[T0] {model_type}/{encoder_name}/h{horizon}/s{seed}: "
        f"{n_test} windows x {horizon} steps = {len(rows)} rows, "
        f"mean CosSim={all_cossim.mean():.4f}"
    )
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
    """T0: Per-window DA11 evaluation for hard-subset mining."""
    import csv as csv_mod

    t_start = time.time()

    seeds = [0, 1, 2] if os.environ.get("FULL", "") == "1" else [0]
    encoders = T0_ENCODERS
    models = T0_MODELS

    n_jobs = len(encoders) * len(seeds) * len(models)
    print(f"\n{'='*70}")
    print(f"T0: Per-Window Evaluation ({n_jobs} jobs, h={T0_HORIZON})")
    print(f"  encoders: {encoders}")
    print(f"  seeds: {seeds}")
    print(f"  models: {models}")
    print(f"{'='*70}")

    futures = []
    for enc in encoders:
        for s in seeds:
            for m in models:
                futures.append((enc, s, m, evaluate_perwindow.spawn(enc, s, m)))

    all_rows = []
    for enc, s, m, future in futures:
        rows = future.get()
        if rows is not None:
            all_rows.extend(rows)

    wall_time = time.time() - t_start
    print(f"\nDone: {len(all_rows)} total rows in {wall_time:.0f}s")

    # Save CSV
    csv_path = Path("artifacts/full/t0_perwindow_results.csv")
    if not csv_path.parent.exists():
        csv_path = Path("code/latent-world-models-av/artifacts/full/t0_perwindow_results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved {len(all_rows)} rows to {csv_path}")
    else:
        print("WARNING: No results to save!")

    # Quick aggregate summary
    print(f"\n{'='*70}")
    print("AGGREGATE SUMMARY (seed-averaged)")
    print(f"{'='*70}")

    import collections
    agg = collections.defaultdict(list)
    for r in all_rows:
        key = (r["encoder"], r["model"], r["horizon_step"])
        agg[key].append(r["cossim"])

    for enc in encoders:
        for m in models:
            far_steps = [12, 13, 14, 15, 16]
            far_cossim = []
            for k in far_steps:
                vals = agg.get((enc, m, k), [])
                if vals:
                    far_cossim.append(sum(vals) / len(vals))
            if far_cossim:
                print(f"  {enc:15s} {m:20s} far-step mean: {sum(far_cossim)/len(far_cossim):.4f}")
