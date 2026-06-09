"""DA10 V0: Compare EMA vs raw DiT-x0 weights on Modal.

Loads DA8 x0-prediction checkpoints and evaluates both:
1. Raw dit_state_dict (same as DA8 evaluation)
2. EMA weights from ema_state_dict (never previously evaluated)

All checkpoints and embeddings live on the Modal volume.
Results downloaded locally as CSV.

Usage::

    modal run scripts/eval_ema_modal.py
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
    app = modal.App("lwm-av-ema-eval")
    vol = modal.Volume.from_name("nuscenes-full")
else:
    app = None
    vol = None

VOL_PATH = "/vol"
EMBED_DIR = f"{VOL_PATH}/embeddings"
DIT_DIR = f"{VOL_PATH}/dits"

TARGET_DIM = 384
HORIZON = 4  # DA8 h=4 checkpoints

NATIVE_DIMS = {
    "vit_s16": 384,
    "dino_vits14": 384,
    "clip_b32": 512,
    "vq_track": 256,
    "vjepa2_rep64": 1024,
    "vjepa2_rep1": 1024,
}

ALL_ENCODERS = sorted(NATIVE_DIMS.keys())
SEEDS = [0, 1, 2]
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
def evaluate_ema_vs_raw(encoder_name: str, seed: int):
    """Evaluate both EMA and raw weights from the same DA8 x0 checkpoint."""
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F

    # -------------------------------------------------------------------
    # Inline model definitions (same as eval_horizon_modal.py)
    # -------------------------------------------------------------------

    class CosineNoiseSchedule(nn.Module):
        def __init__(self, n_steps: int = 1000, s: float = 0.008):
            super().__init__()
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
    n_test = len(z_t_test)
    print(f"[ema-eval] {encoder_name}/s{seed}: {n_test} test windows")

    # -------------------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------------------

    ckpt_path = f"{DIT_DIR}/{encoder_name}/conditioned__x0/seed_{seed}/checkpoint.pt"
    if not os.path.exists(ckpt_path):
        print(f"[ema-eval] MISSING: {ckpt_path}")
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

    # Check EMA availability
    has_ema = "ema_state_dict" in ckpt
    if not has_ema:
        print(f"[ema-eval] WARNING: no ema_state_dict in {ckpt_path}")

    # -------------------------------------------------------------------
    # Helper: run DDIM evaluation with given weights
    # -------------------------------------------------------------------

    def run_ddim_eval(dit, fourier_embed, label):
        schedule = CosineNoiseSchedule(n_steps=DIFFUSION_CONFIG["n_train_steps"]).to(device)
        alphas_cumprod = schedule.alphas_cumprod

        n_ddim_steps = 50
        T = DIFFUSION_CONFIG["n_train_steps"]
        stride = T // n_ddim_steps
        timesteps = list(reversed(list(range(0, T, stride))[:n_ddim_steps]))

        z_t_dev = z_t_test.to(device)
        act_dev = act_test.to(device)
        zf_dev = zf_test.to(device)

        cossim_sums = [0.0] * horizon
        copy_sums = [0.0] * horizon
        total = 0

        torch.manual_seed(seed)

        with torch.no_grad():
            for start in range(0, n_test, EVAL_BATCH_SIZE):
                end = min(start + EVAL_BATCH_SIZE, n_test)
                B = end - start
                z_t_batch = z_t_dev[start:end]
                act_batch = act_dev[start:end]
                zf_batch = zf_dev[start:end]
                H = horizon

                z_t_adapted = (adapter(z_t_batch) - z_mean) / z_std
                zf_adapted = (
                    adapter(zf_batch.reshape(B * H, -1)).reshape(B, H, TARGET_DIM)
                    - z_mean
                ) / z_std

                a_embed = fourier_embed(act_batch)

                # DDIM x0-prediction
                x = torch.randn(B, horizon, TARGET_DIM, device=device)
                for i, t_val in enumerate(timesteps):
                    t = torch.full((B,), t_val, device=device, dtype=torch.long)
                    pred_x0 = dit(x, z_t=z_t_adapted, a_embed=a_embed, timestep=t)

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

                z_hat = x * z_std + z_mean
                zf_orig = zf_adapted * z_std + z_mean
                z_t_unnorm = z_t_adapted * z_std + z_mean

                for k in range(horizon):
                    cs = F.cosine_similarity(z_hat[:, k], zf_orig[:, k], dim=-1)
                    cossim_sums[k] += cs.sum().item()
                    copy_cs = F.cosine_similarity(z_t_unnorm, zf_orig[:, k], dim=-1)
                    copy_sums[k] += copy_cs.sum().item()

                total += B

        result = {
            "cossim_by_step": [round(s / total, 6) for s in cossim_sums],
            "copy_by_step": [round(s / total, 6) for s in copy_sums],
            "n_test": total,
        }
        print(
            f"  [{label}] {encoder_name}/s{seed}: "
            f"h1={result['cossim_by_step'][0]:.4f} "
            f"h4={result['cossim_by_step'][-1]:.4f}"
        )
        return result

    # -------------------------------------------------------------------
    # Evaluate raw weights
    # -------------------------------------------------------------------

    dit = LatentDiT(**{**DIT_CONFIG, "horizon": horizon}).to(device)
    dit.load_state_dict(ckpt["dit_state_dict"])
    dit.eval()

    fourier_embed = FourierActionEmbedding(action_dim=2, **FOURIER_CONFIG).to(device)
    fourier_embed.load_state_dict(ckpt["fourier_embed_state_dict"])
    fourier_embed.eval()

    raw_result = run_ddim_eval(dit, fourier_embed, "raw")

    # -------------------------------------------------------------------
    # Evaluate EMA weights
    # -------------------------------------------------------------------

    ema_result = None
    if has_ema:
        ema_sd = ckpt["ema_state_dict"]

        # Extract dit.* and fourier.* prefixed keys
        dit_ema_sd = {}
        fourier_ema_sd = {}
        for k, v in ema_sd.items():
            if k.startswith("dit."):
                dit_ema_sd[k[4:]] = v  # strip "dit." prefix
            elif k.startswith("fourier."):
                fourier_ema_sd[k[8:]] = v  # strip "fourier." prefix

        if dit_ema_sd:
            dit.load_state_dict(dit_ema_sd)
            # Fourier EMA only has parameters (proj weights), not buffers (freqs).
            # Use strict=False to skip missing buffer keys.
            if fourier_ema_sd:
                fourier_embed.load_state_dict(fourier_ema_sd, strict=False)
            ema_result = run_ddim_eval(dit, fourier_embed, "ema")
        else:
            print(f"[ema-eval] EMA keys don't match expected prefixes. Keys: {list(ema_sd.keys())[:5]}...")

    return {
        "encoder": encoder_name,
        "seed": seed,
        "raw": raw_result,
        "ema": ema_result,
        "has_ema": has_ema,
    }


# ===================================================================
# Entrypoint
# ===================================================================


def _modal_entrypoint_decorator(fn):
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_modal_entrypoint_decorator
def main():
    """DA10 V0: EMA vs raw weight evaluation."""
    import csv as csv_mod

    t_start = time.time()

    n_jobs = len(ALL_ENCODERS) * len(SEEDS)
    print(f"\n{'='*70}")
    print(f"DA10 V0: EMA vs Raw Weight Evaluation ({n_jobs} jobs)")
    print(f"  encoders: {ALL_ENCODERS}")
    print(f"  seeds: {SEEDS}")
    print(f"  horizon: {HORIZON}")
    print(f"{'='*70}")

    # Launch all jobs
    futures = []
    for enc in ALL_ENCODERS:
        for seed in SEEDS:
            futures.append((enc, seed, evaluate_ema_vs_raw.spawn(enc, seed)))

    # Collect results
    all_results = []
    for enc, seed, future in futures:
        result = future.get()
        if result is not None:
            all_results.append(result)

    wall_time = time.time() - t_start
    print(f"\nDone: {len(all_results)} results in {wall_time:.0f}s")

    # Build CSV rows
    csv_rows = []
    for r in all_results:
        if r["raw"] is None:
            continue
        for k in range(HORIZON):
            row = {
                "encoder": r["encoder"],
                "seed": r["seed"],
                "horizon_step": k + 1,
                "raw_cossim": r["raw"]["cossim_by_step"][k],
                "copy_cossim": r["raw"]["copy_by_step"][k],
                "n_test": r["raw"]["n_test"],
            }
            if r["ema"] is not None:
                row["ema_cossim"] = r["ema"]["cossim_by_step"][k]
                row["delta"] = round(r["ema"]["cossim_by_step"][k] - r["raw"]["cossim_by_step"][k], 6)
            else:
                row["ema_cossim"] = None
                row["delta"] = None
            csv_rows.append(row)

    # Save CSV
    csv_path = Path("artifacts/full/da10_ema_eval.csv")
    if not csv_path.parent.exists():
        csv_path = Path("code/latent-world-models-av/artifacts/full/da10_ema_eval.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(csv_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved {len(csv_rows)} rows to {csv_path}")

    # Print summary
    print(f"\n{'='*90}")
    print(f"{'Encoder':<16} {'Seed':>4} {'Raw h1':>10} {'EMA h1':>10} {'Delta h1':>10} "
          f"{'Raw h4':>10} {'EMA h4':>10} {'Delta h4':>10}")
    print("-" * 90)

    for r in sorted(all_results, key=lambda x: (x["encoder"], x["seed"])):
        if r["raw"] is None or r["ema"] is None:
            continue
        raw_h1 = r["raw"]["cossim_by_step"][0]
        ema_h1 = r["ema"]["cossim_by_step"][0]
        raw_h4 = r["raw"]["cossim_by_step"][3]
        ema_h4 = r["ema"]["cossim_by_step"][3]
        print(
            f"{r['encoder']:<16} {r['seed']:>4} "
            f"{raw_h1:>10.4f} {ema_h1:>10.4f} {ema_h1 - raw_h1:>+10.4f} "
            f"{raw_h4:>10.4f} {ema_h4:>10.4f} {ema_h4 - raw_h4:>+10.4f}"
        )

    # Aggregate: mean delta per encoder
    print(f"\n{'='*70}")
    print(f"{'Encoder':<16} {'Mean delta (all h)':>18} {'Improves?':>10}")
    print("-" * 70)

    for enc in ALL_ENCODERS:
        enc_results = [r for r in all_results if r["encoder"] == enc and r["ema"] is not None]
        if not enc_results:
            continue
        deltas = []
        for r in enc_results:
            for k in range(HORIZON):
                deltas.append(r["ema"]["cossim_by_step"][k] - r["raw"]["cossim_by_step"][k])
        mean_delta = sum(deltas) / len(deltas)
        improves = mean_delta > 0.002
        print(f"{enc:<16} {mean_delta:>+18.6f} {'YES' if improves else 'no':>10}")

    # Gate G1
    improving_encoders = 0
    total_delta = 0.0
    n_delta = 0
    for enc in ALL_ENCODERS:
        enc_results = [r for r in all_results if r["encoder"] == enc and r["ema"] is not None]
        if not enc_results:
            continue
        deltas = []
        for r in enc_results:
            for k in range(HORIZON):
                d = r["ema"]["cossim_by_step"][k] - r["raw"]["cossim_by_step"][k]
                deltas.append(d)
                total_delta += d
                n_delta += 1
        mean_delta = sum(deltas) / len(deltas)
        if mean_delta > 0.002:
            improving_encoders += 1

    mean_overall = total_delta / n_delta if n_delta > 0 else 0.0

    print(f"\n{'='*70}")
    print(f"GATE G1 RESULT")
    print(f"  Encoders with mean delta > 0.002: {improving_encoders}/6")
    print(f"  Mean overall delta: {mean_overall:+.6f}")
    if improving_encoders >= 3 and mean_overall > 0:
        print(f"  -> PASS: EMA broadly helps. Report as sensitivity analysis with asterisk.")
    elif improving_encoders >= 1:
        print(f"  -> MARGINAL: EMA helps {improving_encoders} encoder(s). Report as footnote.")
    else:
        print(f"  -> FAIL: EMA provides no systematic benefit.")
