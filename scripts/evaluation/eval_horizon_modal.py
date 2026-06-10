"""DA9 Exp 3: Evaluate DiT-x0, MLP-fair, MLP-residual at h=8,16 on Modal.

All checkpoints and embeddings live on the Modal volume.
DA8 h=4 baseline results loaded from CSV on volume.
Results downloaded locally as CSV.

Usage::

    modal run scripts/eval_horizon_modal.py
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
    app = modal.App("lwm-av-eval-horizon")
    vol = modal.Volume.from_name("nuscenes-full")
else:
    app = None
    vol = None

VOL_PATH = "/vol"
EMBED_DIR = f"{VOL_PATH}/embeddings"
DIT_DIR = f"{VOL_PATH}/dits"
MLP_DIR = f"{VOL_PATH}/outputs"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_DIM = 384

NATIVE_DIMS = {
    "vit_s16": 384,
    "dino_vits14": 384,
    "clip_b32": 512,
    "vq_track": 256,
    "vjepa2_rep64": 1024,
    "vjepa2_rep1": 1024,
}

ALL_ENCODERS = sorted(NATIVE_DIMS.keys())
HORIZONS = [8, 16]
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
def evaluate_one(
    encoder_name: str,
    seed: int,
    horizon: int,
    model_type: str,  # "dit_x0", "mlp_fair", "mlp_residual"
):
    """Evaluate a single model checkpoint. Returns per-step CosSim + copy baseline."""
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    # -------------------------------------------------------------------
    # Inline model definitions
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
    print(f"[eval] {model_type}/{encoder_name}/h{horizon}/s{seed}: {n_test} test windows")

    # -------------------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------------------

    if model_type == "dit_x0":
        ckpt_path = f"{DIT_DIR}/{encoder_name}/conditioned__x0__h{horizon}/seed_{seed}/checkpoint.pt"
    elif model_type == "mlp_fair":
        ckpt_path = f"{MLP_DIR}/latent_predictors_fair_h{horizon}/{encoder_name}/conditioned/seed_{seed}/checkpoint.pt"
    elif model_type == "mlp_residual":
        ckpt_path = f"{MLP_DIR}/latent_predictors_residual_h{horizon}/{encoder_name}/conditioned/seed_{seed}/checkpoint.pt"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if not os.path.exists(ckpt_path):
        print(f"[eval] MISSING: {ckpt_path}")
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

    fourier_embed = FourierActionEmbedding(
        action_dim=2, **FOURIER_CONFIG,
    ).to(device)
    fourier_embed.load_state_dict(ckpt["fourier_embed_state_dict"])
    fourier_embed.eval()

    if model_type == "dit_x0":
        dit = LatentDiT(**{**DIT_CONFIG, "horizon": horizon}).to(device)
        dit.load_state_dict(ckpt["dit_state_dict"])
        dit.eval()
        schedule = CosineNoiseSchedule(n_steps=DIFFUSION_CONFIG["n_train_steps"]).to(device)
        alphas_cumprod = schedule.alphas_cumprod.to(device)
    else:
        predictor = LatentPredictor(
            z_dim=TARGET_DIM, a_dim=TARGET_DIM, horizon=horizon,
        ).to(device)
        predictor.load_state_dict(ckpt["predictor_state_dict"])
        predictor.eval()

    is_residual = model_type == "mlp_residual"

    # -------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------

    z_t_test = z_t_test.to(device)
    act_test = act_test.to(device)
    zf_test = zf_test.to(device)

    test_ds = TensorDataset(z_t_test, act_test, zf_test)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    cossim_sums = [0.0] * horizon
    copy_sums = [0.0] * horizon
    total = 0

    torch.manual_seed(seed)
    np.random.seed(seed)

    # DDIM settings for DiT
    if model_type == "dit_x0":
        n_ddim_steps = 50
        T = DIFFUSION_CONFIG["n_train_steps"]
        stride = T // n_ddim_steps
        timesteps = list(reversed(list(range(0, T, stride))[:n_ddim_steps]))

    with torch.no_grad():
        for z_t_batch, act_batch, zf_batch in test_loader:
            B, H, _ = zf_batch.shape

            z_t_adapted = (adapter(z_t_batch) - z_mean) / z_std
            zf_adapted = (
                adapter(zf_batch.reshape(B * H, -1)).reshape(B, H, TARGET_DIM)
                - z_mean
            ) / z_std

            a_embed = fourier_embed(act_batch)

            if model_type == "dit_x0":
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

                z_hat_norm = x
            elif is_residual:
                z_hat_delta = predictor(z_t_adapted, a_embed)
                z_hat_norm = z_hat_delta + z_t_adapted.unsqueeze(1).expand(-1, H, -1)
            else:
                z_hat_norm = predictor(z_t_adapted, a_embed)

            z_hat = z_hat_norm * z_std + z_mean
            zf_orig = zf_adapted * z_std + z_mean
            z_t_unnorm = z_t_adapted * z_std + z_mean

            for k in range(horizon):
                cs = F.cosine_similarity(z_hat[:, k], zf_orig[:, k], dim=-1)
                cossim_sums[k] += cs.sum().item()
                copy_cs = F.cosine_similarity(z_t_unnorm, zf_orig[:, k], dim=-1)
                copy_sums[k] += copy_cs.sum().item()

            total += B

    result = {
        "encoder": encoder_name,
        "seed": seed,
        "horizon": horizon,
        "model": model_type,
        "n_test_windows": total,
        "cossim_by_step": [round(s / total, 6) for s in cossim_sums],
        "copy_by_step": [round(s / total, 6) for s in copy_sums],
    }
    print(
        f"[eval] {model_type}/{encoder_name}/h{horizon}/s{seed}: "
        f"CosSim@h1={result['cossim_by_step'][0]:.4f} "
        f"CosSim@hN={result['cossim_by_step'][-1]:.4f}"
    )
    return result


# ===================================================================
# Entrypoint
# ===================================================================


def _modal_entrypoint_decorator(fn):
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_modal_entrypoint_decorator
def main():
    """DA9 Exp 3: Full horizon evaluation on Modal."""
    import csv as csv_mod

    t_start = time.time()

    # Load DA8 h=4 baseline from local CSV
    da8_path = Path("code/latent-world-models-av/artifacts/full/da8_tierb_full.csv")
    if not da8_path.exists():
        da8_path = Path("artifacts/full/da8_tierb_full.csv")
    da8_rows = []
    if da8_path.exists():
        with open(da8_path) as f:
            for row in csv_mod.DictReader(f):
                da8_rows.append(row)
        print(f"Loaded {len(da8_rows)} DA8 h=4 reference rows")

    # Build job list
    jobs = []
    for enc in ALL_ENCODERS:
        for h in HORIZONS:
            for seed in SEEDS:
                for model in ["dit_x0", "mlp_fair", "mlp_residual"]:
                    jobs.append((enc, seed, h, model))

    n_jobs = len(jobs)
    print(f"\n{'='*70}")
    print(f"DA9 Exp 3: Full horizon evaluation ({n_jobs} jobs)")
    print(f"  encoders: {ALL_ENCODERS}")
    print(f"  horizons: {HORIZONS}")
    print(f"  seeds: {SEEDS}")
    print(f"  models: dit_x0, mlp_fair, mlp_residual")
    print(f"{'='*70}")

    # Launch all jobs
    futures = []
    for enc, seed, h, model in jobs:
        futures.append((enc, seed, h, model, evaluate_one.spawn(enc, seed, h, model)))

    # Collect results
    all_results = []
    for enc, seed, h, model, future in futures:
        result = future.get()
        if result is not None:
            all_results.append(result)

    wall_time = time.time() - t_start
    print(f"\nEvaluation complete: {len(all_results)} results in {wall_time:.0f}s")

    # Build CSV rows
    csv_rows = []

    # DA8 h=4 baseline rows
    for da8_row in da8_rows:
        enc = da8_row["encoder"]
        model = da8_row["model"]
        seed = int(da8_row["seed"])
        for k in range(1, 5):
            csv_rows.append({
                "encoder": enc,
                "seed": seed,
                "horizon_trained": 4,
                "horizon_step": k,
                "model": model,
                "cossim": float(da8_row[f"cossim_h{k}"]),
                "source": "da8",
            })

    # DA9 h=8, h=16 rows
    for r in all_results:
        for k, (cs, copy_cs) in enumerate(
            zip(r["cossim_by_step"], r["copy_by_step"])
        ):
            csv_rows.append({
                "encoder": r["encoder"],
                "seed": r["seed"],
                "horizon_trained": r["horizon"],
                "horizon_step": k + 1,
                "model": r["model"],
                "cossim": cs,
                "copy_cossim": copy_cs,
                "n_windows": r["n_test_windows"],
                "source": "da9",
            })

        # Copy baseline (one per encoder/seed/horizon, from any model)
        if r["model"] == "dit_x0":
            for k, copy_cs in enumerate(r["copy_by_step"]):
                csv_rows.append({
                    "encoder": r["encoder"],
                    "seed": r["seed"],
                    "horizon_trained": r["horizon"],
                    "horizon_step": k + 1,
                    "model": "copy",
                    "cossim": copy_cs,
                    "n_windows": r["n_test_windows"],
                    "source": "da9",
                })

    # Save CSV locally
    csv_path = Path("artifacts/full/da9_horizon_full.csv")
    if not csv_path.parent.exists():
        csv_path = Path("code/latent-world-models-av/artifacts/full/da9_horizon_full.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    all_keys = set()
    for r in csv_rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)

    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved {len(csv_rows)} rows to {csv_path}")

    # Save raw results as JSON
    json_path = csv_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved raw results to {json_path}")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'Encoder':<14} {'Model':<14} {'H':>3} {'CosSim@1':>10} {'CosSim@N':>10} {'Copy@N':>10}")
    print("-" * 90)
    for r in sorted(all_results, key=lambda x: (x["encoder"], x["model"], x["horizon"])):
        # Average across seeds would be better but this prints per-result
        if r["seed"] == 0:
            print(
                f"{r['encoder']:<14} {r['model']:<14} {r['horizon']:>3} "
                f"{r['cossim_by_step'][0]:>10.4f} {r['cossim_by_step'][-1]:>10.4f} "
                f"{r['copy_by_step'][-1]:>10.4f}"
            )
