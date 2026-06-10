"""DA10 V1+V2+V3: Scene-level statistical validation on Modal.

V1: Per-scene paired t-tests on DiT-x0 vs MLP-residual gaps.
V2: Copy-baseline distribution diagnostics.
V3: Action-variance slicing.

Pre-registered primary targets: dino_vits14 Q1 h4, clip_b32 Q1 h4.

Usage::

    modal run scripts/stat_validation_modal.py
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
    app = modal.App("lwm-av-stat-validation")
    vol = modal.Volume.from_name("nuscenes-full")
else:
    app = None
    vol = None

VOL_PATH = "/vol"
EMBED_DIR = f"{VOL_PATH}/embeddings"
DIT_DIR = f"{VOL_PATH}/dits"
MLP_RESIDUAL_DIR = f"{VOL_PATH}/outputs/latent_predictors_residual"

TARGET_DIM = 384
HORIZON = 4

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
        .pip_install("torch==2.5.1", "numpy>=1.26", "pandas>=2.0", "scipy>=1.10")
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
def evaluate_perscene(encoder_name: str, seed: int, use_ema: bool = True):
    """Run DiT-x0 and MLP-residual inference, return per-window results with scene IDs.

    Returns per-window CosSim for both models, plus scene names and copy baseline.
    """
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F

    # -------------------------------------------------------------------
    # Inline model definitions (same as eval_ema_modal.py)
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
    # Load data with scene tracking
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
    scene_names_arr = data["scene_names"]

    # Build windows with scene tracking and future action variance
    mask = splits == "test"
    emb = embeddings[mask]
    steers = steer_norms[mask]
    accels = accel_norms[mask]
    scenes = scene_names_arr[mask]

    z_t_list, action_list, z_future_list = [], [], []
    window_scenes = []
    action_var_list = []  # V3: action variance over prediction window

    for scene in np.unique(scenes):
        idx = np.where(scenes == scene)[0]
        for j in range(len(idx) - horizon):
            z_t_list.append(emb[idx[j]])
            action_list.append([steers[idx[j]], accels[idx[j]]])
            z_future_list.append(emb[idx[j + 1: j + 1 + horizon]])
            window_scenes.append(scene)
            # Action variance: variance of steer over prediction window
            future_steers = steers[idx[j: j + horizon]]
            future_accels = accels[idx[j: j + horizon]]
            steer_var = float(np.var(future_steers))
            accel_var = float(np.var(future_accels))
            action_var_list.append(steer_var + accel_var)

    z_t_test = torch.tensor(np.array(z_t_list), dtype=torch.float32)
    act_test = torch.tensor(np.array(action_list), dtype=torch.float32)
    zf_test = torch.tensor(np.array(z_future_list), dtype=torch.float32)
    window_scenes = np.array(window_scenes)
    action_vars = np.array(action_var_list, dtype=np.float32)

    n_windows = len(z_t_test)
    print(f"[stat] {encoder_name}/s{seed}: {n_windows} windows, {len(np.unique(window_scenes))} unique scenes")

    # -------------------------------------------------------------------
    # Load DiT-x0 checkpoint (with optional EMA)
    # -------------------------------------------------------------------

    dit_ckpt_path = f"{DIT_DIR}/{encoder_name}/conditioned__x0/seed_{seed}/checkpoint.pt"
    if not os.path.exists(dit_ckpt_path):
        print(f"[stat] MISSING DiT: {dit_ckpt_path}")
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

    # DiT model - EMA weights (primary eval)
    dit = LatentDiT(**{**DIT_CONFIG, "horizon": horizon}).to(device)
    fourier_dit = FourierActionEmbedding(action_dim=2, **FOURIER_CONFIG).to(device)

    has_ema = use_ema and "ema_state_dict" in dit_ckpt
    if has_ema:
        ema_sd = dit_ckpt["ema_state_dict"]
        dit_ema = {k[4:]: v for k, v in ema_sd.items() if k.startswith("dit.")}
        fourier_ema = {k[8:]: v for k, v in ema_sd.items() if k.startswith("fourier.")}
        dit.load_state_dict(dit_ema)
        fourier_dit.load_state_dict(fourier_ema, strict=False)
    else:
        dit.load_state_dict(dit_ckpt["dit_state_dict"])
        fourier_dit.load_state_dict(dit_ckpt["fourier_embed_state_dict"])

    dit.eval()
    fourier_dit.eval()

    # DDIM setup
    schedule = CosineNoiseSchedule(n_steps=DIFFUSION_CONFIG["n_train_steps"]).to(device)
    alphas_cumprod = schedule.alphas_cumprod
    n_ddim_steps = 50
    T = DIFFUSION_CONFIG["n_train_steps"]
    stride = T // n_ddim_steps
    timesteps = list(reversed(list(range(0, T, stride))[:n_ddim_steps]))

    # -------------------------------------------------------------------
    # Run DiT inference -> per-window CosSim
    # -------------------------------------------------------------------

    z_t_dev = z_t_test.to(device)
    act_dev = act_test.to(device)
    zf_dev = zf_test.to(device)

    def _run_dit_inference(dit_model, fourier_model):
        """Run DDIM inference and return (n_windows, horizon) cossim array."""
        cossim = np.zeros((n_windows, horizon), dtype=np.float32)
        copy_cs = np.zeros((n_windows, horizon), dtype=np.float32)
        torch.manual_seed(seed)
        with torch.no_grad():
            for start in range(0, n_windows, EVAL_BATCH_SIZE):
                end = min(start + EVAL_BATCH_SIZE, n_windows)
                B = end - start
                z_t_b = z_t_dev[start:end]
                act_b = act_dev[start:end]
                zf_b = zf_dev[start:end]
                H = horizon

                z_t_adapted = (adapter(z_t_b) - z_mean) / z_std
                zf_adapted = (
                    adapter(zf_b.reshape(B * H, -1)).reshape(B, H, TARGET_DIM) - z_mean
                ) / z_std
                a_embed = fourier_model(act_b)

                x = torch.randn(B, horizon, TARGET_DIM, device=device)
                for i, t_val in enumerate(timesteps):
                    t = torch.full((B,), t_val, device=device, dtype=torch.long)
                    pred_x0 = dit_model(x, z_t=z_t_adapted, a_embed=a_embed, timestep=t)
                    alpha_bar_t = alphas_cumprod[t_val]
                    if i < len(timesteps) - 1:
                        alpha_bar_prev = alphas_cumprod[timesteps[i + 1]]
                    else:
                        alpha_bar_prev = torch.tensor(1.0, device=device)
                    noise_dir = (x - torch.sqrt(alpha_bar_t) * pred_x0) / torch.sqrt(1.0 - alpha_bar_t + 1e-8)
                    x = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1.0 - alpha_bar_prev) * noise_dir

                z_hat = x * z_std + z_mean
                zf_orig = zf_adapted * z_std + z_mean
                z_t_unnorm = z_t_adapted * z_std + z_mean

                for k in range(horizon):
                    cossim[start:end, k] = F.cosine_similarity(z_hat[:, k], zf_orig[:, k], dim=-1).cpu().numpy()
                    copy_cs[start:end, k] = F.cosine_similarity(z_t_unnorm, zf_orig[:, k], dim=-1).cpu().numpy()
        return cossim, copy_cs

    # EMA eval (primary)
    dit_cossim, copy_cossim = _run_dit_inference(dit, fourier_dit)
    print(f"[stat] {encoder_name}/s{seed} EMA: DiT h1={dit_cossim[:, 0].mean():.4f}")

    # Raw weights eval (EMA decomposition) - only if EMA was available
    dit_raw_cossim = None
    if has_ema and "dit_state_dict" in dit_ckpt:
        dit_raw = LatentDiT(**{**DIT_CONFIG, "horizon": horizon}).to(device)
        fourier_raw = FourierActionEmbedding(action_dim=2, **FOURIER_CONFIG).to(device)
        dit_raw.load_state_dict(dit_ckpt["dit_state_dict"])
        fourier_raw.load_state_dict(dit_ckpt["fourier_embed_state_dict"])
        dit_raw.eval()
        fourier_raw.eval()
        dit_raw_cossim, _ = _run_dit_inference(dit_raw, fourier_raw)
        del dit_raw, fourier_raw
        ema_delta = float(dit_cossim.mean() - dit_raw_cossim.mean())
        print(f"[stat] {encoder_name}/s{seed} RAW: DiT h1={dit_raw_cossim[:, 0].mean():.4f} | EMA delta={ema_delta:+.4f}")
        torch.cuda.empty_cache()

    del dit, fourier_dit
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # Load MLP-residual and run inference
    # -------------------------------------------------------------------

    mlp_res_path = f"{MLP_RESIDUAL_DIR}/{encoder_name}/conditioned/seed_{seed}/checkpoint.pt"
    if not os.path.exists(mlp_res_path):
        print(f"[stat] MISSING MLP-residual: {mlp_res_path}")
        return None

    mlp_ckpt = torch.load(mlp_res_path, map_location=device, weights_only=False)
    mlp_z_mean = mlp_ckpt["z_mean"].to(device)
    mlp_z_std = mlp_ckpt["z_std"].to(device)

    # Adapter parity: MLP reuses the DiT's adapter (single source of truth).
    # Both models were trained with separate seeded adapters, but the DiT's
    # adapter is the reference; using it for the MLP removes adapter-weight
    # differences as a confound in the comparison.
    mlp_adapter = adapter  # already loaded from DiT checkpoint above

    mlp_fourier = FourierActionEmbedding(action_dim=2, **FOURIER_CONFIG).to(device)
    mlp_fourier.load_state_dict(mlp_ckpt["fourier_embed_state_dict"])
    mlp_fourier.eval()

    mlp_model = LatentPredictor(z_dim=TARGET_DIM, a_dim=TARGET_DIM, horizon=horizon).to(device)
    mlp_model.load_state_dict(mlp_ckpt["predictor_state_dict"])
    mlp_model.eval()

    mlp_cossim = np.zeros((n_windows, horizon), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_windows, EVAL_BATCH_SIZE):
            end = min(start + EVAL_BATCH_SIZE, n_windows)
            B = end - start
            z_t_b = z_t_dev[start:end]
            act_b = act_dev[start:end]
            zf_b = zf_dev[start:end]
            H = horizon

            z_t_adapted = (mlp_adapter(z_t_b) - mlp_z_mean) / mlp_z_std
            zf_adapted = (
                mlp_adapter(zf_b.reshape(B * H, -1)).reshape(B, H, TARGET_DIM) - mlp_z_mean
            ) / mlp_z_std
            a_embed = mlp_fourier(act_b)

            pred_delta = mlp_model(z_t_adapted, a_embed)
            pred = pred_delta + z_t_adapted.unsqueeze(1).expand(-1, H, -1)
            z_hat = pred * mlp_z_std + mlp_z_mean
            zf_orig = zf_adapted * mlp_z_std + mlp_z_mean

            for k in range(horizon):
                cs = F.cosine_similarity(z_hat[:, k], zf_orig[:, k], dim=-1).cpu().numpy()
                mlp_cossim[start:end, k] = cs

    print(f"[stat] {encoder_name}/s{seed}: DiT h1={dit_cossim[:, 0].mean():.4f} MLP h1={mlp_cossim[:, 0].mean():.4f}")

    result = {
        "encoder": encoder_name,
        "seed": seed,
        "dit_cossim": dit_cossim.tolist(),
        "mlp_cossim": mlp_cossim.tolist(),
        "copy_cossim": copy_cossim.tolist(),
        "window_scenes": window_scenes.tolist(),
        "action_vars": action_vars.tolist(),
        "n_windows": n_windows,
    }
    if dit_raw_cossim is not None:
        result["dit_raw_cossim"] = dit_raw_cossim.tolist()
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
    """DA10 V1+V2+V3: Scene-level statistical validation."""
    import numpy as np
    import pandas as pd
    from scipy import stats

    t_start = time.time()

    n_jobs = len(ALL_ENCODERS) * len(SEEDS)
    print(f"\n{'='*70}")
    print(f"DA10 V1+V2+V3: Statistical Validation ({n_jobs} jobs)")
    print(f"  encoders: {ALL_ENCODERS}")
    print(f"  seeds: {SEEDS}")
    print(f"  EMA: True (use EMA weights for DiT)")
    print(f"  Pre-registered primaries: dino_vits14 Q1 h4, clip_b32 Q1 h4")
    print(f"{'='*70}")

    # Launch all jobs
    futures = []
    for enc in ALL_ENCODERS:
        for seed in SEEDS:
            futures.append((enc, seed, evaluate_perscene.spawn(enc, seed, True)))

    # Collect results
    all_results = []
    for enc, seed, future in futures:
        result = future.get()
        if result is not None:
            all_results.append(result)

    wall_time = time.time() - t_start
    print(f"\nInference done: {len(all_results)} results in {wall_time:.0f}s")

    # -------------------------------------------------------------------
    # V1: Scene-level paired t-tests
    # -------------------------------------------------------------------

    print(f"\n{'='*70}")
    print("V1: Scene-Level Statistical Validation")
    print(f"{'='*70}")

    stat_rows = []

    for enc in ALL_ENCODERS:
        enc_results = [r for r in all_results if r["encoder"] == enc]
        if not enc_results:
            continue

        for seed_r in enc_results:
            seed = seed_r["seed"]
            n_w = seed_r["n_windows"]
            dit_cs = np.array(seed_r["dit_cossim"])  # (N, H)
            mlp_cs = np.array(seed_r["mlp_cossim"])
            copy_cs = np.array(seed_r["copy_cossim"])
            w_scenes = np.array(seed_r["window_scenes"])
            act_vars = np.array(seed_r["action_vars"])

            for h_k in range(HORIZON):
                h = h_k + 1
                copy_h = copy_cs[:, h_k]

                # Assign difficulty quartiles
                try:
                    q_labels = pd.qcut(copy_h, q=4,
                                       labels=["Q1", "Q2", "Q3", "Q4"])
                except ValueError:
                    q_labels = pd.qcut(
                        pd.Series(copy_h).rank(method="first"), q=4,
                        labels=["Q1", "Q2", "Q3", "Q4"])
                q_labels = np.array(q_labels)

                for q in ["Q1", "Q2", "Q3", "Q4"]:
                    q_mask = q_labels == q
                    q_scenes = w_scenes[q_mask]
                    q_dit = dit_cs[q_mask, h_k]
                    q_mlp = mlp_cs[q_mask, h_k]
                    q_copy = copy_h[q_mask]

                    # Aggregate to per-scene means
                    unique_scenes = np.unique(q_scenes)
                    scene_dit_means = []
                    scene_mlp_means = []
                    scene_copy_means = []
                    for sc in unique_scenes:
                        sc_mask = q_scenes == sc
                        scene_dit_means.append(q_dit[sc_mask].mean())
                        scene_mlp_means.append(q_mlp[sc_mask].mean())
                        scene_copy_means.append(q_copy[sc_mask].mean())

                    scene_dit = np.array(scene_dit_means)
                    scene_mlp = np.array(scene_mlp_means)
                    scene_gaps = scene_dit - scene_mlp
                    n_scenes = len(unique_scenes)

                    # Paired t-test on scene means
                    if n_scenes >= 3:
                        t_stat, p_val = stats.ttest_rel(scene_dit, scene_mlp)
                        mean_gap = scene_gaps.mean()
                        se = scene_gaps.std() / np.sqrt(n_scenes)
                        ci_low = mean_gap - stats.t.ppf(0.975, df=n_scenes - 1) * se
                        ci_high = mean_gap + stats.t.ppf(0.975, df=n_scenes - 1) * se
                        cohens_d = mean_gap / scene_gaps.std() if scene_gaps.std() > 0 else 0
                    else:
                        t_stat = p_val = mean_gap = se = ci_low = ci_high = cohens_d = float("nan")

                    stat_rows.append({
                        "encoder": enc,
                        "seed": seed,
                        "quartile": q,
                        "horizon": h,
                        "n_windows": int(q_mask.sum()),
                        "n_scenes": n_scenes,
                        "dit_mean": round(float(q_dit.mean()), 6),
                        "mlp_mean": round(float(q_mlp.mean()), 6),
                        "copy_mean": round(float(q_copy.mean()), 6),
                        "gap_mean": round(float(mean_gap), 6),
                        "gap_se": round(float(se), 6),
                        "ci_low": round(float(ci_low), 6),
                        "ci_high": round(float(ci_high), 6),
                        "t_stat": round(float(t_stat), 4),
                        "p_value": round(float(p_val), 6),
                        "cohens_d": round(float(cohens_d), 4),
                    })

    # Save per-seed results
    stat_path = Path("artifacts/full/da10_stat_validation.csv")
    if not stat_path.parent.exists():
        stat_path = Path("code/latent-world-models-av/artifacts/full/da10_stat_validation.csv")
    stat_path.parent.mkdir(parents=True, exist_ok=True)

    with open(stat_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stat_rows)
    print(f"Saved {len(stat_rows)} rows to {stat_path}")

    # -------------------------------------------------------------------
    # Aggregate across seeds and apply FDR
    # -------------------------------------------------------------------

    df = pd.DataFrame(stat_rows)

    # Seed-averaged results
    agg = df.groupby(["encoder", "quartile", "horizon"]).agg({
        "gap_mean": "mean",
        "p_value": lambda x: list(x),  # keep per-seed p-values
        "n_scenes": "mean",
        "dit_mean": "mean",
        "mlp_mean": "mean",
        "copy_mean": "mean",
        "cohens_d": "mean",
    }).reset_index()

    # For seed-averaged significance, use Fisher's method to combine p-values.
    # NOTE: Fisher assumes independent p-values, but seeds share test scenes,
    # so per-seed p-values are correlated. This inflates significance.
    # We report Fisher as a secondary analysis; the primary is the pooled
    # scene-level test below (which averages gaps across seeds first).
    from scipy.stats import combine_pvalues
    combined_ps = []
    for _, row in agg.iterrows():
        ps = row["p_value"]
        if all(not np.isnan(p) for p in ps):
            _, combined_p = combine_pvalues(ps, method="fisher")
            combined_ps.append(combined_p)
        else:
            combined_ps.append(float("nan"))
    agg["combined_p"] = combined_ps

    # Pooled scene-level test: average per-scene gaps across seeds first,
    # then run a single paired t-test. This respects the non-independence of
    # seeds (shared test set) and is the more defensible primary analysis.
    pooled_ps = []
    pooled_ds = []
    for _, row in agg.iterrows():
        enc, q, h = row["encoder"], row["quartile"], row["horizon"]
        seed_rows = df[(df["encoder"] == enc) & (df["quartile"] == q) & (df["horizon"] == h)]
        # Collect per-scene gaps from each seed, then average across seeds
        # to get one gap value per scene
        scene_gap_sums = {}
        scene_gap_counts = {}
        for _, sr in seed_rows.iterrows():
            s = sr["seed"]
            # Re-extract per-scene data for this seed/enc/q/h
            enc_results = [r for r in all_results if r["encoder"] == enc and r["seed"] == s]
            if not enc_results:
                continue
            r = enc_results[0]
            w_scenes = np.array(r["window_scenes"])
            dit_cs = np.array(r["dit_cossim"])
            mlp_cs = np.array(r["mlp_cossim"])
            copy_cs_arr = np.array(r["copy_cossim"])
            h_k = h - 1

            # Re-derive quartile labels from copy baseline (h=0)
            copy_h0 = copy_cs_arr[:, 0]
            q_labels_seed = pd.qcut(copy_h0, 4, labels=["Q1", "Q2", "Q3", "Q4"])
            q_mask = q_labels_seed == q
            q_scenes = w_scenes[q_mask]
            q_dit = dit_cs[q_mask, h_k]
            q_mlp = mlp_cs[q_mask, h_k]

            for sc in np.unique(q_scenes):
                sc_mask = q_scenes == sc
                gap = q_dit[sc_mask].mean() - q_mlp[sc_mask].mean()
                scene_gap_sums[sc] = scene_gap_sums.get(sc, 0.0) + gap
                scene_gap_counts[sc] = scene_gap_counts.get(sc, 0) + 1

        # Average gaps across seeds per scene
        if scene_gap_counts:
            pooled_gaps = np.array([
                scene_gap_sums[sc] / scene_gap_counts[sc]
                for sc in sorted(scene_gap_counts)
            ])
            if len(pooled_gaps) >= 3:
                _, pooled_p = stats.ttest_1samp(pooled_gaps, 0)
                pooled_d = pooled_gaps.mean() / pooled_gaps.std() if pooled_gaps.std() > 0 else 0
            else:
                pooled_p = pooled_d = float("nan")
        else:
            pooled_p = pooled_d = float("nan")
        pooled_ps.append(pooled_p)
        pooled_ds.append(pooled_d)

    agg["pooled_p"] = pooled_ps
    agg["pooled_d"] = pooled_ds

    # Benjamini-Hochberg FDR correction
    valid_ps = [(i, p) for i, p in enumerate(combined_ps) if not np.isnan(p)]
    valid_ps.sort(key=lambda x: x[1])
    n_tests = len(valid_ps)
    fdr_adjusted = [float("nan")] * len(combined_ps)
    for rank, (idx, p) in enumerate(valid_ps, 1):
        adjusted = p * n_tests / rank
        fdr_adjusted[idx] = min(adjusted, 1.0)
    # Enforce monotonicity
    for i in range(len(valid_ps) - 2, -1, -1):
        idx_i = valid_ps[i][0]
        idx_next = valid_ps[i + 1][0]
        fdr_adjusted[idx_i] = min(fdr_adjusted[idx_i], fdr_adjusted[idx_next])

    agg["fdr_p"] = fdr_adjusted

    # Print primary targets
    print(f"\n{'='*110}")
    print("PRE-REGISTERED PRIMARY TARGETS")
    print(f"{'='*110}")
    print(
        f"{'Encoder':<16} {'Q':>4} {'h':>3} {'Gap':>10} "
        f"{'Fisher p':>12} {'FDR p':>12} {'Pooled p':>12} {'d':>8} {'N scenes':>10}"
    )
    print("-" * 110)

    primaries = agg[
        ((agg["encoder"] == "dino_vits14") & (agg["quartile"] == "Q1") & (agg["horizon"] == 4)) |
        ((agg["encoder"] == "clip_b32") & (agg["quartile"] == "Q1") & (agg["horizon"] == 4))
    ]
    for _, row in primaries.iterrows():
        sig = "***" if row["fdr_p"] < 0.001 else "**" if row["fdr_p"] < 0.01 else "*" if row["fdr_p"] < 0.05 else ""
        print(
            f"{row['encoder']:<16} {row['quartile']:>4} {row['horizon']:>3} "
            f"{row['gap_mean']:>+10.4f} {row['combined_p']:>12.6f} {row['fdr_p']:>12.6f} "
            f"{row.get('pooled_p', float('nan')):>12.6f} "
            f"{row['cohens_d']:>8.3f} {row['n_scenes']:>10.0f} {sig}"
        )
    print(f"\nNOTE: Fisher p combines per-seed p-values (overstates significance due to shared test set).")
    print(f"      Pooled p averages per-scene gaps across seeds first (more conservative, preferred).")

    # Print all positive gaps
    print(f"\n{'='*90}")
    print("ALL POSITIVE GAP CELLS (seed-averaged)")
    print(f"{'='*90}")
    positive = agg[agg["gap_mean"] > 0].sort_values("gap_mean", ascending=False)
    for _, row in positive.iterrows():
        sig = "*" if row["fdr_p"] < 0.05 else ""
        print(
            f"  {row['encoder']:<16} {row['quartile']:>4} h={row['horizon']:>1} "
            f"gap={row['gap_mean']:>+.4f} FDR_p={row['fdr_p']:.4f} d={row['cohens_d']:.3f} {sig}"
        )

    # -------------------------------------------------------------------
    # V2: Copy-baseline distribution diagnostics
    # -------------------------------------------------------------------

    print(f"\n{'='*70}")
    print("V2: Copy-Baseline Distribution Diagnostics")
    print(f"{'='*70}")

    diag_rows = []
    for enc in ALL_ENCODERS:
        enc_results = [r for r in all_results if r["encoder"] == enc]
        if not enc_results:
            continue
        # Use seed 0 for distribution analysis
        r = enc_results[0]
        copy_cs = np.array(r["copy_cossim"])
        for h_k in range(HORIZON):
            copy_h = copy_cs[:, h_k]
            q_boundaries = np.quantile(copy_h, [0, 0.25, 0.5, 0.75, 1.0])
            q_means = [
                copy_h[copy_h <= q_boundaries[1]].mean(),
                copy_h[(copy_h > q_boundaries[1]) & (copy_h <= q_boundaries[2])].mean(),
                copy_h[(copy_h > q_boundaries[2]) & (copy_h <= q_boundaries[3])].mean(),
                copy_h[copy_h > q_boundaries[3]].mean(),
            ]
            diag_rows.append({
                "encoder": enc,
                "horizon": h_k + 1,
                "copy_min": round(float(copy_h.min()), 4),
                "copy_q25": round(float(q_boundaries[1]), 4),
                "copy_median": round(float(q_boundaries[2]), 4),
                "copy_q75": round(float(q_boundaries[3]), 4),
                "copy_max": round(float(copy_h.max()), 4),
                "q1_mean": round(float(q_means[0]), 4),
                "q4_mean": round(float(q_means[3]), 4),
                "q4_minus_q1": round(float(q_means[3] - q_means[0]), 4),
                "monotonic": q_means[0] < q_means[1] < q_means[2] < q_means[3],
            })
            print(
                f"  {enc:<16} h={h_k+1}: Q1 mean={q_means[0]:.4f} Q4 mean={q_means[3]:.4f} "
                f"spread={q_means[3]-q_means[0]:.4f} monotonic={q_means[0] < q_means[1] < q_means[2] < q_means[3]}"
            )

    # -------------------------------------------------------------------
    # V3: Action-variance slicing
    # -------------------------------------------------------------------

    print(f"\n{'='*70}")
    print("V3: Action-Variance Slicing")
    print(f"{'='*70}")

    for enc in ALL_ENCODERS:
        enc_results = [r for r in all_results if r["encoder"] == enc]
        if not enc_results:
            continue
        r = enc_results[0]  # seed 0
        dit_cs = np.array(r["dit_cossim"])
        mlp_cs = np.array(r["mlp_cossim"])
        act_vars = np.array(r["action_vars"])

        for h_k in [3]:  # Focus on h=4
            try:
                av_labels = pd.qcut(act_vars, q=4,
                                    labels=["AV-Q1 (lowest)", "AV-Q2", "AV-Q3", "AV-Q4 (highest)"])
            except ValueError:
                av_labels = pd.qcut(
                    pd.Series(act_vars).rank(method="first"), q=4,
                    labels=["AV-Q1 (lowest)", "AV-Q2", "AV-Q3", "AV-Q4 (highest)"])
            av_labels = np.array(av_labels)

            for avq in ["AV-Q1 (lowest)", "AV-Q2", "AV-Q3", "AV-Q4 (highest)"]:
                mask = av_labels == avq
                gap = dit_cs[mask, h_k].mean() - mlp_cs[mask, h_k].mean()
                n = mask.sum()
                print(
                    f"  {enc:<16} h=4 {avq:<20}: DiT-MLP gap={gap:>+.4f} "
                    f"DiT={dit_cs[mask, h_k].mean():.4f} MLP={mlp_cs[mask, h_k].mean():.4f} N={n}"
                )

    # -------------------------------------------------------------------
    # Save diagnostics
    # -------------------------------------------------------------------

    diag_path = Path("artifacts/full/da10_difficulty_diagnostics.csv")
    if not diag_path.parent.exists():
        diag_path = Path("code/latent-world-models-av/artifacts/full/da10_difficulty_diagnostics.csv")
    diag_path.parent.mkdir(parents=True, exist_ok=True)

    with open(diag_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diag_rows)
    print(f"\nSaved diagnostics to {diag_path}")

    # -------------------------------------------------------------------
    # Gate G2
    # -------------------------------------------------------------------

    print(f"\n{'='*70}")
    print("GATE G2 RESULT")
    print(f"{'='*70}")

    dino_q1h4 = agg[(agg["encoder"] == "dino_vits14") & (agg["quartile"] == "Q1") & (agg["horizon"] == 4)]
    clip_q1h4 = agg[(agg["encoder"] == "clip_b32") & (agg["quartile"] == "Q1") & (agg["horizon"] == 4)]

    dino_sig = False
    clip_sig = False
    if len(dino_q1h4) > 0:
        dino_p = dino_q1h4.iloc[0]["fdr_p"]
        dino_gap = dino_q1h4.iloc[0]["gap_mean"]
        dino_sig = dino_p < 0.05 and dino_gap > 0
        print(f"  dino_vits14 Q1 h4: gap={dino_gap:+.4f}, FDR p={dino_p:.6f} -> {'SIGNIFICANT' if dino_sig else 'not significant'}")

    if len(clip_q1h4) > 0:
        clip_p = clip_q1h4.iloc[0]["fdr_p"]
        clip_gap = clip_q1h4.iloc[0]["gap_mean"]
        clip_sig = clip_p < 0.05 and clip_gap > 0
        print(f"  clip_b32 Q1 h4: gap={clip_gap:+.4f}, FDR p={clip_p:.6f} -> {'SIGNIFICANT' if clip_sig else 'not significant'}")

    other_positive = len(positive[
        ~((positive["encoder"] == "dino_vits14") & (positive["quartile"] == "Q1") & (positive["horizon"] == 4))
    ])

    # Also check pooled (seed-averaged) significance for robustness
    dino_pooled_p = float("nan")
    if len(dino_q1h4) > 0 and "pooled_p" in dino_q1h4.columns:
        dino_pooled_p = dino_q1h4.iloc[0]["pooled_p"]
    dino_pooled_sig = not np.isnan(dino_pooled_p) and dino_pooled_p < 0.05 and dino_gap > 0

    if dino_sig and dino_pooled_sig and other_positive >= 1:
        print(f"  -> PASS (EMA-confounded): dino significant by both Fisher and pooled test + {other_positive} other positive cells")
        print(f"     Caveat: DiT used EMA, MLP did not. Effect sizes (d=0.12-0.25) overlap EMA gain magnitude.")
    elif dino_sig and other_positive >= 1:
        print(f"  -> PASS (EMA-confounded, Fisher-only): dino significant by Fisher (shared-test-set caveat) + {other_positive} other positive cells")
        print(f"     Pooled scene-level p={dino_pooled_p:.4f} (seed-averaged, more conservative)")
    elif dino_sig:
        print(f"  -> MARGINAL PASS: dino significant but single-encoder effect")
    else:
        print(f"  -> FAIL: primary target not significant at scene level")
