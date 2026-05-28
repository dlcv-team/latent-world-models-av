"""DA8: Warm-start DDIM evaluation -- can the DDIM bottleneck be improved?

Tests inference-only modifications to DDIM sampling using existing
DiT-DDIM checkpoints (no retraining). Three experiments:

1. **Step sweep:** vary DDIM step count at full noise (t_start=980)
2. **Warm-start sweep:** vary t_start (SDEdit-style) with best step count
3. **Two-model guidance:** mix conditioned + unconditioned predictions

Also reports z_hat_norm / z_real_norm ratio as a norm diagnostic.

Usage
-----
    python -m evaluation.warmstart_eval --encoder vit_s16
    python -m evaluation.warmstart_eval --encoder vit_s16 --mode steps
    python -m evaluation.warmstart_eval --encoder vit_s16 --mode warmstart
    python -m evaluation.warmstart_eval --encoder vit_s16 --mode guidance
    python -m evaluation.warmstart_eval  # all encoders, warmstart mode
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from evaluation.dit_utils import (
    DEFAULT_HORIZON,
    NATIVE_DIMS,
    TARGET_DIM,
    build_windows,
    load_embeddings,
)
from models.diffusion import CosineNoiseSchedule, DDIMSampler
from models.fourier_embed import FourierActionEmbedding

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIT_CKPT_ROOT = Path("outputs/dits")
SWEEP_CSV_PATH = Path("artifacts/full/warmstart_sweep.csv")

ENCODER_NAMES = sorted(NATIVE_DIMS.keys())
SEEDS = [0, 1, 2]

# DiT architecture (must match configs/dit.yaml)
DIT_CONFIG = {
    "n_blocks": 4,
    "n_heads": 6,
    "z_dim": 384,
    "horizon": 4,
    "cond_dim": 384,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
}

DIFFUSION_CONFIG = {"n_train_steps": 1000}
FOURIER_CONFIG = {"n_frequencies": 64, "base": 2.0, "out_dim": 384}
EVAL_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Inline DiT model (matches scripts/rollout_dit.py exactly)
# ---------------------------------------------------------------------------


def _modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


class TimestepEmbedding(nn.Module):
    def __init__(self, cond_dim=384):
        super().__init__()
        self.cond_dim = cond_dim
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
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


class DiTBlock(nn.Module):
    def __init__(self, dim=384, cond_dim=384, n_heads=6, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.adaln_linear = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.adaln_linear.weight)
        nn.init.zeros_(self.adaln_linear.bias)

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
        nn.init.zeros_(self.final_adaln.weight)
        nn.init.zeros_(self.final_adaln.bias)
        self.final_linear = nn.Linear(z_dim, z_dim)

    def forward(self, x_noisy, z_t=None, a_embed=None, timestep=None):
        cond = self.timestep_embed(timestep) + self.z_t_proj(z_t) + a_embed
        x = self.input_proj(x_noisy)
        for block in self.blocks:
            x = block(x, cond)
        mod = self.final_adaln(cond).unsqueeze(1)
        shift, scale, gate = mod.chunk(3, dim=-1)
        x = gate * self.final_linear(
            _modulate(self.final_norm(x), shift, scale)
        )
        return x


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_dit_checkpoint(encoder_name: str, variant: str, seed: int, device: torch.device):
    """Load a trained DiT checkpoint and return (dit, adapter, fourier, z_mean, z_std)."""
    ckpt_path = DIT_CKPT_ROOT / encoder_name / variant / f"seed_{seed}" / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    z_mean = ckpt["z_mean"].to(device)
    z_std = ckpt["z_std"].to(device)

    native_dim = NATIVE_DIMS[encoder_name]
    needs_adapter = native_dim != TARGET_DIM
    if needs_adapter:
        adapter = nn.Linear(native_dim, TARGET_DIM, bias=False).to(device)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        for p in adapter.parameters():
            p.requires_grad_(False)
    else:
        adapter = nn.Identity().to(device)

    fourier_embed = FourierActionEmbedding(
        action_dim=2,
        n_frequencies=FOURIER_CONFIG["n_frequencies"],
        base=FOURIER_CONFIG["base"],
        out_dim=FOURIER_CONFIG["out_dim"],
    ).to(device)
    fourier_embed.load_state_dict(ckpt["fourier_embed_state_dict"])
    fourier_embed.eval()

    dit = LatentDiT(**DIT_CONFIG).to(device)
    dit.load_state_dict(ckpt["dit_state_dict"])
    dit.eval()

    return dit, adapter, fourier_embed, z_mean, z_std


# ---------------------------------------------------------------------------
# Test data preparation
# ---------------------------------------------------------------------------


def prepare_test_data(encoder_name: str, device: torch.device):
    """Load embeddings, build test windows, return tensors on device."""
    data = load_embeddings(encoder_name)
    result = build_windows(data, split="test", horizon=DEFAULT_HORIZON)
    if result is None:
        raise RuntimeError(f"No test windows for {encoder_name}")
    z_t, act, zf = result
    return z_t.to(device), act.to(device), zf.to(device)


# ---------------------------------------------------------------------------
# Core evaluation: run DDIM (standard or warm-start) and compute metrics
# ---------------------------------------------------------------------------


def evaluate_ddim(
    dit: nn.Module,
    adapter: nn.Module,
    fourier_embed: nn.Module,
    z_mean: torch.Tensor,
    z_std: torch.Tensor,
    z_t_test: torch.Tensor,
    act_test: torch.Tensor,
    zf_test: torch.Tensor,
    n_steps: int = 50,
    t_start: int | None = None,
    guidance_w: float = 1.0,
    dit_uncond: nn.Module | None = None,
    fourier_uncond: nn.Module | None = None,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run DDIM evaluation with optional warm-start and guidance.

    Parameters
    ----------
    t_start : int or None
        If None, standard DDIM from noise. If int, warm-start from
        z_t noised to that timestep.
    guidance_w : float
        Guidance weight. 1.0 = no guidance. > 1.0 = amplify conditioning.
        Requires dit_uncond to be provided.
    dit_uncond, fourier_uncond : nn.Module or None
        Unconditioned model for two-model guidance.
    """
    horizon = DEFAULT_HORIZON

    schedule = CosineNoiseSchedule(n_steps=DIFFUSION_CONFIG["n_train_steps"]).to(device)
    sampler = DDIMSampler(schedule, n_steps=n_steps)

    torch.manual_seed(seed)
    np.random.seed(seed)

    from torch.utils.data import DataLoader, TensorDataset

    test_ds = TensorDataset(z_t_test, act_test, zf_test)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    cossim_sums = [0.0] * horizon
    mse_sums = [0.0] * horizon
    copy_cossim_sums = [0.0] * horizon
    z_hat_norm_sum = 0.0
    z_real_norm_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for z_t_batch, act_batch, zf_batch in test_loader:
            z_t_batch = z_t_batch.to(device)
            act_batch = act_batch.to(device)
            zf_batch = zf_batch.to(device)
            B = z_t_batch.shape[0]

            # Adapter + normalize
            B_f, H, _ = zf_batch.shape
            z_t_adapted = (adapter(z_t_batch) - z_mean) / z_std
            zf_adapted = (
                adapter(zf_batch.reshape(B_f * H, -1)).reshape(B_f, H, TARGET_DIM)
                - z_mean
            ) / z_std

            a_embed = fourier_embed(act_batch)

            # Build noise prediction function (with optional guidance)
            if guidance_w != 1.0 and dit_uncond is not None:
                a_embed_zero = torch.zeros_like(a_embed)

                def noise_pred_fn(x, timestep, z_t, a_embed):
                    eps_cond = dit(x, z_t=z_t, a_embed=a_embed, timestep=timestep)
                    eps_uncond = dit_uncond(
                        x, z_t=z_t, a_embed=a_embed_zero, timestep=timestep
                    )
                    return eps_uncond + guidance_w * (eps_cond - eps_uncond)
            else:

                def noise_pred_fn(x, timestep, z_t, a_embed):
                    return dit(x, z_t=z_t, a_embed=a_embed, timestep=timestep)

            cond_kwargs = {"z_t": z_t_adapted, "a_embed": a_embed}

            if t_start is not None:
                # Warm-start: initialize from z_t
                x_init = z_t_adapted.unsqueeze(1).expand(-1, horizon, -1).clone()
                z_hat_norm = sampler.sample_warm_start(
                    noise_pred_fn, x_init, t_start=t_start,
                    cond_kwargs=cond_kwargs, device=device,
                )
            else:
                # Standard DDIM from noise
                z_hat_norm = sampler.sample(
                    noise_pred_fn, shape=(B, horizon, TARGET_DIM),
                    cond_kwargs=cond_kwargs, device=device,
                )

            # Inverse-transform
            z_hat = z_hat_norm * z_std + z_mean
            zf_orig = zf_adapted * z_std + z_mean
            z_t_orig = z_t_adapted * z_std + z_mean

            # Norm diagnostic
            z_hat_norm_sum += z_hat.norm(dim=-1).sum().item()
            z_real_norm_sum += zf_orig.norm(dim=-1).sum().item()

            # Per-horizon metrics
            for k in range(horizon):
                z_hat_k = z_hat[:, k]
                z_real_k = zf_orig[:, k]

                cossim = F.cosine_similarity(z_hat_k, z_real_k, dim=-1)
                mse = ((z_hat_k - z_real_k) ** 2).mean(dim=-1)
                copy_cossim = F.cosine_similarity(z_t_orig, z_real_k, dim=-1)

                cossim_sums[k] += cossim.sum().item()
                mse_sums[k] += mse.sum().item()
                copy_cossim_sums[k] += copy_cossim.sum().item()

            total_samples += B

    return {
        "cossim_by_horizon": [s / total_samples for s in cossim_sums],
        "mse_by_horizon": [s / total_samples for s in mse_sums],
        "copy_cossim_by_horizon": [s / total_samples for s in copy_cossim_sums],
        "z_hat_norm_ratio": z_hat_norm_sum / (z_real_norm_sum + 1e-8),
        "n_test_windows": total_samples,
    }


# ---------------------------------------------------------------------------
# Sweep runners
# ---------------------------------------------------------------------------


def run_step_sweep(encoder_name, seed, device):
    """Sweep n_steps at full DDIM (no warm-start)."""
    dit, adapter, fourier, z_mean, z_std = load_dit_checkpoint(
        encoder_name, "conditioned", seed, device
    )
    z_t, act, zf = prepare_test_data(encoder_name, device)

    step_values = [5, 10, 20, 50, 100, 200]
    results = []
    for n_steps in step_values:
        r = evaluate_ddim(
            dit, adapter, fourier, z_mean, z_std, z_t, act, zf,
            n_steps=n_steps, t_start=None, seed=seed, device=device,
        )
        results.append({
            "encoder": encoder_name, "variant": "conditioned", "seed": seed,
            "t_start": "noise", "n_ddim_steps": n_steps, "guidance_w": 1.0,
            **{f"cossim_h{k+1}": r["cossim_by_horizon"][k] for k in range(4)},
            "mse_h1": r["mse_by_horizon"][0],
            "z_hat_norm_ratio": r["z_hat_norm_ratio"],
        })
        print(
            f"  steps={n_steps:>3}: CosSim@h1={r['cossim_by_horizon'][0]:.4f}  "
            f"norm_ratio={r['z_hat_norm_ratio']:.3f}"
        )
    return results


def run_warmstart_sweep(encoder_name, seed, n_steps, device):
    """Sweep t_start at fixed n_steps."""
    dit, adapter, fourier, z_mean, z_std = load_dit_checkpoint(
        encoder_name, "conditioned", seed, device
    )
    z_t, act, zf = prepare_test_data(encoder_name, device)

    t_start_values = [0, 40, 100, 200, 400, 600, 800, 980]
    results = []
    for t_start in t_start_values:
        r = evaluate_ddim(
            dit, adapter, fourier, z_mean, z_std, z_t, act, zf,
            n_steps=n_steps, t_start=t_start, seed=seed, device=device,
        )
        results.append({
            "encoder": encoder_name, "variant": "conditioned", "seed": seed,
            "t_start": t_start, "n_ddim_steps": n_steps, "guidance_w": 1.0,
            **{f"cossim_h{k+1}": r["cossim_by_horizon"][k] for k in range(4)},
            "mse_h1": r["mse_by_horizon"][0],
            "z_hat_norm_ratio": r["z_hat_norm_ratio"],
        })
        # Count effective steps
        schedule = CosineNoiseSchedule(n_steps=1000)
        sampler = DDIMSampler(schedule, n_steps=n_steps)
        eff_steps = len([t for t in sampler.timesteps if t <= t_start]) if t_start > 0 else 0
        print(
            f"  t_start={t_start:>3} ({eff_steps:>2} steps): "
            f"CosSim@h1={r['cossim_by_horizon'][0]:.4f}  "
            f"norm_ratio={r['z_hat_norm_ratio']:.3f}"
        )
    return results


def run_guidance_sweep(encoder_name, seed, n_steps, t_start, device):
    """Sweep guidance weight using two independently trained models."""
    dit_cond, adapter, fourier, z_mean, z_std = load_dit_checkpoint(
        encoder_name, "conditioned", seed, device
    )
    dit_uncond, _, fourier_uncond, _, _ = load_dit_checkpoint(
        encoder_name, "unconditioned", seed, device
    )
    z_t, act, zf = prepare_test_data(encoder_name, device)

    w_values = [1.0, 1.5, 2.0, 3.0]
    results = []
    for w in w_values:
        r = evaluate_ddim(
            dit_cond, adapter, fourier, z_mean, z_std, z_t, act, zf,
            n_steps=n_steps, t_start=t_start, guidance_w=w,
            dit_uncond=dit_uncond, fourier_uncond=fourier_uncond,
            seed=seed, device=device,
        )
        results.append({
            "encoder": encoder_name, "variant": "conditioned", "seed": seed,
            "t_start": t_start if t_start is not None else "noise",
            "n_ddim_steps": n_steps, "guidance_w": w,
            **{f"cossim_h{k+1}": r["cossim_by_horizon"][k] for k in range(4)},
            "mse_h1": r["mse_by_horizon"][0],
            "z_hat_norm_ratio": r["z_hat_norm_ratio"],
        })
        print(
            f"  w={w:.1f}: CosSim@h1={r['cossim_by_horizon'][0]:.4f}  "
            f"norm_ratio={r['z_hat_norm_ratio']:.3f}"
        )
    return results


# ---------------------------------------------------------------------------
# Copy bypass (true copy baseline reference)
# ---------------------------------------------------------------------------


def compute_copy_baseline(encoder_name, seed, device):
    """Return z_t as the prediction -- true copy baseline."""
    dit, adapter, fourier, z_mean, z_std = load_dit_checkpoint(
        encoder_name, "conditioned", seed, device
    )
    z_t, act, zf = prepare_test_data(encoder_name, device)

    from torch.utils.data import DataLoader, TensorDataset

    test_ds = TensorDataset(z_t, act, zf)
    loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    cossim_sums = [0.0] * DEFAULT_HORIZON
    total = 0

    with torch.no_grad():
        for z_t_b, _, zf_b in loader:
            z_t_b, zf_b = z_t_b.to(device), zf_b.to(device)
            B, H, _ = zf_b.shape
            z_t_adapted = adapter(z_t_b)
            zf_adapted = adapter(zf_b.reshape(B * H, -1)).reshape(B, H, TARGET_DIM)
            for k in range(DEFAULT_HORIZON):
                cs = F.cosine_similarity(z_t_adapted, zf_adapted[:, k], dim=-1)
                cossim_sums[k] += cs.sum().item()
            total += B

    return [s / total for s in cossim_sums]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="DA8: Warm-start DDIM evaluation")
    parser.add_argument("--encoder", nargs="*", default=None,
                        help="Encoder(s) to evaluate (default: vit_s16 only for pilot)")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    parser.add_argument("--mode", choices=["steps", "warmstart", "guidance", "pilot"],
                        default="pilot", help="Which sweep to run")
    parser.add_argument("--n-steps", type=int, default=50, help="DDIM steps for warmstart/guidance")
    parser.add_argument("--t-start", type=int, default=None,
                        help="Fixed t_start for guidance sweep")
    args = parser.parse_args()

    encoders = args.encoder or ["vit_s16"]
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    print(f"DA8: Warm-start DDIM evaluation")
    print(f"  encoders: {encoders}")
    print(f"  seeds: {seeds}")
    print(f"  mode: {args.mode}")
    print(f"  device: {device}")
    print()

    all_results = []

    if args.mode == "pilot":
        # Full staged pilot: steps -> warmstart -> evaluate gate -> optional guidance
        for enc in encoders:
            print(f"{'='*60}")
            print(f"Pilot: {enc}")
            print(f"{'='*60}")

            # Sanity: copy baseline
            print(f"\n[sanity] Copy baseline:")
            copy_cs = compute_copy_baseline(enc, seeds[0], device)
            print(f"  CosSim@h1={copy_cs[0]:.4f}")

            # Stage 1: step sweep
            print(f"\n[stage 1] Step count sweep (t_start=noise):")
            step_results = []
            for seed in seeds:
                step_results.extend(run_step_sweep(enc, seed, device))

            # Find best n_steps (by mean CosSim@h1 across seeds)
            from collections import defaultdict
            step_means = defaultdict(list)
            for r in step_results:
                step_means[r["n_ddim_steps"]].append(r["cossim_h1"])
            best_steps = max(step_means, key=lambda k: np.mean(step_means[k]))
            print(f"\n  -> Best n_steps={best_steps} "
                  f"(mean CosSim@h1={np.mean(step_means[best_steps]):.4f})")

            all_results.extend(step_results)

            # Stage 2: warm-start sweep
            print(f"\n[stage 2] Warm-start sweep (n_steps={best_steps}):")
            ws_results = []
            for seed in seeds:
                ws_results.extend(run_warmstart_sweep(enc, seed, best_steps, device))

            # Find best t_start
            ts_means = defaultdict(list)
            for r in ws_results:
                ts_means[r["t_start"]].append(r["cossim_h1"])
            best_t = max(ts_means, key=lambda k: np.mean(ts_means[k]))
            best_cossim = np.mean(ts_means[best_t])
            baseline_cossim = np.mean(ts_means.get(980, ts_means.get("noise", [0.0])))
            print(f"\n  -> Best t_start={best_t} "
                  f"(mean CosSim@h1={best_cossim:.4f}, "
                  f"baseline={baseline_cossim:.4f})")

            all_results.extend(ws_results)

            # Gate GA: compute gap recovery
            # Reference values from DA7.5
            dit_direct_ref = {
                "vit_s16": 0.891, "dino_vits14": 0.888, "clip_b32": 0.941,
                "vjepa2_rep1": 0.983, "vjepa2_rep64": 0.988, "vq_track": 0.993,
            }
            dit_ddim_ref = {
                "vit_s16": 0.504, "dino_vits14": 0.448, "clip_b32": 0.621,
                "vjepa2_rep1": 0.752, "vjepa2_rep64": 0.735, "vq_track": 0.501,
            }

            gap = dit_direct_ref.get(enc, 0.9) - dit_ddim_ref.get(enc, 0.5)
            recovery = (best_cossim - dit_ddim_ref.get(enc, 0.5)) / gap if gap > 0 else 0
            print(f"\n[gate GA] Gap recovery: {recovery:.1%}")
            print(f"  gap = {gap:.3f}, improved = {best_cossim:.4f}, "
                  f"baseline DDIM = {dit_ddim_ref.get(enc, 0.5):.4f}")

            # Stage 3: guidance (only if recovery >= 10%)
            if recovery >= 0.10:
                t_for_guidance = best_t if best_t != 0 else None
                print(f"\n[stage 3] Two-model guidance (n_steps={best_steps}, "
                      f"t_start={t_for_guidance}):")
                print("  NOTE: two independently trained models, not canonical CFG")
                for seed in seeds:
                    guide_results = run_guidance_sweep(
                        enc, seed, best_steps, t_for_guidance, device
                    )
                    all_results.extend(guide_results)
            else:
                print(f"\n[stage 3] Skipped guidance (recovery {recovery:.1%} < 10%)")

    elif args.mode == "steps":
        for enc in encoders:
            print(f"\n[steps] {enc}:")
            for seed in seeds:
                all_results.extend(run_step_sweep(enc, seed, device))

    elif args.mode == "warmstart":
        for enc in encoders:
            print(f"\n[warmstart] {enc}:")
            for seed in seeds:
                all_results.extend(
                    run_warmstart_sweep(enc, seed, args.n_steps, device)
                )

    elif args.mode == "guidance":
        for enc in encoders:
            print(f"\n[guidance] {enc}:")
            for seed in seeds:
                all_results.extend(
                    run_guidance_sweep(enc, seed, args.n_steps, args.t_start, device)
                )

    # Save CSV
    if all_results:
        SWEEP_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(all_results[0].keys())
        with open(SWEEP_CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[saved] {SWEEP_CSV_PATH} ({len(all_results)} rows)")


if __name__ == "__main__":
    main()
