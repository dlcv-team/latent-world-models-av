"""Evaluate Latent DiT via DDIM rollout on test embeddings (DA5+DA7).

Loads trained DiT checkpoints from the Modal volume, generates 4-step
rollouts via 50-step DDIM sampling on the test set, and computes
per-horizon CosSim/MSE metrics against ground-truth future latents.

Combines DA5 (rollout generation) and DA7 (evaluation) into a single
pass to avoid serializing large z_hat tensors.  Returns only small
aggregate metric dicts from each Modal job.

Model architectures are reimplemented inline (same as ``train_dit.py``)
because Modal remote functions run in a minimal container without
project source.  The local entrypoint validates all constants against
``configs/dit.yaml`` before dispatching jobs.

Usage:
  modal run scripts/rollout_dit.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

try:
    import modal
    app = modal.App("lwm-av-dit-rollout")
    vol = modal.Volume.from_name("nuscenes-full")
except ImportError:
    modal = None  # allow importing constants without modal installed
    app = None
    vol = None

VOL_PATH = "/vol"
EMBED_DIR = f"{VOL_PATH}/embeddings"
DIT_DIR = f"{VOL_PATH}/dits"

# ---------------------------------------------------------------------------
# Canonical constants -- MUST mirror configs/dit.yaml and
# configs/canonical.yaml::latent_predictor::fourier_action_embed.
# Validated by _validate_dit_config() in the local entrypoint.
# ---------------------------------------------------------------------------

DIT_CANONICAL = {
    "n_blocks": 4,
    "n_heads": 6,
    "z_dim": 384,
    "horizon": 4,
    "cond_dim": 384,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
}

DIFFUSION_CANONICAL = {
    "n_train_steps": 1000,
}

FOURIER_CANONICAL = {
    "n_frequencies": 64,
    "base": 2.0,
    "out_dim": 384,
}

EVAL_CANONICAL = {
    "n_sample_steps": 50,
    "batch_size": 64,
}

# All 6 encoders, 3 seeds, both variants = 36 jobs.
ENCODER_NAMES = ["vit_s16", "dino_vits14", "clip_b32", "vq_track", "vjepa2_rep64", "vjepa2_rep1"]
SEEDS = [0, 1, 2]
VARIANTS = ["conditioned", "unconditioned"]

NATIVE_DIMS = {
    "vit_s16": 384,
    "dino_vits14": 384,
    "clip_b32": 512,
    "vq_track": 256,
    "vjepa2_rep64": 1024,
    "vjepa2_rep1": 1024,
}

if modal is not None:
    base_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch==2.5.1", "numpy>=1.26", "tqdm")
    )
else:
    base_image = None


# ===================================================================
# Remote function -- runs on Modal GPU
# ===================================================================


def _modal_function_decorator(fn):
    """Apply Modal decorator only when modal is available."""
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
def rollout_eval(encoder_name: str, seed: int, variant: str):
    """Evaluate a single DiT checkpoint via DDIM rollout on test set."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    # ---------------------------------------------------------------
    # Inline model definitions (reimplemented from project modules)
    # ---------------------------------------------------------------

    class CosineNoiseSchedule(nn.Module):
        """Cosine beta schedule (Nichol & Dhariwal, 2021)."""

        def __init__(self, n_steps: int = 1000, s: float = 0.008):
            super().__init__()
            self.n_steps = n_steps
            steps = torch.arange(n_steps + 1, dtype=torch.float64)
            f_t = torch.cos(((steps / n_steps) + s) / (1.0 + s) * (torch.pi / 2.0)) ** 2
            alphas_cumprod = f_t / f_t[0]
            alphas_cumprod = alphas_cumprod[:n_steps].float()
            self.register_buffer("alphas_cumprod", alphas_cumprod)
            self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
            self.register_buffer(
                "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
            )

        def _extract(self, arr, t, x_shape):
            out = arr.gather(0, t.long())
            return out.view(-1, *([1] * (len(x_shape) - 1)))

        def add_noise(self, x_0, t, noise=None):
            if noise is None:
                noise = torch.randn_like(x_0)
            sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
            sqrt_one_minus = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
            )
            return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise

    class DDIMSampler:
        """Deterministic DDIM sampler (Song et al., 2020)."""

        def __init__(self, noise_schedule, n_steps: int = 50):
            self.schedule = noise_schedule
            self.n_steps = n_steps
            T = noise_schedule.n_steps
            stride = T // n_steps
            self.timesteps = list(reversed(list(range(0, T, stride))[:n_steps]))

        @torch.no_grad()
        def sample(self, noise_pred_fn, shape, cond_kwargs, device="cpu"):
            alphas_cumprod = self.schedule.alphas_cumprod.to(device)
            x = torch.randn(shape, device=device)
            for i, t_val in enumerate(self.timesteps):
                t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
                noise_pred = noise_pred_fn(x, timestep=t, **cond_kwargs)
                alpha_bar_t = alphas_cumprod[t_val]
                pred_x0 = (
                    x - torch.sqrt(1.0 - alpha_bar_t) * noise_pred
                ) / torch.sqrt(alpha_bar_t)
                if i < len(self.timesteps) - 1:
                    t_prev = self.timesteps[i + 1]
                    alpha_bar_prev = alphas_cumprod[t_prev]
                else:
                    alpha_bar_prev = torch.tensor(1.0, device=device)
                noise_direction = (
                    x - torch.sqrt(alpha_bar_t) * pred_x0
                ) / torch.sqrt(1.0 - alpha_bar_t + 1e-8)
                x = (
                    torch.sqrt(alpha_bar_prev) * pred_x0
                    + torch.sqrt(1.0 - alpha_bar_prev) * noise_direction
                )
            return x

    class TimestepEmbedding(nn.Module):
        """Sinusoidal timestep -> MLP embedding."""

        def __init__(self, cond_dim: int = 384):
            super().__init__()
            self.cond_dim = cond_dim
            self.mlp = nn.Sequential(
                nn.Linear(cond_dim, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
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
        """Transformer block with adaLN-Zero conditioning."""

        def __init__(self, dim=384, cond_dim=384, n_heads=6, mlp_ratio=4.0, dropout=0.0):
            super().__init__()
            self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False)
            self.attn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True
            )
            self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
            mlp_hidden = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, dim)
            )
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
        """DiT for latent future prediction (epsilon prediction)."""

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

        def forward(self, x_noisy, z_t, a_embed, timestep):
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

    class FourierActionEmbedding(nn.Module):
        """Fourier features for (steer, accel) -> dense embedding."""

        def __init__(self, action_dim=2, n_frequencies=64, base=2.0, out_dim=384):
            super().__init__()
            self.action_dim = action_dim
            self.n_frequencies = n_frequencies
            freqs = base ** torch.arange(n_frequencies, dtype=torch.float32) * torch.pi
            self.register_buffer("freqs", freqs)
            fourier_dim = action_dim * 2 * n_frequencies
            self.proj = nn.Sequential(
                nn.Linear(fourier_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )

        def forward(self, action):
            x = action.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
            x = torch.cat([x.sin(), x.cos()], dim=-1)
            x = x.flatten(1)
            return self.proj(x)

    # ---------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------

    print(f"[rollout] encoder={encoder_name}, seed={seed}, variant={variant}")

    embed_path = f"{EMBED_DIR}/{encoder_name}.npz"
    with np.load(embed_path, allow_pickle=True) as f:
        embeddings = f["embeddings"]
        splits = f["splits"]
        steer_norms = f["steer_norms"]
        accel_norms = f["accel_norms"]
        scene_names = f["scene_names"]

    native_dim = NATIVE_DIMS[encoder_name]
    target_dim = DIT_CANONICAL["z_dim"]
    needs_adapter = native_dim != target_dim
    horizon = DIT_CANONICAL["horizon"]

    # Build test sliding windows
    def build_windows(split_name):
        mask = splits == split_name
        emb = embeddings[mask]
        steers = steer_norms[mask]
        accels = accel_norms[mask]
        scenes = scene_names[mask]

        z_t_list, action_list, z_future_list = [], [], []
        unique_scenes = np.unique(scenes)
        for scene in unique_scenes:
            scene_mask = scenes == scene
            idx = np.where(scene_mask)[0]
            n_scene = len(idx)
            for j in range(n_scene - horizon):
                t_idx = idx[j]
                future_idx = idx[j + 1 : j + 1 + horizon]
                z_t_list.append(emb[t_idx])
                action_list.append([steers[t_idx], accels[t_idx]])
                z_future_list.append(emb[future_idx])

        if not z_t_list:
            return None, None, None
        return (
            torch.tensor(np.array(z_t_list), dtype=torch.float32),
            torch.tensor(np.array(action_list), dtype=torch.float32),
            torch.tensor(np.array(z_future_list), dtype=torch.float32),
        )

    z_t_test, act_test, zf_test = build_windows("test")
    if z_t_test is None:
        return {"error": f"No test windows for {encoder_name}"}

    n_test = len(z_t_test)
    print(f"[rollout] Test: {n_test} windows")

    # ---------------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"{DIT_DIR}/{encoder_name}/{variant}/seed_{seed}/checkpoint.pt"
    print(f"[rollout] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Require normalization stats (DA3.1+ checkpoints)
    if "z_mean" not in ckpt or "z_std" not in ckpt:
        raise RuntimeError(
            f"Checkpoint missing z_mean/z_std -- this is a pre-DA3.1 "
            f"checkpoint. Re-train with normalized latents."
        )

    # Validate checkpoint matches the requested encoder/variant/adapter
    if "encoder_name" in ckpt and ckpt["encoder_name"] != encoder_name:
        raise RuntimeError(
            f"Checkpoint encoder mismatch: requested {encoder_name} "
            f"but checkpoint has {ckpt['encoder_name']}"
        )
    if "variant" in ckpt and ckpt["variant"] != variant:
        raise RuntimeError(
            f"Checkpoint variant mismatch: requested {variant} "
            f"but checkpoint has {ckpt['variant']}"
        )
    if needs_adapter and ckpt.get("adapter_state_dict") is None:
        raise RuntimeError(
            f"Encoder {encoder_name} requires adapter but checkpoint has none"
        )
    if not needs_adapter and ckpt.get("adapter_state_dict") is not None:
        raise RuntimeError(
            f"Encoder {encoder_name} is identity but checkpoint has adapter"
        )

    z_mean = ckpt["z_mean"].to(device)  # (384,)
    z_std = ckpt["z_std"].to(device)    # (384,)
    print(f"[rollout] Loaded normalization: per-elem std mean={z_std.mean():.4f}")

    # Reconstruct adapter (frozen, orthogonal init -- load saved weights)
    if needs_adapter:
        adapter = nn.Linear(native_dim, target_dim, bias=False).to(device)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        for p in adapter.parameters():
            p.requires_grad_(False)
    else:
        adapter = nn.Identity().to(device)

    # Reconstruct FourierActionEmbedding
    fourier_embed = FourierActionEmbedding(
        action_dim=2,
        n_frequencies=FOURIER_CANONICAL["n_frequencies"],
        base=FOURIER_CANONICAL["base"],
        out_dim=FOURIER_CANONICAL["out_dim"],
    ).to(device)
    fourier_embed.load_state_dict(ckpt["fourier_embed_state_dict"])

    # Reconstruct DiT
    dit = LatentDiT(
        z_dim=DIT_CANONICAL["z_dim"],
        cond_dim=DIT_CANONICAL["cond_dim"],
        n_blocks=DIT_CANONICAL["n_blocks"],
        n_heads=DIT_CANONICAL["n_heads"],
        horizon=DIT_CANONICAL["horizon"],
        mlp_ratio=DIT_CANONICAL["mlp_ratio"],
        dropout=DIT_CANONICAL["dropout"],
    ).to(device)

    # Default to main weights; EMA is optional and unproven
    dit.load_state_dict(ckpt["dit_state_dict"])

    dit.eval()
    fourier_embed.eval()

    # Build DDIM sampler
    schedule = CosineNoiseSchedule(
        n_steps=DIFFUSION_CANONICAL["n_train_steps"]
    ).to(device)
    sampler = DDIMSampler(schedule, n_steps=EVAL_CANONICAL["n_sample_steps"])

    print(f"[rollout] Model loaded, DDIM {EVAL_CANONICAL['n_sample_steps']} steps")

    # ---------------------------------------------------------------
    # Diagnostic: reconstruction test at various noise levels
    # ---------------------------------------------------------------

    # Project + normalize (same as training)
    B_diag, H_diag, _ = zf_test[:16].shape
    diag_z_t = (adapter(z_t_test[:16].to(device)) - z_mean) / z_std
    diag_zf = (
        adapter(zf_test[:16].to(device).reshape(B_diag * H_diag, -1))
        .reshape(B_diag, H_diag, target_dim) - z_mean
    ) / z_std
    diag_act = act_test[:16].to(device)
    diag_a_embed = fourier_embed(diag_act)
    if variant == "unconditioned":
        diag_a_embed = torch.zeros_like(diag_a_embed)

    print(f"\n[diag] z_real (norm) L2 norm: {diag_zf.norm(dim=-1).mean():.2f}")
    print(f"[diag] z_real (norm) per-element std: {diag_zf.std():.4f}")

    with torch.no_grad():
        for test_t in [10, 100, 500, 900, 980]:
            t_diag = torch.full((16,), test_t, device=device, dtype=torch.long)
            x_noisy, noise = schedule.add_noise(diag_zf, t_diag)
            noise_pred = dit(x_noisy, diag_z_t, diag_a_embed, t_diag)

            # Noise prediction quality
            noise_mse = ((noise_pred - noise) ** 2).mean().item()

            # Reconstruct x_0
            alpha_bar = schedule.alphas_cumprod[test_t]
            pred_x0 = (
                x_noisy - torch.sqrt(1.0 - alpha_bar) * noise_pred
            ) / torch.sqrt(alpha_bar)

            recon_cossim = F.cosine_similarity(
                pred_x0.reshape(16, -1), diag_zf.reshape(16, -1), dim=-1
            ).mean().item()
            recon_mse = ((pred_x0 - diag_zf) ** 2).mean().item()

            print(
                f"[diag] t={test_t:>3}: noise_MSE={noise_mse:.4f}  "
                f"recon_CosSim={recon_cossim:.4f}  recon_MSE={recon_mse:.2f}  "
                f"pred_x0_norm={pred_x0.norm(dim=-1).mean():.2f}"
            )

    # Also test DDIM generation on this batch
    torch.manual_seed(seed)
    with torch.no_grad():
        z_hat_diag = sampler.sample(
            noise_pred_fn=dit,
            shape=(16, horizon, target_dim),
            cond_kwargs={"z_t": diag_z_t, "a_embed": diag_a_embed},
            device=device,
        )
        gen_cossim = F.cosine_similarity(
            z_hat_diag.reshape(16, -1), diag_zf.reshape(16, -1), dim=-1
        ).mean().item()
        print(
            f"[diag] DDIM gen: CosSim={gen_cossim:.4f}  "
            f"z_hat_norm={z_hat_diag.norm(dim=-1).mean():.2f}  "
            f"z_real_norm={diag_zf.norm(dim=-1).mean():.2f}"
        )
    print()

    # ---------------------------------------------------------------
    # DDIM rollout on test set
    # ---------------------------------------------------------------

    # Seed for reproducible DDIM sampling (initial noise)
    torch.manual_seed(seed)
    np.random.seed(seed)

    test_ds = TensorDataset(z_t_test, act_test, zf_test)
    test_loader = DataLoader(
        test_ds,
        batch_size=EVAL_CANONICAL["batch_size"],
        shuffle=False,
    )

    # Accumulators per horizon
    cossim_sums = [0.0] * horizon
    mse_sums = [0.0] * horizon
    copy_cossim_sums = [0.0] * horizon
    total_samples = 0

    t0 = time.time()

    with torch.no_grad():
        for z_t_batch, act_batch, zf_batch in tqdm(
            test_loader, desc=f"[rollout] {encoder_name}/{variant}/s{seed}"
        ):
            z_t_batch = z_t_batch.to(device)
            act_batch = act_batch.to(device)
            zf_batch = zf_batch.to(device)
            B = z_t_batch.shape[0]

            # Adapter projection + normalize
            B_f, H, _ = zf_batch.shape
            z_t_adapted = (adapter(z_t_batch) - z_mean) / z_std  # (B, 384)
            zf_adapted = (
                adapter(zf_batch.reshape(B_f * H, -1))
                .reshape(B_f, H, target_dim) - z_mean
            ) / z_std  # (B, horizon, 384)

            # Action embedding
            a_embed = fourier_embed(act_batch)  # (B, 384)
            if variant == "unconditioned":
                a_embed = torch.zeros_like(a_embed)

            # DDIM sampling (in normalized space)
            z_hat_norm = sampler.sample(
                noise_pred_fn=dit,
                shape=(B, horizon, target_dim),
                cond_kwargs={"z_t": z_t_adapted, "a_embed": a_embed},
                device=device,
            )  # (B, horizon, 384) in normalized space

            # Inverse-transform for metrics in original adapted space
            z_hat = z_hat_norm * z_std + z_mean
            zf_orig = zf_adapted * z_std + z_mean
            z_t_orig = z_t_adapted * z_std + z_mean

            # Per-horizon metrics (in original adapted space)
            for k in range(horizon):
                z_hat_k = z_hat[:, k]       # (B, 384)
                z_real_k = zf_orig[:, k]     # (B, 384)

                # CosSim(z_hat, z_real)
                cs = F.cosine_similarity(z_hat_k, z_real_k, dim=-1)
                cossim_sums[k] += cs.sum().item()

                # MSE(z_hat, z_real)
                mse = ((z_hat_k - z_real_k) ** 2).mean(dim=-1)
                mse_sums[k] += mse.sum().item()

                # Copy baseline: CosSim(z_t, z_real_k) -- in original space
                copy_cs = F.cosine_similarity(z_t_orig, z_real_k, dim=-1)
                copy_cossim_sums[k] += copy_cs.sum().item()

            total_samples += B

    elapsed = time.time() - t0

    # Compute means
    cossim_by_horizon = [s / total_samples for s in cossim_sums]
    mse_by_horizon = [s / total_samples for s in mse_sums]
    copy_baseline_cossim = [s / total_samples for s in copy_cossim_sums]

    # Print summary
    print(f"\n[rollout] {encoder_name}/{variant}/seed={seed} ({elapsed:.1f}s)")
    print(f"  {'k':>3}  {'CosSim':>8}  {'MSE':>8}  {'CopyBL':>8}")
    for k in range(horizon):
        print(
            f"  k={k+1}:  {cossim_by_horizon[k]:>8.4f}  "
            f"{mse_by_horizon[k]:>8.4f}  {copy_baseline_cossim[k]:>8.4f}"
        )

    return {
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


# ===================================================================
# Local entrypoint
# ===================================================================


def _validate_dit_config():
    """Assert duplicated constants match configs/dit.yaml.

    Runs locally (not on Modal) so we can read the project config.
    """
    import yaml

    dit_yaml = Path(__file__).resolve().parent.parent.parent / "configs" / "dit.yaml"
    with open(dit_yaml) as f:
        raw = yaml.safe_load(f)

    # Validate DiT architecture
    dit_cfg = raw["dit"]
    for key in ["n_blocks", "n_heads", "z_dim", "horizon", "cond_dim", "mlp_ratio", "dropout"]:
        expected = dit_cfg[key]
        actual = DIT_CANONICAL[key]
        if isinstance(expected, float):
            assert abs(actual - expected) < 1e-9, (
                f"DIT_CANONICAL[{key!r}] = {actual} but dit.yaml says {expected}"
            )
        else:
            assert actual == expected, (
                f"DIT_CANONICAL[{key!r}] = {actual} but dit.yaml says {expected}"
            )

    # Validate diffusion
    diff_cfg = raw["diffusion"]
    assert DIFFUSION_CANONICAL["n_train_steps"] == diff_cfg["n_train_steps"], (
        f"n_train_steps mismatch: {DIFFUSION_CANONICAL['n_train_steps']} vs {diff_cfg['n_train_steps']}"
    )

    # Validate n_sample_steps
    assert EVAL_CANONICAL["n_sample_steps"] == diff_cfg["n_sample_steps"], (
        f"n_sample_steps mismatch: {EVAL_CANONICAL['n_sample_steps']} vs {diff_cfg['n_sample_steps']}"
    )

    # Validate Fourier params against canonical.yaml
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from config import load_canonical

    cfg = load_canonical()
    lp_cfg = cfg.latent_predictor()
    fourier_cfg = lp_cfg["fourier_action_embed"]
    for key in ["n_frequencies", "base", "out_dim"]:
        expected = fourier_cfg[key]
        actual = FOURIER_CANONICAL[key]
        if isinstance(expected, float):
            assert abs(actual - expected) < 1e-9, (
                f"FOURIER_CANONICAL[{key!r}] = {actual} but canonical.yaml says {expected}"
            )
        else:
            assert int(actual) == int(expected), (
                f"FOURIER_CANONICAL[{key!r}] = {actual} but canonical.yaml says {expected}"
            )

    print("[validate] All constants match configs/dit.yaml + canonical.yaml")


def _build_summary(all_results):
    """Build per-encoder seed-averaged summary from raw results."""
    from collections import defaultdict

    grouped = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if "error" in r:
            continue
        key = r["encoder"]
        var = r["variant"]
        grouped[key][var].append(r["metrics"])

    summary = {}
    for enc, variants in sorted(grouped.items()):
        summary[enc] = {}
        for var, metrics_list in sorted(variants.items()):
            n_seeds = len(metrics_list)
            horizon = len(metrics_list[0]["cossim_by_horizon"])

            cossim_means = []
            cossim_stds = []
            mse_means = []
            mse_stds = []
            copy_means = []

            for k in range(horizon):
                vals = [m["cossim_by_horizon"][k] for m in metrics_list]
                cossim_means.append(sum(vals) / n_seeds)
                if n_seeds > 1:
                    var_val = sum((v - cossim_means[-1]) ** 2 for v in vals) / (n_seeds - 1)
                    cossim_stds.append(var_val ** 0.5)
                else:
                    cossim_stds.append(0.0)

                vals = [m["mse_by_horizon"][k] for m in metrics_list]
                mse_means.append(sum(vals) / n_seeds)
                if n_seeds > 1:
                    var_val = sum((v - mse_means[-1]) ** 2 for v in vals) / (n_seeds - 1)
                    mse_stds.append(var_val ** 0.5)
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


def _modal_entrypoint_decorator(fn):
    """Apply Modal local_entrypoint decorator only when modal is available."""
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_modal_entrypoint_decorator
def main(variants: str = ""):
    """Run DDIM rollout evaluation for all encoder x seed x variant.

    Args:
        variants: Comma-separated list of variants to evaluate
                  (e.g. "conditioned" or "conditioned,unconditioned").
                  Empty string means all variants.
    """
    _validate_dit_config()

    if variants:
        run_variants = [v.strip() for v in variants.split(",")]
        for v in run_variants:
            if v not in VARIANTS:
                raise ValueError(f"Unknown variant {v!r}, choose from {VARIANTS}")
    else:
        run_variants = list(VARIANTS)

    t_start = time.time()
    print("=" * 60)
    print("DiT Rollout Evaluation (DDIM Sampling)")
    print(f"  encoders:  {ENCODER_NAMES}")
    print(f"  seeds:     {SEEDS}")
    print(f"  variants:  {run_variants}")
    n_jobs = len(ENCODER_NAMES) * len(SEEDS) * len(run_variants)
    print(f"  total jobs: {n_jobs}")
    print(f"  DDIM steps: {EVAL_CANONICAL['n_sample_steps']}")
    print(f"  batch size: {EVAL_CANONICAL['batch_size']}")
    print("=" * 60)

    # Launch all jobs in parallel
    futures = []
    for enc_name in ENCODER_NAMES:
        for seed in SEEDS:
            for variant in run_variants:
                print(f"  Launching {enc_name}/{variant}/seed={seed} ...")
                futures.append(
                    (enc_name, seed, variant, rollout_eval.spawn(enc_name, seed, variant))
                )

    # Collect results
    all_results = []
    for enc_name, seed, variant, future in futures:
        print(f"  Waiting for {enc_name}/{variant}/seed={seed} ...")
        result = future.get()
        all_results.append(result)
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            m = result["metrics"]
            print(
                f"  {enc_name}/{variant}/seed={seed}: "
                f"CosSim=[{', '.join(f'{v:.4f}' for v in m['cossim_by_horizon'])}] "
                f"({result['time_s']}s)"
            )

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Encoder':<16} {'Variant':<14} {'Seed':>4} "
          f"{'CS_k1':>7} {'CS_k2':>7} {'CS_k3':>7} {'CS_k4':>7} {'Time':>6}")
    print("-" * 80)
    for r in sorted(all_results, key=lambda x: (x.get("encoder", ""), x.get("variant", ""), x.get("seed", 0))):
        if "error" in r:
            continue
        cs = r["metrics"]["cossim_by_horizon"]
        print(
            f"{r['encoder']:<16} {r['variant']:<14} {r['seed']:>4} "
            f"{cs[0]:>7.4f} {cs[1]:>7.4f} {cs[2]:>7.4f} {cs[3]:>7.4f} "
            f"{r['time_s']:>5.0f}s"
        )

    # Copy baseline
    print(f"\n{'Copy Baseline (CosSim(z_t, z_real_k))':}")
    print(f"{'Encoder':<16} {'Variant':<14} "
          f"{'CB_k1':>7} {'CB_k2':>7} {'CB_k3':>7} {'CB_k4':>7}")
    print("-" * 65)
    for r in sorted(all_results, key=lambda x: (x.get("encoder", ""), x.get("variant", ""), x.get("seed", 0))):
        if "error" in r:
            continue
        cb = r["metrics"]["copy_baseline_cossim"]
        print(
            f"{r['encoder']:<16} {r['variant']:<14} "
            f"{cb[0]:>7.4f} {cb[1]:>7.4f} {cb[2]:>7.4f} {cb[3]:>7.4f}"
        )

    wall_time = time.time() - t_start
    print(f"\nTotal wall time: {wall_time:.0f}s ({wall_time / 60:.1f}min)")

    # Build summary and save -- merge with existing if partial run
    summary_path = "artifacts/full/rollout_results.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    merged_results = list(all_results)

    if set(run_variants) != set(VARIANTS) and os.path.exists(summary_path):
        # Partial run: merge new results into existing JSON
        print(f"\n[merge] Partial variants={run_variants}, merging with existing {summary_path}")
        with open(summary_path) as f:
            existing = json.load(f)
        existing_results = existing.get("results", [])

        # Build key set from new results for replacement
        new_keys = {
            (r["encoder"], r["variant"], r["seed"])
            for r in all_results if "error" not in r
        }

        # Keep existing rows that are not being replaced
        for r in existing_results:
            key = (r.get("encoder"), r.get("variant"), r.get("seed"))
            if key not in new_keys:
                merged_results.append(r)
        print(f"[merge] {len(all_results)} new + {len(merged_results) - len(all_results)} kept = {len(merged_results)} total")

    summary = _build_summary(merged_results)

    output = {
        "results": merged_results,
        "summary": summary,
        "config": {
            "dit": DIT_CANONICAL,
            "diffusion": DIFFUSION_CANONICAL,
            "fourier": FOURIER_CANONICAL,
            "eval": EVAL_CANONICAL,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(summary_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {summary_path}")
