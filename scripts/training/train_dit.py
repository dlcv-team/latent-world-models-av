"""Train Latent DiT on pre-computed embeddings via Modal.

Reads pre-computed embeddings from the Modal volume, builds temporal
sliding windows, and trains a LatentDiT (DA1) with CosineNoiseSchedule
(DA2) using the epsilon-prediction diffusion objective.

Supports both **conditioned** (real action embeddings via Fourier
projection) and **unconditioned** (zeroed action embeddings) variants.

The model architectures (LatentDiT, CosineNoiseSchedule,
FourierActionEmbedding) are reimplemented inline rather than imported,
because Modal remote functions run in a minimal container image without
the project's source tree.  The local entrypoint validates all
duplicated constants against ``configs/dit.yaml`` before dispatching
jobs, so any drift is caught immediately.

Usage:
  modal run scripts/train_dit.py
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
except ImportError:
    modal = None  # allow importing constants without modal installed

if modal is not None:
    app = modal.App("lwm-av-dit")
    vol = modal.Volume.from_name("nuscenes-full")
else:
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

TRAINING_CANONICAL = {
    "epochs": 100,
    "lr": 1e-4,
    "batch_size": 256,
    "ema_decay": 0.999,
    "gradient_clip": 1.0,
    "seed": 0,
    "normalize_latents": True,
    "adapter_frozen": True,
}

FOURIER_CANONICAL = {
    "n_frequencies": 64,
    "base": 2.0,
    "out_dim": 384,
}

# Phase 2: full matrix -- all encoders, 3 seeds, both variants.
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
            timeout=7200,
            memory=16384,
        )(fn)
    return fn


@_modal_function_decorator
def train_dit(encoder_name: str, seed: int, variant: str):
    """Train a single DiT: one encoder, one seed, one variant."""
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

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
            # action: (B, 2)
            # Expand: (B, 2, 1) * (1, 1, n_freq) -> (B, 2, n_freq)
            x = action.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
            x = torch.cat([x.sin(), x.cos()], dim=-1)  # (B, 2, 2*n_freq)
            x = x.flatten(1)  # (B, 2 * 2 * n_freq)
            return self.proj(x)

    class EMAWeights:
        """Lightweight EMA tracker for model parameters."""

        def __init__(self, model, decay=0.9999):
            self.decay = decay
            self.shadow = {n: p.data.clone() for n, p in model.named_parameters()}

        def update(self, model):
            with torch.no_grad():
                for n, p in model.named_parameters():
                    self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

        def state_dict(self):
            return {k: v.cpu() for k, v in self.shadow.items()}

        def apply_to(self, model):
            for n, p in model.named_parameters():
                p.data.copy_(self.shadow[n])

    # ---------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------

    print(f"[dit] encoder={encoder_name}, seed={seed}, variant={variant}")

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

    # Build temporal sliding windows per split
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

    z_t_train, act_train, zf_train = build_windows("train")
    z_t_val, act_val, zf_val = build_windows("val")

    print(f"[dit] Train: {len(z_t_train)} windows, Val: {len(z_t_val)} windows")
    print(f"[dit] native_dim={native_dim}, adapter={needs_adapter}")

    # ---------------------------------------------------------------
    # Model construction
    # ---------------------------------------------------------------

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if needs_adapter:
        adapter = nn.Linear(native_dim, target_dim, bias=False).to(device)
        nn.init.orthogonal_(adapter.weight)
        # Freeze adapter so normalization stats remain valid
        for p in adapter.parameters():
            p.requires_grad_(False)
    else:
        adapter = nn.Identity().to(device)

    # Compute normalization stats from ALL training tokens after frozen adapter
    with torch.no_grad():
        z_t_proj = adapter(z_t_train.to(device))                       # (N, 384)
        B_tr, H_tr, _ = zf_train.shape
        zf_proj = adapter(
            zf_train.reshape(-1, zf_train.shape[-1]).to(device)
        )                                                               # (N*H, 384)
        all_proj = torch.cat([z_t_proj, zf_proj], dim=0)
        z_mean = all_proj.mean(dim=0)                                   # (384,)
        z_std = all_proj.std(dim=0).clamp(min=1e-6)                     # (384,)
        del z_t_proj, zf_proj, all_proj

    print(
        f"[dit] Normalization: per-elem std mean={z_std.mean():.4f}, "
        f"range=[{z_std.min():.4f}, {z_std.max():.4f}], "
        f"per-elem mean norm={z_mean.norm():.4f}"
    )

    fourier_embed = FourierActionEmbedding(
        action_dim=2,
        n_frequencies=FOURIER_CANONICAL["n_frequencies"],
        base=FOURIER_CANONICAL["base"],
        out_dim=FOURIER_CANONICAL["out_dim"],
    ).to(device)

    dit = LatentDiT(
        z_dim=DIT_CANONICAL["z_dim"],
        cond_dim=DIT_CANONICAL["cond_dim"],
        n_blocks=DIT_CANONICAL["n_blocks"],
        n_heads=DIT_CANONICAL["n_heads"],
        horizon=DIT_CANONICAL["horizon"],
        mlp_ratio=DIT_CANONICAL["mlp_ratio"],
        dropout=DIT_CANONICAL["dropout"],
    ).to(device)

    schedule = CosineNoiseSchedule(
        n_steps=DIFFUSION_CANONICAL["n_train_steps"]
    ).to(device)

    # EMA tracks DiT + Fourier (adapter is frozen)
    class _TrainableGroup(nn.Module):
        def __init__(self, dit, fourier):
            super().__init__()
            self.dit = dit
            self.fourier = fourier

    trainable_group = _TrainableGroup(dit, fourier_embed)
    ema = EMAWeights(trainable_group, decay=TRAINING_CANONICAL["ema_decay"])

    # Optimizer over DiT + Fourier only (adapter frozen)
    params = list(dit.parameters()) + list(fourier_embed.parameters())
    optimizer = torch.optim.Adam(params, lr=TRAINING_CANONICAL["lr"])
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in dit.parameters())
    print(f"[dit] DiT params: {n_params:,}")

    # DataLoaders
    train_ds = TensorDataset(z_t_train, act_train, zf_train)
    val_ds = TensorDataset(z_t_val, act_val, zf_val)
    train_loader = DataLoader(
        train_ds, batch_size=TRAINING_CANONICAL["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=TRAINING_CANONICAL["batch_size"], shuffle=False
    )

    T = DIFFUSION_CANONICAL["n_train_steps"]

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------

    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()

    for epoch in range(TRAINING_CANONICAL["epochs"]):
        # --- Train ---
        dit.train()
        fourier_embed.train()

        train_loss_sum = 0.0
        train_n = 0

        for z_t_batch, act_batch, zf_batch in train_loader:
            z_t_batch = z_t_batch.to(device)
            act_batch = act_batch.to(device)
            zf_batch = zf_batch.to(device)

            # Adapt + normalize embeddings
            B, H, _ = zf_batch.shape
            z_t_adapted = (adapter(z_t_batch) - z_mean) / z_std
            zf_adapted = (
                adapter(zf_batch.reshape(B * H, -1)).reshape(B, H, target_dim)
                - z_mean
            ) / z_std

            # Action embedding
            a_embed = fourier_embed(act_batch)                  # (B, 384)
            if variant == "unconditioned":
                a_embed = torch.zeros_like(a_embed)

            # Diffusion forward process
            t = torch.randint(0, T, (B,), device=device)
            x_noisy, noise = schedule.add_noise(zf_adapted, t)  # (B, H, 384)

            # Predict noise
            noise_pred = dit(x_noisy, z_t_adapted, a_embed, t)

            loss = criterion(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                params, TRAINING_CANONICAL["gradient_clip"]
            )

            optimizer.step()
            ema.update(trainable_group)

            train_loss_sum += loss.item() * B
            train_n += B

        # --- Validate ---
        dit.eval()
        fourier_embed.eval()

        val_loss_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for z_t_batch, act_batch, zf_batch in val_loader:
                z_t_batch = z_t_batch.to(device)
                act_batch = act_batch.to(device)
                zf_batch = zf_batch.to(device)

                B, H, _ = zf_batch.shape
                z_t_adapted = (adapter(z_t_batch) - z_mean) / z_std
                zf_adapted = (
                    adapter(zf_batch.reshape(B * H, -1)).reshape(B, H, target_dim)
                    - z_mean
                ) / z_std

                a_embed = fourier_embed(act_batch)
                if variant == "unconditioned":
                    a_embed = torch.zeros_like(a_embed)

                t = torch.randint(0, T, (B,), device=device)
                x_noisy, noise = schedule.add_noise(zf_adapted, t)
                noise_pred = dit(x_noisy, z_t_adapted, a_embed, t)
                loss = criterion(noise_pred, noise)

                val_loss_sum += loss.item() * B
                val_n += B

        train_loss = train_loss_sum / train_n
        val_loss = val_loss_sum / val_n
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"[dit] Epoch {epoch + 1}/{TRAINING_CANONICAL['epochs']}: "
                f"train={train_loss:.6f} val={val_loss:.6f} ({elapsed:.0f}s)"
            )

    elapsed = time.time() - t0
    print(
        f"[dit] {encoder_name}/{variant}/seed={seed}: "
        f"final_train={history['train_loss'][-1]:.6f} "
        f"final_val={history['val_loss'][-1]:.6f} time={elapsed:.1f}s"
    )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    out_dir = f"{DIT_DIR}/{encoder_name}/{variant}/seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)

    # Checkpoint: DiT + EMA + Fourier + adapter + normalization
    checkpoint = {
        "dit_state_dict": dit.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "fourier_embed_state_dict": fourier_embed.state_dict(),
        "adapter_state_dict": adapter.state_dict() if needs_adapter else None,
        "encoder_name": encoder_name,
        "variant": variant,
        "seed": seed,
        "epochs": TRAINING_CANONICAL["epochs"],
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "z_mean": z_mean.cpu(),
        "z_std": z_std.cpu(),
        "normalize_latents": True,
        "adapter_frozen": True,
        "adapter_init": "orthogonal",
    }
    torch.save(checkpoint, f"{out_dir}/checkpoint.pt")

    # Training log
    with open(f"{out_dir}/train_log.json", "w") as f:
        json.dump(history, f)

    # Provenance
    provenance = {
        "encoder_name": encoder_name,
        "variant": variant,
        "seed": seed,
        "native_dim": native_dim,
        "target_dim": target_dim,
        "needs_adapter": needs_adapter,
        "dit": DIT_CANONICAL,
        "diffusion": DIFFUSION_CANONICAL,
        "training": TRAINING_CANONICAL,
        "fourier": FOURIER_CANONICAL,
        "n_train_windows": int(len(z_t_train)),
        "n_val_windows": int(len(z_t_val)),
        "dit_params": n_params,
        "time_s": elapsed,
        "source": "scripts/train_dit.py",
    }
    with open(f"{out_dir}/provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    vol.commit()

    return {
        "encoder": encoder_name,
        "variant": variant,
        "seed": seed,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "time_s": elapsed,
    }


# ===================================================================
# Local entrypoint
# ===================================================================


def _validate_dit_config():
    """Assert duplicated constants match configs/dit.yaml.

    Runs locally (not on Modal) so we can read the project config.
    """
    import yaml

    dit_yaml = Path(__file__).resolve().parent.parent / "configs" / "dit.yaml"
    with open(dit_yaml) as f:
        raw = yaml.safe_load(f)

    # Validate DiT architecture
    dit_cfg = raw["dit"]
    for key in ["n_blocks", "n_heads", "z_dim", "horizon", "cond_dim", "mlp_ratio", "dropout"]:
        expected = dit_cfg[key]
        actual = DIT_CANONICAL[key]
        # Compare with type coercion for int/float
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

    # Validate training
    train_cfg = raw["training"]
    for key in ["epochs", "lr", "batch_size", "ema_decay", "gradient_clip",
                "normalize_latents", "adapter_frozen"]:
        expected = train_cfg[key]
        actual = TRAINING_CANONICAL[key]
        if isinstance(expected, float):
            assert abs(actual - expected) < 1e-9, (
                f"TRAINING_CANONICAL[{key!r}] = {actual} but dit.yaml says {expected}"
            )
        else:
            assert actual == expected, (
                f"TRAINING_CANONICAL[{key!r}] = {actual} but dit.yaml says {expected}"
            )

    # Validate Fourier params against canonical.yaml
    project_root = str(Path(__file__).resolve().parent.parent)
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


def _modal_entrypoint_decorator(fn):
    """Apply Modal local_entrypoint decorator only when modal is available."""
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_modal_entrypoint_decorator
def main(variants: str = ""):
    """Train DiT for all encoder x seed x variant combinations.

    Args:
        variants: Comma-separated list of variants to train
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
    print("DiT Training (Latent Diffusion Transformer)")
    print(f"  encoders:  {ENCODER_NAMES}")
    print(f"  seeds:     {SEEDS}")
    print(f"  variants:  {run_variants}")
    n_jobs = len(ENCODER_NAMES) * len(SEEDS) * len(run_variants)
    print(f"  total jobs: {n_jobs}")
    print("=" * 60)

    # Launch all jobs in parallel
    futures = []
    for enc_name in ENCODER_NAMES:
        for seed in SEEDS:
            for variant in run_variants:
                print(f"  Launching {enc_name}/{variant}/seed={seed} ...")
                futures.append(
                    (enc_name, seed, variant, train_dit.spawn(enc_name, seed, variant))
                )

    # Collect results
    all_results = []
    for enc_name, seed, variant, future in futures:
        print(f"  Waiting for {enc_name}/{variant}/seed={seed} ...")
        result = future.get()
        all_results.append(result)
        print(
            f"  {enc_name}/{variant}/seed={seed}: "
            f"train={result['final_train_loss']:.6f} val={result['final_val_loss']:.6f}"
        )

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Encoder':<16} {'Variant':<14} {'Seed':>4} {'Train Loss':>11} {'Val Loss':>10} {'Time':>6}")
    print("-" * 70)
    for r in sorted(all_results, key=lambda x: (x["encoder"], x["variant"], x["seed"])):
        print(
            f"{r['encoder']:<16} {r['variant']:<14} {r['seed']:>4} "
            f"{r['final_train_loss']:>11.6f} {r['final_val_loss']:>10.6f} "
            f"{r['time_s']:>5.0f}s"
        )

    wall_time = time.time() - t_start
    print(f"\nTotal wall time: {wall_time:.0f}s ({wall_time / 60:.1f}min)")

    # Save aggregate results locally
    summary_path = "artifacts/full/dit_results.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {summary_path}")
