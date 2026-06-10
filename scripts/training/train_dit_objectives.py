"""Train Latent DiT with alternative objectives via Modal (DA8 Tier B).

Fork of ``train_dit.py`` that supports three prediction objectives
and an optional residual formulation:

**Objectives (B1):**

- ``epsilon`` -- predict the noise that was added (original)
- ``x0`` -- predict the clean target directly
- ``v`` -- predict the velocity v = sqrt(alpha)*eps - sqrt(1-alpha)*x0
  (Salimans & Ho, 2022)

**Residual formulation (B2):**

When ``residual=True``, the model diffuses and denoises
``delta_z = z_future - z_t`` (in normalized space) instead of
absolute ``z_future``. This centers the target around zero and
builds the copy baseline into the math.

Usage::

    modal run scripts/train_dit_objectives.py
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
    modal = None

if modal is not None:
    app = modal.App("lwm-av-dit-objectives")
    vol = modal.Volume.from_name("nuscenes-full")
else:
    app = None
    vol = None

VOL_PATH = "/vol"
EMBED_DIR = f"{VOL_PATH}/embeddings"
DIT_DIR = f"{VOL_PATH}/dits"

# ---------------------------------------------------------------------------
# Canonical constants (same as train_dit.py)
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

DIFFUSION_CANONICAL = {"n_train_steps": 1000}

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

NATIVE_DIMS = {
    "vit_s16": 384,
    "dino_vits14": 384,
    "clip_b32": 512,
    "vq_track": 256,
    "vjepa2_rep64": 1024,
    "vjepa2_rep1": 1024,
}

# Full expansion: all encoders, 3 seeds, conditioned
ENCODER_NAMES = sorted(NATIVE_DIMS.keys())
SEEDS = [0, 1, 2]
VARIANT = "conditioned"

PREDICTION_TYPES = ["epsilon", "x0", "v"]

if modal is not None:
    base_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch==2.5.1", "numpy>=1.26", "tqdm")
    )
else:
    base_image = None


# ===================================================================
# Remote function
# ===================================================================


def _modal_function_decorator(fn):
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
def train_dit_objective(
    encoder_name: str,
    seed: int,
    variant: str,
    prediction: str = "epsilon",
    residual: bool = False,
):
    """Train a single DiT with specified objective and formulation."""
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # ---------------------------------------------------------------
    # Inline model definitions (identical to train_dit.py)
    # ---------------------------------------------------------------

    class CosineNoiseSchedule(nn.Module):
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

    class EMAWeights:
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
    # Data loading (same as train_dit.py)
    # ---------------------------------------------------------------

    tag = prediction
    if residual:
        tag = "residual"
    print(
        f"[dit-obj] encoder={encoder_name}, seed={seed}, "
        f"variant={variant}, prediction={prediction}, residual={residual}"
    )

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

    print(f"[dit-obj] Train: {len(z_t_train)} windows, Val: {len(z_t_val)} windows")

    # ---------------------------------------------------------------
    # Model construction (same as train_dit.py)
    # ---------------------------------------------------------------

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if needs_adapter:
        adapter = nn.Linear(native_dim, target_dim, bias=False).to(device)
        nn.init.orthogonal_(adapter.weight)
        for p in adapter.parameters():
            p.requires_grad_(False)
    else:
        adapter = nn.Identity().to(device)

    with torch.no_grad():
        z_t_proj = adapter(z_t_train.to(device))
        B_tr, H_tr, _ = zf_train.shape
        zf_proj = adapter(
            zf_train.reshape(-1, zf_train.shape[-1]).to(device)
        )
        all_proj = torch.cat([z_t_proj, zf_proj], dim=0)
        z_mean = all_proj.mean(dim=0)
        z_std = all_proj.std(dim=0).clamp(min=1e-6)
        del z_t_proj, zf_proj, all_proj

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

    class _TrainableGroup(nn.Module):
        def __init__(self, dit, fourier):
            super().__init__()
            self.dit = dit
            self.fourier = fourier

    trainable_group = _TrainableGroup(dit, fourier_embed)
    ema = EMAWeights(trainable_group, decay=TRAINING_CANONICAL["ema_decay"])

    params = list(dit.parameters()) + list(fourier_embed.parameters())
    optimizer = torch.optim.Adam(params, lr=TRAINING_CANONICAL["lr"])
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in dit.parameters())
    print(f"[dit-obj] DiT params: {n_params:,}")

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
    # Training loop -- with objective/residual variants
    # ---------------------------------------------------------------

    def compute_loss(dit_out, zf_target, noise, t, schedule):
        """Compute loss based on prediction type."""
        if prediction == "epsilon":
            return criterion(dit_out, noise)
        elif prediction == "x0":
            return criterion(dit_out, zf_target)
        elif prediction == "v":
            # v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x_0
            sqrt_alpha = schedule._extract(
                schedule.sqrt_alphas_cumprod, t, zf_target.shape
            )
            sqrt_one_minus = schedule._extract(
                schedule.sqrt_one_minus_alphas_cumprod, t, zf_target.shape
            )
            v_target = sqrt_alpha * noise - sqrt_one_minus * zf_target
            return criterion(dit_out, v_target)
        else:
            raise ValueError(f"Unknown prediction type: {prediction}")

    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()

    for epoch in range(TRAINING_CANONICAL["epochs"]):
        dit.train()
        fourier_embed.train()

        train_loss_sum = 0.0
        train_n = 0

        for z_t_batch, act_batch, zf_batch in train_loader:
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

            # Determine diffusion target
            if residual:
                # B2: diffuse the residual (in normalized space)
                z_t_expanded = z_t_adapted.unsqueeze(1).expand(-1, H, -1)
                diffusion_target = zf_adapted - z_t_expanded  # (B, H, 384)
            else:
                diffusion_target = zf_adapted

            t = torch.randint(0, T, (B,), device=device)
            x_noisy, noise = schedule.add_noise(diffusion_target, t)

            dit_out = dit(x_noisy, z_t_adapted, a_embed, t)
            loss = compute_loss(dit_out, diffusion_target, noise, t, schedule)

            optimizer.zero_grad()
            loss.backward()
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

                if residual:
                    z_t_expanded = z_t_adapted.unsqueeze(1).expand(-1, H, -1)
                    diffusion_target = zf_adapted - z_t_expanded
                else:
                    diffusion_target = zf_adapted

                t = torch.randint(0, T, (B,), device=device)
                x_noisy, noise = schedule.add_noise(diffusion_target, t)
                dit_out = dit(x_noisy, z_t_adapted, a_embed, t)
                loss = compute_loss(dit_out, diffusion_target, noise, t, schedule)

                val_loss_sum += loss.item() * B
                val_n += B

        train_loss = train_loss_sum / train_n
        val_loss = val_loss_sum / val_n
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"[dit-obj] Epoch {epoch + 1}/{TRAINING_CANONICAL['epochs']}: "
                f"train={train_loss:.6f} val={val_loss:.6f} ({elapsed:.0f}s)"
            )

    elapsed = time.time() - t0
    print(
        f"[dit-obj] {encoder_name}/{variant}/{tag}/seed={seed}: "
        f"final_train={history['train_loss'][-1]:.6f} "
        f"final_val={history['val_loss'][-1]:.6f} time={elapsed:.1f}s"
    )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    out_dir = f"{DIT_DIR}/{encoder_name}/{variant}__{tag}/seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)

    checkpoint = {
        "dit_state_dict": dit.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "fourier_embed_state_dict": fourier_embed.state_dict(),
        "adapter_state_dict": adapter.state_dict() if needs_adapter else None,
        "encoder_name": encoder_name,
        "variant": variant,
        "seed": seed,
        "prediction": prediction,
        "residual": residual,
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

    with open(f"{out_dir}/train_log.json", "w") as f:
        json.dump(history, f)

    provenance = {
        "encoder_name": encoder_name,
        "variant": variant,
        "seed": seed,
        "prediction": prediction,
        "residual": residual,
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
        "source": "scripts/train_dit_objectives.py",
    }
    with open(f"{out_dir}/provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    vol.commit()

    return {
        "encoder": encoder_name,
        "variant": variant,
        "seed": seed,
        "prediction": prediction,
        "residual": residual,
        "tag": tag,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "time_s": elapsed,
    }


# ===================================================================
# Local entrypoint
# ===================================================================


def _validate_dit_config():
    """Assert duplicated constants match configs/dit.yaml."""
    import yaml

    dit_yaml = Path(__file__).resolve().parent.parent / "configs" / "dit.yaml"
    with open(dit_yaml) as f:
        raw = yaml.safe_load(f)

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

    diff_cfg = raw["diffusion"]
    assert DIFFUSION_CANONICAL["n_train_steps"] == diff_cfg["n_train_steps"]

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

    print("[validate] All constants match configs/dit.yaml")


def _modal_entrypoint_decorator(fn):
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_modal_entrypoint_decorator
def main():
    """Train DiT with x0-prediction across all encoders and seeds.

    Full expansion after pilot showed x0-prediction recovers 88.5% of
    the epsilon-prediction gap (GB1 passed).  v-prediction also passed
    (80.6%) but x0 dominates, so we only expand x0.
    Residual (B2) failed the gate (6.7%), excluded.
    """
    _validate_dit_config()

    t_start = time.time()
    print("=" * 60)
    print("DA8 Tier B Full: x0-prediction DiT (all encoders x 3 seeds)")
    print(f"  encoders: {ENCODER_NAMES}")
    print(f"  seeds:    {SEEDS}")
    print(f"  variant:  {VARIANT}")
    n_jobs = len(ENCODER_NAMES) * len(SEEDS)
    print(f"  jobs: {n_jobs} (x0-prediction only)")
    print("=" * 60)

    # Only x0-prediction (gate winner)
    configs = [
        # (prediction, residual)
        ("x0", False),
    ]

    futures = []
    for encoder_name in ENCODER_NAMES:
        for seed in SEEDS:
            for pred, resid in configs:
                tag = "residual" if resid else pred
                print(f"  Launching {encoder_name}/{VARIANT}__{tag}/seed={seed} ...")
                futures.append(
                    (
                        encoder_name, seed, tag,
                        train_dit_objective.spawn(
                            encoder_name, seed, VARIANT, pred, resid
                        ),
                    )
                )

    all_results = []
    for enc_name, seed, tag, future in futures:
        print(f"  Waiting for {enc_name}/{VARIANT}__{tag}/seed={seed} ...")
        result = future.get()
        all_results.append(result)
        print(
            f"  {enc_name}/{tag}/seed={seed}: "
            f"train={result['final_train_loss']:.6f} "
            f"val={result['final_val_loss']:.6f}"
        )

    # Summary
    print("\n" + "=" * 80)
    print(
        f"{'Encoder':<14} {'Tag':<12} {'Seed':>4} "
        f"{'Train Loss':>11} {'Val Loss':>10} {'Time':>6}"
    )
    print("-" * 80)
    for r in sorted(all_results, key=lambda x: (x["encoder"], x["tag"], x["seed"])):
        print(
            f"{r['encoder']:<14} {r['tag']:<12} {r['seed']:>4} "
            f"{r['final_train_loss']:>11.6f} {r['final_val_loss']:>10.6f} "
            f"{r['time_s']:>5.0f}s"
        )

    wall_time = time.time() - t_start
    print(f"\nTotal wall time: {wall_time:.0f}s ({wall_time / 60:.1f}min)")

    summary_path = "artifacts/full/dit_objectives_results.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {summary_path}")
