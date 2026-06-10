"""Train Direct-DiT predictors on Modal (DA7.5).

Same DiT transformer backbone (4 blocks, 6 heads, adaLN-Zero, 384-d)
as the diffusion DiT, but trained as a **direct deterministic regressor**:
no noise schedule, no timestep embedding, no DDIM sampling.

Architecture differences from DiT-DDIM:
  - No CosineNoiseSchedule, no TimestepEmbedding
  - Input tokens: z_t repeated H times + learned 1D positional embeddings
  - Conditioning: z_t_proj(z_t) + a_embed (no timestep component)
  - Output: direct prediction of z_future in normalized space
  - Loss: MSE(pred, zf_norm)
  - Single forward pass at inference

This creates a clean ablation:
  | Model      | Architecture   | Objective    | Inference       |
  |------------|----------------|--------------|-----------------|
  | MLP        | 3-layer MLP    | MSE on z     | 1 forward pass  |
  | DiT-direct | 4-block DiT    | MSE on z     | 1 forward pass  |
  | DiT-DDIM   | 4-block DiT    | MSE on noise | 50-step DDIM    |

Uses the same frozen orthogonal adapter + per-element normalization
as DiT-DDIM and MLP.

Usage:
  modal run scripts/train_dit_direct.py                          # pilot
  modal run scripts/train_dit_direct.py --full                   # all 18 jobs
  modal run scripts/train_dit_direct.py --sanity-uncond          # +1 uncond
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
    app = modal.App("lwm-av-dit-direct")
    vol = modal.Volume.from_name("nuscenes-full")
else:
    app = None
    vol = None

VOL_PATH = "/vol"
EMBED_DIR = f"{VOL_PATH}/embeddings"
OUT_DIR = f"{VOL_PATH}/dit_direct"

# ---------------------------------------------------------------------------
# Canonical constants (same architecture as DiT-DDIM)
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

FOURIER_CANONICAL = {
    "n_frequencies": 64,
    "base": 2.0,
    "out_dim": 384,
}

TRAINING_CANONICAL = {
    "epochs": 50,
    "batch_size": 128,
    "gradient_clip": 1.0,
    "normalize_latents": True,
    "adapter_frozen": True,
}

PILOT_LRS = [1e-4, 3e-4, 1e-3]

ENCODER_NAMES = [
    "vit_s16", "dino_vits14", "clip_b32",
    "vq_track", "vjepa2_rep64", "vjepa2_rep1",
]
SEEDS = [0, 1, 2]

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
def train_dit_direct(encoder_name: str, seed: int, variant: str, lr: float):
    """Train a single Direct-DiT: one encoder, one seed, one variant, one LR."""
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # ---------------------------------------------------------------
    # Inline model definitions
    # ---------------------------------------------------------------

    def _modulate(x, shift, scale):
        return x * (1.0 + scale) + shift

    class DiTBlock(nn.Module):
        """Transformer block with adaLN-Zero conditioning."""

        def __init__(self, dim=384, cond_dim=384, n_heads=6,
                     mlp_ratio=4.0, dropout=0.0):
            super().__init__()
            self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False)
            self.attn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=n_heads,
                dropout=dropout, batch_first=True,
            )
            self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
            mlp_hidden = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden), nn.GELU(),
                nn.Linear(mlp_hidden, dim),
            )
            self.drop = (nn.Dropout(dropout) if dropout > 0.0
                         else nn.Identity())
            self.adaln_linear = nn.Linear(cond_dim, 6 * dim)
            nn.init.zeros_(self.adaln_linear.weight)
            nn.init.zeros_(self.adaln_linear.bias)

        def forward(self, x, cond):
            mod = self.adaln_linear(cond).unsqueeze(1)
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
                mod.chunk(6, dim=-1)
            )
            h = _modulate(self.norm_attn(x), shift_a, scale_a)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + gate_a * self.drop(attn_out)
            h = _modulate(self.norm_mlp(x), shift_m, scale_m)
            x = x + gate_m * self.drop(self.mlp(h))
            return x

    class LatentDiTDirect(nn.Module):
        """DiT for direct latent future prediction (no diffusion).

        Input: z_t repeated H times + learned positional embeddings.
        Conditioning: z_t_proj(z_t) + a_embed (via adaLN-Zero).
        Output: direct prediction of z_future in normalized space.
        """

        def __init__(self, z_dim=384, cond_dim=384, n_blocks=4,
                     n_heads=6, horizon=4, mlp_ratio=4.0, dropout=0.0):
            super().__init__()
            self.z_dim = z_dim
            self.horizon = horizon

            # Input projection (same as DiT-DDIM)
            self.input_proj = nn.Linear(z_dim, z_dim)

            # Learned 1D positional embeddings for H horizon positions.
            # Critical: without these, the transformer is
            # permutation-equivariant over identical z_t copies and
            # produces the same output for all H tokens.
            self.pos_embed = nn.Embedding(horizon, z_dim)

            # Conditioning: z_t projection + action embedding
            # (no timestep -- that's the key difference from DiT-DDIM)
            self.z_t_proj = nn.Linear(z_dim, cond_dim)

            # Transformer blocks (identical to DiT-DDIM)
            self.blocks = nn.ModuleList([
                DiTBlock(
                    dim=z_dim, cond_dim=cond_dim, n_heads=n_heads,
                    mlp_ratio=mlp_ratio, dropout=dropout,
                )
                for _ in range(n_blocks)
            ])

            # Final layer (same as DiT-DDIM)
            self.final_norm = nn.LayerNorm(z_dim, elementwise_affine=False)
            self.final_adaln = nn.Linear(cond_dim, 3 * z_dim)
            nn.init.zeros_(self.final_adaln.weight)
            nn.init.zeros_(self.final_adaln.bias)
            self.final_linear = nn.Linear(z_dim, z_dim)

        def forward(self, z_t, a_embed):
            """Forward pass.

            Args:
                z_t: (B, z_dim) -- current frame latent (normalized).
                a_embed: (B, cond_dim) -- action embedding.

            Returns:
                (B, H, z_dim) -- predicted future latents.
            """
            B = z_t.shape[0]

            # Conditioning (no timestep)
            cond = self.z_t_proj(z_t) + a_embed

            # Input tokens: z_t repeated H times + positional embeddings
            pos_ids = torch.arange(
                self.horizon, device=z_t.device
            )  # (H,)
            x = self.input_proj(z_t).unsqueeze(1).expand(
                B, self.horizon, -1
            )  # (B, H, z_dim)
            x = x + self.pos_embed(pos_ids).unsqueeze(0)  # broadcast (1, H, z_dim)

            # Transformer blocks
            for block in self.blocks:
                x = block(x, cond)

            # Final layer
            mod = self.final_adaln(cond).unsqueeze(1)
            shift, scale, gate = mod.chunk(3, dim=-1)
            x = gate * self.final_linear(
                _modulate(self.final_norm(x), shift, scale)
            )
            return x  # (B, H, z_dim)

    class FourierActionEmbedding(nn.Module):
        """Fourier features for (steer, accel) -> dense embedding."""

        def __init__(self, action_dim=2, n_frequencies=64,
                     base=2.0, out_dim=384):
            super().__init__()
            self.action_dim = action_dim
            self.n_frequencies = n_frequencies
            freqs = (
                base ** torch.arange(n_frequencies, dtype=torch.float32)
                * torch.pi
            )
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

    print(f"[dit-direct] encoder={encoder_name}, seed={seed}, "
          f"variant={variant}, lr={lr}")

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
        for scene in np.unique(scenes):
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

    print(f"[dit-direct] Train: {len(z_t_train)} windows, "
          f"Val: {len(z_t_val)} windows")
    print(f"[dit-direct] native_dim={native_dim}, adapter={needs_adapter}")

    # ---------------------------------------------------------------
    # Model construction
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

    # Normalization stats from all training tokens (same as DiT-DDIM)
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

    print(
        f"[dit-direct] Normalization: std mean={z_std.mean():.4f}, "
        f"range=[{z_std.min():.4f}, {z_std.max():.4f}]"
    )

    fourier_embed = FourierActionEmbedding(
        action_dim=2,
        n_frequencies=FOURIER_CANONICAL["n_frequencies"],
        base=FOURIER_CANONICAL["base"],
        out_dim=FOURIER_CANONICAL["out_dim"],
    ).to(device)

    dit = LatentDiTDirect(
        z_dim=DIT_CANONICAL["z_dim"],
        cond_dim=DIT_CANONICAL["cond_dim"],
        n_blocks=DIT_CANONICAL["n_blocks"],
        n_heads=DIT_CANONICAL["n_heads"],
        horizon=DIT_CANONICAL["horizon"],
        mlp_ratio=DIT_CANONICAL["mlp_ratio"],
        dropout=DIT_CANONICAL["dropout"],
    ).to(device)

    params = list(dit.parameters()) + list(fourier_embed.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.MSELoss()

    n_dit_params = sum(p.numel() for p in dit.parameters())
    n_fourier_params = sum(p.numel() for p in fourier_embed.parameters())
    print(f"[dit-direct] DiT params: {n_dit_params:,}, "
          f"Fourier params: {n_fourier_params:,}")

    # DataLoaders
    train_ds = TensorDataset(z_t_train, act_train, zf_train)
    val_ds = TensorDataset(z_t_val, act_val, zf_val)
    train_loader = DataLoader(
        train_ds, batch_size=TRAINING_CANONICAL["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=TRAINING_CANONICAL["batch_size"], shuffle=False,
    )

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

            B, H, _ = zf_batch.shape
            z_t_norm = (adapter(z_t_batch) - z_mean) / z_std
            zf_norm = (
                adapter(zf_batch.reshape(B * H, -1)).reshape(
                    B, H, target_dim
                )
                - z_mean
            ) / z_std

            a_embed = fourier_embed(act_batch)
            if variant == "unconditioned":
                a_embed = torch.zeros_like(a_embed)

            # Direct prediction (no diffusion)
            z_hat = dit(z_t_norm, a_embed)  # (B, H, 384)
            loss = criterion(z_hat, zf_norm)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                params, TRAINING_CANONICAL["gradient_clip"]
            )
            optimizer.step()

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
                z_t_norm = (adapter(z_t_batch) - z_mean) / z_std
                zf_norm = (
                    adapter(zf_batch.reshape(B * H, -1)).reshape(
                        B, H, target_dim
                    )
                    - z_mean
                ) / z_std

                a_embed = fourier_embed(act_batch)
                if variant == "unconditioned":
                    a_embed = torch.zeros_like(a_embed)

                z_hat = dit(z_t_norm, a_embed)
                loss = criterion(z_hat, zf_norm)

                val_loss_sum += loss.item() * B
                val_n += B

        train_loss = train_loss_sum / train_n
        val_loss = val_loss_sum / val_n
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(
                f"[dit-direct] Epoch {epoch + 1}/{TRAINING_CANONICAL['epochs']}: "
                f"train={train_loss:.6f} val={val_loss:.6f} ({elapsed:.0f}s)"
            )

    elapsed = time.time() - t0
    print(
        f"[dit-direct] {encoder_name}/{variant}/seed={seed}/lr={lr}: "
        f"final_train={history['train_loss'][-1]:.6f} "
        f"final_val={history['val_loss'][-1]:.6f} time={elapsed:.1f}s"
    )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    # Include LR in path to avoid collisions during pilot sweeps
    lr_tag = f"_lr{lr:.0e}".replace("+", "").replace("-0", "-")
    out_dir = f"{OUT_DIR}/{encoder_name}/{variant}/seed_{seed}{lr_tag}"
    os.makedirs(out_dir, exist_ok=True)

    checkpoint = {
        "dit_direct_state_dict": dit.state_dict(),
        "fourier_embed_state_dict": fourier_embed.state_dict(),
        "adapter_state_dict": (adapter.state_dict()
                               if needs_adapter else None),
        "encoder_name": encoder_name,
        "variant": variant,
        "seed": seed,
        "lr": lr,
        "epochs": TRAINING_CANONICAL["epochs"],
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "z_mean": z_mean.cpu(),
        "z_std": z_std.cpu(),
        "normalize_latents": True,
        "adapter_frozen": True,
        "adapter_init": "orthogonal",
        "model_type": "dit_direct",
    }
    torch.save(checkpoint, f"{out_dir}/checkpoint.pt")

    with open(f"{out_dir}/train_log.json", "w") as f:
        json.dump(history, f)

    provenance = {
        "encoder_name": encoder_name,
        "variant": variant,
        "seed": seed,
        "lr": lr,
        "native_dim": native_dim,
        "target_dim": target_dim,
        "needs_adapter": needs_adapter,
        "dit": DIT_CANONICAL,
        "training": TRAINING_CANONICAL,
        "fourier": FOURIER_CANONICAL,
        "n_train_windows": int(len(z_t_train)),
        "n_val_windows": int(len(z_t_val)),
        "dit_params": n_dit_params,
        "fourier_params": n_fourier_params,
        "time_s": elapsed,
        "source": "scripts/train_dit_direct.py",
    }
    with open(f"{out_dir}/provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    vol.commit()

    return {
        "encoder": encoder_name,
        "variant": variant,
        "seed": seed,
        "lr": lr,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "time_s": elapsed,
    }


# ===================================================================
# Local entrypoint
# ===================================================================


def _modal_entrypoint_decorator(fn):
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_modal_entrypoint_decorator
def main(
    full: bool = False,
    sanity_uncond: bool = False,
    pilot_encoder: str = "vit_s16",
    pilot_seed: int = 0,
    lr: float = 0.0,
):
    """Train Direct-DiT for encoder benchmarking ablation.

    Default (no flags): pilot run -- 1 encoder x 1 seed x 3 LRs.
    --full: all 6 encoders x 3 seeds x conditioned (18 jobs).
    --sanity-uncond: add 1 unconditioned run on best encoder/seed.

    Args:
        full: Run full matrix (conditioned only).
        sanity_uncond: Add 1 unconditioned sanity check.
        pilot_encoder: Encoder for pilot (default: vit_s16).
        pilot_seed: Seed for pilot (default: 0).
        lr: Fixed LR (if 0, uses pilot sweep for pilot or
            best pilot LR for full run).
    """
    t_start = time.time()

    if not full:
        # Pilot: sweep LRs
        lrs = PILOT_LRS if lr == 0.0 else [lr]
        print("=" * 60)
        print("Direct-DiT PILOT")
        print(f"  encoder:  {pilot_encoder}")
        print(f"  seed:     {pilot_seed}")
        print(f"  LRs:      {lrs}")
        print(f"  variant:  conditioned")
        print(f"  jobs:     {len(lrs)}")
        print("=" * 60)

        futures = []
        for pilot_lr in lrs:
            print(f"  Launching lr={pilot_lr} ...")
            futures.append((
                pilot_lr,
                train_dit_direct.spawn(
                    pilot_encoder, pilot_seed, "conditioned", pilot_lr
                ),
            ))

        results = []
        for pilot_lr, future in futures:
            result = future.get()
            results.append(result)
            print(
                f"  lr={pilot_lr}: "
                f"train={result['final_train_loss']:.6f} "
                f"val={result['final_val_loss']:.6f} "
                f"({result['time_s']:.0f}s)"
            )

        best = min(results, key=lambda r: r["final_val_loss"])
        print(f"\n  Best LR: {best['lr']} "
              f"(val_loss={best['final_val_loss']:.6f})")

    else:
        # Full run: conditioned only, all encoders x seeds
        run_lr = lr if lr > 0 else None
        if run_lr is None:
            print("[ERROR] --full requires --lr (use best LR from pilot)")
            sys.exit(1)

        jobs = []
        for enc in ENCODER_NAMES:
            for s in SEEDS:
                jobs.append((enc, s, "conditioned", run_lr))

        if sanity_uncond:
            jobs.append((ENCODER_NAMES[0], SEEDS[0], "unconditioned", run_lr))

        print("=" * 60)
        print("Direct-DiT FULL RUN")
        print(f"  encoders:  {ENCODER_NAMES}")
        print(f"  seeds:     {SEEDS}")
        print(f"  variant:   conditioned" +
              (" + 1 unconditioned" if sanity_uncond else ""))
        print(f"  lr:        {run_lr}")
        print(f"  jobs:      {len(jobs)}")
        print("=" * 60)

        futures = []
        for enc, s, var, job_lr in jobs:
            print(f"  Launching {enc}/{var}/seed={s} ...")
            futures.append((
                enc, s, var,
                train_dit_direct.spawn(enc, s, var, job_lr),
            ))

        all_results = []
        for enc, s, var, future in futures:
            result = future.get()
            all_results.append(result)
            print(
                f"  {enc}/{var}/seed={s}: "
                f"train={result['final_train_loss']:.6f} "
                f"val={result['final_val_loss']:.6f}"
            )

        # Summary table
        print("\n" + "=" * 70)
        print(f"{'Encoder':<16} {'Variant':<14} {'Seed':>4} "
              f"{'Train Loss':>11} {'Val Loss':>10} {'Time':>6}")
        print("-" * 70)
        for r in sorted(all_results,
                        key=lambda x: (x["encoder"], x["variant"],
                                       x["seed"])):
            print(
                f"{r['encoder']:<16} {r['variant']:<14} {r['seed']:>4} "
                f"{r['final_train_loss']:>11.6f} "
                f"{r['final_val_loss']:>10.6f} "
                f"{r['time_s']:>5.0f}s"
            )

    wall_time = time.time() - t_start
    print(f"\nTotal wall time: {wall_time:.0f}s ({wall_time / 60:.1f}min)")
