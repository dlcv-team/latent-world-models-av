"""DA11: Train DiT-x0 with per-token action-sequence conditioning on Modal.

Fork of train_dit_horizon.py with three key changes:
1. build_windows() extracts action sequences (B, H, 2) not single actions (B, 2)
2. FourierActionEmbedding handles (B, H, 2) -> (B, H, 384) with .flatten(-2)
3. LatentDiT uses per-token conditioning: each horizon token k gets action at step t+k
4. Positional embeddings added to distinguish horizon positions

This enables DiT's self-attention to route per-token action information --
the minimum viable change that gives attention something useful to do.

Usage::

    modal run scripts/train_dit_actseq_modal.py
    FULL=1 modal run scripts/train_dit_actseq_modal.py
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
    app = modal.App("lwm-av-dit-actseq")
    vol = modal.Volume.from_name("nuscenes-full")
else:
    app = None
    vol = None

VOL_PATH = "/vol"
EMBED_DIR = f"{VOL_PATH}/embeddings"
DIT_DIR = f"{VOL_PATH}/dits"

# ---------------------------------------------------------------------------
# Canonical constants (same as train_dit_horizon.py)
# ---------------------------------------------------------------------------

DIT_CANONICAL = {
    "n_blocks": 4,
    "n_heads": 6,
    "z_dim": 384,
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

# Pilot: 3 harder encoders, h=8+16, seed 0
PILOT_ENCODERS = ["vit_s16", "clip_b32", "dino_vits14"]
PILOT_HORIZONS = [8, 16]
PILOT_SEEDS = [0]

# Full: same 3 encoders, 3 seeds
FULL_ENCODERS = ["vit_s16", "clip_b32", "dino_vits14"]
FULL_HORIZONS = [8, 16]
FULL_SEEDS = [0, 1, 2]

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
            timeout=7200,
            memory=16384,
        )(fn)
    return fn


@_modal_function_decorator
def train_dit_actseq(
    encoder_name: str,
    seed: int,
    horizon: int,
):
    """Train DiT-x0 with per-token action-sequence conditioning."""
    import numpy as np
    import torch
    from torch import nn
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

    # KEY CHANGE: DiTBlock accepts per-token conditioning (B, H, D)
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
            nn.init.zeros_(self.adaln_linear.weight)
            nn.init.zeros_(self.adaln_linear.bias)

        def forward(self, x, cond):
            # cond is (B, H, D) -- per-token conditioning
            # adaln_linear output is (B, H, 6*D) -- no unsqueeze needed
            mod = self.adaln_linear(cond)
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mod.chunk(6, dim=-1)
            h = _modulate(self.norm_attn(x), shift_a, scale_a)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + gate_a * self.drop(attn_out)
            h = _modulate(self.norm_mlp(x), shift_m, scale_m)
            x = x + gate_m * self.drop(self.mlp(h))
            return x

    # KEY CHANGE: LatentDiT with per-token action conditioning + positional embeddings
    class LatentDiT(nn.Module):
        def __init__(self, z_dim=384, cond_dim=384, n_blocks=4, n_heads=6,
                     horizon=4, mlp_ratio=4.0, dropout=0.0):
            super().__init__()
            self.z_dim = z_dim
            self.horizon = horizon
            self.input_proj = nn.Linear(z_dim, z_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, horizon, z_dim))  # NEW
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

        def forward(self, x_noisy, z_t, a_embed_seq, timestep):
            # a_embed_seq: (B, H, cond_dim) -- per-token action embeddings
            # Global conditioning: timestep + z_t (shared across tokens)
            cond_global = self.timestep_embed(timestep) + self.z_t_proj(z_t)  # (B, D)
            # Per-token conditioning: global + per-token action
            cond = cond_global.unsqueeze(1) + a_embed_seq  # (B, H, D)

            x = self.input_proj(x_noisy) + self.pos_embed[:, :x_noisy.shape[1], :]
            for block in self.blocks:
                x = block(x, cond)  # cond is (B, H, D)
            # Final layer: per-token modulation
            mod = self.final_adaln(cond)  # (B, H, 3*D) -- no unsqueeze
            shift, scale, gate = mod.chunk(3, dim=-1)
            x = gate * self.final_linear(
                _modulate(self.final_norm(x), shift, scale)
            )
            return x

    # KEY CHANGE: FourierActionEmbedding handles (B, H, 2) -> (B, H, 384)
    class FourierActionEmbedding(nn.Module):
        def __init__(self, action_dim=2, n_frequencies=64, base=2.0, out_dim=384):
            super().__init__()
            self.action_dim = action_dim
            self.n_frequencies = n_frequencies
            freqs = base ** torch.arange(n_frequencies, dtype=torch.float32) * torch.pi
            self.register_buffer("freqs", freqs)
            fourier_dim = action_dim * 2 * n_frequencies
            self.proj = nn.Sequential(
                nn.Linear(fourier_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim),
            )

        def forward(self, action):
            # action: (B, H, 2) -- per-token action sequence
            # freqs: (n_freq,)
            x = action.unsqueeze(-1) * self.freqs  # (B, H, 2, n_freq)
            x = torch.cat([x.sin(), x.cos()], dim=-1)  # (B, H, 2, 2*n_freq)
            x = x.flatten(-2)  # (B, H, 2*2*n_freq) -- NOT .flatten(1)!
            return self.proj(x)  # (B, H, out_dim) -- Linear broadcasts

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

    # -------------------------------------------------------------------
    # Data loading with ACTION SEQUENCES
    # -------------------------------------------------------------------

    print(
        f"[dit-actseq] encoder={encoder_name}, seed={seed}, "
        f"horizon={horizon}, prediction=x0"
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

    # KEY CHANGE: build_windows returns action sequences (B, H, 2)
    def build_windows(split_name):
        mask = splits == split_name
        emb = embeddings[mask]
        steers = steer_norms[mask]
        accels = accel_norms[mask]
        scenes = scene_names[mask]

        z_t_list, action_seq_list, z_future_list = [], [], []
        for scene in np.unique(scenes):
            scene_mask = scenes == scene
            idx = np.where(scene_mask)[0]
            n_scene = len(idx)
            for j in range(n_scene - horizon):
                t_idx = idx[j]
                future_idx = idx[j + 1: j + 1 + horizon]
                z_t_list.append(emb[t_idx])
                # Action sequence: action at each step t, t+1, ..., t+H-1
                # action[k] governs transition from z_{t+k} to z_{t+k+1}
                action_seq = np.stack([
                    np.array([steers[idx[j + k]], accels[idx[j + k]]])
                    for k in range(horizon)
                ])  # (H, 2)
                action_seq_list.append(action_seq)
                z_future_list.append(emb[future_idx])

        if not z_t_list:
            return None, None, None
        return (
            torch.tensor(np.array(z_t_list), dtype=torch.float32),
            torch.tensor(np.array(action_seq_list), dtype=torch.float32),  # (N, H, 2)
            torch.tensor(np.array(z_future_list), dtype=torch.float32),
        )

    z_t_train, act_seq_train, zf_train = build_windows("train")
    z_t_val, act_seq_val, zf_val = build_windows("val")

    print(f"[dit-actseq] Train: {len(z_t_train)} windows, Val: {len(z_t_val)} windows")
    print(f"[dit-actseq] Action seq shape: {act_seq_train.shape}")  # should be (N, H, 2)

    # -------------------------------------------------------------------
    # Model construction
    # -------------------------------------------------------------------

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
        zf_proj = adapter(zf_train.reshape(-1, zf_train.shape[-1]).to(device))
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
        horizon=horizon,
        mlp_ratio=DIT_CANONICAL["mlp_ratio"],
        dropout=DIT_CANONICAL["dropout"],
    ).to(device)

    schedule = CosineNoiseSchedule(n_steps=DIFFUSION_CANONICAL["n_train_steps"]).to(device)

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
    print(f"[dit-actseq] DiT params: {n_params:,}")

    train_ds = TensorDataset(z_t_train, act_seq_train, zf_train)
    val_ds = TensorDataset(z_t_val, act_seq_val, zf_val)
    train_loader = DataLoader(train_ds, batch_size=TRAINING_CANONICAL["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TRAINING_CANONICAL["batch_size"], shuffle=False)

    T = DIFFUSION_CANONICAL["n_train_steps"]

    # -------------------------------------------------------------------
    # Training loop -- x0-prediction with per-token action sequences
    # -------------------------------------------------------------------

    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()

    for epoch in range(TRAINING_CANONICAL["epochs"]):
        dit.train()
        fourier_embed.train()
        train_loss_sum = 0.0
        train_n = 0

        for z_t_batch, act_seq_batch, zf_batch in train_loader:
            z_t_batch = z_t_batch.to(device)
            act_seq_batch = act_seq_batch.to(device)  # (B, H, 2)
            zf_batch = zf_batch.to(device)

            B, H, _ = zf_batch.shape
            z_t_adapted = (adapter(z_t_batch) - z_mean) / z_std
            zf_adapted = (
                adapter(zf_batch.reshape(B * H, -1)).reshape(B, H, target_dim)
                - z_mean
            ) / z_std

            # Per-token action embeddings: (B, H, 2) -> (B, H, 384)
            a_embed_seq = fourier_embed(act_seq_batch)

            diffusion_target = zf_adapted  # x0-prediction

            t = torch.randint(0, T, (B,), device=device)
            x_noisy, noise = schedule.add_noise(diffusion_target, t)

            dit_out = dit(x_noisy, z_t_adapted, a_embed_seq, t)
            loss = criterion(dit_out, diffusion_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, TRAINING_CANONICAL["gradient_clip"])
            optimizer.step()
            ema.update(trainable_group)

            train_loss_sum += loss.item() * B
            train_n += B

        # Validate
        dit.eval()
        fourier_embed.eval()
        val_loss_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for z_t_batch, act_seq_batch, zf_batch in val_loader:
                z_t_batch = z_t_batch.to(device)
                act_seq_batch = act_seq_batch.to(device)
                zf_batch = zf_batch.to(device)

                B, H, _ = zf_batch.shape
                z_t_adapted = (adapter(z_t_batch) - z_mean) / z_std
                zf_adapted = (
                    adapter(zf_batch.reshape(B * H, -1)).reshape(B, H, target_dim)
                    - z_mean
                ) / z_std

                a_embed_seq = fourier_embed(act_seq_batch)
                diffusion_target = zf_adapted

                t = torch.randint(0, T, (B,), device=device)
                x_noisy, noise = schedule.add_noise(diffusion_target, t)
                dit_out = dit(x_noisy, z_t_adapted, a_embed_seq, t)
                loss = criterion(dit_out, diffusion_target)

                val_loss_sum += loss.item() * B
                val_n += B

        train_loss = train_loss_sum / train_n
        val_loss = val_loss_sum / val_n
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"[dit-actseq] Epoch {epoch + 1}/{TRAINING_CANONICAL['epochs']}: "
                f"train={train_loss:.6f} val={val_loss:.6f} ({elapsed:.0f}s)"
            )

    elapsed = time.time() - t0
    print(
        f"[dit-actseq] {encoder_name}/h{horizon}/seed={seed}: "
        f"final_train={history['train_loss'][-1]:.6f} "
        f"final_val={history['val_loss'][-1]:.6f} time={elapsed:.1f}s"
    )

    # -------------------------------------------------------------------
    # Save checkpoint
    # -------------------------------------------------------------------

    out_dir = f"{DIT_DIR}/{encoder_name}/conditioned__x0__actseq__h{horizon}/seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)

    checkpoint = {
        "dit_state_dict": dit.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "fourier_embed_state_dict": fourier_embed.state_dict(),
        "adapter_state_dict": adapter.state_dict() if needs_adapter else None,
        "encoder_name": encoder_name,
        "seed": seed,
        "prediction": "x0",
        "actseq": True,
        "horizon": horizon,
        "epochs": TRAINING_CANONICAL["epochs"],
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "z_mean": z_mean.cpu(),
        "z_std": z_std.cpu(),
    }
    torch.save(checkpoint, f"{out_dir}/checkpoint.pt")

    with open(f"{out_dir}/train_log.json", "w") as f:
        json.dump(history, f)

    provenance = {
        "encoder_name": encoder_name,
        "seed": seed,
        "prediction": "x0",
        "actseq": True,
        "horizon": horizon,
        "native_dim": native_dim,
        "target_dim": target_dim,
        "dit": {**DIT_CANONICAL, "horizon": horizon},
        "training": TRAINING_CANONICAL,
        "fourier": FOURIER_CANONICAL,
        "n_train_windows": int(len(z_t_train)),
        "n_val_windows": int(len(z_t_val)),
        "dit_params": n_params,
        "time_s": elapsed,
        "source": "scripts/train_dit_actseq_modal.py",
    }
    with open(f"{out_dir}/provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    vol.commit()

    return {
        "encoder": encoder_name,
        "seed": seed,
        "horizon": horizon,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "n_train_windows": len(z_t_train),
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
def main():
    """DA11: Train DiT-x0 with action-sequence conditioning.

    Pilot: 3 encoders x 2 horizons x 1 seed = 6 jobs.
    Full: 3 encoders x 2 horizons x 3 seeds = 18 jobs.

    Set FULL=1 for full expansion:
        FULL=1 modal run scripts/train_dit_actseq_modal.py
    """
    if os.environ.get("FULL", "") == "1":
        encoders = FULL_ENCODERS
        horizons = FULL_HORIZONS
        seeds = FULL_SEEDS
        label = "FULL"
    else:
        encoders = PILOT_ENCODERS
        horizons = PILOT_HORIZONS
        seeds = PILOT_SEEDS
        label = "PILOT"

    t_start = time.time()
    n_jobs = len(encoders) * len(horizons) * len(seeds)
    print("=" * 60)
    print(f"DA11 {label}: DiT-x0 with action-sequence conditioning")
    print(f"  encoders: {encoders}")
    print(f"  horizons: {horizons}")
    print(f"  seeds:    {seeds}")
    print(f"  jobs: {n_jobs}")
    print("=" * 60)

    futures = []
    for enc in encoders:
        for h in horizons:
            for s in seeds:
                print(f"  Launching {enc}/h{h}/seed={s} ...")
                futures.append((enc, h, s, train_dit_actseq.spawn(enc, s, h)))

    all_results = []
    for enc, h, s, future in futures:
        result = future.get()
        all_results.append(result)
        tl = result["final_train_loss"]
        vl = result["final_val_loss"]
        print(f"  {enc}/h{h}/seed={s}: train={tl:.6f} val={vl:.6f}")

    print("\n" + "=" * 90)
    print(f"{'Encoder':<14} {'H':>3} {'Seed':>4} {'Train':>11} {'Val':>10} {'Time':>6}")
    print("-" * 90)
    for r in sorted(all_results, key=lambda x: (x["encoder"], x["horizon"], x["seed"])):
        print(
            f"{r['encoder']:<14} {r['horizon']:>3} {r['seed']:>4} "
            f"{r['final_train_loss']:>11.6f} {r['final_val_loss']:>10.6f} "
            f"{r['time_s']:>5.0f}s"
        )

    wall_time = time.time() - t_start
    print(f"\nTotal wall time: {wall_time:.0f}s ({wall_time / 60:.1f}min)")
