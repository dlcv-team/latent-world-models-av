"""DA11: Train MLP-residual with flattened action-sequence conditioning on Modal.

Fair baseline for DA11 DiT comparison. The MLP receives the SAME ordered
action information as DiT, but processes it through a wider feedforward
network instead of per-token attention.

Architecture:
  Input: concat(z_t_norm(384), flatten(a_embed_seq(H*384))) = 384 + H*384
  Hidden: 1024 (wider than standard MLP's 512 to handle larger input)
  Output: z_dim * H residuals (delta_z prediction)

No intermediate projection bottleneck -- the MLP directly receives all
ordered action features. DiT's advantage (if any) comes purely from
per-token attention routing, not from information asymmetry.

Usage::

    modal run scripts/train_mlp_actseq_modal.py
    FULL=1 modal run scripts/train_mlp_actseq_modal.py
"""

from __future__ import annotations

import json
import math
import os
import time

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("lwm-av-mlp-actseq")
    vol = modal.Volume.from_name("nuscenes-full")
else:
    app = None
    vol = None

VOL_PATH = "/vol"
EMBED_DIR = f"{VOL_PATH}/embeddings"
MLP_DIR = f"{VOL_PATH}/outputs"

TARGET_DIM = 384
MLP_HIDDEN = 1024  # wider than standard 512 to handle H*384 action input

FOURIER_CANONICAL = {
    "n_frequencies": 64,
    "base": 2.0,
    "out_dim": 384,
}

TRAINING_CANONICAL = {
    "epochs": 50,
    "lr": 1e-3,
    "batch_size": 128,
    "seed": 0,
}

NATIVE_DIMS = {
    "vit_s16": 384,
    "dino_vits14": 384,
    "clip_b32": 512,
    "vq_track": 256,
    "vjepa2_rep64": 1024,
    "vjepa2_rep1": 1024,
}

PILOT_ENCODERS = ["vit_s16", "clip_b32", "dino_vits14"]
PILOT_HORIZONS = [8, 16]
PILOT_SEEDS = [0]

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
def train_mlp_actseq(
    encoder_name: str,
    seed: int,
    horizon: int,
):
    """Train MLP-residual with flattened ordered action sequences."""
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # -------------------------------------------------------------------
    # Inline model definitions
    # -------------------------------------------------------------------

    class FourierActionEmbedding(nn.Module):
        """Same as DiT version -- handles (B, H, 2) -> (B, H, 384)."""
        def __init__(self, action_dim=2, n_frequencies=64, base=2.0, out_dim=384):
            super().__init__()
            freqs = base ** torch.arange(n_frequencies, dtype=torch.float32) * torch.pi
            self.register_buffer("freqs", freqs)
            fourier_dim = action_dim * 2 * n_frequencies
            self.proj = nn.Sequential(
                nn.Linear(fourier_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim),
            )

        def forward(self, action):
            # action: (B, H, 2)
            x = action.unsqueeze(-1) * self.freqs  # (B, H, 2, n_freq)
            x = torch.cat([x.sin(), x.cos()], dim=-1)  # (B, H, 2, 2*n_freq)
            x = x.flatten(-2)  # (B, H, fourier_dim)
            return self.proj(x)  # (B, H, out_dim)

    class LatentPredictorActSeq(nn.Module):
        """MLP with flattened ordered action-sequence input.

        Input: concat(z_t_norm, flatten(a_embed_seq)) = z_dim + H*a_dim
        Output: z_dim * H residuals
        """
        def __init__(self, z_dim=384, a_dim=384, horizon=4, hidden=1024):
            super().__init__()
            self.horizon = horizon
            input_dim = z_dim + a_dim * horizon  # 384 + H*384
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, z_dim * horizon),
            )

        def forward(self, z_t, a_embed_flat):
            # z_t: (B, z_dim)
            # a_embed_flat: (B, H*a_dim)
            x = torch.cat([z_t, a_embed_flat], dim=-1)
            return self.net(x).view(z_t.shape[0], self.horizon, -1)

    # -------------------------------------------------------------------
    # Data loading with action sequences
    # -------------------------------------------------------------------

    print(f"[mlp-actseq] encoder={encoder_name}, seed={seed}, horizon={horizon}")

    embed_path = f"{EMBED_DIR}/{encoder_name}.npz"
    with np.load(embed_path, allow_pickle=True) as f:
        embeddings = f["embeddings"]
        splits = f["splits"]
        steer_norms = f["steer_norms"]
        accel_norms = f["accel_norms"]
        scene_names = f["scene_names"]

    native_dim = NATIVE_DIMS[encoder_name]
    needs_adapter = native_dim != TARGET_DIM

    def build_windows(split_name):
        mask = splits == split_name
        emb = embeddings[mask]
        steers = steer_norms[mask]
        accels = accel_norms[mask]
        scenes = scene_names[mask]

        z_t_list, action_seq_list, z_future_list = [], [], []
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

        if not z_t_list:
            return None, None, None
        return (
            torch.tensor(np.array(z_t_list), dtype=torch.float32),
            torch.tensor(np.array(action_seq_list), dtype=torch.float32),
            torch.tensor(np.array(z_future_list), dtype=torch.float32),
        )

    z_t_train, act_seq_train, zf_train = build_windows("train")
    z_t_val, act_seq_val, zf_val = build_windows("val")

    print(f"[mlp-actseq] Train: {len(z_t_train)}, Val: {len(z_t_val)}")

    # -------------------------------------------------------------------
    # Model construction
    # -------------------------------------------------------------------

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if needs_adapter:
        adapter = nn.Linear(native_dim, TARGET_DIM, bias=False).to(device)
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
        action_dim=2, **FOURIER_CANONICAL,
    ).to(device)

    predictor = LatentPredictorActSeq(
        z_dim=TARGET_DIM,
        a_dim=FOURIER_CANONICAL["out_dim"],
        horizon=horizon,
        hidden=MLP_HIDDEN,
    ).to(device)

    params = list(predictor.parameters()) + list(fourier_embed.parameters())
    optimizer = torch.optim.Adam(params, lr=TRAINING_CANONICAL["lr"])
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in predictor.parameters())
    print(f"[mlp-actseq] MLP params: {n_params:,}, input_dim={TARGET_DIM + FOURIER_CANONICAL['out_dim'] * horizon}")

    train_ds = TensorDataset(z_t_train, act_seq_train, zf_train)
    val_ds = TensorDataset(z_t_val, act_seq_val, zf_val)
    train_loader = DataLoader(train_ds, batch_size=TRAINING_CANONICAL["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TRAINING_CANONICAL["batch_size"], shuffle=False)

    # -------------------------------------------------------------------
    # Training loop -- residual prediction
    # -------------------------------------------------------------------

    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()

    for epoch in range(TRAINING_CANONICAL["epochs"]):
        predictor.train()
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
                adapter(zf_batch.reshape(B * H, -1)).reshape(B, H, TARGET_DIM)
                - z_mean
            ) / z_std

            # Embed action sequence then flatten: (B, H, 384) -> (B, H*384)
            a_embed_seq = fourier_embed(act_seq_batch)  # (B, H, 384)
            a_embed_flat = a_embed_seq.reshape(B, -1)  # (B, H*384)

            # Residual target: delta_z = z_future - z_t (in normalized space)
            z_t_expanded = z_t_adapted.unsqueeze(1).expand(-1, H, -1)
            delta_target = zf_adapted - z_t_expanded

            delta_pred = predictor(z_t_adapted, a_embed_flat)
            loss = criterion(delta_pred, delta_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * B
            train_n += B

        # Validate
        predictor.eval()
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
                    adapter(zf_batch.reshape(B * H, -1)).reshape(B, H, TARGET_DIM)
                    - z_mean
                ) / z_std

                a_embed_seq = fourier_embed(act_seq_batch)
                a_embed_flat = a_embed_seq.reshape(z_t_batch.shape[0], -1)

                z_t_expanded = z_t_adapted.unsqueeze(1).expand(-1, H, -1)
                delta_target = zf_adapted - z_t_expanded

                delta_pred = predictor(z_t_adapted, a_embed_flat)
                loss = criterion(delta_pred, delta_target)

                val_loss_sum += loss.item() * B
                val_n += B

        train_loss = train_loss_sum / train_n
        val_loss = val_loss_sum / val_n
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"[mlp-actseq] Epoch {epoch + 1}/{TRAINING_CANONICAL['epochs']}: "
                f"train={train_loss:.6f} val={val_loss:.6f} ({elapsed:.0f}s)"
            )

    elapsed = time.time() - t0
    print(
        f"[mlp-actseq] {encoder_name}/h{horizon}/seed={seed}: "
        f"train={history['train_loss'][-1]:.6f} "
        f"val={history['val_loss'][-1]:.6f} time={elapsed:.1f}s"
    )

    # -------------------------------------------------------------------
    # Save checkpoint
    # -------------------------------------------------------------------

    out_dir = (
        f"{MLP_DIR}/latent_predictors_residual_actseq_h{horizon}"
        f"/{encoder_name}/conditioned/seed_{seed}"
    )
    os.makedirs(out_dir, exist_ok=True)

    checkpoint = {
        "predictor_state_dict": predictor.state_dict(),
        "fourier_embed_state_dict": fourier_embed.state_dict(),
        "adapter_state_dict": adapter.state_dict() if needs_adapter else None,
        "encoder_name": encoder_name,
        "seed": seed,
        "horizon": horizon,
        "actseq": True,
        "mode": "residual",
        "hidden": MLP_HIDDEN,
        "epochs": TRAINING_CANONICAL["epochs"],
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "z_mean": z_mean.cpu(),
        "z_std": z_std.cpu(),
    }
    torch.save(checkpoint, f"{out_dir}/checkpoint.pt")

    with open(f"{out_dir}/train_log.json", "w") as f:
        json.dump(history, f)

    vol.commit()

    return {
        "encoder": encoder_name,
        "seed": seed,
        "horizon": horizon,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "n_train_windows": len(z_t_train),
        "mlp_params": n_params,
        "time_s": elapsed,
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
    """DA11: Train MLP-flat-actseq baseline.

    Pilot: 3 encoders x 2 horizons x 1 seed = 6 jobs.
    Full: 3 encoders x 2 horizons x 3 seeds = 18 jobs.
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
    print(f"DA11 {label}: MLP-flat-actseq baseline")
    print(f"  encoders: {encoders}")
    print(f"  horizons: {horizons}")
    print(f"  seeds:    {seeds}")
    print(f"  hidden:   {MLP_HIDDEN}")
    print(f"  jobs: {n_jobs}")
    print("=" * 60)

    futures = []
    for enc in encoders:
        for h in horizons:
            for s in seeds:
                futures.append((enc, h, s, train_mlp_actseq.spawn(enc, s, h)))

    all_results = []
    for enc, h, s, future in futures:
        result = future.get()
        all_results.append(result)
        tl = result["final_train_loss"]
        vl = result["final_val_loss"]
        print(f"  {enc}/h{h}/seed={s}: train={tl:.6f} val={vl:.6f} params={result['mlp_params']:,}")

    print("\n" + "=" * 90)
    print(f"{'Encoder':<14} {'H':>3} {'Seed':>4} {'Train':>11} {'Val':>10} {'Params':>10} {'Time':>6}")
    print("-" * 90)
    for r in sorted(all_results, key=lambda x: (x["encoder"], x["horizon"], x["seed"])):
        print(
            f"{r['encoder']:<14} {r['horizon']:>3} {r['seed']:>4} "
            f"{r['final_train_loss']:>11.6f} {r['final_val_loss']:>10.6f} "
            f"{r['mlp_params']:>10,} {r['time_s']:>5.0f}s"
        )

    wall_time = time.time() - t_start
    print(f"\nTotal wall time: {wall_time:.0f}s ({wall_time / 60:.1f}min)")
