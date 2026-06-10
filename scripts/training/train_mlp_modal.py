"""Train MLP-fair and MLP-residual predictors at longer horizons via Modal.

Runs both model types in parallel on GPU. Reads embeddings from the shared
Modal volume, saves checkpoints there, then downloads locally.

Usage::

    modal run scripts/train_mlp_modal.py                    # remaining jobs only
    FULL=1 modal run scripts/train_mlp_modal.py             # force all 72 jobs
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("lwm-av-mlp-horizon")
    vol = modal.Volume.from_name("nuscenes-full")
else:
    app = None
    vol = None

VOL_PATH = "/vol"
EMBED_DIR = f"{VOL_PATH}/embeddings"

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
ALL_HORIZONS = [8, 16]
ALL_SEEDS = [0, 1, 2]
ALL_MODES = ["fair", "residual"]

# MLP training hyperparameters (from canonical config)
MLP_EPOCHS = 50
MLP_LR = 0.001
MLP_BATCH_SIZE = 128

# Fourier embedding config
FOURIER_CANONICAL = {
    "n_frequencies": 64,
    "base": 2.0,
    "out_dim": 384,
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
            timeout=3600,
            memory=16384,
        )(fn)
    return fn


@_modal_function_decorator
def train_mlp_horizon(
    encoder_name: str,
    seed: int,
    horizon: int,
    mode: str = "fair",  # "fair" or "residual"
):
    """Train a single MLP predictor at specified horizon."""
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    assert mode in ("fair", "residual"), f"Unknown mode: {mode}"

    # -------------------------------------------------------------------
    # Inline model definitions
    # -------------------------------------------------------------------

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

    class LatentPredictor(nn.Module):
        def __init__(self, z_dim=384, a_dim=384, horizon=4, hidden=512):
            super().__init__()
            self.horizon = horizon
            self.net = nn.Sequential(
                nn.Linear(z_dim + a_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, z_dim * horizon),
            )

        def forward(self, z_t, a_embed):
            x = torch.cat([z_t, a_embed], dim=-1)
            out = self.net(x)
            return out.view(z_t.shape[0], self.horizon, -1)

    # -------------------------------------------------------------------
    # Data loading from Modal volume
    # -------------------------------------------------------------------

    target_dim = TARGET_DIM
    native_dim = NATIVE_DIMS[encoder_name]
    needs_adapter = native_dim != target_dim

    embed_path = f"{EMBED_DIR}/{encoder_name}.npz"
    print(f"[mlp-h] Loading {embed_path}")
    data = np.load(embed_path, allow_pickle=True)

    embeddings = data["embeddings"]
    splits = data["splits"]
    steer_norms = data["steer_norms"]
    accel_norms = data["accel_norms"]
    scene_names = data["scene_names"]

    # Output directory on volume
    out_dir = f"{VOL_PATH}/outputs/latent_predictors_{mode}_h{horizon}/{encoder_name}/conditioned/seed_{seed}"
    ckpt_path = f"{out_dir}/checkpoint.pt"

    # Check for existing checkpoint
    if os.path.exists(ckpt_path):
        print(f"[mlp-h] SKIP: {ckpt_path} already exists")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return {
            "encoder": encoder_name,
            "seed": seed,
            "horizon": horizon,
            "mode": mode,
            "final_train_loss": ckpt.get("final_train_loss", -1),
            "final_val_loss": ckpt.get("final_val_loss", -1),
            "skipped": True,
        }

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
    n_train = len(z_t_train)
    n_val = len(z_t_val)
    print(f"[mlp-h] Train: {n_train} windows, Val: {n_val} windows")

    # -------------------------------------------------------------------
    # Adapter + normalization (same as DiT training)
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
        # Use h=4 windows for adapter stats (matching DA8)
        z_t_h4_list, _, zf_h4_list = [], [], []
        for split_name in ["train"]:
            mask = splits == split_name
            emb = embeddings[mask]
            steers = steer_norms[mask]
            accels = accel_norms[mask]
            scenes = scene_names[mask]
            unique_scenes = np.unique(scenes)
            for scene in unique_scenes:
                scene_mask = scenes == scene
                idx = np.where(scene_mask)[0]
                for j in range(len(idx) - 4):
                    t_idx = idx[j]
                    future_idx = idx[j + 1 : j + 5]
                    z_t_h4_list.append(emb[t_idx])
                    zf_h4_list.append(emb[future_idx])

        z_t_h4 = torch.tensor(np.array(z_t_h4_list), dtype=torch.float32).to(device)
        zf_h4 = torch.tensor(np.array(zf_h4_list), dtype=torch.float32).to(device)
        B_h4, H_h4, _ = zf_h4.shape

        z_t_proj = adapter(z_t_h4)
        zf_proj = adapter(zf_h4.reshape(-1, zf_h4.shape[-1]))
        all_proj = torch.cat([z_t_proj, zf_proj], dim=0)
        z_mean = all_proj.mean(dim=0)
        z_std = all_proj.std(dim=0).clamp(min=1e-6)
        del z_t_h4, zf_h4, z_t_proj, zf_proj, all_proj

    # Normalize training data
    with torch.no_grad():
        z_t_norm = (adapter(z_t_train.to(device)) - z_mean) / z_std
        B, H, _ = zf_train.shape
        zf_norm = (
            adapter(zf_train.reshape(B * H, -1).to(device)).reshape(B, H, target_dim)
            - z_mean
        ) / z_std
        act_train_dev = act_train.to(device)

        # Validation data
        z_t_val_norm = (adapter(z_t_val.to(device)) - z_mean) / z_std
        Bv, Hv, _ = zf_val.shape
        zf_val_norm = (
            adapter(zf_val.reshape(Bv * Hv, -1).to(device)).reshape(Bv, Hv, target_dim)
            - z_mean
        ) / z_std
        act_val_dev = act_val.to(device)

    # Compute targets
    if mode == "residual":
        z_t_expanded = z_t_norm.unsqueeze(1).expand(-1, H, -1)
        targets_train = zf_norm - z_t_expanded
        z_t_val_expanded = z_t_val_norm.unsqueeze(1).expand(-1, Hv, -1)
        targets_val = zf_val_norm - z_t_val_expanded
    else:
        targets_train = zf_norm
        targets_val = zf_val_norm

    # -------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------

    predictor = LatentPredictor(
        z_dim=target_dim,
        a_dim=target_dim,
        horizon=horizon,
    ).to(device)

    fourier_embed = FourierActionEmbedding(
        action_dim=2,
        n_frequencies=FOURIER_CANONICAL["n_frequencies"],
        base=FOURIER_CANONICAL["base"],
        out_dim=FOURIER_CANONICAL["out_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(
        list(predictor.parameters()) + list(fourier_embed.parameters()),
        lr=MLP_LR,
    )
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(z_t_norm, act_train_dev, targets_train)
    train_loader = DataLoader(train_ds, batch_size=MLP_BATCH_SIZE, shuffle=True)

    val_ds = TensorDataset(z_t_val_norm, act_val_dev, targets_val)
    val_loader = DataLoader(val_ds, batch_size=MLP_BATCH_SIZE, shuffle=False)

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------

    t0 = time.time()
    train_hist, val_hist = [], []

    for epoch in range(MLP_EPOCHS):
        predictor.train()
        fourier_embed.train()
        epoch_loss, epoch_n = 0.0, 0

        for z_t_b, act_b, target_b in train_loader:
            a_embed = fourier_embed(act_b)
            z_hat = predictor(z_t_b, a_embed)
            loss = loss_fn(z_hat, target_b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * z_t_b.shape[0]
            epoch_n += z_t_b.shape[0]

        train_hist.append(epoch_loss / epoch_n)

        predictor.eval()
        fourier_embed.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for z_t_b, act_b, target_b in val_loader:
                a_embed = fourier_embed(act_b)
                z_hat = predictor(z_t_b, a_embed)
                loss = loss_fn(z_hat, target_b)
                val_loss += loss.item() * z_t_b.shape[0]
                val_n += z_t_b.shape[0]
        val_hist.append(val_loss / val_n)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[mlp-h] Epoch {epoch+1}/{MLP_EPOCHS}: "
                f"train={train_hist[-1]:.6f} val={val_hist[-1]:.6f} "
                f"({time.time()-t0:.0f}s)"
            )

    # -------------------------------------------------------------------
    # Save checkpoint
    # -------------------------------------------------------------------

    os.makedirs(out_dir, exist_ok=True)

    checkpoint = {
        "predictor_state_dict": predictor.state_dict(),
        "fourier_embed_state_dict": fourier_embed.state_dict(),
        "adapter_state_dict": adapter.state_dict() if needs_adapter else {},
        "z_mean": z_mean.cpu(),
        "z_std": z_std.cpu(),
        "final_train_loss": train_hist[-1],
        "final_val_loss": val_hist[-1],
        "epochs": MLP_EPOCHS,
        "learning_rate": MLP_LR,
        "batch_size": MLP_BATCH_SIZE,
        "horizon": horizon,
        **({"residual": True} if mode == "residual" else {}),
        "provenance": {
            "formulation": mode,
            "target": "delta_z" if mode == "residual" else "z_future_norm",
            "space": "normalized (z - z_mean) / z_std",
            "adapter_source": "reconstructed_orthogonal",
            "n_train_windows": n_train,
            "n_val_windows": n_val,
            "torch_version": torch.__version__,
            "encoder_name": encoder_name,
            "variant": "conditioned",
            "seed": seed,
            "horizon": horizon,
            "source": "scripts/train_mlp_modal.py",
        },
    }
    torch.save(checkpoint, ckpt_path)
    vol.commit()

    elapsed = time.time() - t0
    print(
        f"[mlp-h] {encoder_name}/{mode}_h{horizon}/seed={seed}: "
        f"train={train_hist[-1]:.6f} val={val_hist[-1]:.6f} "
        f"time={elapsed:.1f}s"
    )

    return {
        "encoder": encoder_name,
        "seed": seed,
        "horizon": horizon,
        "mode": mode,
        "final_train_loss": train_hist[-1],
        "final_val_loss": val_hist[-1],
        "n_train_windows": n_train,
        "time_s": round(elapsed, 1),
        "skipped": False,
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
    """DA9 Exp 3: Train MLP-fair + MLP-residual at longer horizons on Modal.

    Checks local filesystem for existing checkpoints and only launches
    jobs for missing ones.

    Set FULL=1 to retrain everything (ignores local checkpoints).
    """
    force_all = os.environ.get("FULL", "") == "1"

    # Determine which jobs need to run
    jobs = []
    for mode in ALL_MODES:
        for horizon in ALL_HORIZONS:
            for enc in ALL_ENCODERS:
                for seed in ALL_SEEDS:
                    local_path = (
                        f"outputs/latent_predictors_{mode}_h{horizon}"
                        f"/{enc}/conditioned/seed_{seed}/checkpoint.pt"
                    )
                    if not force_all and os.path.exists(local_path):
                        continue
                    jobs.append((enc, seed, horizon, mode))

    if not jobs:
        print("All checkpoints already exist locally. Nothing to do.")
        return

    t_start = time.time()
    print("=" * 70)
    print(f"DA9 Exp 3: MLP training on Modal ({len(jobs)} jobs)")
    print(f"  modes: {sorted(set(j[3] for j in jobs))}")
    print(f"  horizons: {sorted(set(j[2] for j in jobs))}")
    print(f"  encoders: {sorted(set(j[0] for j in jobs))}")
    print(f"  seeds: {sorted(set(j[1] for j in jobs))}")
    print("=" * 70)

    futures = []
    for enc, seed, horizon, mode in jobs:
        print(f"  Launching {mode}/{enc}/h{horizon}/seed={seed} ...")
        futures.append(
            (
                enc, seed, horizon, mode,
                train_mlp_horizon.spawn(enc, seed, horizon, mode),
            )
        )

    all_results = []
    for enc, seed, horizon, mode, future in futures:
        print(f"  Waiting for {mode}/{enc}/h{horizon}/seed={seed} ...")
        result = future.get()
        all_results.append(result)
        skipped = result.get("skipped", False)
        if skipped:
            status = "SKIPPED"
        else:
            tl = result["final_train_loss"]
            vl = result["final_val_loss"]
            status = f"train={tl:.6f} val={vl:.6f}"
        print(f"  {mode}/{enc}/h{horizon}/seed={seed}: {status}")

    # Summary table
    trained = [r for r in all_results if not r.get("skipped", False)]
    print(f"\n{'='*90}")
    print(
        f"{'Mode':<10} {'Encoder':<14} {'H':>3} {'Seed':>4} "
        f"{'Train Loss':>11} {'Val Loss':>10} {'Time':>6}"
    )
    print("-" * 90)
    for r in sorted(trained, key=lambda x: (x["mode"], x["encoder"], x["horizon"], x["seed"])):
        print(
            f"{r['mode']:<10} {r['encoder']:<14} {r['horizon']:>3} {r['seed']:>4} "
            f"{r['final_train_loss']:>11.6f} {r['final_val_loss']:>10.6f} "
            f"{r.get('time_s', 0):>5.0f}s"
        )

    wall_time = time.time() - t_start
    print(f"\nTotal wall time: {wall_time:.0f}s ({wall_time / 60:.1f}min)")

    # Save results
    summary_path = "artifacts/full/da9_mlp_horizon_results.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {summary_path}")
