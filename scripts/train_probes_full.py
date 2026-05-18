"""Train linear probes on full-dataset embeddings via Modal.

Reads pre-computed embeddings from the Modal volume, trains ActionProbe
with canonical params (batch 256, 50 epochs, no early stopping, seeds [0,1,2]).

Usage:
  modal run scripts/train_probes_full.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

app = modal.App("lwm-av-full-probes")
vol = modal.Volume.from_name("nuscenes-full")

VOL_PATH = "/vol"
EMBED_DIR = f"{VOL_PATH}/embeddings"
PROBE_DIR = f"{VOL_PATH}/probes"

# Canonical probe training params (from configs/canonical.yaml)
CANONICAL = {
    "batch_size": 256,
    "epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "hidden_dim": 256,
    "dropout": 0.1,
    "output_dim": 2,
    "target_dim": 384,
    "seeds": [0, 1, 2],
}

ENCODER_NAMES = ["vit_s16", "dino_vits14", "clip_b32", "vq_track", "vjepa2_rep64", "vjepa2_rep1"]

# Native dims for adapter construction
NATIVE_DIMS = {
    "vit_s16": 384,
    "dino_vits14": 384,
    "clip_b32": 512,
    "vq_track": 256,
    "vjepa2_rep64": 1024,
    "vjepa2_rep1": 1024,
}

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy>=1.26", "tqdm")
)


@app.function(
    volumes={VOL_PATH: vol},
    image=base_image,
    gpu="T4",
    timeout=3600,
    memory=16384,
)
def train_probe_for_encoder(encoder_name: str, seed: int):
    """Train a single probe: one encoder, one seed, canonical params."""
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    print(f"[probe] encoder={encoder_name}, seed={seed}")

    # Load embeddings
    embed_path = f"{EMBED_DIR}/{encoder_name}.npz"
    with np.load(embed_path, allow_pickle=True) as f:
        embeddings = f["embeddings"]
        splits = f["splits"]
        steer_norms = f["steer_norms"]
        accel_norms = f["accel_norms"]
        scene_names = f["scene_names"]

    native_dim = NATIVE_DIMS[encoder_name]
    target_dim = CANONICAL["target_dim"]
    needs_adapter = (native_dim != target_dim)

    # Split data
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    X_train = torch.tensor(embeddings[train_mask], dtype=torch.float32)
    X_val = torch.tensor(embeddings[val_mask], dtype=torch.float32)
    X_test = torch.tensor(embeddings[test_mask], dtype=torch.float32)

    actions = np.stack([steer_norms, accel_norms], axis=1)
    y_train = torch.tensor(actions[train_mask], dtype=torch.float32)
    y_val = torch.tensor(actions[val_mask], dtype=torch.float32)
    y_test = torch.tensor(actions[test_mask], dtype=torch.float32)

    print(f"[probe] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"[probe] Embedding dim: {native_dim}, adapter: {needs_adapter}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build adapter + probe
    if needs_adapter:
        adapter = nn.Linear(native_dim, target_dim, bias=False).to(device)
    else:
        adapter = nn.Identity().to(device)

    # Probe: Linear(384,256) -> GELU -> Dropout(0.1) -> Linear(256,2)
    probe = nn.Sequential(
        nn.Linear(target_dim, CANONICAL["hidden_dim"]),
        nn.GELU(),
        nn.Dropout(CANONICAL["dropout"]),
        nn.Linear(CANONICAL["hidden_dim"], CANONICAL["output_dim"]),
    ).to(device)

    # Optimizer over adapter + probe params
    params = list(probe.parameters())
    if needs_adapter:
        params += list(adapter.parameters())
    optimizer = torch.optim.Adam(
        params, lr=CANONICAL["learning_rate"], weight_decay=CANONICAL["weight_decay"]
    )
    criterion = nn.MSELoss()

    # DataLoaders
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=CANONICAL["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CANONICAL["batch_size"], shuffle=False)

    # Training loop — canonical: 50 epochs, NO early stopping
    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()

    for epoch in range(CANONICAL["epochs"]):
        # Train
        probe.train()
        if needs_adapter:
            adapter.train()
        train_loss_sum = 0.0
        train_n = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            z = adapter(X_batch)
            pred = probe(z)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(X_batch)
            train_n += len(X_batch)

        # Validate
        probe.eval()
        if needs_adapter:
            adapter.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                z = adapter(X_batch)
                pred = probe(z)
                loss = criterion(pred, y_batch)
                val_loss_sum += loss.item() * len(X_batch)
                val_n += len(X_batch)

        train_loss = train_loss_sum / train_n
        val_loss = val_loss_sum / val_n
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"[probe] Epoch {epoch+1}/{CANONICAL['epochs']}: "
                  f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    # Test evaluation
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=CANONICAL["batch_size"], shuffle=False)

    # Per-scene test RMSE
    test_scene_names = scene_names[test_mask]
    all_preds = []
    all_targets = []

    probe.eval()
    if needs_adapter:
        adapter.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            z = adapter(X_batch)
            pred = probe(z)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    test_mse = float(np.mean((preds - targets) ** 2))
    test_rmse = float(np.sqrt(test_mse))

    elapsed = time.time() - t0
    print(f"[probe] {encoder_name}/seed={seed}: test_rmse={test_rmse:.6f}, time={elapsed:.1f}s")

    # Save results
    out_dir = f"{PROBE_DIR}/{encoder_name}/seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)

    # Save probe checkpoint
    checkpoint = {
        "probe_state_dict": probe.state_dict(),
        "adapter_state_dict": adapter.state_dict() if needs_adapter else None,
        "encoder_name": encoder_name,
        "seed": seed,
        "test_rmse": test_rmse,
        "test_mse": test_mse,
        "epochs": CANONICAL["epochs"],
        "history": history,
    }
    torch.save(checkpoint, f"{out_dir}/checkpoint.pt")

    # Save per-scene RMSE
    per_scene = {}
    errors_sq = (preds - targets) ** 2  # (N, 2)
    for i, scene in enumerate(test_scene_names):
        scene = str(scene)
        if scene not in per_scene:
            per_scene[scene] = {"steer_sq": [], "accel_sq": []}
        per_scene[scene]["steer_sq"].append(float(errors_sq[i, 0]))
        per_scene[scene]["accel_sq"].append(float(errors_sq[i, 1]))

    scene_rmse = {}
    for scene, errs in per_scene.items():
        scene_rmse[scene] = {
            "steer_rmse": float(np.sqrt(np.mean(errs["steer_sq"]))),
            "accel_rmse": float(np.sqrt(np.mean(errs["accel_sq"]))),
            "n": len(errs["steer_sq"]),
        }

    with open(f"{out_dir}/per_scene_rmse.json", "w") as f:
        json.dump(scene_rmse, f, indent=2)

    # Save training log
    with open(f"{out_dir}/train_log.json", "w") as f:
        json.dump(history, f)

    vol.commit()

    return {
        "encoder": encoder_name,
        "seed": seed,
        "test_rmse": test_rmse,
        "test_mse": test_mse,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "time_s": elapsed,
    }


@app.local_entrypoint()
def main():
    """Train probes for all encoders × seeds in parallel."""
    t_start = time.time()
    print("=" * 60)
    print("Full-Dataset Probe Training (Canonical Params)")
    print(f"  batch_size={CANONICAL['batch_size']}, epochs={CANONICAL['epochs']}")
    print(f"  early_stopping=False, seeds={CANONICAL['seeds']}")
    print("=" * 60)

    # Launch all encoder×seed jobs in parallel
    futures = []
    for enc_name in ENCODER_NAMES:
        for seed in CANONICAL["seeds"]:
            print(f"  Launching {enc_name}/seed={seed} ...")
            futures.append((enc_name, seed, train_probe_for_encoder.spawn(enc_name, seed)))

    # Collect results
    all_results = []
    for enc_name, seed, future in futures:
        print(f"  Waiting for {enc_name}/seed={seed} ...")
        result = future.get()
        all_results.append(result)
        print(f"  {enc_name}/seed={seed}: test_rmse={result['test_rmse']:.6f}")

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Encoder':<16} {'Seed':>4} {'Test RMSE':>10} {'Train Loss':>11} {'Val Loss':>10}")
    print("-" * 60)
    for r in sorted(all_results, key=lambda x: (x["encoder"], x["seed"])):
        print(f"{r['encoder']:<16} {r['seed']:>4} {r['test_rmse']:>10.6f} "
              f"{r['final_train_loss']:>11.6f} {r['final_val_loss']:>10.6f}")

    wall_time = time.time() - t_start
    print(f"\nTotal wall time: {wall_time:.0f}s ({wall_time/60:.1f}min)")

    # Save aggregate results
    summary_path = "artifacts/full/probe_results.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {summary_path}")
