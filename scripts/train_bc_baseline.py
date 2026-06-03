"""Train the BC baseline head on pre-computed embeddings (A19.10).

Loads pre-computed 384-d embeddings from ``data.embeddings``, trains a
:class:`~models.bc_baseline.BCBaseline` head with early stopping per
``configs/canonical.yaml::bc_baseline``, evaluates on the held-out test
set, and emits a single-row summary CSV.

The BC head is architecturally identical to the ActionProbe
(``Linear(384, 256) -> GELU -> Dropout(0.1) -> Linear(256, 2)``), but
trains with early stopping (patience=10) instead of fixed epochs.

Output layout under ``<output-root>/<encoder>/seed_<seed>/``::

    train_log.csv      -- per-epoch (epoch, train_loss, val_loss)
    checkpoint.pt      -- bc_model + adapter state dicts
    provenance.json    -- encoder, config, git SHA, seed
    bc_baseline_row.csv -- single-row summary with RMSE + hyperparams

Usage
-----
    python scripts/train_bc_baseline.py --encoder vjepa2_rep64
    python scripts/train_bc_baseline.py --encoder vjepa2_rep64 --epochs 2  # smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Native embedding dimensions per encoder. Matches
# configs/canonical.yaml::encoders::*::output_dim_native.
# "rep64" refers to the V-JEPA2 checkpoint variant facebook/vjepa2-vitl-fpc64-256
# (fpc64 = "frames per clip 64" in the pre-training recipe). Our canonical input
# is 16 frames (canonical.yaml::dataset::vjepa2::clip_frames: 16); the model
# interpolates temporal positional embeddings to handle the mismatch.
NATIVE_DIMS: dict[str, int] = {
    "vit_s16": 384,
    "dino_vits14": 384,
    "clip_b32": 512,
    "vq_track": 256,
    "vjepa2_rep64": 1024,
    "vjepa2_rep1": 1024,
}


def _git_sha() -> str:
    """Return short git SHA or ``'unknown'``."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train BC baseline on pre-computed embeddings."
    )
    parser.add_argument(
        "--encoder",
        required=True,
        choices=sorted(NATIVE_DIMS.keys()),
        help="Encoder whose embeddings to train on.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/bc"),
        help="Root output directory (default: outputs/bc).",
    )
    parser.add_argument(
        "--embed-dir",
        type=Path,
        default=None,
        help="Override embedding directory (default: auto-detect via data.embeddings).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override canonical epoch count (default: 50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override canonical batch size (default: 256).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: auto-detect cuda/cpu).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    # Add project root to path for imports
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from config import load_canonical
    from data.embeddings import load_encoder_embedding
    from models.bc_baseline import BCBaseline, train_bc

    cfg = load_canonical()
    bc_cfg = cfg.bc()
    target_dim = cfg.target_embedding_dim
    native_dim = NATIVE_DIMS[args.encoder]
    needs_adapter = native_dim != target_dim

    epochs = args.epochs if args.epochs is not None else int(bc_cfg["epochs"])
    batch_size = (
        args.batch_size if args.batch_size is not None else int(bc_cfg["batch_size"])
    )
    patience = int(bc_cfg["early_stopping_patience"])
    lr = float(bc_cfg["learning_rate"])
    wd = float(bc_cfg["weight_decay"])
    seed = args.seed

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = args.output_root / args.encoder / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bc] encoder={args.encoder} seed={seed}")
    print(f"[bc] native_dim={native_dim} target_dim={target_dim} adapter={needs_adapter}")
    print(f"[bc] epochs={epochs} batch_size={batch_size} patience={patience} lr={lr}")
    print(f"[bc] out_dir={out_dir} device={device}")

    # ------------------------------------------------------------------
    # Load pre-computed embeddings
    # ------------------------------------------------------------------
    data = load_encoder_embedding(args.encoder, directory=args.embed_dir)
    embeddings = data["embeddings"]  # (N, native_dim)
    splits = data["splits"]          # (N,) str: "train"/"val"/"test"
    steer_norms = data["steer_norms"]  # (N,)
    accel_norms = data["accel_norms"]  # (N,)
    scene_names = data["scene_names"]  # (N,)

    # Split
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

    print(f"[bc] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ------------------------------------------------------------------
    # Build model + adapter
    # ------------------------------------------------------------------
    if needs_adapter:
        adapter: nn.Module = nn.Linear(native_dim, target_dim, bias=False).to(device)
    else:
        adapter = nn.Identity().to(device)

    bc_model = BCBaseline.from_canonical(cfg).to(device)

    # Optimizer: BC head + adapter (when applicable)
    params = list(bc_model.parameters())
    if needs_adapter:
        params += list(adapter.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    # ------------------------------------------------------------------
    # Adapt embeddings through adapter, then train BC on Identity encoder
    # ------------------------------------------------------------------
    # For the fast path, we pre-project all embeddings through the adapter
    # during training. Since train_bc() expects an encoder argument, we use
    # a combined adapter+identity approach: train_bc gets nn.Identity() as
    # the encoder, and we handle the adapter in the training loop ourselves.
    #
    # However, train_bc's _epoch_loss expects encoder(batch) -> embeddings.
    # If we pass pre-adapted tensors with nn.Identity, it just passes them
    # through. But when needs_adapter=True, we need the adapter in the loop.
    #
    # Simplest correct approach: use the direct training loop (same as
    # train_probes_full.py) rather than train_bc(), since train_bc()
    # doesn't save/restore adapter state separately from the BC head,
    # and we need the adapter checkpoint for downstream evaluation.

    # DataLoaders
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Training loop with early stopping
    # ------------------------------------------------------------------
    criterion = nn.MSELoss()
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    patience_counter = 0

    log_csv_path = out_dir / "train_log.csv"
    log_file = open(log_csv_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_loss"])

    try:
        for epoch in range(epochs):
            # Train
            bc_model.train()
            if needs_adapter:
                adapter.train()
            train_loss_sum = 0.0
            train_n = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                z = adapter(X_batch)
                pred = bc_model(z)
                loss = criterion(pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * len(X_batch)
                train_n += len(X_batch)

            # Validate
            bc_model.eval()
            if needs_adapter:
                adapter.eval()
            val_loss_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    z = adapter(X_batch)
                    pred = bc_model(z)
                    loss = criterion(pred, y_batch)
                    val_loss_sum += loss.item() * len(X_batch)
                    val_n += len(X_batch)

            train_loss = train_loss_sum / train_n
            val_loss = val_loss_sum / val_n
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            log_writer.writerow([epoch + 1, f"{train_loss:.8f}", f"{val_loss:.8f}"])
            log_file.flush()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"[bc] Epoch {epoch + 1}/{epochs}: "
                    f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
                )

            # Early stopping
            if val_loss < best_val_loss - 1e-12:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = {
                    "bc": {k: v.clone() for k, v in bc_model.state_dict().items()},
                    "adapter": (
                        {k: v.clone() for k, v in adapter.state_dict().items()}
                        if needs_adapter
                        else None
                    ),
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[bc] Early stopping at epoch {epoch + 1} (best={best_epoch})")
                    break
    finally:
        log_file.close()

    stopped_early = patience_counter >= patience
    epochs_run = epoch + 1

    # Restore best weights
    bc_model.load_state_dict(best_state["bc"])
    if needs_adapter and best_state["adapter"] is not None:
        adapter.load_state_dict(best_state["adapter"])

    print(f"[bc] Training complete. best_epoch={best_epoch} best_val_loss={best_val_loss:.6f}")

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    bc_model.eval()
    if needs_adapter:
        adapter.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            z = adapter(X_batch)
            pred = bc_model(z)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # Per-channel RMSE
    steer_rmse = float(np.sqrt(np.mean((preds[:, 0] - targets[:, 0]) ** 2)))
    accel_rmse = float(np.sqrt(np.mean((preds[:, 1] - targets[:, 1]) ** 2)))

    # Count unique test scenes
    test_scene_names = scene_names[test_mask]
    n_test_scenes = len(set(test_scene_names))

    print(f"[bc] Test: steer_rmse={steer_rmse:.6f}, accel_rmse={accel_rmse:.6f}")
    print(f"[bc] Test scenes: {n_test_scenes}")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    # Checkpoint
    checkpoint = {
        "bc_state_dict": best_state["bc"],
        "adapter_state_dict": best_state["adapter"],
        "encoder_name": args.encoder,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "stopped_early": stopped_early,
        "best_val_loss": best_val_loss,
        "steer_rmse": steer_rmse,
        "accel_rmse": accel_rmse,
        "history": history,
    }
    torch.save(checkpoint, out_dir / "checkpoint.pt")
    print(f"[bc] Saved checkpoint to {out_dir / 'checkpoint.pt'}")

    # Provenance
    provenance = {
        "encoder_name": args.encoder,
        "seed": seed,
        "git_sha": _git_sha(),
        "native_dim": native_dim,
        "target_dim": target_dim,
        "needs_adapter": needs_adapter,
        "lr": lr,
        "weight_decay": wd,
        "batch_size": batch_size,
        "epochs_max": epochs,
        "epochs_run": epochs_run,
        "early_stopping_patience": patience,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
    }
    with open(out_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
        f.write("\n")

    # Summary CSV: bc_baseline_row.csv
    # This is the deliverable for task C3/A19.10 -- single-row summary
    # with all hyperparams and test RMSE.
    summary_csv_path = out_dir / "bc_baseline_row.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "encoder",
                "lr",
                "wd",
                "batch",
                "epochs_run",
                "early_stop_epoch",
                "seed",
                "steer_rmse",
                "accel_rmse",
                "n_test_scenes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "encoder": args.encoder,
                "lr": lr,
                "wd": wd,
                "batch": batch_size,
                "epochs_run": epochs_run,
                "early_stop_epoch": best_epoch if stopped_early else "",
                "seed": seed,
                "steer_rmse": f"{steer_rmse:.6f}",
                "accel_rmse": f"{accel_rmse:.6f}",
                "n_test_scenes": n_test_scenes,
            },
        )
    print(f"[bc] Saved summary to {summary_csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
