"""CLI entry point for training the latent predictor (P1 pipeline).

Trains a :class:`~models.latent_pred.LatentPredictor` on pre-computed
embeddings from :class:`~data.temporal.TemporalEmbeddingDataset`.
Supports both **conditioned** (real action embeddings via
FourierActionEmbedding) and **unconditional** (zeroed action embeddings)
variants.

For encoders whose native embedding dimension differs from the
project-wide ``target_embedding_dim`` (384), a trainable
``nn.Linear(native_dim, 384, bias=False)`` adapter is included in the
optimizer, following the same pattern as ``scripts/train_probes_full.py``.

Output sidecars under ``outputs/latent_predictors/<encoder>/<variant>/seed_<seed>/``:

* ``train_log.csv`` — per-epoch (epoch, train_loss, val_loss)
* ``checkpoint.pt`` — predictor + fourier_embed + adapter state dicts
* ``provenance.json`` — encoder, variant, config, git SHA, seed

Usage
-----
    python -m training.train_latent_predictor --encoder vjepa2_rep64 --variant conditioned
    python -m training.train_latent_predictor --encoder vjepa2_rep64 --variant unconditioned
    python -m training.train_latent_predictor --encoder vit_s16 --epochs 2  # smoke test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import load_canonical
from data.temporal import TemporalEmbeddingDataset
from models.fourier_embed import FourierActionEmbedding
from models.latent_pred import LatentPredictor, train_latent_predictor

# Native embedding dimensions per encoder. Matches
# configs/canonical.yaml::encoders::*::output_dim_native and
# scripts/train_probes_full.py::NATIVE_DIMS.
NATIVE_DIMS: dict[str, int] = {
    "vit_s16": 384,
    "dino_vits14": 384,
    "clip_b32": 512,
    "vq_track": 256,
    "vjepa2_rep64": 1024,
    "vjepa2_rep1": 1024,
}


def _git_sha() -> str:
    """Return short git SHA or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the latent predictor on pre-computed embeddings."
    )
    parser.add_argument(
        "--encoder",
        required=True,
        choices=sorted(NATIVE_DIMS.keys()),
        help="Encoder whose embeddings to train on.",
    )
    parser.add_argument(
        "--variant",
        default="conditioned",
        choices=["conditioned", "unconditioned"],
        help="Conditioned uses real action embeddings; unconditioned zeros them.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/latent_predictors"),
        help="Root output directory (default: outputs/latent_predictors).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override canonical epoch count.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override canonical batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: auto-detect cuda/cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    cfg = load_canonical()
    lp_cfg = cfg.latent_predictor()
    target_dim = cfg.target_embedding_dim
    native_dim = NATIVE_DIMS[args.encoder]
    needs_adapter = native_dim != target_dim

    epochs = args.epochs if args.epochs is not None else int(lp_cfg["epochs"])
    batch_size = (
        args.batch_size if args.batch_size is not None else int(lp_cfg["batch_size"])
    )
    seed = args.seed
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    torch.manual_seed(seed)

    out_dir = args.output_root / args.encoder / args.variant / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_lp] encoder={args.encoder} variant={args.variant}")
    print(f"[train_lp] native_dim={native_dim} target_dim={target_dim} adapter={needs_adapter}")
    print(f"[train_lp] out_dir={out_dir} device={device} seed={seed}")
    print(f"[train_lp] epochs={epochs} batch_size={batch_size}")

    # Build datasets and loaders
    train_ds = TemporalEmbeddingDataset.from_encoder(
        args.encoder, split="train", horizon=int(lp_cfg["prediction_horizon"])
    )
    val_ds = TemporalEmbeddingDataset.from_encoder(
        args.encoder, split="val", horizon=int(lp_cfg["prediction_horizon"])
    )
    print(f"[train_lp] train={len(train_ds)} val={len(val_ds)} sequences")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Build models
    if needs_adapter:
        adapter: nn.Module = nn.Linear(native_dim, target_dim, bias=False).to(device)
    else:
        adapter = nn.Identity().to(device)

    fourier_embed = FourierActionEmbedding.from_canonical(cfg).to(device)
    predictor = LatentPredictor.from_canonical(cfg).to(device)

    # Optimizer: predictor + fourier_embed + adapter (when applicable)
    params = list(predictor.parameters()) + list(fourier_embed.parameters())
    if needs_adapter:
        params += list(adapter.parameters())

    optimizer = torch.optim.Adam(
        params, lr=float(lp_cfg["learning_rate"])
    )

    # Train
    history = train_latent_predictor(
        predictor=predictor,
        fourier_embed=fourier_embed,
        adapter=adapter,
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=epochs,
        variant=args.variant,
        val_loader=val_loader,
        device=device,
        log_csv_path=out_dir / "train_log.csv",
    )

    # Save checkpoint
    torch.save(
        {
            "predictor_state_dict": predictor.state_dict(),
            "fourier_embed_state_dict": fourier_embed.state_dict(),
            "adapter_state_dict": adapter.state_dict() if needs_adapter else None,
            "encoder_name": args.encoder,
            "variant": args.variant,
            "seed": seed,
            "epochs": epochs,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": (
                history["val_loss"][-1]
                if history["val_loss"] is not None
                else None
            ),
        },
        out_dir / "checkpoint.pt",
    )

    # Write provenance
    provenance = {
        "encoder_name": args.encoder,
        "variant": args.variant,
        "native_dim": native_dim,
        "target_dim": target_dim,
        "needs_adapter": needs_adapter,
        "prediction_horizon": int(lp_cfg["prediction_horizon"]),
        "fourier_n_frequencies": int(lp_cfg["fourier_action_embed"]["n_frequencies"]),
        "fourier_base": float(lp_cfg["fourier_action_embed"]["base"]),
        "fourier_out_dim": int(lp_cfg["fourier_action_embed"]["out_dim"]),
        "learning_rate": float(lp_cfg["learning_rate"]),
        "batch_size": batch_size,
        "epochs": epochs,
        "loss": lp_cfg["loss"],
        "seed": seed,
        "action_alignment": (
            "action_t is the nearest CAN bus sample to frame t's timestamp "
            "(typically <5 ms offset at 100 Hz CAN rate); approximately "
            "start-of-interval for the 0.5 s keyframe transition at 2 Hz"
        ),
        "git_sha": _git_sha(),
        "torch_version": torch.__version__,
        "source": "training/train_latent_predictor.py",
    }
    with (out_dir / "provenance.json").open("w") as fh:
        json.dump(provenance, fh, indent=2)

    final_train = history["train_loss"][-1]
    final_val = (
        history["val_loss"][-1] if history["val_loss"] is not None else None
    )
    print(f"[train_lp] done. train_loss={final_train:.6f} val_loss={final_val}")
    print(f"[train_lp] wrote sidecars under {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
