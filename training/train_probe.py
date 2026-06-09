"""CLI entry point for training an ActionProbe on top of a frozen encoder.

Builds the encoder/probe pair from the canonical config, trains the probe
via :func:`models.probe.train_probe`, and writes a fixed sidecar set
under ``outputs/probes/<encoder>/``:

* ``train_log.csv`` — per-epoch (epoch, train_loss, val_loss)
* ``checkpoint.pt`` — probe state dict + minimal training metadata
* ``per_scene_rmse.csv`` — one row per (encoder, scene, scenario, fold,
  steer_rmse, accel_rmse, n); single fold for this CLI (fold_id=0)
* ``provenance.json`` — encoder identity, config + manifest SHAs, git
  SHA, and (for VQ) the ``fallback_caveat`` string

The CLI exists as a reproducibility / strict-closure artifact. M1's
headline numbers for the milestone come from the pre-existing pilot
artifacts (adopted by ``scripts/adopt_pilot_artifacts.py``); this CLI is
what we run when we need a fresh canonical re-run for any encoder.

Usage
-----
    python -m training.train_probe --encoder vits16
    python -m training.train_probe --encoder vjepa2 --batch-size 4

The ``--no-pretrained`` flag swaps every wrapper into its random-init
test-mode path and is the escape hatch used by the smoke tests in
``tests/test_training_train_probe.py``.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import (
    CanonicalConfig,
    load_canonical,
    resolve_action_labels_path,
)
from models.probe import ActionProbe, train_probe


# ---------------------------------------------------------------------------
# Encoder registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncoderSpec:
    """Static metadata for an encoder the CLI knows how to drive."""

    module_path: str
    class_name: str
    mode: str  # "single_frame" or "clip"
    pilot_name: str  # name used in pilot artifacts + canonical output paths
    pretrained_weights_id: str  # upstream weights identifier (timm tag,
    #                            # torch.hub name, HF repo, or open_clip
    #                            # pretrained tag)


ENCODER_REGISTRY: dict[str, EncoderSpec] = {
    "vits16": EncoderSpec(
        "encoders.vits16", "ViTS16Wrapper", "single_frame", "vit_s16",
        "vit_small_patch16_224",
    ),
    "dinov2": EncoderSpec(
        "encoders.dinov2", "DINOv2S14Wrapper", "single_frame", "dino_vits14",
        "dinov2_vits14",
    ),
    "clip": EncoderSpec(
        "encoders.clip_enc", "CLIPB32Wrapper", "single_frame", "clip_b32",
        "openai",
    ),
    "vqvae": EncoderSpec(
        "encoders.vqvae", "VQVAEWrapper", "single_frame", "vq_track",
        "vqgan-imagenet-f16-16384",
    ),
    # "rep64" = checkpoint variant fpc64 (pre-trained on 64-frame clips),
    # NOT our input frame count (16 frames per canonical.yaml::clip_frames).
    "vjepa2": EncoderSpec(
        "encoders.vjepa2", "VJEPA2Wrapper", "clip", "vjepa2_rep64",
        "facebook/vjepa2-vitl-fpc64-256",
    ),
}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_encoder(name: str, pretrained: bool) -> nn.Module:
    """Construct an encoder wrapper by registry name.

    ``pretrained=False`` triggers the random-init test path on every
    wrapper. For CLIP the wrapper takes ``pretrained: Optional[str]``
    rather than a bool, so we map ``True → "openai"`` and ``False → None``.
    """
    if name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder {name!r}; choices are "
            f"{sorted(ENCODER_REGISTRY)}"
        )
    spec = ENCODER_REGISTRY[name]
    module = importlib.import_module(spec.module_path)
    cls = getattr(module, spec.class_name)

    if name == "clip":
        return cls(pretrained="openai" if pretrained else None)
    return cls(pretrained=pretrained)


def build_optimizer(
    probe: ActionProbe,
    encoder: nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Adam over probe params plus any encoder-adapter (trainable) params.

    This mirrors the canonical pattern documented on
    :func:`models.probe.train_probe`.
    """
    params: list[nn.Parameter] = list(probe.parameters())
    if hasattr(encoder, "trainable_parameters"):
        params += list(encoder.trainable_parameters())
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def build_loaders(
    cfg: CanonicalConfig,
    mode: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from :class:`NuScenesFrameDataset`.

    Imported lazily so unit tests that monkey-patch
    ``training.train_probe.NuScenesFrameDataset`` to an in-memory stub
    don't trip on nuScenes import-time side effects.
    """
    from data.dataset import NuScenesFrameDataset  # noqa: WPS433 (lazy)

    train_ds = NuScenesFrameDataset(split="p0_train", mode=mode)
    val_ds = NuScenesFrameDataset(split="p0_val", mode=mode)
    test_ds = NuScenesFrameDataset(split="p0_test", mode=mode)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Test-pass writer
# ---------------------------------------------------------------------------


def write_per_scene_rmse(
    encoder: nn.Module,
    probe: ActionProbe,
    test_loader: Iterable[dict[str, Any]],
    out_path: Path,
    pilot_name: str,
    device: Optional[torch.device] = None,
) -> None:
    """Run one inference pass and write per-scene aggregate RMSE.

    Columns: ``encoder, scene_name, scenario, fold_id, steer_rmse,
    accel_rmse, n``. ``fold_id`` is always 0 for this CLI (single split
    run); the multi-fold CV is what produced the pilot artifacts and
    can be re-implemented later if we ever need to reproduce them
    end-to-end. ``scenario`` is left empty here — it's added by
    downstream scenario classification (see ``evaluation/metrics.py``).
    """
    encoder.eval()
    probe.eval()

    # scene_name -> list of (steer_err^2, accel_err^2)
    per_scene_errors: dict[str, list[tuple[float, float]]] = {}

    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"]
            actions = batch["actions"]
            scene_names = batch["scene_name"]
            if device is not None:
                image = image.to(device)
                actions = actions.to(device)

            embedding = encoder(image)
            pred = probe(embedding)
            err_sq = (pred - actions) ** 2  # (B, 2)
            for i, scene in enumerate(scene_names):
                per_scene_errors.setdefault(scene, []).append(
                    (float(err_sq[i, 0].item()), float(err_sq[i, 1].item()))
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["encoder", "scene_name", "scenario", "fold_id", "steer_rmse", "accel_rmse", "n"]
        )
        for scene in sorted(per_scene_errors):
            samples = per_scene_errors[scene]
            n = len(samples)
            steer_mse = sum(s for s, _ in samples) / n
            accel_mse = sum(a for _, a in samples) / n
            writer.writerow(
                [
                    pilot_name,
                    scene,
                    "",  # scenario added by downstream classification
                    0,
                    f"{steer_mse ** 0.5:.10g}",
                    f"{accel_mse ** 0.5:.10g}",
                    n,
                ]
            )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Best-effort git rev-parse; returns 'unknown' if anything goes wrong."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _action_labels_sha256(cfg: CanonicalConfig) -> str:
    """Return the configured action-labels SHA256 (does not re-hash on disk)."""
    return str(cfg.raw["dataset"]["action_labels"]["sha256"])


def write_provenance(
    encoder: nn.Module,
    encoder_name: str,
    out_path: Path,
    cfg: CanonicalConfig,
    pretrained: bool,
    seed: int,
) -> None:
    """Emit ``provenance.json`` for downstream figure scripts.

    The ``fallback_caveat`` field is empty for encoders without a fallback;
    figure scripts read this field by data, not by hardcoded literals
    (per A11 task spec).
    """
    spec = ENCODER_REGISTRY[encoder_name]
    caveat = getattr(encoder, "fallback_caveat", "")
    provenance = {
        "encoder_name": encoder_name,
        "pilot_name": spec.pilot_name,
        "wrapper_class": spec.class_name,
        "pretrained_weights_id": spec.pretrained_weights_id,
        "pretrained": bool(pretrained),
        "config_version": cfg.version,
        "manifest_sha256": cfg.manifest_sha256,
        "action_labels_sha256": _action_labels_sha256(cfg),
        "git_sha": _git_sha(),
        "fallback_caveat": str(caveat),
        "torch_version": torch.__version__,
        "seed": int(seed),
        "source": "training/train_probe.py",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(provenance, fh, indent=2, sort_keys=True)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_probe",
        description=(
            "Train an ActionProbe on top of a frozen encoder using "
            "canonical hyperparameters."
        ),
    )
    parser.add_argument(
        "--encoder",
        required=True,
        choices=sorted(ENCODER_REGISTRY),
        help="Encoder wrapper to use.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/probes"),
        help="Per-encoder sidecars go under <output-root>/<pilot_name>/.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override canonical probe.epochs. Default: from canonical config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override canonical probe.batch_size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Default: cuda if available else cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Default: canonical.reproducibility.global_seed.",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=True,
        help="(Default) Load pretrained encoder weights.",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Use random-init weights (test/smoke mode).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    cfg = load_canonical()
    probe_cfg = cfg.probe()
    epochs = args.epochs if args.epochs is not None else int(probe_cfg["epochs"])
    batch_size = (
        args.batch_size if args.batch_size is not None else int(probe_cfg["batch_size"])
    )
    seed = args.seed if args.seed is not None else cfg.global_seed
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    torch.manual_seed(seed)

    spec = ENCODER_REGISTRY[args.encoder]
    out_dir: Path = args.output_root / spec.pilot_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_probe] encoder={args.encoder} pilot_name={spec.pilot_name}")
    print(f"[train_probe] out_dir={out_dir} device={device} seed={seed}")
    print(f"[train_probe] epochs={epochs} batch_size={batch_size}")

    encoder = build_encoder(args.encoder, pretrained=args.pretrained).to(device)
    probe = ActionProbe.from_canonical(cfg).to(device)

    train_loader, val_loader, test_loader = build_loaders(
        cfg=cfg,
        mode=spec.mode,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )

    optimizer = build_optimizer(
        probe=probe,
        encoder=encoder,
        lr=float(probe_cfg["learning_rate"]),
        weight_decay=float(probe_cfg["weight_decay"]),
    )

    history = train_probe(
        encoder=encoder,
        probe=probe,
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=epochs,
        val_loader=val_loader,
        device=device,
        log_csv_path=out_dir / "train_log.csv",
    )

    # Save checkpoint (probe weights + adapter weights + minimal metadata).
    # Encoder backbone weights are deterministic from `pretrained=True`, but
    # the adapter is trained, so we save it for downstream tasks.
    torch.save(
        {
            "probe_state_dict": probe.state_dict(),
            "adapter_state_dict": encoder.adapter.state_dict() if not isinstance(encoder.adapter, nn.Identity) else None,
            "encoder_name": args.encoder,
            "pilot_name": spec.pilot_name,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": (
                history["val_loss"][-1] if history["val_loss"] is not None else None
            ),
            "epochs": epochs,
            "seed": seed,
        },
        out_dir / "checkpoint.pt",
    )

    write_per_scene_rmse(
        encoder=encoder,
        probe=probe,
        test_loader=test_loader,
        out_path=out_dir / "per_scene_rmse.csv",
        pilot_name=spec.pilot_name,
        device=device,
    )

    write_provenance(
        encoder=encoder,
        encoder_name=args.encoder,
        out_path=out_dir / "provenance.json",
        cfg=cfg,
        pretrained=args.pretrained,
        seed=seed,
    )

    print(f"[train_probe] wrote sidecars under {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
