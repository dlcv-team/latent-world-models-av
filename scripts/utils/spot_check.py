"""Spot-check validation for cached embeddings.

Loads 10 random samples per encoder, runs them through the live encoder,
and verifies the cached embeddings match within fp32 tolerance.

Usage:
  python scripts/spot_check.py --embeddings-dir artifacts/full/embeddings/

Requires GPU and nuScenes data for live forward passes.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def spot_check_encoder(
    encoder_name: str,
    pilot_name: str,
    embeddings_dir: Path,
    nusc_root: Path,
    n_samples: int = 10,
    atol: float = 1e-5,
) -> dict:
    """Spot-check a single encoder's cached embeddings against live forward pass."""
    from scripts.training.train_probe import build_encoder

    # Load cached embeddings
    embed_path = embeddings_dir / f"{pilot_name}.npz"
    if not embed_path.exists():
        return {"encoder": pilot_name, "status": "SKIP", "reason": f"File not found: {embed_path}"}

    with np.load(embed_path, allow_pickle=True) as f:
        cached_embeddings = f["embeddings"]
        image_paths = f["image_paths"]

    n = len(cached_embeddings)
    indices = random.sample(range(n), min(n_samples, n))

    # Load encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_encoder(
        encoder_name if encoder_name != "vjepa2_rep1" else "vjepa2",
        pretrained=True,
    ).to(device)
    encoder.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    max_diff = 0.0
    failures = []

    with torch.inference_mode():
        for idx in indices:
            img_path = nusc_root / str(image_paths[idx])
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            # For V-JEPA2, wrap as clip (single frame)
            if "vjepa2" in pilot_name:
                x = x.unsqueeze(1)  # (1, 1, 3, 224, 224)

            live = encoder._encode(x).cpu().numpy().flatten()
            cached = cached_embeddings[idx]

            diff = np.max(np.abs(live - cached))
            max_diff = max(max_diff, diff)

            if diff > atol:
                failures.append({
                    "index": idx,
                    "image": str(image_paths[idx]),
                    "max_diff": float(diff),
                })

    status = "PASS" if not failures else "FAIL"
    result = {
        "encoder": pilot_name,
        "status": status,
        "samples_checked": len(indices),
        "max_diff": float(max_diff),
        "atol": atol,
        "failures": failures,
    }

    icon = "+" if status == "PASS" else "x"
    print(f"  [{icon}] {pilot_name}: max_diff={max_diff:.2e} ({status})")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--nusc-root", type=Path, default=None,
                        help="nuScenes dataroot (for loading images)")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    ENCODERS = {
        "vits16":   "vit_s16",
        "dinov2":   "dino_vits14",
        "clip":     "clip_b32",
        "vqvae":    "vq_track",
    }

    print(f"Spot-checking embeddings in {args.embeddings_dir}")
    print(f"  Samples per encoder: {args.n_samples}")
    print()

    results = []
    for enc_name, pilot_name in ENCODERS.items():
        result = spot_check_encoder(
            encoder_name=enc_name,
            pilot_name=pilot_name,
            embeddings_dir=args.embeddings_dir,
            nusc_root=args.nusc_root or args.embeddings_dir.parent.parent / "nuscenes",
            n_samples=args.n_samples,
        )
        results.append(result)

    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    total = sum(1 for r in results if r["status"] != "SKIP")
    print(f"\n{passed}/{total} encoders passed spot-check")

    if any(r["status"] == "FAIL" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
