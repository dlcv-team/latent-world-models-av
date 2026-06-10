"""Extract spatial patch-token embeddings with 2x2 block pooling.

For ViT-S/16: 14x14 patches -> 2x2 pool -> 7x7=49 tokens x 384-d
For DINOv2-S/14: 16x16 patches -> 2x2 pool -> 8x8=64 tokens x 384-d

Saves NPZ with same structure as pooled embeddings but with spatial dims.

Usage::

    modal run scripts/embed_spatial_modal.py --encoder vit_s16
    modal run scripts/embed_spatial_modal.py --encoder dino_vits14
"""

from __future__ import annotations

import os
import time
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("lwm-av-embed-spatial")
    vol = modal.Volume.from_name("nuscenes-full")
else:
    app = None
    vol = None

VOL_PATH = "/vol"
EMBED_DIR = f"{VOL_PATH}/embeddings"
SPATIAL_DIR = f"{VOL_PATH}/embeddings/spatial"

# Encoder configs: native patch grid before pooling
ENCODER_CONFIGS = {
    "vit_s16": {
        "patch_grid": (14, 14),   # 224/16 = 14
        "pooled_grid": (7, 7),    # 14/2 = 7
        "native_dim": 384,
        "model_id": "vit_small_patch16_224",
        "loader": "timm",
    },
    "dino_vits14": {
        "patch_grid": (16, 16),   # 224/14 = 16
        "pooled_grid": (8, 8),    # 16/2 = 8
        "native_dim": 384,
        "model_id": "dinov2_vits14",
        "loader": "dinov2",
    },
}

if modal is not None:
    base_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.5.1",
            "torchvision==0.20.1",
            "numpy>=1.26",
            "timm>=1.0.3",
            "Pillow>=10.0",
        )
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
def extract_spatial_embeddings(encoder_name: str):
    """Extract spatial patch tokens for all frames, apply 2x2 block pooling."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ENCODER_CONFIGS[encoder_name]
    pH, pW = config["patch_grid"]
    sH, sW = config["pooled_grid"]
    native_dim = config["native_dim"]

    print(f"[spatial-embed] {encoder_name}: {pH}x{pW} patches -> {sH}x{sW} pooled, dim={native_dim}")

    # ---- Load encoder ----
    if config["loader"] == "timm":
        import timm
        import timm.data
        model = timm.create_model(config["model_id"], pretrained=True, num_classes=0)
        model = model.to(device).eval()
        cfg = timm.data.resolve_data_config({}, model=model)
        img_mean = torch.tensor(cfg["mean"], dtype=torch.float32).view(1, 3, 1, 1).to(device)
        img_std = torch.tensor(cfg["std"], dtype=torch.float32).view(1, 3, 1, 1).to(device)

        def extract_patches(x):
            """x: (B, 3, 224, 224) -> (B, pH*pW, dim)"""
            x = (x - img_mean) / img_std
            features = model.forward_features(x)  # (B, 1+pH*pW, dim) with CLS
            patches = features[:, 1:]  # drop CLS token -> (B, pH*pW, dim)
            return patches

    elif config["loader"] == "dinov2":
        model = torch.hub.load(
            "facebookresearch/dinov2", config["model_id"],
            pretrained=True, trust_repo=True, verbose=False,
        )
        model = model.to(device).eval()
        _mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        _std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        def extract_patches(x):
            """x: (B, 3, 224, 224) -> (B, pH*pW, dim)"""
            x = (x - _mean) / _std
            # DINOv2: get_intermediate_layers returns list of (B, N, D) patch tokens
            # n=1 means last layer only
            try:
                patches_list = model.get_intermediate_layers(x, n=1)
                patches = patches_list[0]  # (B, pH*pW, dim)
            except (AttributeError, TypeError):
                # Fallback: forward_features or manual extraction
                out = model.forward_features(x)
                if isinstance(out, dict):
                    patches = out.get("x_norm_patchtokens", out.get("x_norm", out.get("x"))[:, 1:])
                else:
                    patches = out[:, 1:]  # drop CLS
            return patches

    else:
        raise ValueError(f"Unknown loader: {config['loader']}")

    # ---- Setup: extract raw images from tar if needed ----
    import subprocess

    DATA_ROOT = f"{VOL_PATH}/nuscenes"
    RAW_DIR = f"{VOL_PATH}/raw"
    cam_dir = f"{DATA_ROOT}/samples/CAM_FRONT"

    os.makedirs(f"{DATA_ROOT}/samples", exist_ok=True)
    if not os.path.isdir(cam_dir) or len(os.listdir(cam_dir)) < 30000:
        tar_path = f"{RAW_DIR}/CAM_FRONT.tar"
        if os.path.exists(tar_path):
            print(f"[spatial-embed] Extracting CAM_FRONT.tar ({os.path.getsize(tar_path)/1e9:.1f} GB)...")
            subprocess.run(["tar", "xf", tar_path, "-C", f"{DATA_ROOT}/samples/"], check=True)
            n_imgs = len(os.listdir(cam_dir))
            print(f"[spatial-embed] Extracted {n_imgs} images to {cam_dir}")
        else:
            print(f"[spatial-embed] ERROR: {tar_path} not found! Cannot extract spatial embeddings.")
            print(f"[spatial-embed] Upload with: modal volume put nuscenes-full CAM_FRONT.tar /raw/CAM_FRONT.tar")
            return None
    else:
        n_imgs = len(os.listdir(cam_dir))
        print(f"[spatial-embed] CAM_FRONT already extracted ({n_imgs} images)")

    # ---- Load existing pooled embeddings to get metadata ----
    pooled_data = np.load(f"{EMBED_DIR}/{encoder_name}.npz", allow_pickle=True)
    scene_names = pooled_data["scene_names"]
    splits = pooled_data["splits"]
    steer_norms = pooled_data["steer_norms"]
    accel_norms = pooled_data["accel_norms"]
    image_paths = pooled_data["image_paths"]

    n_total = len(scene_names)
    print(f"[spatial-embed] {n_total} frames to process")

    # ---- Verify first image is accessible ----
    first_path = f"{DATA_ROOT}/{image_paths[0]}"
    if not os.path.exists(first_path):
        print(f"[spatial-embed] ERROR: First image not found at {first_path}")
        print(f"[spatial-embed] Listing {DATA_ROOT}/samples/CAM_FRONT/:")
        if os.path.isdir(cam_dir):
            files = os.listdir(cam_dir)[:3]
            print(f"  {files}")
        return None
    print(f"[spatial-embed] Image access verified: {first_path}")

    # ---- Load images and extract spatial embeddings ----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    from PIL import Image

    batch_size = 64
    all_spatial = []
    n_missing = 0

    with torch.no_grad():
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch_imgs = []

            for i in range(start, end):
                img_path = str(image_paths[i])
                full_path = f"{DATA_ROOT}/{img_path}"
                if not os.path.exists(full_path):
                    n_missing += 1
                    if n_missing <= 5:
                        print(f"  WARNING: image not found: {full_path}")
                    batch_imgs.append(torch.zeros(3, 224, 224))
                    continue
                img = Image.open(full_path).convert("RGB")
                batch_imgs.append(transform(img))

            batch_tensor = torch.stack(batch_imgs).to(device)
            patches = extract_patches(batch_tensor)  # (B, pH*pW, dim)

            # Reshape to spatial grid and apply 2x2 average pooling
            B = patches.shape[0]
            grid = patches.reshape(B, pH, pW, native_dim)
            grid = grid.permute(0, 3, 1, 2)  # (B, dim, pH, pW)
            pooled = F.avg_pool2d(grid, kernel_size=2, stride=2)  # (B, dim, sH, sW)
            pooled = pooled.permute(0, 2, 3, 1)  # (B, sH, sW, dim)
            pooled = pooled.reshape(B, sH * sW, native_dim)  # (B, S, dim)

            all_spatial.append(pooled.cpu().numpy())

            if (start // batch_size) % 50 == 0:
                print(f"  [{start + B}/{n_total}] processed")

    spatial_embeddings = np.concatenate(all_spatial, axis=0)
    print(f"[spatial-embed] Final shape: {spatial_embeddings.shape}")
    if n_missing > 0:
        print(f"[spatial-embed] WARNING: {n_missing}/{n_total} images missing!")
    else:
        print(f"[spatial-embed] All {n_total} images processed successfully")
    # Expected: (n_total, S, dim) where S=49 for ViT or S=64 for DINOv2

    # ---- Save ----
    os.makedirs(SPATIAL_DIR, exist_ok=True)
    out_path = f"{SPATIAL_DIR}/{encoder_name}_spatial.npz"
    np.savez_compressed(
        out_path,
        spatial_embeddings=spatial_embeddings,
        scene_names=scene_names,
        splits=splits,
        steer_norms=steer_norms,
        accel_norms=accel_norms,
        image_paths=image_paths,
        encoder=encoder_name,
        spatial_grid=np.array(config["pooled_grid"]),
        native_dim=native_dim,
    )
    file_size_mb = os.path.getsize(out_path) / 1e6
    print(f"[spatial-embed] Saved {out_path} ({file_size_mb:.0f} MB)")

    vol.commit()
    return {
        "encoder": encoder_name,
        "shape": list(spatial_embeddings.shape),
        "spatial_grid": config["pooled_grid"],
        "file_size_mb": round(file_size_mb, 1),
    }


# ===================================================================
# Entrypoint
# ===================================================================

def _modal_entrypoint_decorator(fn):
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_modal_entrypoint_decorator
def main(encoder: str = "vit_s16"):
    """Extract spatial embeddings for one encoder."""
    if encoder not in ENCODER_CONFIGS:
        print(f"ERROR: encoder must be one of {list(ENCODER_CONFIGS.keys())}")
        return

    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"Spatial Embedding Extraction: {encoder}")
    print(f"  Grid: {ENCODER_CONFIGS[encoder]['patch_grid']} -> {ENCODER_CONFIGS[encoder]['pooled_grid']}")
    print(f"{'='*60}")

    result = extract_spatial_embeddings.remote(encoder)
    wall = time.time() - t_start
    print(f"\nDone in {wall:.0f}s: {result}")
