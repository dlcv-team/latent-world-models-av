"""Extract SD VAE latent grids for all nuScenes CAM_FRONT frames.

Encodes 256x256 driving images through the Stable Diffusion VAE encoder
to produce 32x32x4 latent grids. These can be patchified to 8x8=64 tokens
of 64-d for DiT training, matching the production world-model pipeline
(GAIA-1, Cosmos, etc.).

Usage::

    modal run scripts/embed_vae_modal.py
    modal run scripts/embed_vae_modal.py --smoke       # 100 frames only
    modal run scripts/embed_vae_modal.py --validate    # decode 8 frames to verify round-trip
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
    app = modal.App("lwm-av-embed-vae")
    vol = modal.Volume.from_name("nuscenes-full")
else:
    app = None
    vol = None

VOL_PATH = "/vol"
EMBED_DIR = f"{VOL_PATH}/embeddings"
SPATIAL_DIR = f"{VOL_PATH}/embeddings/spatial"

# SD VAE config
VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"
TARGET_SIZE = 256  # 256x256 -> 32x32x4 latent grid
SCALING_FACTOR = 0.18215  # from vae.config.scaling_factor
BATCH_SIZE = 32  # VAE is memory-heavier than ViT/DINO

if modal is not None:
    base_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.5.1",
            "torchvision==0.20.1",
            "numpy>=1.26",
            "Pillow>=10.0",
            "diffusers>=0.27",
            "accelerate",
            "transformers>=4.50",
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
def extract_vae_latents(smoke: bool = False, validate: bool = True):
    """Extract SD VAE latent grids for all nuScenes CAM_FRONT frames."""
    import numpy as np
    import subprocess
    import torch
    from torchvision import transforms
    from PIL import Image
    from diffusers import AutoencoderKL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[vae-embed] device={device}, smoke={smoke}, validate={validate}")

    # ---- Load SD VAE ----
    print(f"[vae-embed] Loading {VAE_MODEL_ID}...")
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID)
    vae = vae.to(device).eval()
    print(f"[vae-embed] VAE loaded. scaling_factor={vae.config.scaling_factor}")

    # ---- Setup: extract raw images from tar if needed ----
    DATA_ROOT = f"{VOL_PATH}/nuscenes"
    RAW_DIR = f"{VOL_PATH}/raw"
    cam_dir = f"{DATA_ROOT}/samples/CAM_FRONT"

    os.makedirs(f"{DATA_ROOT}/samples", exist_ok=True)
    if not os.path.isdir(cam_dir) or len(os.listdir(cam_dir)) < 30000:
        tar_path = f"{RAW_DIR}/CAM_FRONT.tar"
        if os.path.exists(tar_path):
            print(f"[vae-embed] Extracting CAM_FRONT.tar ({os.path.getsize(tar_path)/1e9:.1f} GB)...")
            subprocess.run(["tar", "xf", tar_path, "-C", f"{DATA_ROOT}/samples/"], check=True)
            n_imgs = len(os.listdir(cam_dir))
            print(f"[vae-embed] Extracted {n_imgs} images")
        else:
            print(f"[vae-embed] ERROR: {tar_path} not found!")
            return None
    else:
        n_imgs = len(os.listdir(cam_dir))
        print(f"[vae-embed] CAM_FRONT already extracted ({n_imgs} images)")

    # ---- Load metadata from existing encoder NPZ ----
    ref_npz = f"{EMBED_DIR}/vit_s16.npz"
    data = np.load(ref_npz, allow_pickle=True)
    scene_names = data["scene_names"]
    splits = data["splits"]
    steer_norms = data["steer_norms"]
    accel_norms = data["accel_norms"]
    image_paths = data["image_paths"]

    n_total = len(scene_names)
    if smoke:
        n_total = min(100, n_total)
        print(f"[vae-embed] SMOKE MODE: processing {n_total} frames only")
    else:
        print(f"[vae-embed] Processing {n_total} frames")

    # ---- Verify first image ----
    first_path = f"{DATA_ROOT}/{image_paths[0]}"
    if not os.path.exists(first_path):
        print(f"[vae-embed] ERROR: First image not found at {first_path}")
        return None
    print(f"[vae-embed] Image access verified: {first_path}")

    # ---- Image preprocessing ----
    # Center-crop to square (1600x900 -> 900x900), then resize to 256x256
    # SD VAE expects images in [-1, 1]
    def load_and_preprocess(path):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        # Center-crop to square
        if w != h:
            crop_size = min(w, h)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            img = img.crop((left, top, left + crop_size, top + crop_size))
        # Resize to target
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        # To tensor and normalize to [-1, 1]
        import torchvision.transforms.functional as TF
        tensor = TF.to_tensor(img)  # [0, 1]
        tensor = tensor * 2.0 - 1.0  # [-1, 1]
        return tensor

    # ---- Encode all frames ----
    all_latents = []
    n_missing = 0
    batch_size = BATCH_SIZE

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
                    batch_imgs.append(torch.zeros(3, TARGET_SIZE, TARGET_SIZE))
                    continue
                batch_imgs.append(load_and_preprocess(full_path))

            batch_tensor = torch.stack(batch_imgs).to(device)

            # Encode through VAE
            posterior = vae.encode(batch_tensor).latent_dist
            latents = posterior.mean  # (B, 4, 32, 32) -- deterministic
            latents = latents * SCALING_FACTOR

            all_latents.append(latents.cpu().float().numpy())

            if (start // batch_size) % 50 == 0:
                print(f"  [{start + len(batch_imgs)}/{n_total}] processed")

    vae_latents = np.concatenate(all_latents, axis=0)
    print(f"[vae-embed] Final shape: {vae_latents.shape}")
    print(f"[vae-embed] Stats: mean={vae_latents.mean():.4f}, std={vae_latents.std():.4f}")
    if n_missing > 0:
        print(f"[vae-embed] WARNING: {n_missing}/{n_total} images missing!")
    else:
        print(f"[vae-embed] All {n_total} images processed successfully")

    # ---- Validation: decode a few frames and check round-trip ----
    if validate:
        print(f"\n[vae-embed] === VALIDATION: decoding 8 frames ===")
        sample_idx = [0, 1, 100, 500, 1000, 5000, 10000, min(20000, n_total - 1)]
        sample_idx = [i for i in sample_idx if i < n_total]

        sample_latents = torch.tensor(vae_latents[sample_idx]).to(device)
        with torch.no_grad():
            decoded = vae.decode(sample_latents / SCALING_FACTOR).sample
            decoded = (decoded.clamp(-1, 1) + 1) / 2  # back to [0, 1]

        # Save decoded images
        val_dir = f"{SPATIAL_DIR}/vae_validation"
        os.makedirs(val_dir, exist_ok=True)
        for j, idx in enumerate(sample_idx):
            img_tensor = decoded[j].cpu()
            img = Image.fromarray(
                (img_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            )
            img.save(f"{val_dir}/decoded_{idx:06d}.png")

        # Compute reconstruction metrics
        for j, idx in enumerate(sample_idx):
            orig_path = f"{DATA_ROOT}/{image_paths[idx]}"
            if os.path.exists(orig_path):
                orig = load_and_preprocess(orig_path)
                recon = decoded[j].cpu()
                # Shift both to [0, 1]
                orig_01 = (orig + 1) / 2
                recon_01 = recon
                mse = ((orig_01 - recon_01) ** 2).mean().item()
                print(f"  Frame {idx}: recon MSE={mse:.6f}")

        print(f"[vae-embed] Validation images saved to {val_dir}/")

    # ---- Degeneracy check: copy-baseline CosSim ----
    print(f"\n[vae-embed] === DEGENERACY CHECK: copy-baseline CosSim ===")
    import torch.nn.functional as F

    # Flatten latents to vectors for CosSim
    flat = torch.tensor(vae_latents).reshape(len(vae_latents), -1)  # (N, 4096)

    # Check cosine similarity between consecutive frames (same scene)
    test_mask = splits[:n_total] == "test"
    test_scenes = scene_names[:n_total][test_mask]
    test_flat = flat[test_mask]

    scene = np.unique(test_scenes)[0]
    scene_idx = np.where(test_scenes == scene)[0]

    for gap in [1, 4, 16]:
        if len(scene_idx) > gap:
            z0 = test_flat[scene_idx[:-gap]]
            z1 = test_flat[scene_idx[gap:]]
            cs = F.cosine_similarity(z0, z1, dim=-1).mean().item()
            print(f"  Copy CosSim gap={gap} ({gap*0.5:.1f}s): {cs:.6f}")

    # ---- Save ----
    if not smoke:
        out_path = f"{SPATIAL_DIR}/sd_vae_latents.npz"
    else:
        out_path = f"{SPATIAL_DIR}/sd_vae_latents_smoke.npz"

    os.makedirs(SPATIAL_DIR, exist_ok=True)
    np.savez_compressed(
        out_path,
        vae_latents=vae_latents,
        scene_names=scene_names[:n_total],
        splits=splits[:n_total],
        steer_norms=steer_norms[:n_total],
        accel_norms=accel_norms[:n_total],
        image_paths=image_paths[:n_total],
        encoder="sd_vae_ft_mse",
        latent_shape=np.array([4, 32, 32]),
        scaling_factor=np.float32(SCALING_FACTOR),
    )
    file_size_mb = os.path.getsize(out_path) / 1e6
    print(f"[vae-embed] Saved {out_path} ({file_size_mb:.0f} MB)")

    vol.commit()
    return {
        "shape": list(vae_latents.shape),
        "file_size_mb": round(file_size_mb, 1),
        "n_missing": n_missing,
        "n_total": n_total,
        "smoke": smoke,
    }


# ===================================================================
# Entrypoint
# ===================================================================

def _modal_entrypoint_decorator(fn):
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_modal_entrypoint_decorator
def main(smoke: bool = False, validate: bool = True):
    """Extract SD VAE latent grids for nuScenes frames."""
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"SD VAE Latent Extraction")
    print(f"  model: {VAE_MODEL_ID}")
    print(f"  target: {TARGET_SIZE}x{TARGET_SIZE} -> 32x32x4 latent grid")
    print(f"  smoke: {smoke}, validate: {validate}")
    print(f"{'='*60}")

    result = extract_vae_latents.remote(smoke, validate)
    wall = time.time() - t_start
    print(f"\nDone in {wall:.0f}s: {result}")
