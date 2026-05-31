"""Extract FULL-GRID spatial patch-token embeddings (no pooling) + HF backup.

Tier C data prep (Opus 4.8 takeover, 2026-05-31). Unlike embed_spatial_modal.py
(which 2x2-pools to 7x7/8x8), this keeps the native patch grid:

  ViT-S/16:    14x14 = 196 tokens x 384-d
  DINOv2-S/14: 16x16 = 256 tokens x 384-d

Writes to NEW filenames (never overwrites the pooled NPZs):
  {encoder}_spatial_fullgrid.npz

Runs a diversity sanity check (guards against the zero-embedding bug), computes
a SHA-256 checksum, and -- if the `huggingface-token` Modal secret is attached --
uploads to the surlac/lwm-av-embeddings HF dataset under spatial/.

Usage::

    modal run scripts/embed_spatial_fullgrid_modal.py --encoder vit_s16
    modal run scripts/embed_spatial_fullgrid_modal.py --encoder dino_vits14
"""

from __future__ import annotations

import time

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("lwm-av-embed-fullgrid")
    vol = modal.Volume.from_name("nuscenes-full")
    try:
        hf_secret = modal.Secret.from_name("huggingface-token")
    except Exception:
        hf_secret = None
else:
    app = None
    vol = None
    hf_secret = None

VOL_PATH = "/vol"
EMBED_DIR = f"{VOL_PATH}/embeddings"
SPATIAL_DIR = f"{VOL_PATH}/embeddings/spatial"

HF_REPO = "surlac/lwm-av-embeddings"

ENCODER_CONFIGS = {
    "vit_s16": {
        "patch_grid": (14, 14),
        "native_dim": 384,
        "model_id": "vit_small_patch16_224",
        "loader": "timm",
    },
    "dino_vits14": {
        "patch_grid": (16, 16),
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
            "huggingface_hub>=0.24",
        )
    )
else:
    base_image = None


def _modal_function_decorator(fn):
    if app is not None:
        secrets = [hf_secret] if hf_secret is not None else []
        return app.function(
            volumes={VOL_PATH: vol},
            image=base_image,
            gpu="A10G",
            timeout=10800,
            memory=32768,
            secrets=secrets,
        )(fn)
    return fn


@_modal_function_decorator
def extract_fullgrid(encoder_name: str, upload_hf: bool = True):
    import hashlib
    import os
    import numpy as np
    import torch
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ENCODER_CONFIGS[encoder_name]
    pH, pW = config["patch_grid"]
    native_dim = config["native_dim"]
    n_tokens = pH * pW

    print(f"[fullgrid] {encoder_name}: {pH}x{pW} = {n_tokens} tokens, dim={native_dim}")

    if config["loader"] == "timm":
        import timm
        import timm.data
        model = timm.create_model(config["model_id"], pretrained=True, num_classes=0)
        model = model.to(device).eval()
        cfg = timm.data.resolve_data_config({}, model=model)
        img_mean = torch.tensor(cfg["mean"], dtype=torch.float32).view(1, 3, 1, 1).to(device)
        img_std = torch.tensor(cfg["std"], dtype=torch.float32).view(1, 3, 1, 1).to(device)

        def extract_patches(x):
            x = (x - img_mean) / img_std
            features = model.forward_features(x)
            return features[:, 1:]

    elif config["loader"] == "dinov2":
        model = torch.hub.load(
            "facebookresearch/dinov2", config["model_id"],
            pretrained=True, trust_repo=True, verbose=False,
        )
        model = model.to(device).eval()
        _mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        _std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        def extract_patches(x):
            x = (x - _mean) / _std
            try:
                patches_list = model.get_intermediate_layers(x, n=1)
                patches = patches_list[0]
            except (AttributeError, TypeError):
                out = model.forward_features(x)
                if isinstance(out, dict):
                    patches = out.get("x_norm_patchtokens", out.get("x_norm", out.get("x"))[:, 1:])
                else:
                    patches = out[:, 1:]
            return patches
    else:
        raise ValueError(f"Unknown loader: {config['loader']}")

    import subprocess
    DATA_ROOT = f"{VOL_PATH}/nuscenes"
    RAW_DIR = f"{VOL_PATH}/raw"
    cam_dir = f"{DATA_ROOT}/samples/CAM_FRONT"

    os.makedirs(f"{DATA_ROOT}/samples", exist_ok=True)
    if not os.path.isdir(cam_dir) or len(os.listdir(cam_dir)) < 30000:
        tar_path = f"{RAW_DIR}/CAM_FRONT.tar"
        if os.path.exists(tar_path):
            print(f"[fullgrid] Extracting CAM_FRONT.tar ({os.path.getsize(tar_path)/1e9:.1f} GB)...")
            subprocess.run(["tar", "xf", tar_path, "-C", f"{DATA_ROOT}/samples/"], check=True)
            print(f"[fullgrid] Extracted {len(os.listdir(cam_dir))} images")
        else:
            print(f"[fullgrid] ERROR: {tar_path} not found!")
            return None
    else:
        print(f"[fullgrid] CAM_FRONT already extracted ({len(os.listdir(cam_dir))} images)")

    pooled_data = np.load(f"{EMBED_DIR}/{encoder_name}.npz", allow_pickle=True)
    scene_names = pooled_data["scene_names"]
    splits = pooled_data["splits"]
    steer_norms = pooled_data["steer_norms"]
    accel_norms = pooled_data["accel_norms"]
    image_paths = pooled_data["image_paths"]
    n_total = len(scene_names)
    print(f"[fullgrid] {n_total} frames to process")

    first_path = f"{DATA_ROOT}/{image_paths[0]}"
    if not os.path.exists(first_path):
        print(f"[fullgrid] ERROR: First image not found at {first_path}")
        return None

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    from PIL import Image

    batch_size = 64
    all_spatial = []
    n_missing = 0

    with torch.no_grad():
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch_imgs = []
            for i in range(start, end):
                full_path = f"{DATA_ROOT}/{str(image_paths[i])}"
                if not os.path.exists(full_path):
                    n_missing += 1
                    batch_imgs.append(torch.zeros(3, 224, 224))
                    continue
                img = Image.open(full_path).convert("RGB")
                batch_imgs.append(transform(img))

            batch_tensor = torch.stack(batch_imgs).to(device)
            patches = extract_patches(batch_tensor)  # (B, n_tokens, dim) -- NO pooling
            all_spatial.append(patches.cpu().numpy().astype(np.float32))

            if (start // batch_size) % 50 == 0:
                print(f"  [{end}/{n_total}] processed")

    spatial_embeddings = np.concatenate(all_spatial, axis=0)
    print(f"[fullgrid] Final shape: {spatial_embeddings.shape}")
    if n_missing > 0:
        print(f"[fullgrid] WARNING: {n_missing}/{n_total} images missing!")

    # ---- Diversity sanity check (guard vs zero-embedding bug) ----
    import torch.nn.functional as F
    emb_t = torch.tensor(spatial_embeddings[:200], dtype=torch.float32)
    diff01 = float(np.abs(spatial_embeddings[0] - spatial_embeddings[1]).mean())
    var100 = float(spatial_embeddings[:100].var(axis=0).mean())
    # per-token CosSim t vs t+16 within first test scene
    test_mask = splits == "test"
    test_emb = spatial_embeddings[test_mask]
    test_scenes = scene_names[test_mask]
    s0 = np.unique(test_scenes)[0]
    idx = np.where(test_scenes == s0)[0]
    cs_t16 = None
    if len(idx) > 16:
        z0 = torch.tensor(test_emb[idx[0]]); z16 = torch.tensor(test_emb[idx[16]])
        cs_t16 = float(F.cosine_similarity(z0, z16, dim=-1).mean())
    diversity = {
        "diff_frame0_frame1": diff01,
        "per_position_var_100": var100,
        "per_token_cossim_t_vs_t16": cs_t16,
        "all_identical": bool(diff01 < 1e-8),
    }
    print(f"[fullgrid] diversity: {diversity}")
    if diversity["all_identical"]:
        print("[fullgrid] ABORT: embeddings identical (zero-embedding bug). Not saving.")
        return {"error": "zero-embedding bug detected", "diversity": diversity}

    os.makedirs(SPATIAL_DIR, exist_ok=True)
    out_path = f"{SPATIAL_DIR}/{encoder_name}_spatial_fullgrid.npz"
    np.savez_compressed(
        out_path,
        spatial_embeddings=spatial_embeddings,
        scene_names=scene_names,
        splits=splits,
        steer_norms=steer_norms,
        accel_norms=accel_norms,
        image_paths=image_paths,
        encoder=encoder_name,
        spatial_grid=np.array(config["patch_grid"]),
        native_dim=native_dim,
    )
    file_size_mb = os.path.getsize(out_path) / 1e6

    sha = hashlib.sha256()
    with open(out_path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    checksum = sha.hexdigest()
    print(f"[fullgrid] Saved {out_path} ({file_size_mb:.0f} MB)  sha256={checksum}")

    vol.commit()

    hf_status = "skipped"
    if upload_hf and os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ["HF_TOKEN"])
            api.upload_file(
                path_or_fileobj=out_path,
                path_in_repo=f"spatial/{encoder_name}_spatial_fullgrid.npz",
                repo_id=HF_REPO,
                repo_type="dataset",
            )
            hf_status = f"uploaded to {HF_REPO}:spatial/{encoder_name}_spatial_fullgrid.npz"
            print(f"[fullgrid] HF {hf_status}")
        except Exception as e:
            hf_status = f"FAILED: {e}"
            print(f"[fullgrid] HF upload {hf_status}")
    else:
        print("[fullgrid] HF upload skipped (no HF_TOKEN secret or upload_hf=False)")

    return {
        "encoder": encoder_name,
        "shape": list(spatial_embeddings.shape),
        "spatial_grid": list(config["patch_grid"]),
        "n_tokens": n_tokens,
        "file_size_mb": round(file_size_mb, 1),
        "sha256": checksum,
        "diversity": diversity,
        "hf_status": hf_status,
        "n_missing": n_missing,
    }


def _modal_entrypoint_decorator(fn):
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_modal_entrypoint_decorator
def main(encoder: str = "vit_s16", upload_hf: bool = True):
    import json
    if encoder not in ENCODER_CONFIGS:
        print(f"ERROR: encoder must be one of {list(ENCODER_CONFIGS.keys())}")
        return
    t_start = time.time()
    print(f"\n{'='*60}\nFull-grid extraction: {encoder} "
          f"({ENCODER_CONFIGS[encoder]['patch_grid']})\n{'='*60}")
    result = extract_fullgrid.remote(encoder, upload_hf)
    print(f"\nDone in {time.time()-t_start:.0f}s")
    print(json.dumps(result, indent=2))
