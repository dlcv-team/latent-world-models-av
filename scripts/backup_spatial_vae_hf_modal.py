"""Back up spatial/VAE artifacts from Modal volume to Hugging Face (no local download).

Usage::

    modal run scripts/backup_spatial_vae_hf_modal.py
    modal run --detach scripts/backup_spatial_vae_hf_modal.py

Requires HF_TOKEN via Modal secret ``huggingface-token`` or env at invoke time.
"""

from __future__ import annotations

import os
import time

try:
    import modal
except ImportError:
    modal = None

VOL_PATH = "/vol"
CHECKPOINTS_REPO = "surlac/lwm-av-checkpoints"
EMBEDDINGS_REPO = "surlac/lwm-av-embeddings"

# (volume_path, hf_repo, path_in_repo, repo_type)
UPLOADS = [
    # Spatial DiT checkpoints (direct, h=16)
    *[
        (
            f"{VOL_PATH}/dits/spatial_anchored/vit_s16/direct/h16/seed_{s}/dit_checkpoint.pt",
            CHECKPOINTS_REPO,
            f"checkpoints/spatial_vit_direct_h16_s{s}.pt",
            "model",
        )
        for s in (0, 1, 2)
    ],
    *[
        (
            f"{VOL_PATH}/dits/spatial_anchored/dino_vits14/direct/h16/seed_{s}/dit_checkpoint.pt",
            CHECKPOINTS_REPO,
            f"checkpoints/spatial_dino_direct_h16_s{s}.pt",
            "model",
        )
        for s in (0, 1, 2)
    ],
    # VAE direct DiT
    (
        f"{VOL_PATH}/dits/vae_latent/h16/seed_0/dit.pt",
        CHECKPOINTS_REPO,
        "checkpoints/vae_dit_direct_h16_s0.pt",
        "model",
    ),
    # Large embeddings (expensive to recreate)
    (
        f"{VOL_PATH}/embeddings/spatial/vit_s16_spatial_fullgrid.npz",
        EMBEDDINGS_REPO,
        "spatial/vit_s16_spatial_fullgrid.npz",
        "dataset",
    ),
    (
        f"{VOL_PATH}/embeddings/spatial/dino_vits14_spatial_fullgrid.npz",
        EMBEDDINGS_REPO,
        "spatial/dino_vits14_spatial_fullgrid.npz",
        "dataset",
    ),
    (
        f"{VOL_PATH}/embeddings/spatial/sd_vae_latents.npz",
        EMBEDDINGS_REPO,
        "spatial/sd_vae_latents.npz",
        "dataset",
    ),
]

if modal is not None:
    app = modal.App("lwm-av-spatial-vae-hf-backup")
    vol = modal.Volume.from_name("nuscenes-full")

    image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub>=0.24")
    HF_SECRETS = [modal.Secret.from_name("huggingface-token")]
else:
    app = None
    vol = None
    image = None


def _decorator(fn):
    if app is not None:
        return app.function(
            volumes={VOL_PATH: vol},
            image=image,
            timeout=14400,
            memory=8192,
            secrets=HF_SECRETS,
        )(fn)
    return fn


@_decorator
def backup_all():
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not available (set huggingface-token secret or pass HF_TOKEN=...)")

    api = HfApi(token=token)
    for repo_id, repo_type in [
        (CHECKPOINTS_REPO, "model"),
        (EMBEDDINGS_REPO, "dataset"),
    ]:
        api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=True)

    uploaded, skipped, errors = [], [], []

    for local_path, repo_id, path_in_repo, repo_type in UPLOADS:
        if not os.path.exists(local_path):
            print(f"  MISSING: {local_path}")
            skipped.append(path_in_repo)
            continue
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  Uploading {path_in_repo} ({size_mb:.1f} MB) ...")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
            )
            uploaded.append(f"{repo_id}:{path_in_repo}")
            print(f"    OK: {repo_id}/{path_in_repo}")
        except Exception as e:
            errors.append({"path": path_in_repo, "error": str(e)})
            print(f"    ERROR: {e}")

    return {"uploaded": uploaded, "skipped": skipped, "errors": errors}


def _entry(fn):
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_entry
def main():
    t0 = time.time()
    print(f"Backing up {len(UPLOADS)} files from Modal volume to HF")
    result = backup_all.remote()
    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"  Uploaded: {len(result['uploaded'])}")
    print(f"  Skipped (missing on volume): {len(result['skipped'])}")
    print(f"  Errors: {len(result['errors'])}")
    if result["errors"]:
        for err in result["errors"]:
            print(f"    {err}")
