"""Back up DA9 checkpoints from Modal volume to HuggingFace.

Runs as a Modal function to avoid downloading large checkpoints locally.
Reads from the Modal volume and uploads directly to HuggingFace.

Usage::

    HF_TOKEN=... modal run scripts/backup_to_hf.py
"""

from __future__ import annotations

import os
import sys
import time

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("lwm-av-hf-backup")
    vol = modal.Volume.from_name("nuscenes-full")
else:
    app = None
    vol = None

VOL_PATH = "/vol"

NATIVE_DIMS = {
    "vit_s16": 384,
    "dino_vits14": 384,
    "clip_b32": 512,
    "vq_track": 256,
    "vjepa2_rep64": 1024,
    "vjepa2_rep1": 1024,
}

ALL_ENCODERS = sorted(NATIVE_DIMS.keys())
HORIZONS = [8, 16]
SEEDS = [0, 1, 2]

HF_REPO = "surlac/lwm-av-checkpoints"

if modal is not None:
    base_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("huggingface_hub>=0.20")
    )
else:
    base_image = None


def _modal_function_decorator(fn):
    if app is not None:
        return app.function(
            volumes={VOL_PATH: vol},
            image=base_image,
            timeout=7200,
            memory=4096,
            secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
        )(fn)
    return fn


@_modal_function_decorator
def upload_checkpoints(hf_repo: str):
    """Upload all DA9 checkpoints from Modal volume to HuggingFace."""
    from huggingface_hub import HfApi

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not set")

    api = HfApi(token=hf_token)

    # Create repo if needed
    try:
        api.create_repo(hf_repo, repo_type="model", exist_ok=True, private=True)
        print(f"Using repo: {hf_repo}")
    except Exception as e:
        print(f"Repo check: {e}")

    uploaded = 0
    skipped = 0
    errors = 0

    # DiT checkpoints
    for enc in ALL_ENCODERS:
        for h in HORIZONS:
            for seed in SEEDS:
                local_path = f"{VOL_PATH}/dits/{enc}/conditioned__x0__h{h}/seed_{seed}/checkpoint.pt"
                hf_path = f"da9/dits/{enc}/conditioned__x0__h{h}/seed_{seed}/checkpoint.pt"

                if not os.path.exists(local_path):
                    print(f"  MISSING: {local_path}")
                    skipped += 1
                    continue

                try:
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=hf_path,
                        repo_id=hf_repo,
                        repo_type="model",
                    )
                    uploaded += 1
                    print(f"  Uploaded: {hf_path}")
                except Exception as e:
                    errors += 1
                    print(f"  ERROR uploading {hf_path}: {e}")

    # MLP checkpoints
    for mode in ["fair", "residual"]:
        for h in HORIZONS:
            for enc in ALL_ENCODERS:
                for seed in SEEDS:
                    local_path = (
                        f"{VOL_PATH}/outputs/latent_predictors_{mode}_h{h}"
                        f"/{enc}/conditioned/seed_{seed}/checkpoint.pt"
                    )
                    hf_path = (
                        f"da9/mlp_{mode}_h{h}/{enc}/conditioned/seed_{seed}/checkpoint.pt"
                    )

                    if not os.path.exists(local_path):
                        print(f"  MISSING: {local_path}")
                        skipped += 1
                        continue

                    try:
                        api.upload_file(
                            path_or_fileobj=local_path,
                            path_in_repo=hf_path,
                            repo_id=hf_repo,
                            repo_type="model",
                        )
                        uploaded += 1
                        print(f"  Uploaded: {hf_path}")
                    except Exception as e:
                        errors += 1
                        print(f"  ERROR uploading {hf_path}: {e}")

    # Also upload DA8 h=4 checkpoints (DiT + MLP) for completeness
    for enc in ALL_ENCODERS:
        for seed in SEEDS:
            # DA8 DiT x0
            local_path = f"{VOL_PATH}/dits/{enc}/conditioned__x0/seed_{seed}/checkpoint.pt"
            hf_path = f"da8/dits/{enc}/conditioned__x0/seed_{seed}/checkpoint.pt"
            if os.path.exists(local_path):
                try:
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=hf_path,
                        repo_id=hf_repo,
                        repo_type="model",
                    )
                    uploaded += 1
                    print(f"  Uploaded: {hf_path}")
                except Exception as e:
                    errors += 1
                    print(f"  ERROR: {e}")

    return {
        "uploaded": uploaded,
        "skipped": skipped,
        "errors": errors,
    }


def _modal_entrypoint_decorator(fn):
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_modal_entrypoint_decorator
def main():
    """Back up checkpoints to HuggingFace."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: Set HF_TOKEN environment variable")
        return

    print(f"Backing up checkpoints to HuggingFace: {HF_REPO}")
    print(f"  DiT: {len(ALL_ENCODERS)} encoders x {len(HORIZONS)} horizons x {len(SEEDS)} seeds = {len(ALL_ENCODERS)*len(HORIZONS)*len(SEEDS)} checkpoints")
    print(f"  MLP: 2 modes x {len(ALL_ENCODERS)} enc x {len(HORIZONS)} h x {len(SEEDS)} seeds = {2*len(ALL_ENCODERS)*len(HORIZONS)*len(SEEDS)} checkpoints")

    t0 = time.time()
    result = upload_checkpoints.remote(HF_REPO)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Uploaded: {result['uploaded']}")
    print(f"  Skipped: {result['skipped']}")
    print(f"  Errors: {result['errors']}")

    if result["errors"] > 0:
        sys.exit(1)
