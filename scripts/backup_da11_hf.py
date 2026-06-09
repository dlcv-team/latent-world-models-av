"""Back up DA11 action-sequence checkpoints from Modal volume to HuggingFace.

Uploads DiT-actseq and MLP-flat-actseq checkpoints directly from Modal volume.

Usage::

    HF_TOKEN=... modal run scripts/backup_da11_hf.py
"""

from __future__ import annotations

import os

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("lwm-av-da11-backup")
    vol = modal.Volume.from_name("nuscenes-full")
else:
    app = None
    vol = None

VOL_PATH = "/vol"

ENCODERS = ["vit_s16", "clip_b32", "dino_vits14"]
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
def upload_da11_checkpoints(hf_repo: str):
    """Upload DA11 checkpoints to HuggingFace."""
    from huggingface_hub import HfApi

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not set")

    api = HfApi(token=hf_token)
    try:
        api.create_repo(hf_repo, repo_type="model", exist_ok=True, private=True)
    except Exception as e:
        print(f"Repo check: {e}")

    uploaded = 0
    skipped = 0

    # DiT-actseq checkpoints
    for enc in ENCODERS:
        for h in HORIZONS:
            for seed in SEEDS:
                local = f"{VOL_PATH}/dits/{enc}/conditioned__x0__actseq__h{h}/seed_{seed}/checkpoint.pt"
                hf_path = f"da11/dits/{enc}/conditioned__x0__actseq__h{h}/seed_{seed}/checkpoint.pt"
                if os.path.exists(local):
                    try:
                        api.upload_file(path_or_fileobj=local, path_in_repo=hf_path,
                                       repo_id=hf_repo, repo_type="model")
                        uploaded += 1
                        print(f"  Uploaded: {hf_path}")
                    except Exception as e:
                        print(f"  ERROR: {hf_path}: {e}")
                else:
                    skipped += 1

    # MLP-flat-actseq checkpoints
    for enc in ENCODERS:
        for h in HORIZONS:
            for seed in SEEDS:
                local = (f"{VOL_PATH}/outputs/latent_predictors_residual_actseq_h{h}"
                        f"/{enc}/conditioned/seed_{seed}/checkpoint.pt")
                hf_path = f"da11/mlp_actseq_h{h}/{enc}/conditioned/seed_{seed}/checkpoint.pt"
                if os.path.exists(local):
                    try:
                        api.upload_file(path_or_fileobj=local, path_in_repo=hf_path,
                                       repo_id=hf_repo, repo_type="model")
                        uploaded += 1
                        print(f"  Uploaded: {hf_path}")
                    except Exception as e:
                        print(f"  ERROR: {hf_path}: {e}")
                else:
                    skipped += 1

    return {"uploaded": uploaded, "skipped": skipped}


def _modal_entrypoint_decorator(fn):
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_modal_entrypoint_decorator
def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: Set HF_TOKEN environment variable")
        return
    print(f"Backing up DA11 checkpoints to {HF_REPO}")
    result = upload_da11_checkpoints.remote(HF_REPO)
    print(f"Done: uploaded={result['uploaded']}, skipped={result['skipped']}")
