"""Back up the new action-conditioned VAE diffusion checkpoint + generative-eval
artifacts from the Modal volume to Hugging Face (durability; volume was deleted once).

    modal run scripts/backup_vae_diffusion_hf_modal.py

Requires HF_TOKEN via Modal secret ``huggingface-token``.
"""

from __future__ import annotations

import os

try:
    import modal
except ImportError:
    modal = None

VOL_PATH = "/vol"
CHECKPOINTS_REPO = "surlac/lwm-av-checkpoints"
EMBEDDINGS_REPO = "surlac/lwm-av-embeddings"

UPLOADS = [
    (f"{VOL_PATH}/dits/vae_latent/diffusion/h16/seed_0/dit.pt",
     CHECKPOINTS_REPO, "vae_latent/diffusion_cfg0.1/h16/seed_0/dit.pt", "model"),
    (f"{VOL_PATH}/dits/vae_latent/diffusion/h16/seed_1/dit.pt",
     CHECKPOINTS_REPO, "vae_latent/diffusion_cfg0.1/h16/seed_1/dit.pt", "model"),
    (f"{VOL_PATH}/dits/vae_latent/diffusion/h16/seed_2/dit.pt",
     CHECKPOINTS_REPO, "vae_latent/diffusion_cfg0.1/h16/seed_2/dit.pt", "model"),
    # P1 scaling probe: 3.0M (n_blocks=2) diffusion ckpt (Future-Work, low-end capacity point)
    (f"{VOL_PATH}/dits/vae_latent/diffusion/h16/seed_0_nb2/dit.pt",
     CHECKPOINTS_REPO, "vae_latent/diffusion_cfg0.1/h16/seed_0_nb2_3.0M/dit.pt", "model"),
    (f"{VOL_PATH}/embeddings/spatial/gen_eval/vae_4row_demo.pdf",
     EMBEDDINGS_REPO, "gen_eval/vae_4row_demo.pdf", "dataset"),
    (f"{VOL_PATH}/embeddings/spatial/gen_eval/fid_eval_diffusion.json",
     EMBEDDINGS_REPO, "gen_eval/fid_eval_full600.json", "dataset"),
    (f"{VOL_PATH}/embeddings/spatial/gen_eval/motion_eval.json",
     EMBEDDINGS_REPO, "gen_eval/motion_eval.json", "dataset"),
    (f"{VOL_PATH}/da_analysis/t0_perwindow_results.csv",
     EMBEDDINGS_REPO, "da_analysis/t0_perwindow_results.csv", "dataset"),
]

if modal is not None:
    app = modal.App("lwm-av-backup-vae-diff")
    vol = modal.Volume.from_name("nuscenes-full")
    image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub>=0.24")
    HF_SECRETS = [modal.Secret.from_name("huggingface-token")]
else:
    app = vol = image = None


def _decorator(fn):
    if app is not None:
        return app.function(volumes={VOL_PATH: vol}, image=image, timeout=3600,
                            memory=8192, secrets=HF_SECRETS)(fn)
    return fn


@_decorator
def backup():
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not available")
    api = HfApi(token=token)
    for repo_id, repo_type in [(CHECKPOINTS_REPO, "model"), (EMBEDDINGS_REPO, "dataset")]:
        api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, private=True)
    uploaded, skipped, errors = [], [], []
    for local_path, repo_id, path_in_repo, repo_type in UPLOADS:
        if not os.path.exists(local_path):
            print(f"  MISSING: {local_path}"); skipped.append(path_in_repo); continue
        mb = os.path.getsize(local_path) / 1e6
        print(f"  Uploading {path_in_repo} ({mb:.1f} MB) ...")
        try:
            api.upload_file(path_or_fileobj=local_path, path_in_repo=path_in_repo,
                            repo_id=repo_id, repo_type=repo_type)
            uploaded.append(f"{repo_id}:{path_in_repo}"); print(f"    OK")
        except Exception as e:
            errors.append({"path": path_in_repo, "error": str(e)}); print(f"    ERR {e}")
    print(f"uploaded={len(uploaded)} skipped={len(skipped)} errors={len(errors)}")
    return {"uploaded": uploaded, "skipped": skipped, "errors": errors}


def _entry(fn):
    return app.local_entrypoint()(fn) if app is not None else fn


@_entry
def main():
    print(backup.remote())
