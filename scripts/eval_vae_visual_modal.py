"""Decode VAE-latent DiT predictions to images for paper figure (3-row layout)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_dit_vae_modal.py"

if modal is not None:
    app = modal.App("lwm-av-vae-visual")
    vol = modal.Volume.from_name("nuscenes-full")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.5.1", "numpy>=1.26", "Pillow>=10.0",
            "diffusers>=0.27", "matplotlib>=3.8", "accelerate", "transformers>=4.50",
            "torchvision>=0.20",
        )
        .add_local_file(str(TRAIN_SCRIPT), remote_path="/root/train_dit_vae_modal.py")
    )
else:
    app = None
    vol = None
    image = None

VOL_PATH = "/vol"
DATA_ROOT = f"{VOL_PATH}/nuscenes"
SPATIAL_DIR = f"{VOL_PATH}/embeddings/spatial"
CKPT_DIR = f"{VOL_PATH}/dits/vae_latent"
VAE_NPZ = f"{SPATIAL_DIR}/sd_vae_latents.npz"
SCALING = 0.18215
TARGET_SIZE = 256
HORIZON = 16
STEPS_SHOW = [0, 4, 8, 12, 15]


def _decorator(fn):
    if app is not None:
        return app.function(volumes={VOL_PATH: vol}, image=image, gpu="A10G", timeout=3600, memory=16384)(fn)
    return fn


@_decorator
def run_visual_eval(n_windows: int = 20, seed: int = 0, horizon: int = 16):
    import numpy as np
    import torch
    import torchvision.transforms.functional as TF
    from diffusers import AutoencoderKL
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    spec = importlib.util.spec_from_file_location("train_dit_vae_modal", "/root/train_dit_vae_modal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    patchify = mod.patchify
    unpatchify = mod.unpatchify
    AnchoredVAEDiT = mod.AnchoredVAEDiT
    FourierActionEmbedding = mod.FourierActionEmbedding
    DIT_CONFIG = mod.DIT_CONFIG
    FOURIER_CONFIG = mod.FOURIER_CONFIG
    PATCH_DIM = mod.PATCH_DIM
    N_SPATIAL = mod.N_SPATIAL

    device = torch.device("cuda")
    ckpt_path = f"{CKPT_DIR}/h{horizon}/seed_{seed}/dit.pt"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint {ckpt_path} missing")
        return None

    data = np.load(VAE_NPZ, allow_pickle=True)
    latents = data["vae_latents"]
    scenes = data["scene_names"]
    splits = data["splits"]
    steers = data["steer_norms"]
    accels = data["accel_norms"]
    image_paths = data["image_paths"] if "image_paths" in data else None

    def load_rgb_256(path):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w != h:
            crop = min(w, h)
            left = (w - crop) // 2
            top = (h - crop) // 2
            img = img.crop((left, top, left + crop, top + crop))
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        t = TF.to_tensor(img) * 2.0 - 1.0
        return t

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    z_mean = ckpt["z_mean"].to(device)
    z_std = ckpt["z_std"].to(device)

    def norm_p(g):
        return (patchify(g) - z_mean) / z_std

    dit = AnchoredVAEDiT(horizon=horizon, n_spatial=N_SPATIAL, **DIT_CONFIG).to(device)
    fourier = FourierActionEmbedding(**FOURIER_CONFIG).to(device)
    dit.load_state_dict(ckpt["dit"])
    fourier.load_state_dict(ckpt["fourier"])
    dit.eval()
    fourier.eval()

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()

    test_idx = np.where(splits == "test")[0]
    picks = []
    for sc in np.unique(scenes[test_idx])[:n_windows]:
        idx = test_idx[scenes[test_idx] == sc]
        if len(idx) > horizon:
            picks.append(int(idx[0]))
    picks = picks[:n_windows]

    out_dir = f"{SPATIAL_DIR}/vae_figure"
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for wi, frame_i in enumerate(picks):
            z_t = torch.tensor(latents[frame_i:frame_i + 1], device=device)
            act = torch.stack([
                torch.tensor([steers[frame_i + k], accels[frame_i + k]], device=device)
                for k in range(horizon)
            ]).unsqueeze(0)
            zf_gt = torch.tensor(latents[frame_i + 1: frame_i + 1 + horizon], device=device).unsqueeze(0)

            z_t_n = norm_p(z_t)
            a_emb = fourier(act)
            z_rep = z_t_n.unsqueeze(1).expand(-1, horizon, -1, -1).reshape(1, horizon * N_SPATIAL, PATCH_DIM)
            t0 = torch.zeros(1, dtype=torch.long, device=device)
            pred_n = dit(z_rep, z_t_n, a_emb, t0)
            pred_tok = (pred_n * z_std + z_mean).reshape(1, horizon, N_SPATIAL, PATCH_DIM)

            fig, axes = plt.subplots(3, len(STEPS_SHOW), figsize=(3 * len(STEPS_SHOW), 9))
            for col, k in enumerate(STEPS_SHOW):
                pred_lat_k = unpatchify(pred_tok[:, k]).clamp(-3, 3)
                vae_gt_dec = vae.decode(zf_gt[:, k] / SCALING).sample
                pr_dec = vae.decode(pred_lat_k / SCALING).sample
                vae_gt_img = ((vae_gt_dec.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
                pr_img = ((pr_dec.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()

                if image_paths is not None:
                    gt_frame = frame_i + 1 + k
                    if gt_frame < len(image_paths):
                        ipath = f"{DATA_ROOT}/{image_paths[gt_frame]}"
                        if os.path.exists(ipath):
                            cam = load_rgb_256(ipath)
                            cam_img = ((cam.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
                            axes[0, col].imshow(cam_img)
                        else:
                            axes[0, col].imshow(vae_gt_img)
                    else:
                        axes[0, col].imshow(vae_gt_img)
                else:
                    axes[0, col].imshow(vae_gt_img)
                axes[0, col].set_title(f"RGB t+{k}")
                axes[0, col].axis("off")

                axes[1, col].imshow(vae_gt_img)
                axes[1, col].set_title(f"VAE GT t+{k}")
                axes[1, col].axis("off")

                axes[2, col].imshow(pr_img)
                axes[2, col].set_title(f"DiT t+{k}")
                axes[2, col].axis("off")

            fig.savefig(f"{out_dir}/window_{wi:02d}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

    pdf_path = f"{out_dir}/vae_pipeline_demo.pdf"
    with PdfPages(pdf_path) as pdf:
        for wi in range(min(3, len(picks))):
            img = np.array(Image.open(f"{out_dir}/window_{wi:02d}.png"))
            fig, ax = plt.subplots(figsize=(12, 9))
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

    vol.commit()
    return {"n_windows": len(picks), "pdf": pdf_path}


def _entry(fn):
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_entry
def main(n_windows: int = 20, seed: int = 0):
    print(run_visual_eval.remote(n_windows, seed))
