"""Smoke test: extract spatial embeddings for 10 frames, verify diversity."""
from __future__ import annotations
import os, subprocess

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("lwm-smoke-spatial")
    vol = modal.Volume.from_name("nuscenes-full")
    base_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy>=1.26", "timm>=1.0.3", "Pillow>=10.0")
    )
else:
    app = None; vol = None; base_image = None

def _dec(fn):
    if app: return app.function(volumes={"/vol": vol}, image=base_image, gpu="T4", timeout=600, memory=16384)(fn)
    return fn

@_dec
def smoke_test():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image
    import timm, timm.data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup: extract images
    DATA_ROOT = "/vol/nuscenes"
    cam_dir = f"{DATA_ROOT}/samples/CAM_FRONT"
    os.makedirs(f"{DATA_ROOT}/samples", exist_ok=True)
    if not os.path.isdir(cam_dir) or len(os.listdir(cam_dir)) < 30000:
        tar_path = "/vol/raw/CAM_FRONT.tar"
        if not os.path.exists(tar_path):
            print(f"ERROR: {tar_path} not found!")
            return {"error": "tar not found"}
        print(f"Extracting CAM_FRONT.tar...")
        subprocess.run(["tar", "xf", tar_path, "-C", f"{DATA_ROOT}/samples/"], check=True)
        print(f"Extracted {len(os.listdir(cam_dir))} images")
    else:
        print(f"CAM_FRONT already extracted: {len(os.listdir(cam_dir))} images")

    # Load ViT-S/16
    model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0).to(device).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    img_mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(device)
    img_std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(device)

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # Load metadata
    data = np.load("/vol/embeddings/vit_s16.npz", allow_pickle=True)
    image_paths = data["image_paths"]

    # Process first 10 frames
    results = []
    with torch.no_grad():
        for i in range(10):
            path = f"{DATA_ROOT}/{image_paths[i]}"
            if not os.path.exists(path):
                print(f"  Frame {i}: NOT FOUND at {path}")
                results.append(None)
                continue

            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            x = (x - img_mean) / img_std
            features = model.forward_features(x)  # (1, 197, 384)
            patches = features[:, 1:]  # (1, 196, 384)

            # 2x2 pool to 7x7
            grid = patches.reshape(1, 14, 14, 384).permute(0, 3, 1, 2)
            pooled = F.avg_pool2d(grid, kernel_size=2, stride=2)  # (1, 384, 7, 7)
            spatial = pooled.permute(0, 2, 3, 1).reshape(49, 384)  # (49, 384)

            results.append(spatial.cpu())
            print(f"  Frame {i}: shape={spatial.shape}, mean={spatial.mean():.4f}, std={spatial.std():.4f}")

    # Check inter-frame diversity
    if results[0] is not None and results[5] is not None:
        cs = F.cosine_similarity(results[0], results[5], dim=-1).mean().item()
        diff = (results[0] - results[5]).abs().mean().item()
        identical = torch.allclose(results[0], results[5])
        print(f"\n  Frame 0 vs 5: per-token CosSim={cs:.6f}, mean_abs_diff={diff:.6f}, identical={identical}")

    if results[0] is not None and results[9] is not None:
        cs = F.cosine_similarity(results[0], results[9], dim=-1).mean().item()
        diff = (results[0] - results[9]).abs().mean().item()
        identical = torch.allclose(results[0], results[9])
        print(f"  Frame 0 vs 9: per-token CosSim={cs:.6f}, mean_abs_diff={diff:.6f}, identical={identical}")

    return {"success": True, "n_frames": len([r for r in results if r is not None])}

def _entry_dec(fn):
    if app: return app.local_entrypoint()(fn)
    return fn

@_entry_dec
def main():
    result = smoke_test.remote()
    print(f"\nResult: {result}")
