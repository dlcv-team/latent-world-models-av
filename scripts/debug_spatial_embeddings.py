"""Quick diagnostic: check if spatial embeddings have inter-frame variation."""

from __future__ import annotations
import os

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("lwm-debug-spatial")
    vol = modal.Volume.from_name("nuscenes-full")
    base_image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy>=1.26", "torch==2.5.1")
else:
    app = None; vol = None; base_image = None

VOL_PATH = "/vol"

def _dec(fn):
    if app: return app.function(volumes={VOL_PATH: vol}, image=base_image, timeout=600, memory=16384)(fn)
    return fn

@_dec
def check_spatial(encoder: str):
    import numpy as np
    import torch
    import torch.nn.functional as F

    path = f"{VOL_PATH}/embeddings/spatial/{encoder}_spatial.npz"
    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}

    data = np.load(path, allow_pickle=True)
    emb = data["spatial_embeddings"]  # (N, S, D)
    splits = data["splits"]
    scenes = data["scene_names"]

    N, S, D = emb.shape
    print(f"Shape: {emb.shape}")
    print(f"dtype: {emb.dtype}")
    print(f"Global stats: mean={emb.mean():.6f}, std={emb.std():.6f}, min={emb.min():.4f}, max={emb.max():.4f}")

    # Check if all frames are identical
    frame0 = emb[0]  # (S, D)
    frame1 = emb[1]
    frame100 = emb[100]
    frame1000 = emb[1000]

    print(f"\nFrame 0 stats: mean={frame0.mean():.6f}, std={frame0.std():.6f}")
    print(f"Frame 1 stats: mean={frame1.mean():.6f}, std={frame1.std():.6f}")
    print(f"Frame 100 stats: mean={frame100.mean():.6f}, std={frame100.std():.6f}")
    print(f"Frame 1000 stats: mean={frame1000.mean():.6f}, std={frame1000.std():.6f}")

    # Check if frames are identical
    diff_01 = np.abs(frame0 - frame1).mean()
    diff_0_100 = np.abs(frame0 - frame100).mean()
    diff_0_1000 = np.abs(frame0 - frame1000).mean()
    print(f"\nMean abs diff: frame0-frame1={diff_01:.8f}, frame0-frame100={diff_0_100:.8f}, frame0-frame1000={diff_0_1000:.8f}")

    # Check if ALL frames are the same
    all_same = np.allclose(emb[0], emb[1]) and np.allclose(emb[0], emb[100])
    print(f"All frames identical? {all_same}")

    # Per-token CosSim between frame 0 and frame 16 (same scene)
    t_emb = torch.tensor(emb, dtype=torch.float32)

    mask = splits == "test"
    test_emb = t_emb[mask]
    test_scenes = scenes[mask]

    scene = np.unique(test_scenes)[0]
    idx = np.where(test_scenes == scene)[0]
    print(f"\nScene: {scene}, {len(idx)} frames")

    z0 = test_emb[idx[0]]   # (S, D)
    z1 = test_emb[idx[1]]   # (S, D)  -- 0.5s later
    z4 = test_emb[idx[4]]   # (S, D)  -- 2s later
    z16 = test_emb[idx[min(16, len(idx)-1)]]  # (S, D) -- 8s later

    # Per-token CosSim
    cs_01 = F.cosine_similarity(z0, z1, dim=-1).mean().item()
    cs_04 = F.cosine_similarity(z0, z4, dim=-1).mean().item()
    cs_016 = F.cosine_similarity(z0, z16, dim=-1).mean().item()

    print(f"Per-token CosSim: t vs t+1: {cs_01:.6f}")
    print(f"Per-token CosSim: t vs t+4: {cs_04:.6f}")
    print(f"Per-token CosSim: t vs t+16: {cs_016:.6f}")

    # Also check per-token variance across first 100 frames
    var_across_frames = emb[:100].var(axis=0).mean()  # variance per (S,D) position
    print(f"\nPer-position variance across first 100 frames: {var_across_frames:.8f}")

    # Check individual token variation
    for s in [0, S//4, S//2, S-1]:
        token_var = emb[:100, s, :].var(axis=0).mean()
        print(f"  Token {s} variance across frames: {token_var:.8f}")

    return {
        "shape": list(emb.shape),
        "all_frames_identical": bool(all_same),
        "diff_01": float(diff_01),
        "diff_0_1000": float(diff_0_1000),
        "per_token_cossim_h1": cs_01,
        "per_token_cossim_h4": cs_04,
        "per_token_cossim_h16": cs_016,
        "per_position_variance": float(var_across_frames),
    }

def _entry_dec(fn):
    if app: return app.local_entrypoint()(fn)
    return fn

@_entry_dec
def main():
    import json
    for enc in ["vit_s16", "dino_vits14"]:
        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC: {enc}")
        print(f"{'='*60}")
        result = check_spatial.remote(enc)
        print(json.dumps(result, indent=2))
