"""Debug: Check if images are accessible on Modal volume."""
from __future__ import annotations
import os

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("lwm-debug-paths")
    vol = modal.Volume.from_name("nuscenes-full")
    base_image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy>=1.26")
else:
    app = None; vol = None; base_image = None

VOL_PATH = "/vol"

def _dec(fn):
    if app: return app.function(volumes={VOL_PATH: vol}, image=base_image, timeout=300)(fn)
    return fn

@_dec
def check_paths():
    import numpy as np

    # Load image paths from embeddings
    data = np.load(f"{VOL_PATH}/embeddings/vit_s16.npz", allow_pickle=True)
    image_paths = data["image_paths"]
    print(f"Total paths: {len(image_paths)}")
    print(f"Sample paths: {image_paths[:3]}")

    # Check various path prefixes
    sample_path = str(image_paths[0])
    prefixes = [
        f"/vol/nuscenes/{sample_path}",
        f"/vol/nuscenes/trainval/{sample_path}",
        f"/vol/{sample_path}",
        sample_path,
    ]
    for p in prefixes:
        exists = os.path.exists(p)
        print(f"  {p} -> {'EXISTS' if exists else 'NOT FOUND'}")

    # Check directory structure
    for d in ["/vol/nuscenes", "/vol/nuscenes/samples", "/vol/nuscenes/samples/CAM_FRONT",
              "/vol/nuscenes/trainval", "/vol/nuscenes/trainval/samples"]:
        if os.path.exists(d):
            n = len(os.listdir(d)) if os.path.isdir(d) else "file"
            print(f"  DIR {d}: {n} items")
        else:
            print(f"  DIR {d}: NOT FOUND")

    # Find actual CAM_FRONT location
    import subprocess
    result = subprocess.run(["find", "/vol/nuscenes", "-name", "CAM_FRONT", "-type", "d"],
                          capture_output=True, text=True, timeout=30)
    print(f"\nfind CAM_FRONT dirs: {result.stdout.strip()}")

    # Check if any jpg exists
    result2 = subprocess.run(["find", "/vol/nuscenes", "-name", "*.jpg", "-type", "f"],
                           capture_output=True, text=True, timeout=30)
    jpgs = result2.stdout.strip().split('\n')
    print(f"Total jpg files found: {len(jpgs)}")
    if jpgs and jpgs[0]:
        print(f"First jpg: {jpgs[0]}")

def _entry_dec(fn):
    if app: return app.local_entrypoint()(fn)
    return fn

@_entry_dec
def main():
    check_paths.remote()
