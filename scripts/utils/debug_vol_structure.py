"""Check what's actually on the Modal volume."""
from __future__ import annotations
import os
try:
    import modal
except ImportError:
    modal = None
if modal is not None:
    app = modal.App("lwm-debug-vol")
    vol = modal.Volume.from_name("nuscenes-full")
    base_image = modal.Image.debian_slim(python_version="3.12")
else:
    app = None; vol = None; base_image = None

def _dec(fn):
    if app: return app.function(volumes={"/vol": vol}, image=base_image, timeout=300)(fn)
    return fn

@_dec
def check_vol():
    for d in ["/vol", "/vol/raw", "/vol/embeddings", "/vol/embeddings/spatial",
              "/vol/dits", "/vol/index", "/vol/hf_cache"]:
        if os.path.exists(d) and os.path.isdir(d):
            items = os.listdir(d)
            print(f"{d}: {len(items)} items")
            for item in sorted(items)[:10]:
                full = os.path.join(d, item)
                if os.path.isfile(full):
                    sz = os.path.getsize(full) / 1e6
                    print(f"  {item} ({sz:.1f} MB)")
                else:
                    print(f"  {item}/")
        else:
            print(f"{d}: NOT FOUND")

def _entry_dec(fn):
    if app: return app.local_entrypoint()(fn)
    return fn

@_entry_dec
def main():
    check_vol.remote()
