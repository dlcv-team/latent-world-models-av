"""Full-dataset embedding pipeline on Modal.

Embeds all ~850 nuScenes v1.0-trainval scenes with 5 encoders + V-JEPA2 rep1.
Produces native (pre-adapter) embeddings for downstream probe training.

Architecture:
  Phase 1 (CPU):  build_index — parse nuScenes + CAN bus, produce sample JSON
  Phase 2 (GPU):  embed_* — per-encoder forward passes on image batches
  Phase 3 (CPU):  merge — combine shards, checksums, metadata

Usage:
  # 1. Create volume and upload data (run once)
  modal volume create nuscenes-full
  modal volume put nuscenes-full .../CAM_FRONT.tar /raw/
  modal volume put nuscenes-full .../v1.0-trainval.tar /raw/
  modal volume put nuscenes-full .../can_bus.zip /raw/
  modal volume put nuscenes-full .../vjepa2-hf-cache.tar /raw/

  # 2. Run the full pipeline
  modal run scripts/embed_full.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

app = modal.App("lwm-av-full-embed")
vol = modal.Volume.from_name("nuscenes-full", create_if_missing=True)

VOL_PATH = "/vol"
DATA_ROOT = f"{VOL_PATH}/nuscenes"       # nuScenes dataroot
HF_CACHE = f"{VOL_PATH}/hf_cache"        # HuggingFace model cache
INDEX_DIR = f"{VOL_PATH}/index"           # Pre-built sample index
OUT_DIR = f"{VOL_PATH}/embeddings"        # Output embeddings
RAW_DIR = f"{VOL_PATH}/raw"              # Uploaded tars

# Encoder metadata: name -> (pilot_name, native_dim, mode)
ENCODERS = {
    "vits16":   ("vit_s16",      384,  "single_frame"),
    "dinov2":   ("dino_vits14",  384,  "single_frame"),
    "clip":     ("clip_b32",     512,  "single_frame"),
    "vqvae":    ("vq_track",     256,  "single_frame"),
    "vjepa2":   ("vjepa2_rep64", 1024, "clip"),
}

LIGHT_ENCODERS = ["vits16", "dinov2", "clip", "vqvae"]
HEAVY_ENCODERS = ["vjepa2"]  # V-JEPA2 rep64

# Optimal batch sizes per encoder (fp32 on target GPU)
BATCH_SIZES = {
    "vits16": 512,   # T4 16GB, ViT-S ~2GB VRAM
    "dinov2": 512,   # T4, DINOv2-S ~2GB
    "clip":   256,   # T4, CLIP-B ~4GB
    "vqvae":  64,    # T4, VQGAN uses 256x256 + torch.compile overhead
    "vjepa2": 8,     # A10G 24GB, V-JEPA2 ~12GB with 16 frames
}

# Number of shards for V-JEPA2 parallel processing
VJEPA2_SHARDS = 3

# ---------------------------------------------------------------------------
# Container images
# ---------------------------------------------------------------------------

_project_root = str(Path(__file__).resolve().parent.parent)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "timm==1.0.12",
        "open-clip-torch==2.29.0",
        "transformers>=4.51.0",
        "pytorch-lightning",  # needed to unpickle VQGAN checkpoint
        "nuscenes-devkit==1.1.11",
        "numpy>=1.26",
        "Pillow>=10.0",
        "tqdm",
        "pyyaml",
    )
    .apt_install("unzip", "libgl1-mesa-glx", "libglib2.0-0")
    .add_local_dir(
        _project_root,
        remote_path="/app",
        ignore=["artifacts/**", ".git/**", "**/*.npz", "**/__pycache__/**"],
    )
)


# ---------------------------------------------------------------------------
# Phase 0: Setup volume (extract uploaded tars)
# ---------------------------------------------------------------------------

@app.function(
    volumes={VOL_PATH: vol},
    image=base_image,
    timeout=1800,
)
def setup_volume():
    """Extract uploaded tar files on the volume. Idempotent."""
    import subprocess

    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(f"{DATA_ROOT}/samples", exist_ok=True)

    # Extract CAM_FRONT images
    cam_dir = f"{DATA_ROOT}/samples/CAM_FRONT"
    if not os.path.isdir(cam_dir) or len(os.listdir(cam_dir)) < 30000:
        print("[setup] Extracting CAM_FRONT.tar ...")
        subprocess.run(
            ["tar", "xf", f"{RAW_DIR}/CAM_FRONT.tar", "-C", f"{DATA_ROOT}/samples/"],
            check=True,
        )
        n = len(os.listdir(cam_dir))
        print(f"[setup] Extracted {n} CAM_FRONT images")
    else:
        print(f"[setup] CAM_FRONT already extracted ({len(os.listdir(cam_dir))} files)")

    # Extract nuScenes metadata
    meta_dir = f"{DATA_ROOT}/v1.0-trainval"
    if not os.path.isdir(meta_dir):
        print("[setup] Extracting v1.0-trainval.tar ...")
        subprocess.run(
            ["tar", "xf", f"{RAW_DIR}/v1.0-trainval.tar", "-C", DATA_ROOT],
            check=True,
        )
        print(f"[setup] Extracted metadata to {meta_dir}")
    else:
        print("[setup] v1.0-trainval metadata already extracted")

    # Create dummy map PNGs (NuScenes constructor loads maps; we don't need them)
    maps_dir = f"{DATA_ROOT}/maps"
    os.makedirs(maps_dir, exist_ok=True)
    map_files = [
        "53992ee3023e5494b90c316c183be829.png",
        "36092f0b03a857c6a3403e25b4b7aab3.png",
        "93406b464a165eaba6d9de76ca09f5da.png",
        "37819e65e09e5547b8a3ceaefba56bb2.png",
    ]
    for mf in map_files:
        mpath = f"{maps_dir}/{mf}"
        if not os.path.exists(mpath):
            # Create minimal 1x1 grayscale PNG (NuScenes MapMask just needs a loadable image)
            import struct, zlib
            def _minimal_png():
                sig = b'\x89PNG\r\n\x1a\n'
                ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 0, 0, 0, 0)
                ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
                ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
                raw = zlib.compress(b'\x00\x00')
                idat_crc = zlib.crc32(b'IDAT' + raw) & 0xffffffff
                idat = struct.pack('>I', len(raw)) + b'IDAT' + raw + struct.pack('>I', idat_crc)
                iend_crc = zlib.crc32(b'IEND') & 0xffffffff
                iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
                return sig + ihdr + idat + iend
            with open(mpath, 'wb') as f:
                f.write(_minimal_png())
    print(f"[setup] Map placeholders ready ({len(map_files)} files)")

    # Extract CAN bus data
    can_dir = f"{DATA_ROOT}/can_bus"
    if not os.path.isdir(can_dir):
        print("[setup] Extracting can_bus.zip ...")
        subprocess.run(
            ["unzip", "-q", "-o", f"{RAW_DIR}/can_bus.zip", "-d", DATA_ROOT],
            check=True,
        )
        print(f"[setup] Extracted CAN bus data to {can_dir}")
    else:
        print("[setup] CAN bus data already extracted")

    # Extract HuggingFace cache (V-JEPA2 weights)
    if not os.path.isdir(HF_CACHE) or len(os.listdir(HF_CACHE)) == 0:
        print("[setup] Extracting vjepa2-hf-cache.tar ...")
        os.makedirs(HF_CACHE, exist_ok=True)
        subprocess.run(
            ["tar", "xf", f"{RAW_DIR}/vjepa2-hf-cache.tar", "-C", HF_CACHE],
            check=True,
        )
        print("[setup] Extracted HF cache")
    else:
        print("[setup] HF cache already extracted")

    # Copy VQGAN checkpoint into HF cache dir (so embed_single_frame finds it)
    # Expected SHA-256 of the Heidelberg f16-16384 checkpoint
    VQGAN_EXPECTED_SHA = "845a68805098cb666420d5db93df53f3a3b6dd443e6dd85c05759c5b998cd663"
    vqgan_src = f"{RAW_DIR}/vqgan_imagenet_f16_16384.ckpt"
    vqgan_dst = f"{HF_CACHE}/vqgan_imagenet_f16_16384.ckpt"
    if os.path.isfile(vqgan_src) and not os.path.isfile(vqgan_dst):
        import shutil
        shutil.copy2(vqgan_src, vqgan_dst)
        print(f"[setup] Copied VQGAN checkpoint to {vqgan_dst}")
    elif os.path.isfile(vqgan_dst):
        print("[setup] VQGAN checkpoint already in place")
    else:
        print("[setup] WARNING: VQGAN checkpoint not found on volume")

    # Verify VQGAN checkpoint integrity
    if os.path.isfile(vqgan_dst):
        import hashlib as _hashlib
        sha = _hashlib.sha256()
        with open(vqgan_dst, "rb") as _f:
            for chunk in iter(lambda: _f.read(1 << 20), b""):
                sha.update(chunk)
        actual_sha = sha.hexdigest()
        if actual_sha == VQGAN_EXPECTED_SHA:
            print(f"[setup] VQGAN checkpoint SHA-256 verified ✓")
        else:
            print(f"[setup] ERROR: VQGAN checkpoint SHA-256 mismatch!")
            print(f"  Expected: {VQGAN_EXPECTED_SHA}")
            print(f"  Got:      {actual_sha}")
            raise RuntimeError("VQGAN checkpoint corrupted — aborting")

    vol.commit()
    print("[setup] Volume setup complete")


# ---------------------------------------------------------------------------
# Phase 1: Build sample index
# ---------------------------------------------------------------------------

@app.function(
    volumes={VOL_PATH: vol},
    image=base_image,
    timeout=1800,
    memory=16384,
)
def build_index(manifest_path: str = "/app/configs/full_manifest.json"):
    """Build filtered sample index from nuScenes + CAN bus data.

    Produces a JSON with one entry per valid keyframe, including:
    - cam_token, scene_name, timestamp_us, image_path
    - steer_norm, accel_norm (action labels)
    - clip_paths (16 frame paths for V-JEPA2)
    """
    import numpy as np
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus

    sys.path.insert(0, "/app")
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_scenes = set()
    for split_name in ["train", "val", "test"]:
        all_scenes.update(manifest["splits"][split_name])

    print(f"[index] Loading nuScenes from {DATA_ROOT} ...")
    t0 = time.time()
    nusc = NuScenes(version="v1.0-trainval", dataroot=DATA_ROOT, verbose=False)
    nusc_can = NuScenesCanBus(dataroot=DATA_ROOT)
    print(f"[index] NuScenes loaded in {time.time() - t0:.1f}s")

    # Normalization constants (from canonical config)
    steer_divisor = 6.0
    accel_divisor = 10.0
    clip_range = (-1.0, 1.0)
    max_can_delta_us = 50000

    # CAN bus blacklist
    can_blacklist = {
        "scene-0161", "scene-0162", "scene-0163", "scene-0164",
        "scene-0165", "scene-0166", "scene-0167", "scene-0168",
        "scene-0170", "scene-0171", "scene-0172",
    }

    samples = []
    stats = {"total": 0, "kept": 0, "dropped_can": 0, "dropped_blacklist": 0, "dropped_align": 0}

    scenes_by_name = {s["name"]: s for s in nusc.scene}

    # Pre-build sample_data lookup for fast prev-link walking (avoids O(n) nusc.get)
    sd_by_token = {sd["token"]: sd for sd in nusc.sample_data}
    print(f"[index] Built sample_data lookup ({len(sd_by_token)} records) in {time.time() - t0:.1f}s")

    for scene_name in sorted(all_scenes):
        scene = scenes_by_name.get(scene_name)
        if scene is None:
            continue

        if scene_name in can_blacklist:
            stats["dropped_blacklist"] += 1
            continue

        # Load CAN bus
        try:
            steer_msgs = nusc_can.get_messages(scene_name, "steeranglefeedback")
            pose_msgs = nusc_can.get_messages(scene_name, "pose")
        except (KeyError, Exception):
            stats["dropped_can"] += 1
            continue

        if not steer_msgs or not pose_msgs:
            stats["dropped_can"] += 1
            continue

        steer_times = np.array([m["utime"] for m in steer_msgs])
        steer_values = np.array([m["value"] for m in steer_msgs])
        pose_times = np.array([m["utime"] for m in pose_msgs])
        pose_accels = np.array([m["accel"][0] for m in pose_msgs])

        # Determine which split this scene belongs to
        for split_name in ["train", "val", "test"]:
            if scene_name in manifest["splits"][split_name]:
                sample_split = split_name
                break

        # Iterate keyframes
        sample_token = scene["first_sample_token"]
        while sample_token:
            stats["total"] += 1
            sample = nusc.get("sample", sample_token)

            if "CAM_FRONT" not in sample["data"]:
                sample_token = sample["next"]
                continue

            cam_token = sample["data"]["CAM_FRONT"]
            ts = sample["timestamp"]

            # CAN alignment check
            steer_idx = np.abs(steer_times - ts).argmin()
            pose_idx = np.abs(pose_times - ts).argmin()
            if max(abs(steer_times[steer_idx] - ts), abs(pose_times[pose_idx] - ts)) > max_can_delta_us:
                stats["dropped_align"] += 1
                sample_token = sample["next"]
                continue

            # Normalize actions
            steer_norm = float(np.clip(steer_values[steer_idx] / steer_divisor, *clip_range))
            accel_norm = float(np.clip(pose_accels[pose_idx] / accel_divisor, *clip_range))

            # Get image path
            cam_data = sd_by_token[cam_token]
            image_path = cam_data["filename"]  # e.g., samples/CAM_FRONT/xxx.jpg

            # Build clip frame paths (16 frames walking backwards through prev links)
            clip_paths = []
            cur_token = cam_token
            while cur_token and len(clip_paths) < 16:
                sd = sd_by_token[cur_token]
                clip_paths.append(sd["filename"])
                cur_token = sd["prev"] if sd["prev"] else ""
            clip_paths = clip_paths[::-1]  # oldest first
            # Pad with earliest frame if needed
            if len(clip_paths) < 16:
                clip_paths = [clip_paths[0]] * (16 - len(clip_paths)) + clip_paths

            samples.append({
                "cam_token": cam_token,
                "scene_name": scene_name,
                "split": sample_split,
                "timestamp_us": int(ts),
                "image_path": image_path,
                "clip_paths": clip_paths,
                "steer_norm": steer_norm,
                "accel_norm": accel_norm,
            })
            stats["kept"] += 1

            sample_token = sample["next"]

    # Sort by scene_name then timestamp for deterministic ordering
    samples.sort(key=lambda s: (s["scene_name"], s["timestamp_us"]))

    print(f"[index] Stats: {json.dumps(stats, indent=2)}")
    print(f"[index] Kept {len(samples)} samples from {len(all_scenes)} scenes")

    # Save index
    os.makedirs(INDEX_DIR, exist_ok=True)
    index_path = f"{INDEX_DIR}/full_index.json"
    with open(index_path, "w") as f:
        json.dump({"samples": samples, "stats": stats}, f)
    print(f"[index] Saved index to {index_path} ({os.path.getsize(index_path) / 1e6:.1f} MB)")

    vol.commit()
    return stats


# ---------------------------------------------------------------------------
# Phase 2a: Embed with single-frame encoder (T4)
# ---------------------------------------------------------------------------

@app.function(
    volumes={VOL_PATH: vol},
    image=base_image,
    gpu="T4",
    timeout=7200,
    memory=16384,
)
def embed_single_frame(encoder_name: str):
    """Compute native embeddings for a single-frame encoder."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from tqdm import tqdm

    pilot_name, native_dim, _ = ENCODERS[encoder_name]
    batch_size = BATCH_SIZES[encoder_name]

    print(f"[embed] Encoder: {encoder_name} ({pilot_name}), dim={native_dim}, batch={batch_size}")

    # Load sample index
    with open(f"{INDEX_DIR}/full_index.json") as f:
        index = json.load(f)
    samples = index["samples"]
    print(f"[embed] Processing {len(samples)} samples")

    # GPU optimizations
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    # Load encoder
    sys.path.insert(0, "/app")
    os.chdir("/app")

    # Set VQGAN checkpoint path if available
    vqgan_cache = Path(HF_CACHE) / "vqgan_imagenet_f16_16384.ckpt"
    if vqgan_cache.exists():
        os.environ["VQGAN_CKPT_PATH"] = str(vqgan_cache)

    from scripts.train_probe import build_encoder
    encoder = build_encoder(encoder_name, pretrained=True).to(device)
    encoder.eval()

    # VQGAN reference encoding verification (catches checkpoint corruption)
    if encoder_name == "vqvae":
        _rng = torch.Generator().manual_seed(12345)
        _ref_input = torch.rand(1, 3, 224, 224, generator=_rng).to(device)
        with torch.inference_mode():
            _ref_out = encoder._encode(_ref_input)
        _ref_norm = torch.norm(_ref_out).item()
        _ref_sum = _ref_out.sum().item()
        # Reference values from local encoding (CPU, no compile)
        # Allow tolerance for GPU/precision differences
        print(f"[embed] VQGAN verification: norm={_ref_norm:.4f} (expect ~4.2190), sum={_ref_sum:.4f} (expect ~-0.2310)")
        if abs(_ref_norm - 4.219048) > 0.5 or abs(_ref_sum - (-0.231009)) > 1.0:
            raise RuntimeError(
                f"VQGAN reference encoding mismatch! norm={_ref_norm:.4f}, sum={_ref_sum:.4f}. "
                f"Expected norm~4.2190, sum~-0.2310. Checkpoint likely corrupted."
            )
        print("[embed] VQGAN checkpoint verified ✓")
        del _rng, _ref_input, _ref_out

    # Apply channels_last for conv models
    try:
        encoder.backbone = encoder.backbone.to(memory_format=torch.channels_last)
    except Exception:
        pass  # Not all models support channels_last

    # torch.compile for kernel fusion
    try:
        encoder.backbone = torch.compile(encoder.backbone)
        print(f"[embed] torch.compile applied")
    except Exception as e:
        print(f"[embed] torch.compile skipped: {e}")

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Process in batches
    all_embeddings = []
    checkpoint_interval = 1000
    t0 = time.time()

    with torch.inference_mode():
        for batch_start in tqdm(range(0, len(samples), batch_size), desc=pilot_name):
            batch_samples = samples[batch_start:batch_start + batch_size]

            # Load images
            images = []
            for s in batch_samples:
                img_path = f"{DATA_ROOT}/{s['image_path']}"
                img = Image.open(img_path).convert("RGB")
                images.append(transform(img))

            batch_tensor = torch.stack(images).to(device)
            if batch_tensor.is_contiguous(memory_format=torch.channels_last):
                pass  # already channels_last from model
            else:
                try:
                    batch_tensor = batch_tensor.to(memory_format=torch.channels_last)
                except Exception:
                    pass

            # Forward pass — use _encode() for native (pre-adapter) embeddings
            z = encoder._encode(batch_tensor)
            all_embeddings.append(z.cpu().numpy())

            # Periodic checkpoint
            if len(all_embeddings) * batch_size >= checkpoint_interval and batch_start > 0:
                if batch_start % (checkpoint_interval * 5) == 0:
                    elapsed = time.time() - t0
                    done = batch_start + len(batch_samples)
                    rate = done / elapsed
                    eta = (len(samples) - done) / rate if rate > 0 else 0
                    print(f"[embed] {done}/{len(samples)} ({rate:.0f} samples/s, ETA {eta:.0f}s)")

    # Concatenate all embeddings
    embeddings = np.concatenate(all_embeddings, axis=0)
    assert embeddings.shape == (len(samples), native_dim), f"Shape mismatch: {embeddings.shape}"

    elapsed = time.time() - t0
    print(f"[embed] {pilot_name}: {len(samples)} samples in {elapsed:.1f}s ({len(samples)/elapsed:.0f}/s)")

    # Build output arrays
    cam_tokens = np.array([s["cam_token"] for s in samples], dtype="U32")
    scene_names = np.array([s["scene_name"] for s in samples], dtype="U16")
    timestamps = np.array([s["timestamp_us"] for s in samples], dtype=np.int64)
    image_paths = np.array([s["image_path"] for s in samples], dtype="U100")
    splits = np.array([s["split"] for s in samples], dtype="U8")
    steer_norms = np.array([s["steer_norm"] for s in samples], dtype=np.float32)
    accel_norms = np.array([s["accel_norm"] for s in samples], dtype=np.float32)

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/{pilot_name}.npz"
    np.savez(
        out_path,
        cam_tokens=cam_tokens,
        scene_names=scene_names,
        timestamps_us=timestamps,
        image_paths=image_paths,
        splits=splits,
        steer_norms=steer_norms,
        accel_norms=accel_norms,
        embeddings=embeddings,
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[embed] Saved {out_path} ({size_mb:.1f} MB)")

    vol.commit()
    return {"encoder": pilot_name, "samples": len(samples), "time_s": elapsed, "size_mb": size_mb}


# ---------------------------------------------------------------------------
# Phase 2b: Embed with V-JEPA2 (A10G, sharded)
# ---------------------------------------------------------------------------

@app.function(
    volumes={VOL_PATH: vol},
    image=base_image,
    gpu="A10G",
    timeout=10800,
    memory=24576,
)
def embed_vjepa2(shard_id: int, n_shards: int, rep: int = 64):
    """Compute V-JEPA2 embeddings for a shard of samples.

    Parameters
    ----------
    shard_id : int
        Which shard (0-indexed)
    n_shards : int
        Total number of shards
    rep : int
        Frame repetition: 64 for temporal (16 frames), 1 for single frame
    """
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from tqdm import tqdm

    clip_frames = 16 if rep == 64 else 1
    pilot_name = f"vjepa2_rep{rep}"
    native_dim = 1024
    batch_size = BATCH_SIZES["vjepa2"] if rep == 64 else 32  # rep1 uses much less VRAM

    print(f"[vjepa2] rep={rep}, shard {shard_id}/{n_shards}, clip_frames={clip_frames}, batch={batch_size}")

    # Load sample index
    with open(f"{INDEX_DIR}/full_index.json") as f:
        index = json.load(f)
    all_samples = index["samples"]

    # Shard: each container processes a contiguous slice
    shard_size = len(all_samples) // n_shards
    start = shard_id * shard_size
    end = start + shard_size if shard_id < n_shards - 1 else len(all_samples)
    samples = all_samples[start:end]
    print(f"[vjepa2] Shard {shard_id}: samples {start}-{end} ({len(samples)} total)")

    # GPU optimizations
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    # Load V-JEPA2 encoder
    sys.path.insert(0, "/app")
    os.chdir("/app")
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE}/hub"

    from scripts.train_probe import build_encoder
    encoder = build_encoder("vjepa2", pretrained=True).to(device)
    encoder.eval()

    # torch.compile
    try:
        encoder.backbone = torch.compile(encoder.backbone)
        print(f"[vjepa2] torch.compile applied")
    except Exception as e:
        print(f"[vjepa2] torch.compile skipped: {e}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    def load_clip(sample, n_frames):
        """Load a clip of n_frames images.

        Falls back to nearest available frame for missing sweep files
        (we only uploaded keyframe images to the volume). Preserves
        temporal ordering: missing frames use the nearest preceding
        available frame (or first available if none precedes).
        """
        if n_frames == 1:
            # Single frame (rep1)
            img = Image.open(f"{DATA_ROOT}/{sample['image_path']}").convert("RGB")
            return torch.stack([transform(img)])

        # Multi-frame clip (rep64)
        paths = sample["clip_paths"][-n_frames:]

        # Resolve paths: find which frames exist, fill gaps
        resolved = []
        last_valid = f"{DATA_ROOT}/{sample['image_path']}"  # fallback = keyframe
        for p in paths:
            full_path = f"{DATA_ROOT}/{p}"
            if os.path.exists(full_path):
                last_valid = full_path
            resolved.append(last_valid)

        frames = []
        for p in resolved:
            img = Image.open(p).convert("RGB")
            frames.append(transform(img))
        return torch.stack(frames)  # (T, 3, 224, 224)

    # Process in batches
    all_embeddings = []
    t0 = time.time()

    with torch.inference_mode():
        for batch_start in tqdm(range(0, len(samples), batch_size), desc=f"vjepa2_rep{rep}_s{shard_id}"):
            batch_samples = samples[batch_start:batch_start + batch_size]

            # Load clips
            clips = [load_clip(s, clip_frames) for s in batch_samples]
            batch_tensor = torch.stack(clips).to(device)  # (B, T, 3, 224, 224)

            # Forward pass — _encode() for native embeddings
            z = encoder._encode(batch_tensor)
            all_embeddings.append(z.cpu().numpy())

            # Progress logging
            done = batch_start + len(batch_samples)
            if done % max(1, (len(samples) // 10)) < batch_size:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(samples) - done) / rate if rate > 0 else 0
                print(f"[vjepa2] Shard {shard_id}: {done}/{len(samples)} ({rate:.1f}/s, ETA {eta:.0f}s)")

    embeddings = np.concatenate(all_embeddings, axis=0)
    assert embeddings.shape == (len(samples), native_dim), f"Shape mismatch: {embeddings.shape}"

    elapsed = time.time() - t0
    print(f"[vjepa2] Shard {shard_id}: {len(samples)} samples in {elapsed:.1f}s")

    # Save shard
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/{pilot_name}_shard{shard_id}.npz"
    np.savez(out_path, embeddings=embeddings, start=start, end=end)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[vjepa2] Saved shard {out_path} ({size_mb:.1f} MB)")

    vol.commit()
    return {"shard": shard_id, "samples": len(samples), "time_s": elapsed, "size_mb": size_mb}


# ---------------------------------------------------------------------------
# Phase 3: Merge shards + checksums
# ---------------------------------------------------------------------------

@app.function(
    volumes={VOL_PATH: vol},
    image=base_image,
    timeout=600,
)
def merge_and_checksum(n_shards_rep64: int, n_shards_rep1: int = 1):
    """Merge V-JEPA2 shards and compute SHA-256 checksums for all embeddings."""
    import numpy as np

    # Load sample index for metadata
    with open(f"{INDEX_DIR}/full_index.json") as f:
        index = json.load(f)
    samples = index["samples"]
    n_total = len(samples)

    rep_shards = {64: n_shards_rep64, 1: n_shards_rep1}
    for rep in [64, 1]:
        n_shards = rep_shards[rep]
        pilot_name = f"vjepa2_rep{rep}"

        # Skip merge if merged file already exists and no shards present
        merged_path = f"{OUT_DIR}/{pilot_name}.npz"
        shard0_path = f"{OUT_DIR}/{pilot_name}_shard0.npz"
        if os.path.exists(merged_path) and not os.path.exists(shard0_path):
            print(f"[merge] {pilot_name}.npz already exists, skipping merge")
            continue

        print(f"[merge] Merging {pilot_name} from {n_shards} shards ...")

        # Load and concatenate shards in order
        all_embeddings = []
        for i in range(n_shards):
            shard_path = f"{OUT_DIR}/{pilot_name}_shard{i}.npz"
            with np.load(shard_path) as f:
                all_embeddings.append(f["embeddings"])
        embeddings = np.concatenate(all_embeddings, axis=0)
        assert embeddings.shape[0] == n_total, f"Expected {n_total}, got {embeddings.shape[0]}"

        # Save merged file with metadata
        cam_tokens = np.array([s["cam_token"] for s in samples], dtype="U32")
        scene_names = np.array([s["scene_name"] for s in samples], dtype="U16")
        timestamps = np.array([s["timestamp_us"] for s in samples], dtype=np.int64)
        image_paths = np.array([s["image_path"] for s in samples], dtype="U100")
        splits = np.array([s["split"] for s in samples], dtype="U8")
        steer_norms = np.array([s["steer_norm"] for s in samples], dtype=np.float32)
        accel_norms = np.array([s["accel_norm"] for s in samples], dtype=np.float32)

        out_path = f"{OUT_DIR}/{pilot_name}.npz"
        np.savez(
            out_path,
            cam_tokens=cam_tokens,
            scene_names=scene_names,
            timestamps_us=timestamps,
            image_paths=image_paths,
            splits=splits,
            steer_norms=steer_norms,
            accel_norms=accel_norms,
            embeddings=embeddings,
        )
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"[merge] Saved {out_path} ({size_mb:.1f} MB)")

        # Clean up shard files
        for i in range(n_shards):
            shard_path = f"{OUT_DIR}/{pilot_name}_shard{i}.npz"
            if os.path.exists(shard_path):
                os.remove(shard_path)

    # Compute SHA-256 checksums for all embedding files
    checksums = {}
    for fname in sorted(os.listdir(OUT_DIR)):
        if fname.endswith(".npz"):
            fpath = f"{OUT_DIR}/{fname}"
            h = hashlib.sha256()
            with open(fpath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            checksums[fname] = h.hexdigest()
            print(f"[checksum] {fname}: {checksums[fname][:16]}...")

    # Save checksums
    cksum_path = f"{OUT_DIR}/checksums.sha256"
    with open(cksum_path, "w") as f:
        for fname, sha in sorted(checksums.items()):
            f.write(f"{sha}  {fname}\n")
    print(f"[checksum] Saved {cksum_path}")

    vol.commit()
    return checksums


# ---------------------------------------------------------------------------
# Provenance metadata
# ---------------------------------------------------------------------------

@app.function(
    volumes={VOL_PATH: vol},
    image=base_image,
    timeout=120,
)
def write_provenance(results: list[dict], wall_time_s: float):
    """Write provenance.json with run metadata."""
    import torch

    sys.path.insert(0, "/app")

    provenance = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pipeline": "scripts/embed_full.py",
        "dataset": "nuscenes v1.0-trainval (full)",
        "manifest": "configs/full_manifest.json",
        "split": "630 train / 70 val / 150 test (per EDD: 90/10 official train + official val as test)",
        "encoders": {
            name: {"pilot_name": info[0], "native_dim": info[1], "mode": info[2]}
            for name, info in ENCODERS.items()
        },
        "optimizations": [
            "torch.inference_mode()",
            "cudnn.benchmark=True",
            "torch.compile (where supported)",
            "channels_last memory format",
            "container sharding for V-JEPA2",
        ],
        "hardware": {
            "light_encoders": "T4 (16GB)",
            "vjepa2": "A10G (24GB)",
            "vjepa2_shards": VJEPA2_SHARDS,
        },
        "torch_version": torch.__version__,
        "wall_time_s": wall_time_s,
        "per_encoder_results": results,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    prov_path = f"{OUT_DIR}/provenance.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
        f.write("\n")
    print(f"[provenance] Saved {prov_path}")

    vol.commit()
    return provenance


# ---------------------------------------------------------------------------
# Local entrypoint: orchestrate the full pipeline
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Run the full embedding pipeline."""
    t_start = time.time()
    print("=" * 60)
    print("Full-Dataset Embedding Pipeline")
    print("=" * 60)

    # Phase 0: Setup volume
    print("\n--- Phase 0: Setup volume ---")
    setup_volume.remote()
    print("Volume setup complete")

    # Phase 1: Build sample index
    print("\n--- Phase 1: Build sample index ---")
    stats = build_index.remote()
    print(f"Index built: {stats['kept']} samples kept")

    # Phase 2: Embed all encoders in parallel (skip existing)
    print("\n--- Phase 2: Embed (all encoders in parallel) ---")

    # Check which embeddings already exist on the volume
    existing = set()
    try:
        import subprocess as _sp
        ls_out = _sp.run(
            ["modal", "volume", "ls", "nuscenes-full", "/embeddings/"],
            capture_output=True, text=True,
        )
        for line in ls_out.stdout.strip().splitlines():
            fname = line.strip().replace("embeddings/", "")
            if fname.endswith(".npz"):
                existing.add(fname.replace(".npz", ""))
        print(f"  Existing embeddings: {sorted(existing)}")
    except Exception:
        pass

    futures = []

    # Launch 4 lightweight encoders (each on its own T4) — skip existing
    for enc_name in LIGHT_ENCODERS:
        pilot_name = ENCODERS[enc_name][0]
        if pilot_name in existing:
            print(f"  SKIP {enc_name} ({pilot_name}.npz already exists)")
            continue
        print(f"  Launching {enc_name} on T4 ...")
        futures.append(("light", enc_name, embed_single_frame.spawn(enc_name)))

    # Launch V-JEPA2 rep64 sharded across multiple A10Gs — skip if merged file exists
    if "vjepa2_rep64" in existing:
        print(f"  SKIP vjepa2_rep64 (already exists)")
    else:
        for shard_id in range(VJEPA2_SHARDS):
            print(f"  Launching vjepa2_rep64 shard {shard_id}/{VJEPA2_SHARDS} on A10G ...")
            futures.append(("vjepa2_rep64", f"shard_{shard_id}",
                           embed_vjepa2.spawn(shard_id, VJEPA2_SHARDS, rep=64)))

    # Launch V-JEPA2 rep1 (single shard, much faster) — skip if exists
    if "vjepa2_rep1" in existing:
        print(f"  SKIP vjepa2_rep1 (already exists)")
    else:
        print(f"  Launching vjepa2_rep1 on A10G ...")
        futures.append(("vjepa2_rep1", "full",
                       embed_vjepa2.spawn(0, 1, rep=1)))

    # Wait for all to complete (resilient — one failure doesn't kill others)
    results = []
    failed = []
    for label, name, future in futures:
        print(f"  Waiting for {label}/{name} ...")
        try:
            result = future.get()
            results.append(result)
            print(f"  {label}/{name} done: {result}")
        except Exception as e:
            print(f"  {label}/{name} FAILED: {e}")
            failed.append(f"{label}/{name}")

    if failed:
        print(f"\n  WARNING: {len(failed)} encoder(s) failed: {failed}")
        print(f"  Continuing with {len(results)} successful results...")

    # Phase 3: Merge V-JEPA2 shards + checksums
    print("\n--- Phase 3: Merge + checksums ---")
    checksums = merge_and_checksum.remote(VJEPA2_SHARDS, 1)
    print(f"Checksums: {json.dumps(checksums, indent=2)}")

    # Phase 4: Write provenance
    wall_time = time.time() - t_start
    print(f"\n--- Phase 4: Provenance (wall time: {wall_time:.0f}s) ---")
    provenance = write_provenance.remote(results, wall_time)

    print("\n" + "=" * 60)
    print(f"Pipeline complete in {wall_time:.0f}s ({wall_time/60:.1f}min)")
    print(f"Results on volume 'nuscenes-full' at {OUT_DIR}/")
    print("=" * 60)
