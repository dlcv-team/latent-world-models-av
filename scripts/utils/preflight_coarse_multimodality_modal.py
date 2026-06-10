"""Gate S': coarse vs fine within-class true-future spread (data-only, Modal CPU).

Verifies that coarse-action grouping hides more multimodality in z_{t+H} than
fine (exact mean-steer) bins on the HV subset (top quartile future_steer_std).

Usage::

    modal run scripts/preflight_coarse_multimodality_modal.py --encoder vit_s16
    modal run scripts/preflight_coarse_multimodality_modal.py --encoder dino_vits14
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("lwm-av-preflight-coarse")
    vol = modal.Volume.from_name("nuscenes-full")
else:
    app = None
    vol = None

VOL_PATH = "/vol"
SPATIAL_DIR = f"{VOL_PATH}/embeddings/spatial"
TARGET_DIM = 384
DEFAULT_THETA = 0.1


def _fn_decorator(fn):
    if app is not None:
        return app.function(
            volumes={VOL_PATH: vol},
            image=modal.Image.debian_slim(python_version="3.12").pip_install(
                "numpy>=1.26"
            ),
            cpu=4,
            memory=16384,
            timeout=7200,
        )(fn)
    return fn


@_fn_decorator
def preflight(
    encoder_name: str = "vit_s16",
    horizon: int = 16,
    theta: float = DEFAULT_THETA,
    max_windows: int | None = None,
    n_pairs_per_group: int = 200,
    min_group_size: int = 8,
):
    import os
    import numpy as np

    path = f"{SPATIAL_DIR}/{encoder_name}_spatial.npz"
    if not os.path.exists(path):
        print(f"ERROR: missing {path}")
        return {"ok": False, "error": "missing_npz"}

    data = np.load(path, allow_pickle=True)
    spatial_emb = data["spatial_embeddings"]
    splits = data["splits"]
    steer_norms = data["steer_norms"]
    accel_norms = data["accel_norms"]
    scene_names = data["scene_names"]

    def build_windows(split_name):
        mask = splits == split_name
        emb = spatial_emb[mask]
        steers = steer_norms[mask]
        accels = accel_norms[mask]
        scenes = scene_names[mask]
        z_t_list, act_list, z_last_list, steer_std_list = [], [], [], []
        for scene in np.unique(scenes):
            idx = np.where(scenes == scene)[0]
            for j in range(len(idx) - horizon):
                act = np.stack(
                    [
                        np.array([steers[idx[j + k]], accels[idx[j + k]]])
                        for k in range(horizon)
                    ]
                )
                act_list.append(act)
                z_t_list.append(emb[idx[j]])
                z_last_list.append(emb[idx[j + horizon]])
                steer_std_list.append(float(np.std(act[:, 0])))
        return (
            np.array(z_t_list, dtype=np.float32),
            np.array(act_list, dtype=np.float32),
            np.array(z_last_list, dtype=np.float32),
            np.array(steer_std_list, dtype=np.float32),
        )

    z_t_tr, act_tr, _, _ = build_windows("train")
    z_t_te, act_te, z_last_te, steer_std_te = build_windows("test")

    mean_steer_tr = act_tr[:, :, 0].mean(axis=1)
    left_c = float(np.percentile(mean_steer_tr, 10))
    right_c = float(np.percentile(mean_steer_tr, 90))

    def coarse_label(ms):
        if ms < -theta:
            return "left"
        if ms > theta:
            return "right"
        return "straight"

    mean_steer_te = act_te[:, :, 0].mean(axis=1)
    coarse_te = np.array([coarse_label(m) for m in mean_steer_te])
    fine_te = np.round(mean_steer_te, 2)

    hv_thr = float(np.quantile(steer_std_te, 0.75))
    hv_mask = steer_std_te >= hv_thr
    hv_idx = np.where(hv_mask)[0]
    print(f"[S'] {encoder_name}: test={len(act_te)} HV={len(hv_idx)} "
          f"hv_thr={hv_thr:.4f} theta={theta}")

    def class_balance(labels):
        u, c = np.unique(labels, return_counts=True)
        return {str(k): int(v) for k, v in zip(u, c)}

    bal = class_balance(coarse_te)
    print(f"[S'] coarse balance (all test): {bal}")
    peak_steer = {}
    for lab in ("left", "straight", "right"):
        m = coarse_te == lab
        if m.any():
            peak_steer[lab] = float(np.max(np.abs(act_te[m, :, 0])))
    print(f"[S'] peak |steer| by coarse class: {peak_steer}")

    def flatten_z_last(indices):
        return z_last_te[indices].reshape(len(indices), -1)

    def within_group_spread(indices, group_key_fn):
        """Mean pairwise L2 and mean (1 - cos) on z_{t+H} flatten."""
        from collections import defaultdict

        groups = defaultdict(list)
        for i in indices:
            groups[group_key_fn(i)].append(i)
        l2_vals, one_minus_cs = [], []
        rng = np.random.default_rng(0)
        for g, idxs in groups.items():
            if len(idxs) < min_group_size:
                continue
            vecs = flatten_z_last(np.array(idxs))
            vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
            n = len(idxs)
            n_pairs = min(n_pairs_per_group, n * (n - 1) // 2)
            for _ in range(n_pairs):
                a, b = rng.choice(n, size=2, replace=False)
                d = vecs[a] - vecs[b]
                l2_vals.append(float(np.linalg.norm(d)))
                one_minus_cs.append(float(1.0 - np.dot(vecs[a], vecs[b])))
        if not l2_vals:
            return {"mean_l2": 0.0, "mean_1_minus_cossim": 0.0, "n_groups": 0}
        return {
            "mean_l2": float(np.mean(l2_vals)),
            "mean_1_minus_cossim": float(np.mean(one_minus_cs)),
            "n_groups": len(groups),
        }

    hv_indices = hv_idx
    if max_windows is not None and len(hv_indices) > max_windows:
        hv_indices = hv_indices[:max_windows]

    coarse_spread = within_group_spread(
        hv_indices, lambda i: coarse_te[i]
    )
    fine_spread = within_group_spread(
        hv_indices, lambda i: fine_te[i]
    )

    ratio_l2 = (
        coarse_spread["mean_l2"] / (fine_spread["mean_l2"] + 1e-8)
    )
    ratio_1mcs = (
        coarse_spread["mean_1_minus_cossim"]
        / (fine_spread["mean_1_minus_cossim"] + 1e-8)
    )
    gate_pass = ratio_l2 > 1.05 and ratio_1mcs > 1.05
    print(f"[S'] HV coarse spread L2={coarse_spread['mean_l2']:.4f} "
          f"1-cos={coarse_spread['mean_1_minus_cossim']:.4f}")
    print(f"[S'] HV fine   spread L2={fine_spread['mean_l2']:.4f} "
          f"1-cos={fine_spread['mean_1_minus_cossim']:.4f}")
    print(f"[S'] ratios L2={ratio_l2:.3f} 1-cos={ratio_1mcs:.3f} "
          f"GATE_S_PRIME={'PASS' if gate_pass else 'FAIL'}")

    out = {
        "encoder": encoder_name,
        "horizon": horizon,
        "theta": theta,
        "left_center": left_c,
        "right_center": right_c,
        "hv_threshold_steer_std": hv_thr,
        "n_test": int(len(act_te)),
        "n_hv": int(len(hv_idx)),
        "coarse_balance_test": bal,
        "peak_abs_steer_by_class": peak_steer,
        "coarse_spread_hv": coarse_spread,
        "fine_spread_hv": fine_spread,
        "ratio_l2_coarse_over_fine": round(ratio_l2, 4),
        "ratio_1_minus_cossim_coarse_over_fine": round(ratio_1mcs, 4),
        "gate_s_prime_pass": gate_pass,
    }
    return out


def _entry(fn):
    if app is not None:
        return app.local_entrypoint()(fn)
    return fn


@_entry
def main(encoder: str = "vit_s16", theta: float = DEFAULT_THETA):
    import os
    import time

    t0 = time.time()
    for enc in ([encoder] if encoder != "both" else ["vit_s16", "dino_vits14"]):
        if app is not None:
            r = preflight.remote(enc, theta=theta)
        else:
            r = preflight(enc, theta=theta)
        out = Path(f"artifacts/full/preflight_gate_s_prime_{enc}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(r, f, indent=2)
        print(json.dumps(r, indent=2))
        print(f"Saved {out}")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__" and modal is None:
    main()
