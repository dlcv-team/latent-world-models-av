"""Corrected Gate S' : SCENE-CONTROLLED conditional multimodality (data-only).

The first preflight
(`preflight_coarse_multimodality_modal.py`) grouped true futures by ACTION
label alone (coarse 3-bin vs round(mean_steer,2) fine), never controlling for
the current scene z_t. That ratio (~1.03) is dominated by scene diversity and
does NOT measure the diffusion-relevant quantity.

This script measures the quantity that decides whether diffusion can help:

    Given a SIMILAR SCENE (z_t nearest-neighbours) and the SAME coarse intent,
    how spread are the true futures z_{t+H}, relative to what the deterministic
    direct model already achieves (eps_direct ~= 1 - 0.81 = 0.19)?

Metrics on the HV subset (top quartile by per-window future steer std):

All spreads are PER-TOKEN mean (1-cos) on the future spatial tokens, matching
the metric behind eps_direct (1 - per-token mean CosSim of the direct model).

  sigma_global       : anchor-future vs random-other-future            [upper bound]
  sigma_scene        : anchor-future vs K scene-NN futures (any intent)
  sigma_scene_coarse : ... scene-NN that SHARE the coarse intent       [KEY]
  sigma_scene_exact  : ... scene-NN, same coarse intent AND |d mean_steer|<tol
                                                                       [floor / non-action]

Decision (per encoder, at the canonical H=16):
  * sigma_scene_coarse << eps_direct  -> futures tight given scene+intent;
    diffusion cannot beat direct. Boundary confirmed -> narrative branch.
  * sigma_scene_coarse >~ eps_direct  -> real multimodality a point predictor
    must average over -> ONE targeted diffusion run is justified.
  * sigma_scene_coarse - sigma_scene_exact : action-recoverable part (already
    exploited by exact-action direct); sigma_scene_exact : irreducible floor.

Horizon sweep H in {4,8,16,24,32} (2,4,8,12,16 s ahead at 2 Hz, stride 1) maps
whether conditional multimodality grows with the prediction time-horizon.

Usage:
    modal run scripts/preflight_conditional_multimodality_modal.py --encoder vit_s16
    modal run scripts/preflight_conditional_multimodality_modal.py --encoder dino_vits14
"""

from __future__ import annotations

import json
import time
from pathlib import Path

try:
    import modal
    app = modal.App("lwm-av-preflight-condmm")
    vol = modal.Volume.from_name("nuscenes-full")
except ImportError:
    modal = None
    app = None
    vol = None

VOL_PATH = "/vol"
SPATIAL_DIR = f"{VOL_PATH}/embeddings/spatial"
TARGET_DIM = 384
HORIZON = 16
THETA = 0.1                 # coarse-intent absolute threshold on mean future steer
K_NEIGHBORS = 12            # scene nearest-neighbours per anchor
STEER_MATCH_TOL = 0.02      # |d mean_steer| for the exact-action control
MIN_NEIGHBORS = 6           # skip anchors that cannot meet a constraint
HORIZONS = [4, 8, 16, 24, 32]   # steps ahead at stride 1 (2,4,8,12,16 s @ 2 Hz)
N_RANDOM_OTHERS = 64        # sample size for sigma_global per anchor

# Reference: direct-anchored spatial DiT mean CosSim (3-seed), from
# spatial_dit_direct_residual_result_*; eps_direct = 1 - mean_cossim.
EPS_DIRECT = {"vit_s16": 1.0 - 0.8115, "dino_vits14": 1.0 - 0.8047}

if modal is not None:
    image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy>=1.26")
else:
    image = None


def _fn(fn):
    if app is not None:
        return app.function(volumes={VOL_PATH: vol}, image=image,
                            cpu=8.0, memory=32768, timeout=3600)(fn)
    return fn


@_fn
def preflight(encoder_name: str):
    import numpy as np

    path = f"{SPATIAL_DIR}/{encoder_name}_spatial.npz"
    data = np.load(path, allow_pickle=True)
    spatial = data["spatial_embeddings"].astype(np.float32)   # (N, S, 384)
    splits = data["splits"]
    steer = data["steer_norms"].astype(np.float32)
    scenes = data["scene_names"]
    N, S, D = spatial.shape
    print(f"[condmm] {encoder_name}: {spatial.shape}")

    test_mask = splits == "test"
    sp = spatial[test_mask]
    st = steer[test_mask]
    sc = scenes[test_mask]

    rng = np.random.default_rng(0)

    def build_windows(H):
        """Scene-respecting sliding windows; future endpoint is z_{t+H} at stride 1.

        Returns z_t (n,S,D), z_last=z_{t+H} (n,S,D), mean/std of steer over t+1..t+H.
        """
        z_t, z_last, mean_steer, steer_std = [], [], [], []
        for scene in np.unique(sc):
            idx = np.where(sc == scene)[0]
            for j in range(len(idx) - H):
                fut = idx[j + 1 : j + 1 + H]   # t+1 .. t+H, contiguous (stride 1)
                z_t.append(sp[idx[j]])
                z_last.append(sp[fut[-1]])     # z_{t+H}
                fs = st[fut]
                mean_steer.append(float(fs.mean()))
                steer_std.append(float(fs.std()))
        return (np.array(z_t, dtype=np.float32), np.array(z_last, dtype=np.float32),
                np.array(mean_steer, dtype=np.float32), np.array(steer_std, dtype=np.float32))

    def l2n(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    def coarse(ms):
        return np.where(ms < -THETA, 0, np.where(ms > THETA, 2, 1))  # L/straight/R

    out = {"encoder": encoder_name, "eps_direct_h16": round(EPS_DIRECT[encoder_name], 4),
           "theta": THETA, "k_neighbors": K_NEIGHBORS, "stride": 1,
           "metric": "per_token_mean_1_minus_cos", "by_horizon": {}}

    for H in HORIZONS:
        z_t, z_last, ms, ss = build_windows(H)
        n = len(z_t)
        if n < 200:
            print(f"[condmm] H={H}: too few windows ({n}), skip")
            continue

        intent = coarse(ms)
        pool = l2n(z_t.reshape(n, S, D).mean(axis=1))        # (n,384) scene gist, unit
        fut_tok = l2n(z_last.reshape(n, S, D))               # (n,S,D) per-token unit
        hv_thr = float(np.quantile(ss, 0.75))
        hv = np.where(ss >= hv_thr)[0]

        # per-token mean (1-cos) between anchor future and a set of neighbour futures
        def spread(a, nb):
            if len(nb) == 0:
                return None
            # (m,S) per-token cos -> (m,) per-window mean -> scalar mean over neighbours
            cos_ms = np.einsum("msd,sd->ms", fut_tok[nb], fut_tok[a])
            return float((1.0 - cos_ms).mean(axis=1).mean())

        sims = pool[hv] @ pool.T                              # (n_hv, n)
        s_global, s_scene, s_scene_coarse, s_scene_exact, nbr_tight = [], [], [], [], []
        cov_coarse = cov_exact = 0

        for r, a in enumerate(hv):
            row = sims[r].copy()
            row[a] = -2.0
            order = np.argsort(-row)                          # scene similarity desc

            others = rng.choice(np.delete(np.arange(n), a),
                                size=min(N_RANDOM_OTHERS, n - 1), replace=False)
            s_global.append(spread(a, others))

            knn = order[:K_NEIGHBORS]
            s_scene.append(spread(a, knn))
            nbr_tight.append(float(row[knn].mean()))

            same_int = order[intent[order] == intent[a]][:K_NEIGHBORS]
            if len(same_int) >= MIN_NEIGHBORS:
                s_scene_coarse.append(spread(a, same_int)); cov_coarse += 1

            cand = order[(intent[order] == intent[a]) &
                         (np.abs(ms[order] - ms[a]) < STEER_MATCH_TOL)][:K_NEIGHBORS]
            if len(cand) >= MIN_NEIGHBORS:
                s_scene_exact.append(spread(a, cand)); cov_exact += 1

        def agg(v):
            return round(float(np.mean(v)), 4) if v else None

        rec = {
            "seconds_ahead": round(H * 0.5, 1), "n_windows": n, "n_hv": len(hv),
            "hv_thr": round(hv_thr, 4),
            "intent_balance": {int(k): int(c) for k, c in zip(*np.unique(intent, return_counts=True))},
            "mean_scene_nn_cossim": agg(nbr_tight),
            "sigma_global": agg(s_global),
            "sigma_scene": agg(s_scene),
            "sigma_scene_coarse": agg(s_scene_coarse),
            "sigma_scene_exact": agg(s_scene_exact),
            "coverage_coarse": round(cov_coarse / len(hv), 3),
            "coverage_exact": round(cov_exact / len(hv), 3),
        }
        if rec["sigma_scene_coarse"] is not None:
            rec["ratio_condmm_over_eps_direct_h16"] = round(
                rec["sigma_scene_coarse"] / EPS_DIRECT[encoder_name], 3)
            rec["action_recoverable"] = (
                round(rec["sigma_scene_coarse"] - rec["sigma_scene_exact"], 4)
                if rec["sigma_scene_exact"] is not None else None)
        out["by_horizon"][str(H)] = rec
        print(f"[condmm] {encoder_name} H={H} ({rec['seconds_ahead']}s): "
              f"scene_coarse={rec['sigma_scene_coarse']} vs eps_direct(h16)={out['eps_direct_h16']} "
              f"ratio={rec.get('ratio_condmm_over_eps_direct_h16')} | "
              f"scene={rec['sigma_scene']} exact={rec['sigma_scene_exact']} "
              f"global={rec['sigma_global']} nn_cos={rec['mean_scene_nn_cossim']} "
              f"cov_coarse={rec['coverage_coarse']}")

    return out


def _entry(fn):
    return app.local_entrypoint()(fn) if app is not None else fn


@_entry
def main(encoder: str = "vit_s16"):
    t = time.time()
    res = preflight.remote(encoder)
    print(json.dumps(res, indent=2))
    p = Path(f"artifacts/full/preflight_condmm_{encoder}.json")
    if not p.parent.exists():
        p = Path(f"code/latent-world-models-av/artifacts/full/preflight_condmm_{encoder}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2))
    print(f"Saved to {p}  ({time.time()-t:.0f}s)")
