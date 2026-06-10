#!/usr/bin/env python3
"""Print pre-registered G1/G1b/G2/G3 gate table from coarse-action result JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="vit_s16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--objective", default="x0", choices=["x0", "vpred"])
    p.add_argument("--artifacts", type=Path, default=Path("artifacts/full"))
    args = p.parse_args()

    enc, h, s, obj = args.encoder, args.horizon, args.seed, args.objective
    coarse = "_coarse"
    diff_path = args.artifacts / (
        f"spatial_dit_anchored_{obj}_result_{enc}_h{h}_s{s}{coarse}.json"
    )
    direct_path = args.artifacts / (
        f"spatial_dit_direct_residual_result_{enc}_h{h}_s{s}{coarse}.json"
    )
    for label, path in [("diffusion", diff_path), ("direct", direct_path)]:
        if not path.exists():
            print(f"MISSING {label}: {path}")
            return
    diff = load(diff_path)
    direct = load(direct_path)

    def hv_best(d, key):
        block = d.get(key, {})
        return block.get("best_of_k_mean_cossim", float("nan"))

    def hv_mean(d, key):
        block = d.get(key, {})
        return block.get("mean_cossim", float("nan"))

    d_best = hv_best(diff, "dit_distributional_hv")
    d_mean = hv_mean(diff, "dit_distributional_hv")
    dit_best = direct.get("dit_hv", {}).get("mean_cossim", float("nan"))
    mlp_best = direct.get("mlp_hv", {}).get("mean_cossim", float("nan"))
    mn_best = hv_best(diff, "matched_noise_distributional_hv")
    div = diff.get("dit_distributional_hv", {}).get("sample_diversity_l2", float("nan"))

    g1 = d_best - dit_best
    g1b = d_best - mn_best
    g2 = d_mean - (dit_best - 0.01)

    print(f"\n=== Coarse-action HV gates ({enc} h{h} s{s} obj={obj}) ===")
    print(f"  diffusion best-of-K:     {d_best:.4f}")
    print(f"  diffusion mean-of-K:     {d_mean:.4f}")
    print(f"  direct DiT (HV):         {dit_best:.4f}")
    print(f"  matched-noise best-of-K: {mn_best:.4f}")
    print(f"  MLP (HV):                {mlp_best:.4f}")
    print(f"  diffusion diversity L2:  {div:.4f}")
    print(f"  G1  (diff-direct >= 0.02):     {g1:+.4f}  {'PASS' if g1 >= 0.02 else 'FAIL'}")
    print(f"  G1b (diff-matched >= 0.01):    {g1b:+.4f}  {'PASS' if g1b >= 0.01 else 'FAIL'}")
    print(f"  G2  (mean >= direct-0.01):     {g2:+.4f}  {'PASS' if g2 >= 0 else 'FAIL'}")
    headline = g1 >= 0.02 and g1b >= 0.01
    print(f"  Headline (G1 ∧ G1b):           {'PASS' if headline else 'FAIL'}")
    if mlp_best >= d_best:
        print("  NOTE: MLP-coarse >= diffusion best-of-K — headline is coarse-conditioning not diffusion.")


if __name__ == "__main__":
    main()
