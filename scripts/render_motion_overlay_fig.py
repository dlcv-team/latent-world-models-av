"""Regenerate the motion-overlay paper figure (fig_motion_overlay.pdf).

Scientific AV visual: a held-out road frame overlaid with the low-frequency (coherent scene motion, hot
colormap) vs high-frequency (texture, cool colormap) consecutive-frame change between t and t+4, for the
ground truth (top) and the compact chain-anchor jump model (bottom). Shows the jump model captures the
spatial distribution of scene-level motion even though decoded frames are blurry.

PROVENANCE / REGENERATION CHAIN (this script was originally a /tmp one-off; preserved here for durability):
  HF ckpt  surlac/lwm-av-checkpoints:vae_latent/motionmini_jump4/direct_smoke2/dit.pt
    -> `modal run scripts/train_motion_mini_modal.py --task demo --tag smoke2 --scene-ids 3217`
       produces the V0 demo strips on the Modal volume (/viz/demo_s3217_*.png), mirrored to
       docs-repo/final-report/figures/motion_mini_preview/{s3217_baseline.png, s3217_full5.png} (committed)
    -> this script reads those two strips and renders fig_motion_overlay.pdf.

Inputs (committed): s3217_baseline.png (row0 = GT frames), s3217_full5.png (row1 = jump prediction).
Both 256x256 frames per cell, 6px white gaps, 5 columns (t+0,4,8,12,16).

Usage:
  python scripts/render_motion_overlay_fig.py \
      [--data-dir <dir with s3217_baseline.png,s3217_full5.png>] [--out <fig_motion_overlay.pdf>]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from PIL import Image

H, GAP, SIGMA, SCENE = 256, 6, 8, 3217
DEFAULT_DATA = "docs-repo/final-report/figures/motion_mini_preview"
DEFAULT_OUT = "docs-repo/final-report/latex/figures/fig_motion_overlay.pdf"


def _cols(strip, row):
    """Extract the 5 256x256 frames from the given row of a gapped strip."""
    y0 = row * (H + GAP)
    return [strip[y0:y0 + H, c * (H + GAP):c * (H + GAP) + H] for c in range(5)]


def _lowhigh(a, b):
    ag, bg = a.mean(2).astype(float), b.mean(2).astype(float)
    al, bl = gaussian_filter(ag, SIGMA), gaussian_filter(bg, SIGMA)
    low = np.abs(bl - al)                          # coherent scene motion
    high = np.abs((bg - bl) - (ag - al))           # texture variation
    return low, high


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    baseline = np.asarray(Image.open(f"{args.data_dir}/s{SCENE}_baseline.png"))  # GT(top)/5.4M/jump rows
    full5 = np.asarray(Image.open(f"{args.data_dir}/s{SCENE}_full5.png"))        # GT(top)/jump-pred(bottom)
    gt = _cols(baseline, 0)            # ground-truth frames (top row of baseline strip)
    pred = _cols(full5, 1)            # jump-model prediction (bottom row of full5 strip)

    gt_low, gt_high = _lowhigh(gt[0], gt[1])
    pr_low, pr_high = _lowhigh(pred[0], pred[1])
    vlo, vhi = max(gt_low.max(), 0.01), max(gt_high.max(), 0.01)  # shared scale GT<->pred

    fig, ax = plt.subplots(2, 3, figsize=(7.0, 4.5))
    rows = [("Ground\nTruth", gt[0], gt_low, gt_high, "GT Present ($t$)", "GT Low-freq motion\n($t{\\to}t{+}4$, coherent)", "GT High-freq change\n($t{\\to}t{+}4$, texture)"),
            ("Jump\nModel", pred[0], pr_low, pr_high, "Jump pred. present", "Pred low-freq motion\n(coherent scene shift)", "Pred high-freq change\n(texture variation)")]
    for r, (_, frame, low, high, t0, t1, t2) in enumerate(rows):
        ax[r, 0].imshow(frame); ax[r, 0].set_title(t0, fontsize=8); ax[r, 0].axis("off")
        ax[r, 1].imshow(frame); ax[r, 1].imshow(low, cmap="hot", alpha=0.6, norm=Normalize(0, vlo))
        ax[r, 1].set_title(t1, fontsize=7); ax[r, 1].axis("off")
        ax[r, 2].imshow(frame); ax[r, 2].imshow(high, cmap="cool", alpha=0.6, norm=Normalize(0, vhi))
        ax[r, 2].set_title(t2, fontsize=7); ax[r, 2].axis("off")
    fig.text(0.01, 0.72, "Ground\nTruth", fontsize=8, fontweight="bold", va="center", rotation=90)
    fig.text(0.01, 0.28, "Jump\nModel", fontsize=8, fontweight="bold", va="center", rotation=90)
    plt.tight_layout(rect=[0.03, 0.0, 1.0, 1.0])  # no suptitle (caption is in the .tex; matches cropped fig)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
