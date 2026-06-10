#!/usr/bin/env python3
"""Render publication-ready DiT architecture diagram.

Generates a box-and-arrow diagram of the Diffusion Transformer (DiT)
architecture for 4-token latent sequence prediction with adaLN-Zero
conditioning.

The diagram visualizes:
- Input: 4 noisy latent tokens (B, 4, 384)
- Three conditioning streams (timestep, current-frame, action)
- Stack of 4 DiT blocks with adaLN-Zero modulation
- Output: noise prediction (B, 4, 384)

References:
- Peebles & Xie (2023): "Scalable Diffusion Models with Transformers"
- models/latent_dit.py (on m1/tier2-dit-model branch)
- configs/dit.yaml (on m1/tier2-dit-model branch)

Usage:
    python figures/render_dit_architecture.py
    python figures/render_dit_architecture.py --out-dir outputs/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# DiT hyperparameters (from configs/dit.yaml on m1/tier2-dit-model branch)
Z_DIM = 384
COND_DIM = 384
N_BLOCKS = 4
N_HEADS = 6
HORIZON = 4
MLP_RATIO = 4.0

DPI = 300

# Project color palette
COLOR_MAIN_FLOW = "#4878CF"  # Blue for main data flow
COLOR_CONDITIONING = "#EE854A"  # Orange for conditioning paths
COLOR_MODULATION = "#6BAA75"  # Green for adaLN modulation
COLOR_BOX_BG = "#F0F0F0"  # Light gray for box backgrounds


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    color: str = COLOR_MAIN_FLOW,
    alpha: float = 0.3,
) -> None:
    """Draw a rounded box with text at (x, y) center."""
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02",
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
        linewidth=2,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
        color="black",
    )


def draw_arrow(
    ax: plt.Axes,
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    color: str = "black",
    linewidth: float = 2,
    style: str = "-",
) -> None:
    """Draw an arrow from (x_start, y_start) to (x_end, y_end)."""
    arrow = FancyArrowPatch(
        (x_start, y_start),
        (x_end, y_end),
        arrowstyle="->",
        color=color,
        linewidth=linewidth,
        linestyle=style,
        mutation_scale=20,
    )
    ax.add_patch(arrow)


def draw_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    fontsize: int = 8,
    color: str = "black",
) -> None:
    """Draw a text label at (x, y)."""
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color)


def render_dit_architecture(out_path: Path) -> None:
    """Render DiT architecture diagram and save to out_path."""
    fig, ax = plt.subplots(figsize=(10, 14), dpi=DPI)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # Title
    ax.text(
        5,
        13.5,
        "Diffusion Transformer (DiT) Architecture",
        ha="center",
        va="top",
        fontsize=14,
        weight="bold",
    )
    ax.text(
        5,
        13.1,
        "4-token latent sequence prediction with adaLN-Zero conditioning",
        ha="center",
        va="top",
        fontsize=10,
        style="italic",
        color="gray",
    )

    # Vertical positions
    y_input = 11.5
    y_cond_streams = 10.5
    y_cond_sum = 9.5
    y_input_proj = 8.5
    y_block1 = 7.5
    y_block2 = 6.5
    y_block3 = 5.5
    y_block4 = 4.5
    y_final = 3.0
    y_output = 1.5

    # --- Input: Noisy tokens ---
    draw_box(ax, 5, y_input, 2.5, 0.4, "x_noisy\n(B, 4, 384)", COLOR_MAIN_FLOW, 0.3)
    draw_arrow(ax, 5, y_input - 0.2, 5, y_input_proj + 0.2, COLOR_MAIN_FLOW)

    # --- Conditioning streams (3 parallel inputs) ---
    # Timestep
    draw_box(ax, 1.5, y_cond_streams, 1.8, 0.4, "timestep t\nsin/cos → MLP", COLOR_CONDITIONING, 0.3)
    draw_label(ax, 1.5, y_cond_streams - 0.5, "(B, 384)", fontsize=7, color=COLOR_CONDITIONING)

    # Current frame
    draw_box(ax, 5, y_cond_streams, 1.8, 0.4, "z_t\nLinear proj", COLOR_CONDITIONING, 0.3)
    draw_label(ax, 5, y_cond_streams - 0.5, "(B, 384)", fontsize=7, color=COLOR_CONDITIONING)

    # Action
    draw_box(ax, 8.5, y_cond_streams, 1.8, 0.4, "a_embed\naction", COLOR_CONDITIONING, 0.3)
    draw_label(ax, 8.5, y_cond_streams - 0.5, "(B, 384)", fontsize=7, color=COLOR_CONDITIONING)

    # Conditioning sum
    draw_arrow(ax, 1.5, y_cond_streams - 0.2, 1.5, y_cond_sum, COLOR_CONDITIONING, 1.5, "--")
    draw_arrow(ax, 5, y_cond_streams - 0.6, 5, y_cond_sum, COLOR_CONDITIONING, 1.5, "--")
    draw_arrow(ax, 8.5, y_cond_streams - 0.2, 8.5, y_cond_sum, COLOR_CONDITIONING, 1.5, "--")

    # Sum arrows converge
    draw_arrow(ax, 1.5, y_cond_sum, 4.5, y_cond_sum, COLOR_CONDITIONING, 1.5, "--")
    draw_arrow(ax, 8.5, y_cond_sum, 5.5, y_cond_sum, COLOR_CONDITIONING, 1.5, "--")

    # Conditioning vector C
    draw_box(ax, 5, y_cond_sum, 2.2, 0.4, "C = Σ\n(B, 384)", COLOR_CONDITIONING, 0.4)

    # --- Input projection ---
    draw_box(ax, 5, y_input_proj, 2.0, 0.4, "Input Proj\nLinear(384→384)", COLOR_MAIN_FLOW, 0.3)
    draw_label(ax, 5, y_input_proj - 0.5, "(B, 4, 384)", fontsize=7, color=COLOR_MAIN_FLOW)
    draw_arrow(ax, 5, y_input_proj - 0.2, 5, y_block1 + 0.3, COLOR_MAIN_FLOW)

    # --- DiT Blocks (×4) ---
    block_ys = [y_block1, y_block2, y_block3, y_block4]
    for i, y_block in enumerate(block_ys):
        # Conditioning input to block
        draw_arrow(ax, 5, y_cond_sum, 8, y_block, COLOR_CONDITIONING, 1.5, "--")

        # DiT block box
        draw_box(ax, 5, y_block, 3.5, 0.6, f"DiT Block {i+1}\nadaLN + Attn + MLP", COLOR_MAIN_FLOW, 0.25)

        # adaLN modulation annotation
        draw_label(ax, 8.2, y_block + 0.15, "shift, scale, gate", fontsize=7, color=COLOR_MODULATION)
        draw_label(ax, 8.2, y_block - 0.15, "(from C)", fontsize=7, color=COLOR_CONDITIONING)

        # Arrow to next block
        if i < len(block_ys) - 1:
            draw_arrow(ax, 5, y_block - 0.3, 5, block_ys[i + 1] + 0.3, COLOR_MAIN_FLOW)

    # Stack indicator
    ax.text(
        1.0,
        (y_block1 + y_block4) / 2,
        f"×{N_BLOCKS}\nblocks",
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
    )

    # --- Final layer ---
    draw_arrow(ax, 5, y_block4 - 0.3, 5, y_final + 0.3, COLOR_MAIN_FLOW)
    draw_box(ax, 5, y_final, 3.0, 0.5, "Final Layer\nadaLN + Linear", COLOR_MAIN_FLOW, 0.3)
    draw_arrow(ax, 5, y_cond_sum, 8, y_final, COLOR_CONDITIONING, 1.5, "--")
    draw_label(ax, 8.2, y_final, "shift, scale, gate", fontsize=7, color=COLOR_MODULATION)

    # --- Output ---
    draw_arrow(ax, 5, y_final - 0.25, 5, y_output + 0.2, COLOR_MAIN_FLOW)
    draw_box(ax, 5, y_output, 2.5, 0.4, "ε_pred\n(B, 4, 384)", COLOR_MAIN_FLOW, 0.4)
    draw_label(ax, 5, y_output - 0.5, "noise prediction", fontsize=8, color="gray")

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(color=COLOR_MAIN_FLOW, label="Main data flow"),
        mpatches.Patch(color=COLOR_CONDITIONING, label="Conditioning streams"),
        mpatches.Patch(color=COLOR_MODULATION, label="adaLN modulation"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    # --- Caption and provenance ---
    caption = (
        f"Architecture: {N_BLOCKS} DiT blocks, {N_HEADS} attention heads, "
        f"{HORIZON}-token sequence, {Z_DIM}-d latent space\n"
        f"adaLN-Zero conditioning from timestep, current-frame embedding (z_t), and action embedding (a_embed)\n"
        f"Reference: Peebles & Xie (2023), models/latent_dit.py (m1/tier2-dit-model branch)"
    )
    ax.text(
        5,
        0.3,
        caption,
        ha="center",
        va="bottom",
        fontsize=7,
        style="italic",
        color="gray",
        wrap=True,
    )

    # Save figure
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[render_dit_architecture] Saved to {out_path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render publication-ready DiT architecture diagram."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Output directory for figure (default: outputs/figures)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out_path = args.out_dir / "dit_architecture_diagram.pdf"

    print("[render_dit_architecture] Generating DiT architecture diagram...")
    render_dit_architecture(out_path)
    print(f"[render_dit_architecture] Done. Output: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
