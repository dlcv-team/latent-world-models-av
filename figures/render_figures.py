#!/usr/bin/env python3
"""Render publication-ready figures for encoder evaluation (B8).

Generates:
- Figure 1: Grouped bar chart of encoder RMSE with error bars and Bonferroni brackets
- Figure 2: Heatmap of per-scenario steering RMSE by encoder

Both figures saved at 300 DPI with appropriate captions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap


def load_data():
    """Load all required CSV files."""
    encoder_summary = pd.read_csv("outputs/encoder_summary_with_ci.csv")
    paired_tests = pd.read_csv("outputs/paired_tests.csv")
    per_scenario = pd.read_csv("outputs/per_scenario_rmse.csv")

    return encoder_summary, paired_tests, per_scenario


def check_vq_fallback(encoder_summary):
    """Check if VQ-VAE is using fallback based on performance.

    VQ fallback copies DINOv2 embeddings, so performance would be identical.
    """
    # Get VQVAE and DINOv2 steering RMSE values
    vqvae_steer = encoder_summary[
        (encoder_summary["encoder"] == "vqvae") &
        (encoder_summary["metric"] == "steer_rmse_norm")
    ]["mean"].values

    dinov2_steer = encoder_summary[
        (encoder_summary["encoder"] == "dinov2_s14") &
        (encoder_summary["metric"] == "steer_rmse_norm")
    ]["mean"].values

    # If both exist and are identical (within floating point tolerance), fallback is active
    if len(vqvae_steer) > 0 and len(dinov2_steer) > 0:
        return np.allclose(vqvae_steer[0], dinov2_steer[0], rtol=1e-6)

    return False  # Default to no fallback if data missing


def render_figure1(encoder_summary, paired_tests, output_path="outputs/figure1_encoder_rmse.pdf"):
    """Render Figure 1: Grouped bar chart with Bonferroni brackets.

    Parameters
    ----------
    encoder_summary : pd.DataFrame
        Encoder-level RMSE with CI columns
    paired_tests : pd.DataFrame
        Pairwise comparisons with p_bonferroni column
    output_path : str
        Path to save PDF
    """
    # Prepare data
    encoders = sorted(encoder_summary["encoder"].unique())
    n_encoders = len(encoders)

    # Get n_comparisons for caption
    n_comparisons = paired_tests["n_comparisons"].iloc[0]

    # Prepare plotting data
    steer_data = encoder_summary[encoder_summary["metric"] == "steer_rmse_norm"].sort_values("encoder")
    accel_data = encoder_summary[encoder_summary["metric"] == "accel_rmse_norm"].sort_values("encoder")

    steer_means = steer_data["mean"].values
    steer_cis = np.array([steer_data["mean"].values - steer_data["ci_lo"].values,
                          steer_data["ci_hi"].values - steer_data["mean"].values])

    accel_means = accel_data["mean"].values
    accel_cis = np.array([accel_data["mean"].values - accel_data["ci_lo"].values,
                          accel_data["ci_hi"].values - accel_data["mean"].values])

    # Create figure with space for text box on the right
    fig = plt.figure(figsize=(12, 6), dpi=300)
    ax = fig.add_subplot(111)

    # Shrink plot area to leave room on the right for text box
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    # Bar positions
    x = np.arange(n_encoders)
    width = 0.35

    # Plot bars
    bars1 = ax.bar(x - width/2, steer_means, width, yerr=steer_cis,
                   label='Steering RMSE', capsize=5, color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, accel_means, width, yerr=accel_cis,
                   label='Acceleration RMSE', capsize=5, color='#ff7f0e', alpha=0.8)

    # Customize axes
    ax.set_ylabel('Normalized RMSE', fontsize=11)
    ax.set_xlabel('Encoder', fontsize=11)
    ax.set_title('Encoder Performance: Steering and Acceleration RMSE\n' +
                 f'(Bonferroni-corrected, n_comparisons={n_comparisons})',
                 fontsize=12, pad=15)
    ax.set_xticks(x)

    # Format encoder names for display
    encoder_labels = [e.replace('_', '-').upper() for e in encoders]
    ax.set_xticklabels(encoder_labels, rotation=0, ha='center')

    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add performance comparison text box (outside graph, top right)
    steer_tests = paired_tests[paired_tests["metric"] == "steer_rmse_norm"]
    encoder_means = dict(zip(steer_data["encoder"], steer_data["mean"]))
    best_encoder = min(encoder_means, key=encoder_means.get)
    worst_encoder = max(encoder_means, key=encoder_means.get)

    # Get p-value for best vs worst comparison
    comparison = steer_tests[
        ((steer_tests["encoder1"] == best_encoder) & (steer_tests["encoder2"] == worst_encoder)) |
        ((steer_tests["encoder1"] == worst_encoder) & (steer_tests["encoder2"] == best_encoder))
    ]

    if len(comparison) > 0:
        p_val = comparison["p_bonferroni"].values[0]

        # Format encoder names for display
        best_display = best_encoder.replace('_', '-').upper()
        worst_display = worst_encoder.replace('_', '-').upper()

        # Create text box content
        textbox_content = (
            f"Best (steer): {best_display}\n"
            f"Worst (steer): {worst_display}\n"
            f"p = {p_val:.4f}\n"
            f"n = {n_comparisons}"
        )

        # Add text box outside plot area, very bottom right
        fig.text(0.92, 0.08, textbox_content,
                fontsize=9,
                verticalalignment='bottom',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='none',
                         edgecolor='black', linewidth=1))

    # Add caption
    vq_fallback = check_vq_fallback(encoder_summary)
    caption = "trainval-mirror subset (180/20/40, seed 42)"
    if vq_fallback:
        caption += "\nVQ-VAE: DINOv2 fallback per FR-08 (checkpoint load failure)"

    fig.text(0.5, 0.02, caption, ha='center', fontsize=9, style='italic', wrap=True)

    # Save figure
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 1 to {output_path}")
    plt.close()


def render_figure2(per_scenario, output_path="outputs/figure2_scenario_heatmap.pdf"):
    """Render Figure 2: Heatmap of per-scenario steering RMSE.

    Parameters
    ----------
    per_scenario : pd.DataFrame
        Per-scenario RMSE data
    output_path : str
        Path to save PDF
    """
    # Filter for steering RMSE only
    steer_data = per_scenario[per_scenario["metric"] == "steer_rmse_norm"].copy()

    # Pivot to create heatmap matrix
    heatmap_data = steer_data.pivot(index="encoder", columns="scenario", values="mean")

    # Reorder columns to standard scenario order
    scenario_order = ["highway", "urban", "intersection", "other"]
    heatmap_data = heatmap_data[scenario_order]

    # Reorder rows by overall performance (best to worst)
    row_means = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.loc[row_means.sort_values().index]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Create heatmap using matplotlib imshow
    im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=heatmap_data.values.max() * 1.1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Steering RMSE (normalized)', fontsize=10)

    # Add value annotations
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            text_color = 'white' if value > heatmap_data.values.max() * 0.6 else 'black'
            ax.text(j, i, f'{value:.4f}', ha='center', va='center',
                   color=text_color, fontsize=9)

    # Set ticks
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))

    # Customize
    ax.set_title('Per-Scenario Steering RMSE by Encoder', fontsize=12, pad=15)
    ax.set_xlabel('Scenario Type', fontsize=11)
    ax.set_ylabel('Encoder', fontsize=11)

    # Format labels
    encoder_labels = [e.replace('_', '-').upper() for e in heatmap_data.index]
    scenario_labels = [s.capitalize() for s in heatmap_data.columns]

    ax.set_yticklabels(encoder_labels, rotation=0)
    ax.set_xticklabels(scenario_labels, rotation=0)

    # Add VQ fallback indicator if applicable
    vq_fallback = check_vq_fallback(per_scenario)
    if vq_fallback and 'vqvae' in heatmap_data.index:
        vq_idx = list(heatmap_data.index).index('vqvae')
        # Add asterisk to VQ row label
        labels = ax.get_yticklabels()
        labels[vq_idx].set_text(labels[vq_idx].get_text() + '*')
        ax.set_yticklabels(labels, rotation=0)

    # Add caption
    caption = "trainval-mirror subset (180/20/40, seed 42)"
    if vq_fallback:
        caption += "\n*VQ-VAE: DINOv2 fallback per FR-08 (checkpoint load failure)"

    fig.text(0.5, 0.02, caption, ha='center', fontsize=9, style='italic', wrap=True)

    # Save figure
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 2 to {output_path}")
    plt.close()


def main():
    """Generate both figures."""
    # Load data
    encoder_summary, paired_tests, per_scenario = load_data()

    # Render figures
    render_figure1(encoder_summary, paired_tests)
    render_figure2(per_scenario)

    print("\nFigures generated successfully!")
    print("Figure 1: outputs/figure1_encoder_rmse.pdf")
    print("Figure 2: outputs/figure2_scenario_heatmap.pdf")


if __name__ == "__main__":
    main()
