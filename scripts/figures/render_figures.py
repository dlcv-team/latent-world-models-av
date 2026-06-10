#!/usr/bin/env python3
"""Render publication-ready figures for encoder evaluation (B8).

Generates:
- Figure 1: Grouped bar chart of encoder RMSE with error bars and Bonferroni brackets
- Figure 2: Heatmap of per-scenario steering RMSE by encoder

Both figures saved at 300 DPI with appropriate captions.

Consumes outputs from M1's analysis.paired_tests pipeline:
- outputs/analysis/encoder_summary_with_ci.csv
- outputs/analysis/paired_tests.csv
- outputs/analysis/per_scenario_rmse.csv (generated separately)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ENCODER_DISPLAY


def load_data(data_dir: Path = Path("outputs/analysis")):
    """Load all required CSV files.

    Parameters
    ----------
    data_dir : Path
        Directory containing analysis CSV files (default: outputs/analysis/)

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        encoder_summary, paired_tests, per_scenario DataFrames
    """
    encoder_summary = pd.read_csv(data_dir / "encoder_summary_with_ci.csv")
    paired_tests = pd.read_csv(data_dir / "paired_tests.csv")
    per_scenario = pd.read_csv(data_dir / "per_scenario_rmse.csv")

    return encoder_summary, paired_tests, per_scenario


def generate_caption(encoder_summary: pd.DataFrame) -> str:
    """Generate dataset caption dynamically from CSV metadata and canonical config.

    Parameters
    ----------
    encoder_summary : pd.DataFrame
        Encoder summary with num_scenes column

    Returns
    -------
    str
        Caption text describing the dataset split
    """
    from config import load_canonical

    # Extract scene count from CSV (same across all encoders)
    n_scenes = int(encoder_summary['num_scenes'].iloc[0])

    # Load canonical config for comparison
    cfg = load_canonical()
    expected = cfg.expected_split_counts

    # Load manifest for policy description and seed
    manifest_path = cfg.manifest_path
    with manifest_path.open('r') as f:
        manifest = json.load(f)

    seed = manifest['seed']

    # Determine dataset type and build caption
    if n_scenes == expected['p0_test']:
        return f"P0 test split ({expected['p0_test']} scenes, seed {seed})"
    elif n_scenes == expected['p0_val']:
        return f"P0 validation split ({expected['p0_val']} scenes, seed {seed})"
    elif n_scenes == expected['p0_train']:
        return f"P0 training split ({expected['p0_train']} scenes, seed {seed})"
    elif n_scenes == expected['p0_all']:
        return f"trainval-mirror subset ({expected['p0_train']}/{expected['p0_val']}/{expected['p0_test']}, seed {seed})"
    else:
        # Full dataset or other - check if we have full_dataset config
        full_cfg = cfg.raw.get('dataset', {}).get('full_dataset')
        if full_cfg and n_scenes == full_cfg.get('n_scenes'):
            # Full dataset
            n_full = full_cfg['n_scenes']
            split_ratio = full_cfg['split_ratio']
            return f"Full v1.0-trainval dataset ({n_full} scenes, {split_ratio})"
        else:
            # Unknown dataset - use generic description
            return f"Custom dataset ({n_scenes} scenes)"


def detect_rmse_units(encoder_summary: pd.DataFrame) -> str:
    """Detect whether RMSE values are normalized or denormalized.

    Parameters
    ----------
    encoder_summary : pd.DataFrame
        Encoder summary with steer_rmse_scene_mean column

    Returns
    -------
    str
        'normalized' if values in [0, 2] range (normalized space)
        'denormalized' if values > 2 (degrees/m/s²)

    Notes
    -----
    Normalized RMSE is bounded by [0, 2] (max distance between -1 and +1).
    Denormalized steering RMSE typically ranges 1-30 degrees.
    """
    # Check maximum steering RMSE value
    steer_col = 'steer_rmse_scene_mean'
    if steer_col not in encoder_summary.columns:
        # Fallback: assume normalized if column missing
        return 'normalized'

    max_steer_rmse = encoder_summary[steer_col].max()

    # Values > 2.0 are impossible in normalized space (max is sqrt(2²) = 2.0)
    return 'denormalized' if max_steer_rmse > 2.0 else 'normalized'


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
    # Prepare data - set encoder as index if it's a column
    if "encoder" in encoder_summary.columns:
        encoder_summary = encoder_summary.set_index("encoder")

    encoders = sorted(encoder_summary.index.unique())
    n_encoders = len(encoders)

    # Get n_comparisons for caption
    n_comparisons = paired_tests.iloc[0]["n_comparisons"]

    # Detect unit system from data
    units = detect_rmse_units(encoder_summary)
    ylabel = 'RMSE (degrees / m/s²)' if units == 'denormalized' else 'RMSE (normalized)'

    # Prepare plotting data
    steer_means = []
    steer_cis_lo = []
    steer_cis_hi = []
    accel_means = []
    accel_cis_lo = []
    accel_cis_hi = []

    for enc in encoders:
        enc_data = encoder_summary.loc[enc]
        steer_means.append(enc_data["steer_rmse_scene_mean"])
        steer_cis_lo.append(enc_data["steer_rmse_scene_mean"] - enc_data["steer_ci95_lo"])
        steer_cis_hi.append(enc_data["steer_ci95_hi"] - enc_data["steer_rmse_scene_mean"])
        accel_means.append(enc_data["accel_rmse_scene_mean"])
        accel_cis_lo.append(enc_data["accel_rmse_scene_mean"] - enc_data["accel_ci95_lo"])
        accel_cis_hi.append(enc_data["accel_ci95_hi"] - enc_data["accel_rmse_scene_mean"])

    steer_cis = np.array([steer_cis_lo, steer_cis_hi])
    accel_cis = np.array([accel_cis_lo, accel_cis_hi])

    # Create figure
    fig = plt.figure(figsize=(12, 6), dpi=300)
    ax = fig.add_subplot(111)

    # Bar positions
    x = np.arange(n_encoders)
    width = 0.35

    # Plot bars
    bars1 = ax.bar(x - width/2, steer_means, width, yerr=steer_cis,
                   label='Steering RMSE', capsize=5, color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, accel_means, width, yerr=accel_cis,
                   label='Acceleration RMSE', capsize=5, color='#ff7f0e', alpha=0.8)

    # Customize axes (ylabel now dynamically set based on units)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel('Encoder', fontsize=11)
    ax.set_title('Encoder Performance: Steering and Acceleration RMSE\n' +
                 f'(Bonferroni-corrected, n_comparisons={n_comparisons})',
                 fontsize=12, pad=15)
    ax.set_xticks(x)

    # Format encoder names for display using canonical mapping
    encoder_labels = [ENCODER_DISPLAY.get(e, e) for e in encoders]
    ax.set_xticklabels(encoder_labels, rotation=0, ha='center')

    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Draw Bonferroni bracket for most significant pairwise comparison
    steer_tests = paired_tests[
        paired_tests["encoder_a"].isin(encoders)
        & paired_tests["encoder_b"].isin(encoders)
    ]
    if len(steer_tests) > 0:
        # Find the most significant comparison (lowest p_bonferroni)
        valid_pvals = steer_tests["p_bonferroni"].dropna()
        if len(valid_pvals) > 0:
            most_sig_idx = valid_pvals.idxmin()
            most_sig_row = steer_tests.loc[most_sig_idx]

            enc1 = most_sig_row["encoder_a"]
            enc2 = most_sig_row["encoder_b"]
            p_val = most_sig_row["p_bonferroni"]

            # Get x positions for these encoders
            encoder_to_x = {enc: i for i, enc in enumerate(encoders)}
            x1 = encoder_to_x.get(enc1)
            x2 = encoder_to_x.get(enc2)

            if x1 is not None and x2 is not None:
                # Get bar heights (steer bars, left side)
                h1 = steer_means[x1] + steer_cis[1, x1]  # top of error bar
                h2 = steer_means[x2] + steer_cis[1, x2]

                # Bracket height: 5% above the tallest bar
                y_max = max(h1, h2)
                bracket_y = y_max * 1.05
                bracket_h = y_max * 0.02  # vertical tick height

                # Draw bracket: left tick, horizontal line, right tick
                ax.plot([x1 - width/2, x1 - width/2], [bracket_y, bracket_y + bracket_h],
                       'k-', linewidth=1.5, clip_on=False)
                ax.plot([x1 - width/2, x2 - width/2], [bracket_y + bracket_h, bracket_y + bracket_h],
                       'k-', linewidth=1.5, clip_on=False)
                ax.plot([x2 - width/2, x2 - width/2], [bracket_y + bracket_h, bracket_y],
                       'k-', linewidth=1.5, clip_on=False)

                # Add p-value label above bracket
                mid_x = (x1 + x2) / 2 - width/2
                ax.text(mid_x, bracket_y + bracket_h * 1.5, f'p = {p_val:.4f}',
                       ha='center', va='bottom', fontsize=9, clip_on=False)

    # Add caption (dynamically generated from dataset metadata)
    caption = generate_caption(encoder_summary)
    fig.text(0.5, 0.02, caption, ha='center', fontsize=9, style='italic', wrap=True)

    # Save figure
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 1 to {output_path}")
    plt.close()


def render_figure2(per_scenario, encoder_summary, output_path="outputs/figure2_scenario_heatmap.pdf"):
    """Render Figure 2: Heatmap of per-scenario steering RMSE.

    Parameters
    ----------
    per_scenario : pd.DataFrame
        Per-scenario RMSE data
    encoder_summary : pd.DataFrame
        Encoder summary with num_scenes column (for caption generation)
    output_path : str
        Path to save PDF
    """
    # Detect units from per_scenario data (check what metric names exist)
    available_metrics = per_scenario["metric"].unique()
    if "steer_rmse_deg" in available_metrics:
        steer_metric = "steer_rmse_deg"
        colorbar_label = 'Steering RMSE (degrees)'
    elif "steer_rmse" in available_metrics:
        steer_metric = "steer_rmse"
        colorbar_label = 'Steering RMSE (normalized)'
    else:
        raise ValueError(
            f"No steering RMSE metric found in per_scenario data. "
            f"Expected 'steer_rmse_deg' or 'steer_rmse'. "
            f"Available metrics: {list(available_metrics)}"
        )

    # Filter for steering RMSE
    steer_data = per_scenario[per_scenario["metric"] == steer_metric].copy()

    # Pivot to create heatmap matrix
    heatmap_data = steer_data.pivot(index="encoder", columns="scenario", values="mean")

    # Reorder columns to standard scenario order (only include scenarios that exist)
    scenario_order = ["highway", "urban", "intersection", "other"]
    available_scenarios = [s for s in scenario_order if s in heatmap_data.columns]
    heatmap_data = heatmap_data[available_scenarios]

    # Reorder rows by overall performance (best to worst)
    row_means = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.loc[row_means.sort_values().index]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Create heatmap using matplotlib imshow
    im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=heatmap_data.values.max() * 1.1)

    # Add colorbar (label dynamically set based on detected units)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, fontsize=10)

    # Add value annotations (2 decimals for degrees/m/s², 4 for normalized)
    fmt = '.2f' if steer_metric == 'steer_rmse_deg' else '.4f'
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            text_color = 'white' if value > heatmap_data.values.max() * 0.6 else 'black'
            ax.text(j, i, f'{value:{fmt}}', ha='center', va='center',
                   color=text_color, fontsize=9)

    # Set ticks
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))

    # Customize
    ax.set_title('Per-Scenario Steering RMSE by Encoder', fontsize=12, pad=15)
    ax.set_xlabel('Scenario Type', fontsize=11)
    ax.set_ylabel('Encoder', fontsize=11)

    # Format labels using canonical mapping
    encoder_labels = [ENCODER_DISPLAY.get(e, e) for e in heatmap_data.index]
    scenario_labels = [s.capitalize() for s in heatmap_data.columns]

    ax.set_yticklabels(encoder_labels, rotation=0)
    ax.set_xticklabels(scenario_labels, rotation=0)

    # Add caption (dynamically generated from dataset metadata)
    caption = generate_caption(encoder_summary)
    fig.text(0.5, 0.02, caption, ha='center', fontsize=9, style='italic', wrap=True)

    # Save figure
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 2 to {output_path}")
    plt.close()


def main():
    """Generate both figures."""
    parser = argparse.ArgumentParser(description="Render publication-ready figures")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/analysis"),
        help="Directory containing analysis CSV files (default: outputs/analysis/)"
    )
    args = parser.parse_args()

    # Load data from specified directory
    encoder_summary, paired_tests, per_scenario = load_data(args.data_dir)

    # Render figures (captions now generated dynamically)
    fig1_path = args.data_dir / "figure1_encoder_rmse.pdf"
    fig2_path = args.data_dir / "figure2_scenario_heatmap.pdf"
    render_figure1(encoder_summary, paired_tests, output_path=fig1_path)
    render_figure2(per_scenario, encoder_summary, output_path=fig2_path)

    print("\nFigures generated successfully!")
    print(f"Figure 1: {fig1_path}")
    print(f"Figure 2: {fig2_path}")


if __name__ == "__main__":
    main()
