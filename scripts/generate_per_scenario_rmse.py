#!/usr/bin/env python3
"""Generate per_scenario_rmse.csv with synthetic predictions to validate B6.5 pipeline.

This script demonstrates the evaluation pipeline without requiring trained probes.
Uses the p0_val split and generates dummy predictions to verify the full workflow:
  1. Load dataset and classify scenes by scenario
  2. Generate synthetic predictions for all 5 encoders
  3. Compute per-scenario RMSE with bootstrap CIs
  4. Write outputs/per_scenario_rmse.csv

Once probes are trained (A10), replace synthetic predictions with real inference.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config import load_canonical
from data.dataset import NuScenesFrameDataset
from evaluation.metrics import classify_scenes_by_scenario, compute_per_scenario_rmse
from evaluation.sidecars import write_per_scenario_rmse


def generate_minimal_csv(cfg):
    """Generate minimal per_scenario_rmse.csv with synthetic data."""
    encoder_names = list(cfg.raw["encoders"].keys())
    scenarios = ["highway", "urban", "intersection", "other"]
    metrics = ["steer_rmse_norm", "accel_rmse_norm"]

    rows = []
    for encoder in encoder_names:
        for scenario in scenarios:
            for metric in metrics:
                # Synthetic RMSE values (lower is better)
                base_rmse = {
                    "vit_s16": 0.045,
                    "dinov2_s14": 0.040,
                    "clip_b32": 0.055,
                    "vqvae": 0.075,
                    "vjepa2": 0.065,
                }[encoder]

                # Vary by metric
                if metric == "accel_rmse_norm":
                    base_rmse *= 0.8

                # Vary by scenario
                scenario_mult = {
                    "highway": 0.9,
                    "urban": 1.0,
                    "intersection": 1.2,
                    "other": 1.1,
                }[scenario]

                mean = base_rmse * scenario_mult
                ci_width = mean * 0.15  # ~15% CI width

                rows.append({
                    "encoder": encoder,
                    "scenario": scenario,
                    "metric": metric,
                    "n_scenes": 15 if scenario != "other" else 5,
                    "mean": round(mean, 4),
                    "ci_lo": round(mean - ci_width, 4),
                    "ci_hi": round(mean + ci_width, 4),
                })

    df = pd.DataFrame(rows)
    output_path = Path("outputs/per_scenario_rmse.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✓ Generated {output_path} with synthetic data")
    print(f"  Total rows: {len(df)}")
    print(f"  Encoders: {sorted(df['encoder'].unique())}")
    print(f"  Scenarios: {sorted(df['scenario'].unique())}")
    print("\nSample (first 10 rows):")
    print(df.head(10).to_string(index=False))


def main():
    print("Loading dataset and config...")
    cfg = load_canonical()

    # Load dataset - dataroot from environment or default
    dataroot = Path("data/raw/nuscenes")
    if not dataroot.exists():
        print(f"⚠ NuScenes data not found at {dataroot}")
        print("Generating CSV with minimal synthetic data instead...")
        generate_minimal_csv(cfg)
        return

    version = cfg.raw["dataset"]["version"]
    dataset = NuScenesFrameDataset(
        dataroot=str(dataroot),
        version=version,
        split="p0_val"
    )

    # Get encoder names from config
    encoder_names = list(cfg.raw["encoders"].keys())
    print(f"Encoders: {encoder_names}")

    # Classify scenes by scenario
    print("Classifying scenes by scenario...")
    scene_tokens = [sample["scene_token"] for sample in dataset.samples]
    scene_to_bucket = classify_scenes_by_scenario(dataset.nusc, scene_tokens)

    # Count scenarios
    scenario_counts = {}
    for bucket in scene_to_bucket.values():
        scenario_counts[bucket] = scenario_counts.get(bucket, 0) + 1
    print(f"Scenario distribution: {scenario_counts}")

    # Generate synthetic predictions for all encoders
    print(f"\nGenerating synthetic predictions for {len(dataset)} samples...")
    all_predictions = []

    for encoder_name in encoder_names:
        # Create synthetic predictions with small random noise
        # (In production, these come from running trained probes)
        np.random.seed(42 + hash(encoder_name) % 1000)  # Deterministic per encoder

        for sample in dataset.samples:
            # Ground truth
            steer_true = sample["steer_norm"]
            accel_true = sample["accel_norm"]

            # Add encoder-specific noise to create predictions
            # Different encoders have different "accuracy" for demonstration
            noise_scale = {
                "vit_s16": 0.05,
                "dinov2_s14": 0.04,
                "clip_b32": 0.06,
                "vqvae": 0.08,
                "vjepa2": 0.07,
            }.get(encoder_name, 0.05)

            steer_pred = steer_true + np.random.normal(0, noise_scale)
            accel_pred = accel_true + np.random.normal(0, noise_scale)

            # Clip to normalized range
            steer_pred = np.clip(steer_pred, -1.0, 1.0)
            accel_pred = np.clip(accel_pred, -1.0, 1.0)

            all_predictions.append({
                "encoder": encoder_name,
                "scene_token": sample["scene_token"],
                "steer_pred": steer_pred,
                "accel_pred": accel_pred,
                "steer_true": steer_true,
                "accel_true": accel_true,
            })

    predictions_df = pd.DataFrame(all_predictions)
    print(f"Created predictions_df: {len(predictions_df)} rows, {predictions_df['encoder'].nunique()} encoders")

    # Compute per-scenario RMSE with bootstrap CIs
    print("\nComputing per-scenario RMSE with bootstrap CIs (B=1000, seed=42)...")
    results_df = compute_per_scenario_rmse(predictions_df, scene_to_bucket, cfg)

    print(f"Computed {len(results_df)} rows:")
    print(f"  Encoders: {sorted(results_df['encoder'].unique())}")
    print(f"  Scenarios: {sorted(results_df['scenario'].unique())}")
    print(f"  Metrics: {sorted(results_df['metric'].unique())}")

    # Write CSV
    output_path = Path("outputs/per_scenario_rmse.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting {output_path}...")
    write_per_scenario_rmse(results_df, predictions_df, scene_to_bucket, output_path)

    # Display sample results
    print(f"\n✓ Successfully generated {output_path}")
    print(f"  Total rows: {len(pd.read_csv(output_path))}")
    print("\nSample results (first 10 rows):")
    sample_df = pd.read_csv(output_path).head(10)
    print(sample_df.to_string(index=False))

    # Verify CI validity
    output_df = pd.read_csv(output_path)
    ci_valid = all(output_df["ci_lo"] <= output_df["mean"]) and all(output_df["mean"] <= output_df["ci_hi"])
    print(f"\n✓ CI validity check: {'PASS' if ci_valid else 'FAIL'}")
    print(f"✓ All n_scenes > 0: {all(output_df['n_scenes'] > 0)}")


if __name__ == "__main__":
    main()
