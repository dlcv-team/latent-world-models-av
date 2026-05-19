"""Example: Generate B6.5 outputs (data_quality_report.json + per_scenario_rmse.csv).

This script demonstrates the full B6.5 workflow:
1. Load dataset → generates data_quality_stats
2. Write data_quality_report.json
3. Collect predictions from encoder evaluation
4. Classify scenes by scenario
5. Compute per-scenario RMSE with bootstrap CI
6. Write per_scenario_rmse.csv

Usage:
    python examples/generate_b6_5_outputs.py

Outputs:
    outputs/data_quality_report.json
    outputs/analysis/per_scenario_rmse.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import load_canonical
from data.dataset import NuScenesFrameDataset
from evaluation.metrics import (
    classify_scenes_by_scenario,
    compute_per_scenario_rmse,
)
from evaluation.sidecars import write_data_quality_report, write_per_scenario_rmse


def main() -> None:
    """Generate B6.5 outputs."""
    # Load configuration
    cfg = load_canonical()

    # Step 1: Load dataset (tracks data quality stats during construction)
    print("Loading dataset...")
    dataset = NuScenesFrameDataset(split="p0_val", mode="single_frame")
    print(f"✓ Loaded {len(dataset)} samples")

    # Step 2: Write data quality report
    print("\nGenerating data_quality_report.json...")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    write_data_quality_report(
        dataset,
        output_dir / "data_quality_report.json",
        cfg=cfg,
    )
    print(f"✓ Wrote data_quality_report.json")

    # Step 3: Simulate predictions (in real evaluation, run inference here)
    print("\nSimulating encoder predictions...")
    # For demonstration, use ground truth as predictions
    # Get scene tokens from samples, actions from dataset items
    scene_tokens = [sample["scene_token"] for sample in dataset.samples]
    actions_list = [dataset[i]["actions"] for i in range(len(dataset))]

    predictions_df = pd.DataFrame({
        "encoder": ["vit_s16"] * len(dataset),
        "scene_token": scene_tokens,
        "steer_pred": [float(action[0]) for action in actions_list],
        "accel_pred": [float(action[1]) for action in actions_list],
        "steer_true": [float(action[0]) for action in actions_list],
        "accel_true": [float(action[1]) for action in actions_list],
    })
    print(f"✓ Generated {len(predictions_df)} predictions")

    # Step 4: Classify scenes by scenario
    print("\nClassifying scenes by scenario...")
    scene_tokens = predictions_df["scene_token"].unique().tolist()
    scene_to_bucket = classify_scenes_by_scenario(dataset.nusc, scene_tokens)
    print(f"✓ Classified {len(scene_tokens)} scenes")

    # Step 5: Compute per-scenario RMSE with bootstrap CI
    print("\nComputing per-scenario RMSE...")
    results_df = compute_per_scenario_rmse(
        predictions_df,
        scene_to_bucket,
        cfg,
    )
    print(f"✓ Computed RMSE for {len(results_df)} (encoder × scenario × metric) groups")

    # Step 6: Write per-scenario RMSE CSV
    print("\nGenerating per_scenario_rmse.csv...")
    write_per_scenario_rmse(
        results_df,
        predictions_df,
        scene_to_bucket,
        analysis_dir / "per_scenario_rmse.csv",
    )
    print(f"✓ Wrote per_scenario_rmse.csv")

    print("\n" + "=" * 60)
    print("✓ B6.5 outputs generated successfully!")
    print(f"  - {output_dir / 'data_quality_report.json'}")
    print(f"  - {analysis_dir / 'per_scenario_rmse.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
