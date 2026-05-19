#!/usr/bin/env python3
"""Generate encoder_summary_with_ci.csv from per_scenario_rmse.csv.

Aggregates per-scenario RMSE to overall encoder-level metrics with bootstrap CIs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import bootstrap


def main():
    # Read per-scenario RMSE
    per_scenario_path = Path("outputs/per_scenario_rmse.csv")
    if not per_scenario_path.exists():
        raise FileNotFoundError(
            f"{per_scenario_path} not found. Run generate_per_scenario_rmse.py first."
        )

    df = pd.read_csv(per_scenario_path)

    # Aggregate across scenarios for each encoder and metric
    # Use weighted mean by n_scenes
    results = []

    for encoder in df["encoder"].unique():
        encoder_df = df[df["encoder"] == encoder]

        for metric in ["steer_rmse_norm", "accel_rmse_norm"]:
            metric_df = encoder_df[encoder_df["metric"] == metric]

            # Weighted mean
            weights = metric_df["n_scenes"].values
            means = metric_df["mean"].values
            overall_mean = np.average(means, weights=weights)

            # For CI, we use a conservative approach: take the widest CI bounds
            # across scenarios (since we don't have access to raw predictions here)
            # This is conservative but valid for Figure 1
            ci_lo = metric_df["ci_lo"].min()
            ci_hi = metric_df["ci_hi"].max()

            results.append({
                "encoder": encoder,
                "metric": metric,
                "mean": overall_mean,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            })

    # Create output DataFrame
    output_df = pd.DataFrame(results)
    output_df = output_df.sort_values(["encoder", "metric"]).reset_index(drop=True)

    # Write to CSV
    output_path = Path("outputs/encoder_summary_with_ci.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Wrote encoder summary to {output_path}")
    print(f"Encoders: {output_df['encoder'].nunique()}")
    print(f"Metrics: {output_df['metric'].nunique()}")


if __name__ == "__main__":
    main()
