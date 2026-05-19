#!/usr/bin/env python3
"""Generate paired_tests.csv with Bonferroni-corrected pairwise comparisons.

Performs pairwise t-tests between all encoder pairs for steering and acceleration
RMSE, with Bonferroni correction for multiple comparisons.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_rel
from itertools import combinations


def main():
    # Read per-scenario RMSE (we'll use scenario-level means as observations)
    per_scenario_path = Path("outputs/per_scenario_rmse.csv")
    if not per_scenario_path.exists():
        raise FileNotFoundError(
            f"{per_scenario_path} not found. Run generate_per_scenario_rmse.py first."
        )

    df = pd.read_csv(per_scenario_path)

    # For paired tests, we use scenario-level RMSE as observations
    # (this respects scenario structure and gives reasonable DoF)

    results = []
    encoders = sorted(df["encoder"].unique())

    for metric in ["steer_rmse_norm", "accel_rmse_norm"]:
        metric_df = df[df["metric"] == metric]

        # Pivot to get encoder × scenario matrix
        pivot = metric_df.pivot(index="scenario", columns="encoder", values="mean")

        # Perform pairwise t-tests
        for enc1, enc2 in combinations(encoders, 2):
            if enc1 not in pivot.columns or enc2 not in pivot.columns:
                continue

            # Get paired observations (scenario-level means)
            obs1 = pivot[enc1].dropna().values
            obs2 = pivot[enc2].dropna().values

            # Ensure same scenarios
            if len(obs1) != len(obs2):
                print(f"Warning: {enc1} vs {enc2} have different scenario counts")
                continue

            # Paired t-test
            t_stat, p_value = ttest_rel(obs1, obs2)

            results.append({
                "encoder1": enc1,
                "encoder2": enc2,
                "metric": metric,
                "t_statistic": t_stat,
                "p_value": p_value,
            })

    # Create DataFrame
    output_df = pd.DataFrame(results)

    # Compute Bonferroni correction
    n_comparisons = len(output_df)
    alpha = 0.05
    bonferroni_alpha = alpha / n_comparisons

    output_df["p_bonferroni"] = output_df["p_value"] * n_comparisons
    output_df["p_bonferroni"] = output_df["p_bonferroni"].clip(upper=1.0)  # Cap at 1
    output_df["n_comparisons"] = n_comparisons
    output_df["alpha"] = alpha
    output_df["bonferroni_alpha"] = bonferroni_alpha
    output_df["significant_bonferroni"] = output_df["p_bonferroni"] < alpha

    # Sort by metric and encoder pairs
    output_df = output_df.sort_values(["metric", "encoder1", "encoder2"]).reset_index(drop=True)

    # Write to CSV
    output_path = Path("outputs/paired_tests.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Wrote paired tests to {output_path}")
    print(f"Total comparisons: {n_comparisons}")
    print(f"Bonferroni alpha: {bonferroni_alpha:.6f}")
    print(f"Significant pairs (Bonferroni): {output_df['significant_bonferroni'].sum()}")


if __name__ == "__main__":
    main()
