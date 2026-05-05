"""Evaluation metrics for encoder benchmarking.

Provides RMSE computation and scenario classification.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from nuscenes.nuscenes import NuScenes

logger = logging.getLogger(__name__)


def compute_rmse(
    predictions: np.ndarray, targets: np.ndarray
) -> tuple[float, float]:
    """Compute root mean squared error in normalized space.

    IMPORTANT: This function computes RMSE in whatever units the inputs are in.
    The dataset outputs normalized actions in [-1, 1] space. To convert steering
    RMSE to degrees, use convert_steer_rmse_to_deg().

    Parameters
    ----------
    predictions
        Predicted values, shape (N, 2) where column 0 is steering (normalized)
        and column 1 is acceleration (normalized).
    targets
        Ground truth values, shape (N, 2) where column 0 is steering (normalized)
        and column 1 is acceleration (normalized).

    Returns
    -------
    tuple[float, float]
        (steer_rmse_norm, accel_rmse_norm) tuple, both in normalized space.
    """
    steer_rmse_norm = float(
        np.sqrt(np.mean((predictions[:, 0] - targets[:, 0]) ** 2))
    )
    accel_rmse_norm = float(np.sqrt(np.mean((predictions[:, 1] - targets[:, 1]) ** 2)))
    return steer_rmse_norm, accel_rmse_norm


def convert_steer_rmse_to_deg(rmse_norm: float, cfg: dict[str, Any] | None = None) -> float:
    """Convert normalized steering RMSE to degrees.

    Uses the eval_back_to_deg_factor from canonical config to convert from
    normalized [-1, 1] space to degrees. Factor is 6 * 180 / pi ≈ 34.377.

    Parameters
    ----------
    rmse_norm
        Steering RMSE in normalized space (from compute_rmse).
    cfg
        Configuration dict with normalization.steering.eval_back_to_deg_factor.
        If None, loads canonical config. Can be either a config object (from
        load_canonical) or a plain dict.

    Returns
    -------
    float
        Steering RMSE in degrees.

    Examples
    --------
    >>> convert_steer_rmse_to_deg(0.1)  # doctest: +SKIP
    3.4377...
    """
    if cfg is None:
        from config import load_canonical
        cfg = load_canonical()

    # Handle both config object (with normalization method) and plain dict (for testing)
    if hasattr(cfg, "normalization"):
        steer_config = cfg.normalization("steering")
    else:
        steer_config = cfg["normalization"]["steering"]

    factor = steer_config["eval_back_to_deg_factor"]
    return rmse_norm * factor


def classify_scenes_by_scenario(
    nusc: NuScenes,
    scene_tokens: list[str],
) -> dict[str, str]:
    """Map scenes to highway / urban / intersection via string matching.

    Parameters
    ----------
    nusc
        NuScenes dataset instance.
    scene_tokens
        List of scene tokens.

    Returns
    -------
    dict[str, str]
        Mapping from scene_token to scenario category
        ("highway", "urban", "intersection", or "other").

    Notes
    -----
    Classification based on string matching on scene descriptions from
    nuScenes metadata. Falls back to "other" if no clear match.

    Warning
    -------
    String-matching heuristic is brittle on free-form nuScenes descriptions.
    Check logged scene counts per bucket — if "other" dominates, will need
    richer classification (location lookup, time-of-day metadata) for B8/B9.
    """
    mapping = {}

    for scene_token in scene_tokens:
        # Find scene record by token
        scene_records = [s for s in nusc.scene if s["token"] == scene_token]
        if not scene_records:
            mapping[scene_token] = "other"
            continue

        scene = scene_records[0]
        description = scene.get("description", "").lower()

        # String matching heuristic
        if "highway" in description or "freeway" in description:
            bucket = "highway"
        elif (
            "urban" in description
            or "city" in description
            or "downtown" in description
        ):
            bucket = "urban"
        elif "intersection" in description or "junction" in description:
            bucket = "intersection"
        else:
            bucket = "other"

        mapping[scene_token] = bucket

    # Log scene counts per bucket to detect classification issues
    from collections import Counter
    bucket_counts = Counter(mapping.values())
    total_scenes = len(scene_tokens)
    logger.info("Scene classification breakdown (total=%d):", total_scenes)
    for bucket in ["highway", "urban", "intersection", "other"]:
        count = bucket_counts.get(bucket, 0)
        pct = 100.0 * count / total_scenes if total_scenes > 0 else 0.0
        logger.info("  %s: %d (%.1f%%)", bucket, count, pct)

    return mapping


def compute_per_scenario_rmse(
    predictions_df: pd.DataFrame,
    scene_to_bucket: dict[str, str],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Compute per-scenario RMSE with bootstrap confidence intervals.

    IMPORTANT: This function computes RMSE in normalized space (matching the
    dataset and model output). To convert steering RMSE to degrees, use
    convert_steer_rmse_to_deg() on the returned values.

    Parameters
    ----------
    predictions_df
        DataFrame with columns:
        ``encoder, scene_token, steer_pred, accel_pred, steer_true, accel_true``.
        All prediction/target columns should be in normalized [-1, 1] space.
        Alternatively, if ``scene_name`` is present instead of ``scene_token``,
        it will be used for scenario mapping.
    scene_to_bucket
        Mapping from scene_token to scenario category
        (output of ``classify_scenes_by_scenario``).
    cfg
        Configuration dict with ``evaluation.bootstrap.seed``,
        ``evaluation.bootstrap.n_resamples``, and
        ``evaluation.bootstrap.confidence_level``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``encoder, scenario, metric, mean, ci_lo, ci_hi``.
        Each encoder × scenario × {steer_rmse_norm, accel_rmse_norm} combination
        gets one row with bootstrapped confidence intervals.
        All RMSE values are in normalized space.

    Notes
    -----
    Uses 1000-resample bootstrap (configurable via cfg) to compute 95% CI.
    Called by evaluation harness (B6.5); output consumed by
    ``evaluation.sidecars.write_per_scenario_rmse``.
    """
    # Extract bootstrap config
    bootstrap_cfg = cfg["evaluation"]["bootstrap"]
    n_resamples = bootstrap_cfg["n_resamples"]
    seed = bootstrap_cfg["seed"]
    confidence_level = bootstrap_cfg["confidence_level"]

    # Map scenes to scenarios
    # Handle both scene_token and scene_name columns
    if "scene_token" in predictions_df.columns:
        scene_col = "scene_token"
    elif "scene_name" in predictions_df.columns:
        scene_col = "scene_name"
    else:
        raise ValueError(
            "predictions_df must have 'scene_token' or 'scene_name' column"
        )

    predictions_df = predictions_df.copy()
    predictions_df["scenario"] = predictions_df[scene_col].map(scene_to_bucket)

    # Drop rows where scenario mapping failed
    predictions_df = predictions_df.dropna(subset=["scenario"])

    # Compute squared errors
    predictions_df["steer_sq_error"] = (
        predictions_df["steer_pred"] - predictions_df["steer_true"]
    ) ** 2
    predictions_df["accel_sq_error"] = (
        predictions_df["accel_pred"] - predictions_df["accel_true"]
    ) ** 2

    # Bootstrap resampling for CI
    rng = np.random.RandomState(seed)
    alpha = 1.0 - confidence_level

    results = []

    for encoder in predictions_df["encoder"].unique():
        encoder_df = predictions_df[predictions_df["encoder"] == encoder]

        for scenario in encoder_df["scenario"].unique():
            scenario_df = encoder_df[encoder_df["scenario"] == scenario]

            if len(scenario_df) == 0:
                continue

            # Bootstrap for steering RMSE
            steer_rmse_samples = []
            for _ in range(n_resamples):
                sample_indices = rng.choice(
                    len(scenario_df), size=len(scenario_df), replace=True
                )
                sample = scenario_df.iloc[sample_indices]
                rmse = np.sqrt(sample["steer_sq_error"].mean())
                steer_rmse_samples.append(rmse)

            steer_rmse_samples = np.array(steer_rmse_samples)
            steer_mean = float(np.mean(steer_rmse_samples))
            steer_ci_lo = float(np.percentile(steer_rmse_samples, 100 * alpha / 2))
            steer_ci_hi = float(
                np.percentile(steer_rmse_samples, 100 * (1 - alpha / 2))
            )

            results.append(
                {
                    "encoder": encoder,
                    "scenario": scenario,
                    "metric": "steer_rmse_norm",
                    "mean": steer_mean,
                    "ci_lo": steer_ci_lo,
                    "ci_hi": steer_ci_hi,
                }
            )

            # Bootstrap for acceleration RMSE
            accel_rmse_samples = []
            for _ in range(n_resamples):
                sample_indices = rng.choice(
                    len(scenario_df), size=len(scenario_df), replace=True
                )
                sample = scenario_df.iloc[sample_indices]
                rmse = np.sqrt(sample["accel_sq_error"].mean())
                accel_rmse_samples.append(rmse)

            accel_rmse_samples = np.array(accel_rmse_samples)
            accel_mean = float(np.mean(accel_rmse_samples))
            accel_ci_lo = float(np.percentile(accel_rmse_samples, 100 * alpha / 2))
            accel_ci_hi = float(
                np.percentile(accel_rmse_samples, 100 * (1 - alpha / 2))
            )

            results.append(
                {
                    "encoder": encoder,
                    "scenario": scenario,
                    "metric": "accel_rmse_norm",
                    "mean": accel_mean,
                    "ci_lo": accel_ci_lo,
                    "ci_hi": accel_ci_hi,
                }
            )

    return pd.DataFrame(results)


