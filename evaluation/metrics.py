"""Evaluation metrics for action prediction."""

import numpy as np
import pandas as pd
from scipy.stats import bootstrap


def compute_rmse(predictions, targets):
    """
    Compute RMSE for steering (in degrees) and acceleration.

    Args:
        predictions: numpy array of shape (N, 2) with normalized [steering, accel] in [-1, 1]
        targets: numpy array of shape (N, 2) with normalized [steering, accel] in [-1, 1]

    Returns:
        steer_rmse_deg: RMSE for steering angle in degrees
        accel_rmse: RMSE for acceleration in m/s^2
    """
    # Denormalize predictions and targets
    # Steering: normalized = raw / 6.0 -> raw = normalized * 6.0
    pred_steer_deg = predictions[:, 0] * 6.0
    target_steer_deg = targets[:, 0] * 6.0

    # Acceleration: normalized = raw / 10.0 -> raw = normalized * 10.0
    pred_accel = predictions[:, 1] * 10.0
    target_accel = targets[:, 1] * 10.0

    # Compute RMSE
    steer_rmse_deg = np.sqrt(np.mean((pred_steer_deg - target_steer_deg) ** 2))
    accel_rmse = np.sqrt(np.mean((pred_accel - target_accel) ** 2))

    return steer_rmse_deg, accel_rmse


def scenario_breakdown(nusc, scene_tokens, rmse_by_frame):
    """
    Map scenes to scenario categories (highway/urban/intersection) via scene description matching.

    Args:
        nusc: NuScenes instance
        scene_tokens: list of scene tokens corresponding to each frame
        rmse_by_frame: numpy array of shape (N, 2) with [steer_rmse, accel_rmse] per frame

    Returns:
        dict mapping scenario types to dict with:
            - 'steer_rmse': mean steering RMSE for this scenario
            - 'accel_rmse': mean acceleration RMSE for this scenario
            - 'count': number of frames in this scenario
    """
    # Initialize scenario bins
    scenarios = {
        'highway': {'steer': [], 'accel': [], 'count': 0},
        'urban': {'steer': [], 'accel': [], 'count': 0},
        'intersection': {'steer': [], 'accel': [], 'count': 0}
    }

    # Build scene -> description lookup
    scene_descriptions = {}
    for scene in nusc.scene:
        scene_descriptions[scene['token']] = scene['description'].lower()

    # Categorize each frame based on scene description
    for i, scene_token in enumerate(scene_tokens):
        desc = scene_descriptions.get(scene_token, '')
        steer_rmse, accel_rmse = rmse_by_frame[i]

        # String matching for scenario classification
        # Priority order: intersection > highway > urban (default)
        if any(keyword in desc for keyword in ['intersection', 'intersect', 'turn', 'junction']):
            category = 'intersection'
        elif any(keyword in desc for keyword in ['highway', 'freeway', 'motorway', 'expressway']):
            category = 'highway'
        else:
            # Default to urban for city streets, residential, parking, etc.
            category = 'urban'

        scenarios[category]['steer'].append(steer_rmse)
        scenarios[category]['accel'].append(accel_rmse)
        scenarios[category]['count'] += 1

    # Compute means
    results = {}
    for scenario_type, data in scenarios.items():
        if data['count'] > 0:
            results[scenario_type] = {
                'steer_rmse': np.mean(data['steer']),
                'accel_rmse': np.mean(data['accel']),
                'count': data['count']
            }
        else:
            results[scenario_type] = {
                'steer_rmse': 0.0,
                'accel_rmse': 0.0,
                'count': 0
            }

    return results


def classify_scenes_by_scenario(nusc, scene_tokens: list[str]) -> dict[str, str]:
    """Map scene tokens to scenario categories.

    Args:
        nusc: NuScenes instance with .scene attribute
        scene_tokens: List of scene token strings to classify

    Returns:
        Dictionary mapping scene_token → scenario_bucket
        Scenario buckets: 'highway', 'urban', 'intersection', 'other'

    Notes:
        Reuses classification logic from scenario_breakdown():
        - intersection: keywords in ['intersection', 'intersect', 'turn', 'junction']
        - highway: keywords in ['highway', 'freeway', 'motorway', 'expressway']
        - urban: default for non-empty descriptions
        - other: missing or empty descriptions
    """
    # Build scene token → description lookup
    scene_descriptions = {}
    for scene in nusc.scene:
        scene_descriptions[scene['token']] = scene['description'].lower()

    # Classify each unique scene token
    scene_to_bucket = {}
    for token in set(scene_tokens):  # Deduplicate
        desc = scene_descriptions.get(token, '')

        # Priority matching (same logic as scenario_breakdown)
        if any(kw in desc for kw in ['intersection', 'intersect', 'turn', 'junction']):
            bucket = 'intersection'
        elif any(kw in desc for kw in ['highway', 'freeway', 'motorway', 'expressway']):
            bucket = 'highway'
        elif desc:  # Non-empty description that didn't match above
            bucket = 'urban'
        else:  # Missing or empty description
            bucket = 'other'

        scene_to_bucket[token] = bucket

    return scene_to_bucket


def compute_per_scenario_rmse(
    predictions_df: pd.DataFrame,
    scene_to_bucket: dict[str, str],
    cfg: dict | object,
) -> pd.DataFrame:
    """Compute per-scenario RMSE with bootstrap confidence intervals.

    Args:
        predictions_df: DataFrame with columns:
            - encoder: str
            - scene_name (or scene_token): str
            - steer_pred, accel_pred: float (normalized)
            - steer_true, accel_true: float (normalized)
        scene_to_bucket: Mapping from scene identifier to scenario bucket
        cfg: Config with evaluation.bootstrap parameters (dict or CanonicalConfig)

    Returns:
        DataFrame with columns:
            - encoder, scenario, metric, mean, ci_lo, ci_hi
            where metric is 'steer_rmse_norm' or 'accel_rmse_norm'

    Notes:
        - Uses scene-level resampling to respect temporal correlation
        - Bootstrap config: n_resamples=1000, seed=42, confidence_level=0.95
        - Scenarios with 1 scene: ci_lo = ci_hi = mean (no variance)
        - Empty scenarios: omitted from output
    """
    # Extract config
    if hasattr(cfg, 'raw'):  # CanonicalConfig
        bootstrap_cfg = cfg.raw["evaluation"]["bootstrap"]
    else:  # dict
        bootstrap_cfg = cfg["evaluation"]["bootstrap"]

    n_resamples = bootstrap_cfg["n_resamples"]
    seed = bootstrap_cfg["seed"]
    confidence_level = bootstrap_cfg["confidence_level"]

    # Handle empty input
    if predictions_df.empty:
        return pd.DataFrame(
            columns=["encoder", "scenario", "metric", "mean", "ci_lo", "ci_hi"]
        )

    # Determine scene column
    if "scene_token" in predictions_df.columns:
        scene_col = "scene_token"
    elif "scene_name" in predictions_df.columns:
        scene_col = "scene_name"
    else:
        raise ValueError("predictions_df must have 'scene_token' or 'scene_name' column")

    # Add scenario column
    df = predictions_df.copy()
    df["scenario"] = df[scene_col].map(scene_to_bucket)
    df = df.dropna(subset=["scenario"])

    # Helper: compute bootstrap CI
    def bootstrap_mean_ci(data: np.ndarray) -> tuple[float, float, float]:
        mean = np.mean(data)
        if len(data) == 1:
            return mean, mean, mean

        rng = np.random.default_rng(seed)
        result = bootstrap(
            (data,),
            np.mean,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_state=rng,
            method='percentile'
        )
        return mean, result.confidence_interval.low, result.confidence_interval.high

    # Compute per-scenario RMSE for each encoder × scenario
    results = []

    for (encoder, scenario), group in df.groupby(["encoder", "scenario"]):
        # Compute scene-level RMSE
        scene_rmses = []

        for scene_name, scene_group in group.groupby(scene_col):
            # Compute errors
            steer_errors = scene_group["steer_pred"].values - scene_group["steer_true"].values
            accel_errors = scene_group["accel_pred"].values - scene_group["accel_true"].values

            # Compute RMSE for this scene (in normalized space)
            steer_rmse = np.sqrt(np.mean(steer_errors ** 2))
            accel_rmse = np.sqrt(np.mean(accel_errors ** 2))

            scene_rmses.append((steer_rmse, accel_rmse))

        # Convert to arrays
        scene_steer = np.array([x[0] for x in scene_rmses])
        scene_accel = np.array([x[1] for x in scene_rmses])

        # Bootstrap steering
        steer_mean, steer_lo, steer_hi = bootstrap_mean_ci(scene_steer)
        results.append({
            "encoder": encoder,
            "scenario": scenario,
            "metric": "steer_rmse_norm",
            "mean": steer_mean,
            "ci_lo": steer_lo,
            "ci_hi": steer_hi,
        })

        # Bootstrap acceleration
        accel_mean, accel_lo, accel_hi = bootstrap_mean_ci(scene_accel)
        results.append({
            "encoder": encoder,
            "scenario": scenario,
            "metric": "accel_rmse_norm",
            "mean": accel_mean,
            "ci_lo": accel_lo,
            "ci_hi": accel_hi,
        })

    # Build output DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["encoder", "scenario", "metric"]).reset_index(drop=True)

    return results_df


def convert_steer_rmse_to_deg(rmse_norm: float, cfg: dict | object | None = None) -> float:
    """Convert normalized steering RMSE to degrees.

    Args:
        rmse_norm: RMSE in normalized space
        cfg: Config object or dict with normalization.steering.eval_back_to_deg_factor
             If None, loads canonical config from disk

    Returns:
        RMSE in degrees

    Notes:
        Conversion factor = 6.0 * 180 / π ≈ 34.3775
        This converts normalized RMSE → radians → degrees
    """
    # Load config if needed
    if cfg is None:
        from config import load_canonical
        cfg = load_canonical()

    # Extract conversion factor
    if hasattr(cfg, 'normalization'):  # CanonicalConfig
        factor = cfg.normalization("steering")["eval_back_to_deg_factor"]
    else:  # dict
        factor = cfg["normalization"]["steering"]["eval_back_to_deg_factor"]

    # Convert
    return rmse_norm * factor
