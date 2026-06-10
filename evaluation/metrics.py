"""Evaluation metrics for encoder benchmarking.

Provides RMSE computation and scenario classification.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nuscenes.nuscenes import NuScenes

from config import CanonicalConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Timestamp utilities for environment classification
# ============================================================================


def parse_logfile_timestamp(logfile: str) -> str:
    """Extract timestamp from nuScenes logfile name.

    Args:
        logfile: e.g., "n015-2018-07-18-11-07-57+0800"

    Returns:
        Time string "HH:MM:SS" or "unknown" if parsing fails

    Note:
        Timestamp represents local time at recording location.
        The ±HHMM offset indicates timezone (e.g., +0800 = Singapore SGT, -0500 = Boston EST).

    Examples:
        >>> parse_logfile_timestamp("n015-2018-07-18-11-07-57+0800")
        '11:07:57'
        >>> parse_logfile_timestamp("invalid")
        'unknown'
    """
    match = re.search(r'-(\d{2})-(\d{2})-(\d{2})[+-]\d{4}$', logfile)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        second = int(match.group(3))
        return f"{hour:02d}:{minute:02d}:{second:02d}"
    return "unknown"


def is_night_from_timestamp(timestamp: str) -> bool:
    """Return True if timestamp is in night range (18:00-05:59).

    Args:
        timestamp: Time string "HH:MM:SS" from parse_logfile_timestamp

    Returns:
        True if night (6pm-6am), False otherwise

    Examples:
        >>> is_night_from_timestamp("18:00:00")
        True
        >>> is_night_from_timestamp("23:59:59")
        True
        >>> is_night_from_timestamp("00:00:00")
        True
        >>> is_night_from_timestamp("05:59:59")
        True
        >>> is_night_from_timestamp("06:00:00")
        False
        >>> is_night_from_timestamp("12:00:00")
        False
        >>> is_night_from_timestamp("unknown")
        False
    """
    if timestamp == "unknown":
        return False
    hour = int(timestamp.split(":")[0])
    return hour >= 18 or hour < 6


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
        (steer_rmse, accel_rmse) tuple, both in normalized space.
    """
    steer_rmse = float(
        np.sqrt(np.mean((predictions[:, 0] - targets[:, 0]) ** 2))
    )
    accel_rmse = float(np.sqrt(np.mean((predictions[:, 1] - targets[:, 1]) ** 2)))
    return steer_rmse, accel_rmse


def bootstrap_mean_ci(
    values: np.ndarray,
    n_resamples: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) via nonparametric bootstrap of the mean.

    Args:
        values: Array of observations to bootstrap
        n_resamples: Number of bootstrap resamples (typically 1000)
        seed: Random seed for reproducibility
        confidence_level: CI level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (mean, ci_lo, ci_hi)

    Notes:
        Uses percentile method with vectorized resampling.
        For single-value inputs, returns (mean, mean, mean).

    Raises:
        ValueError: If values array is empty
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("bootstrap_mean_ci received zero-length input")

    mean = float(np.mean(values))

    if len(values) == 1:
        return mean, mean, mean

    # Vectorized bootstrap resampling
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = values[indices].mean(axis=1)

    # Compute percentile-based CI
    alpha = 1.0 - confidence_level
    ci_lo = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return mean, ci_lo, ci_hi


def bootstrap_ratio_ci(
    numerator_values: np.ndarray,
    denominator_values: np.ndarray,
    n_resamples: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float, float]:
    """Return (ratio, ci_lo, ci_hi) via bootstrap of ratio of two independent means.

    Args:
        numerator_values: Array of observations for numerator
        denominator_values: Array of observations for denominator
        n_resamples: Number of bootstrap resamples (typically 1000)
        seed: Random seed for reproducibility
        confidence_level: CI level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (ratio, ci_lo, ci_hi)

    Notes:
        Resamples numerator and denominator independently (different scene subsets).
        Uses percentile method with vectorized resampling.
        For single-value inputs, returns (ratio, ratio, ratio).

    Raises:
        ValueError: If either array is empty
    """
    numerator_values = np.asarray(numerator_values, dtype=float)
    denominator_values = np.asarray(denominator_values, dtype=float)

    if len(numerator_values) == 0 or len(denominator_values) == 0:
        raise ValueError("bootstrap_ratio_ci received zero-length input")

    ratio = float(np.mean(numerator_values) / np.mean(denominator_values))

    if len(numerator_values) == 1 and len(denominator_values) == 1:
        return ratio, ratio, ratio

    # Vectorized bootstrap resampling (independent)
    rng = np.random.default_rng(seed)
    n_num = numerator_values.shape[0]
    n_den = denominator_values.shape[0]

    num_indices = rng.integers(0, n_num, size=(n_resamples, n_num))
    den_indices = rng.integers(0, n_den, size=(n_resamples, n_den))

    boot_num_means = numerator_values[num_indices].mean(axis=1)
    boot_den_means = denominator_values[den_indices].mean(axis=1)
    boot_ratios = boot_num_means / boot_den_means

    # Compute percentile-based CI
    alpha = 1.0 - confidence_level
    ci_lo = float(np.percentile(boot_ratios, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_ratios, 100 * (1 - alpha / 2)))

    return ratio, ci_lo, ci_hi


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

        # String matching heuristic (check specific before general:
        # "intersection" before "urban" since descriptions may contain both)
        if "highway" in description or "freeway" in description:
            bucket = "highway"
        elif "intersection" in description or "junction" in description:
            bucket = "intersection"
        elif (
            "urban" in description
            or "city" in description
            or "downtown" in description
        ):
            bucket = "urban"
        else:
            bucket = "other"

        mapping[scene_token] = bucket

    # Log scene counts per bucket to detect classification issues
    bucket_counts = Counter(mapping.values())
    total_scenes = len(scene_tokens)
    logger.info("Scene classification breakdown (total=%d):", total_scenes)
    for bucket in ["highway", "urban", "intersection", "other"]:
        count = bucket_counts.get(bucket, 0)
        pct = 100.0 * count / total_scenes if total_scenes > 0 else 0.0
        logger.info("  %s: %d (%.1f%%)", bucket, count, pct)

    return mapping


def classify_scenes_by_environment(
    scene_names: list[str],
    night_scenes: set[str],
    rain_scenes: set[str],
) -> dict[str, list[str]]:
    """Map scenes to independent night/rain/day_clear environmental subsets.

    Parameters
    ----------
    scene_names
        List of scene names to classify.
    night_scenes
        Set of scene names identified as night/low-light conditions.
    rain_scenes
        Set of scene names identified as rain/wet conditions.

    Returns
    -------
    dict[str, list[str]]
        Mapping from environment type to list of scene names.
        Keys: "night", "rain", "day_clear"
        Scenes can appear in multiple subsets (e.g., night+rain scenes are in both).

    Notes
    -----
    This function creates INDEPENDENT, possibly OVERLAPPING subsets:
    - "night": All night scenes (including night+rain)
    - "rain": All rain scenes (including night+rain)
    - "day_clear": Baseline scenes with neither night nor rain conditions

    Night+rain scenes contribute to both night and rain performance metrics,
    which is scientifically correct for isolating each environmental effect.

    Examples
    --------
    >>> result = classify_scenes_by_environment(
    ...     ["scene-0001", "scene-0002", "scene-0003", "scene-0004"],
    ...     night_scenes={"scene-0001", "scene-0002"},
    ...     rain_scenes={"scene-0002", "scene-0003"}
    ... )
    >>> result["night"]
    ['scene-0001', 'scene-0002']
    >>> result["rain"]
    ['scene-0002', 'scene-0003']
    >>> result["day_clear"]
    ['scene-0004']
    """
    night_list = [s for s in scene_names if s in night_scenes]
    rain_list = [s for s in scene_names if s in rain_scenes]
    day_clear_list = [s for s in scene_names if s not in night_scenes and s not in rain_scenes]

    # Validate that every YAML scene is present in the probe set
    # Catches both format mismatches (scene-XXXX vs scene_XXXX) and missing data
    scene_names_set = set(scene_names)

    # Helper to check format mismatch before count mismatch
    def _validate_scene_list(yaml_scenes: set[str], scene_type: str, matched_list: list[str]) -> None:
        missing = yaml_scenes - scene_names_set
        if not missing:
            return  # All scenes found

        # Check format consistency FIRST
        # Pick a missing scene to check format
        if yaml_scenes and scene_names_set and missing:
            missing_sample = next(iter(missing))
            probe_sample = next(iter(scene_names_set))

            # Check case mismatch: if lowercase match exists but exact doesn't
            probe_lower_set = {s.lower() for s in scene_names_set}
            if missing_sample.lower() in probe_lower_set and missing_sample not in scene_names_set:
                raise ValueError(
                    f"Scene-name format mismatch: YAML {scene_type}_scenes uses different case. "
                    f"Example: YAML={missing_sample}, probe uses lowercase. "
                    f"This affects {len(missing)} {scene_type} scenes."
                )

            # Check dash vs underscore
            yaml_has_dash = "-" in missing_sample
            probe_has_dash = "-" in probe_sample

            if yaml_has_dash != probe_has_dash:
                raise ValueError(
                    f"Scene-name format mismatch: YAML uses {'scene-XXXX' if yaml_has_dash else 'scene_XXXX'}, "
                    f"probe uses {'scene-XXXX' if probe_has_dash else 'scene_XXXX'}. "
                    f"Example: YAML={missing_sample}, probe={probe_sample}. "
                    f"This affects {len(missing)} {scene_type} scenes."
                )

        # Format looks consistent, so this is missing data
        if len(missing) == len(yaml_scenes):
            raise ValueError(
                f"All {len(yaml_scenes)} {scene_type} scenes missing from probe data. "
                f"YAML {scene_type}_scenes: {sorted(yaml_scenes)[:3]}, probe scene_names: {sorted(scene_names)[:3]}. "
                f"Check that probe training included these scenes."
            )
        else:
            raise ValueError(
                f"Incomplete probe data: {len(missing)} {scene_type} scenes missing from probe results. "
                f"Missing: {sorted(missing)[:5]}. YAML defines {len(yaml_scenes)} {scene_type} scenes "
                f"but probe only has {len(matched_list)}."
            )

    if night_scenes:
        _validate_scene_list(night_scenes, "night", night_list)

    if rain_scenes:
        _validate_scene_list(rain_scenes, "rain", rain_list)

    # Find overlaps
    overlap = set(night_list) & set(rain_list)

    # Log distribution
    total_scenes = len(scene_names)
    logger.info("Environment subset distribution (total=%d):", total_scenes)
    logger.info("  night:     %d (%.1f%%)", len(night_list), 100.0 * len(night_list) / total_scenes if total_scenes > 0 else 0.0)
    logger.info("  rain:      %d (%.1f%%)", len(rain_list), 100.0 * len(rain_list) / total_scenes if total_scenes > 0 else 0.0)
    logger.info("  day_clear: %d (%.1f%%)", len(day_clear_list), 100.0 * len(day_clear_list) / total_scenes if total_scenes > 0 else 0.0)
    if overlap:
        logger.info("  overlap (night+rain): %d scenes", len(overlap))

    return {
        "night": sorted(night_list),
        "rain": sorted(rain_list),
        "day_clear": sorted(day_clear_list),
    }


def compute_robustness_ratios(
    encoder_df: pd.DataFrame,
    env_subsets: dict[str, list[str]],
    steer_denorm_factor: float,
    accel_denorm_factor: float,
    n_resamples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> list[dict]:
    """Compute robustness ratios with bootstrap CIs for one encoder.

    Processes both steering and acceleration metrics, computing night/day and
    rain/day RMSE ratios with bootstrap confidence intervals.

    Args:
        encoder_df: Per-scene RMSE for one encoder (columns: scene_name, steer_rmse, accel_rmse)
        env_subsets: Environment scene lists from classify_scenes_by_environment
        steer_denorm_factor: Multiply steer_rmse by this to get degrees
        accel_denorm_factor: Multiply accel_rmse by this to get m/s²
        n_resamples: Bootstrap resamples (typically 1000)
        bootstrap_seed: Seed for reproducibility
        confidence_level: CI level (typically 0.95)

    Returns:
        List of dicts with columns: encoder, metric, rmse_night, rmse_rain, rmse_day_clear,
        ratio_night_day, ci_lo_night_day, ci_hi_night_day, ratio_rain_day, ci_lo_rain_day,
        ci_hi_rain_day, n_night, n_rain, n_day_clear

    Notes:
        - Skips metrics when n_day_clear == 0 (no baseline for ratio)
        - Sets ratio columns to None when environment subset is empty
        - Returns 2 rows (one per metric) if both have valid baselines
    """
    results = []

    # Process both metrics (values start in normalized [-1, 1] space)
    for metric_col, metric_name, denorm_factor in [
        ("steer_rmse", "steer_rmse_deg", steer_denorm_factor),
        ("accel_rmse", "accel_rmse_mps2", accel_denorm_factor),
    ]:
        # Extract per-scene RMSE for each environment (normalized [-1, 1] space)
        night_df = encoder_df[encoder_df["scene_name"].isin(env_subsets["night"])]
        rain_df = encoder_df[encoder_df["scene_name"].isin(env_subsets["rain"])]
        day_clear_df = encoder_df[encoder_df["scene_name"].isin(env_subsets["day_clear"])]

        night_values_norm = night_df[metric_col].values  # normalized [-1, 1]
        rain_values_norm = rain_df[metric_col].values    # normalized [-1, 1]
        day_clear_values_norm = day_clear_df[metric_col].values  # normalized [-1, 1]

        # Convert to physical units (degrees for steering, m/s² for acceleration)
        night_values = night_values_norm * denorm_factor  # → degrees or m/s²
        rain_values = rain_values_norm * denorm_factor    # → degrees or m/s²
        day_clear_values = day_clear_values_norm * denorm_factor  # → degrees or m/s²

        n_night = len(night_values)
        n_rain = len(rain_values)
        n_day_clear = len(day_clear_values)

        # Validate baseline
        if n_day_clear == 0:
            continue  # Skip this metric (no baseline for ratio)

        # Compute mean RMSE for each environment
        rmse_night = float(night_values.mean()) if n_night > 0 else None
        rmse_rain = float(rain_values.mean()) if n_rain > 0 else None
        rmse_day_clear = float(day_clear_values.mean())

        # Compute night/day ratio with bootstrap CI
        if n_night > 0:
            ratio_night_day, ci_lo_night_day, ci_hi_night_day = bootstrap_ratio_ci(
                night_values,
                day_clear_values,
                n_resamples=n_resamples,
                seed=bootstrap_seed,
                confidence_level=confidence_level,
            )
        else:
            ratio_night_day = None
            ci_lo_night_day = None
            ci_hi_night_day = None

        # Compute rain/day ratio with bootstrap CI
        if n_rain > 0:
            ratio_rain_day, ci_lo_rain_day, ci_hi_rain_day = bootstrap_ratio_ci(
                rain_values,
                day_clear_values,
                n_resamples=n_resamples,
                seed=bootstrap_seed,
                confidence_level=confidence_level,
            )
        else:
            ratio_rain_day = None
            ci_lo_rain_day = None
            ci_hi_rain_day = None

        results.append({
            "metric": metric_name,
            "rmse_night": rmse_night,
            "rmse_rain": rmse_rain,
            "rmse_day_clear": rmse_day_clear,
            "ratio_night_day": ratio_night_day,
            "ci_lo_night_day": ci_lo_night_day,
            "ci_hi_night_day": ci_hi_night_day,
            "ratio_rain_day": ratio_rain_day,
            "ci_lo_rain_day": ci_lo_rain_day,
            "ci_hi_rain_day": ci_hi_rain_day,
            "n_night": n_night,
            "n_rain": n_rain,
            "n_day_clear": n_day_clear,
        })

    return results


def compute_per_scenario_rmse(
    predictions_df: pd.DataFrame,
    scene_to_bucket: dict[str, str],
    cfg: CanonicalConfig | dict[str, Any],
) -> pd.DataFrame:
    """Compute per-scenario RMSE with bootstrap confidence intervals.

    IMPORTANT: This function computes RMSE in normalized space (matching the
    dataset and model output). To convert steering RMSE to degrees, use
    convert_steer_rmse_to_deg() on the returned values.

    Args:
        predictions_df: DataFrame with columns:
            - encoder: str
            - scene_name (or scene_token): str
            - steer_pred, accel_pred: float (normalized)
            - steer_true, accel_true: float (normalized)
        scene_to_bucket: Mapping from scene identifier to scenario bucket
        cfg: Canonical config or plain dict with evaluation.bootstrap parameters

    Returns:
        DataFrame with columns:
            - encoder, scenario, metric, mean, ci_lo, ci_hi
            where metric is 'steer_rmse' or 'accel_rmse'
            All RMSE values are in normalized space.

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

        # Bootstrap steering (normalized)
        steer_mean, steer_lo, steer_hi = bootstrap_mean_ci(
            scene_steer, n_resamples, seed, confidence_level
        )
        results.append({
            "encoder": encoder,
            "scenario": scenario,
            "metric": "steer_rmse",
            "mean": steer_mean,
            "ci_lo": steer_lo,
            "ci_hi": steer_hi,
        })

        # Bootstrap acceleration (normalized)
        accel_mean, accel_lo, accel_hi = bootstrap_mean_ci(
            scene_accel, n_resamples, seed, confidence_level
        )
        results.append({
            "encoder": encoder,
            "scenario": scenario,
            "metric": "accel_rmse",
            "mean": accel_mean,
            "ci_lo": accel_lo,
            "ci_hi": accel_hi,
        })

    # Build output DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["encoder", "scenario", "metric"]).reset_index(drop=True)

    return results_df


def load_per_scene_rmse(probe_root: Path, metric: str) -> dict[str, pd.Series]:
    """Scan <probe_root>/*/per_scene_rmse.csv and return per-encoder series.

    Each returned series is indexed by scene_name so paired tests can align
    cleanly across encoders. Raises ValueError if scene sets don't match.

    Args:
        probe_root: Directory containing <encoder>/per_scene_rmse.csv
        metric: RMSE metric column name (e.g., 'steer_rmse', 'accel_rmse')

    Returns:
        dict mapping encoder_name -> pd.Series(scene_name -> rmse_value)

    Raises:
        FileNotFoundError: If probe_root doesn't exist or no CSVs found
        ValueError: If scene sets don't match across encoders or columns missing
    """
    probe_root = Path(probe_root)
    if not probe_root.exists():
        raise FileNotFoundError(f"probe-root does not exist: {probe_root}")

    per_encoder: dict[str, pd.Series] = {}
    for enc_dir in sorted(p for p in probe_root.iterdir() if p.is_dir()):
        csv_path = enc_dir / "per_scene_rmse.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if metric not in df.columns or "scene_name" not in df.columns:
            raise ValueError(
                f"{csv_path}: missing required columns "
                f"({metric!r}, 'scene_name'); got {list(df.columns)!r}"
            )
        # If fold_id is present and >0 rows exist per scene, mean across folds
        if "fold_id" in df.columns:
            df = df.groupby("scene_name", as_index=True)[metric].mean()
        else:
            df = df.set_index("scene_name")[metric]
        per_encoder[enc_dir.name] = df.sort_index()

    if not per_encoder:
        raise FileNotFoundError(
            f"no per_scene_rmse.csv files found under {probe_root}"
        )

    # Require identical scene sets across encoders
    reference_scenes = next(iter(per_encoder.values())).index
    for enc, series in per_encoder.items():
        if not series.index.equals(reference_scenes):
            missing = set(reference_scenes) - set(series.index)
            extra = set(series.index) - set(reference_scenes)
            raise ValueError(
                f"encoder {enc!r} has a mismatched scene set; "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

    return per_encoder


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


def denormalize_rmse_dataframe(
    df: pd.DataFrame,
    cfg: CanonicalConfig | None = None,
    value_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Convert normalized RMSE metrics to physical units in a DataFrame.

    This function is for OUTPUT/PRESENTATION only. The internal evaluation workflow
    operates in normalized space; denormalization happens only when writing CSVs
    for human consumption.

    Args:
        df: DataFrame with 'metric' column containing 'steer_rmse' and/or
            'accel_rmse' and numeric value columns to convert.
        cfg: Canonical config with normalization factors. If None, loads canonical config.
        value_cols: List of numeric columns to convert (e.g., ["mean", "ci_lo", "ci_hi"]).
            If None, defaults to ["mean", "ci_lo", "ci_hi"].

    Returns:
        Modified DataFrame with denormalized values and renamed metrics:
        - steer_rmse → steer_rmse_deg (multiplied by eval_back_to_deg_factor)
        - accel_rmse → accel_rmse_mps2 (multiplied by divisor)

    Examples:
        >>> df = pd.DataFrame({
        ...     "encoder": ["vit_s16", "vit_s16"],
        ...     "metric": ["steer_rmse", "accel_rmse"],
        ...     "mean": [0.1, 0.05],
        ...     "ci_lo": [0.09, 0.04],
        ...     "ci_hi": [0.11, 0.06]
        ... })
        >>> denormalized = denormalize_rmse_dataframe(df)
        >>> denormalized["metric"].tolist()  # doctest: +SKIP
        ['steer_rmse_deg', 'accel_rmse_mps2']
    """
    if cfg is None:
        from config import load_canonical
        cfg = load_canonical()

    if value_cols is None:
        value_cols = ["mean", "ci_lo", "ci_hi"]

    # Get normalization factors
    if hasattr(cfg, 'normalization'):
        steer_factor = cfg.normalization("steering")["eval_back_to_deg_factor"]
        accel_factor = cfg.normalization("acceleration")["divisor"]
    else:
        steer_factor = cfg["normalization"]["steering"]["eval_back_to_deg_factor"]
        accel_factor = cfg["normalization"]["acceleration"]["divisor"]

    df = df.copy()

    # Convert steering
    steer_mask = df["metric"] == "steer_rmse"
    for col in value_cols:
        if col in df.columns:
            df.loc[steer_mask, col] *= steer_factor
    df.loc[steer_mask, "metric"] = "steer_rmse_deg"

    # Convert acceleration
    accel_mask = df["metric"] == "accel_rmse"
    for col in value_cols:
        if col in df.columns:
            df.loc[accel_mask, col] *= accel_factor
    df.loc[accel_mask, "metric"] = "accel_rmse_mps2"

    return df


# --- Merged from main-tier2 ---

