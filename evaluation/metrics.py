"""Evaluation metrics for action prediction."""

import numpy as np


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
