"""Validate ViT-S probe on nuScenes validation set."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

from data.dataset import NuScenesActionDataset, ImageNetNormalizedDataset
from evaluation.metrics import compute_rmse, scenario_breakdown


def load_vit_probe(checkpoint_path):
    """
    Load ViT-S probe checkpoint from A10.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        model: Loaded model in eval mode
    """
    # Placeholder - update with actual model architecture
    # from models.vit_probe import ViTProbe
    # model = ViTProbe.from_pretrained('google/vit-base-patch16-224', num_actions=2)
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()
    raise NotImplementedError("Update with actual ViT-S probe architecture")


def validate(model, dataloader, device='cuda'):
    """
    Run validation and collect predictions.

    Args:
        model: ViT-S probe model
        dataloader: Validation dataloader
        device: Device to run on

    Returns:
        predictions: numpy array (N, 2)
        targets: numpy array (N, 2)
        scene_tokens: list of scene tokens per frame
    """
    model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []
    all_scene_tokens = []

    with torch.no_grad():
        for frames, actions, scene_tokens, timestamps in dataloader:
            frames = frames.to(device)

            # Forward pass
            preds = model(frames)

            # Collect results
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(actions.numpy())
            all_scene_tokens.extend(scene_tokens)

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return predictions, targets, all_scene_tokens


def main():
    parser = argparse.ArgumentParser(description='Validate ViT-S probe on nuScenes val set')
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to nuScenes dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to ViT-S probe checkpoint from A10')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='nuScenes version')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for validation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    args = parser.parse_args()

    print(f"Loading ViT-S probe from {args.checkpoint}...")
    model = load_vit_probe(args.checkpoint)

    print(f"Loading validation dataset from {args.dataroot}...")
    base_dataset = NuScenesActionDataset(
        dataroot=args.dataroot,
        version=args.version,
        split='val',
        mode='frame'
    )
    val_dataset = ImageNetNormalizedDataset(base_dataset)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Validating on {len(val_dataset)} frames...")
    predictions, targets, scene_tokens = validate(model, val_loader, device=args.device)

    # Compute overall RMSE
    steer_rmse_deg, accel_rmse = compute_rmse(predictions, targets)
    print(f"\nOverall Metrics:")
    print(f"  Steering RMSE: {steer_rmse_deg:.3f}°")
    print(f"  Acceleration RMSE: {accel_rmse:.3f} m/s²")

    # Compute per-frame RMSE for scenario breakdown
    steer_rmse_per_frame = np.abs(predictions[:, 0] * 6.0 - targets[:, 0] * 6.0)
    accel_rmse_per_frame = np.abs(predictions[:, 1] * 10.0 - targets[:, 1] * 10.0)
    rmse_by_frame = np.stack([steer_rmse_per_frame, accel_rmse_per_frame], axis=1)

    # Scenario breakdown
    nusc = base_dataset.nusc
    scenario_results = scenario_breakdown(nusc, scene_tokens, rmse_by_frame)

    print(f"\nScenario Breakdown:")
    for scenario, metrics in scenario_results.items():
        if metrics['count'] > 0:
            print(f"  {scenario.capitalize():12s} ({metrics['count']:4d} frames): "
                  f"Steer {metrics['steer_rmse']:.3f}°, "
                  f"Accel {metrics['accel_rmse']:.3f} m/s²")

    print("\nValidation complete!")


if __name__ == '__main__':
    main()
