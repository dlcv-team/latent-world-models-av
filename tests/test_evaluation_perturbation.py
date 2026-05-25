"""Unit tests for evaluation.perturbation module."""

from pathlib import Path
from unittest import mock

import pytest
import torch
from torch import nn

from evaluation.perturbation import (
    apply_perturbation,
    compute_frame_drmse,
    evaluate_frame,
    load_encoder_and_probe,
    select_validation_frames,
)
from models.probe import ActionProbe


class TestSelectValidationFrames:
    """Tests for random frame selection."""

    def test_correct_number_of_frames(self):
        """Should return exactly n_frames indices."""
        indices = select_validation_frames(
            dataset_length=805, n_frames=50, seed=42
        )
        assert len(indices) == 50

    def test_indices_in_valid_range(self):
        """All indices should be within dataset bounds."""
        indices = select_validation_frames(
            dataset_length=805, n_frames=50, seed=42
        )
        assert all(0 <= idx < 805 for idx in indices)

    def test_indices_are_unique(self):
        """No duplicate indices should be selected."""
        indices = select_validation_frames(
            dataset_length=805, n_frames=50, seed=42
        )
        assert len(indices) == len(set(indices))

    def test_indices_are_sorted(self):
        """Returned indices should be sorted for efficient access."""
        indices = select_validation_frames(
            dataset_length=805, n_frames=50, seed=42
        )
        assert indices == sorted(indices)

    def test_reproducibility(self):
        """Same seed should give same indices."""
        indices1 = select_validation_frames(
            dataset_length=805, n_frames=50, seed=42
        )
        indices2 = select_validation_frames(
            dataset_length=805, n_frames=50, seed=42
        )
        assert indices1 == indices2

    def test_different_seeds_give_different_results(self):
        """Different seeds should give different indices."""
        indices1 = select_validation_frames(
            dataset_length=805, n_frames=50, seed=42
        )
        indices2 = select_validation_frames(
            dataset_length=805, n_frames=50, seed=123
        )
        assert indices1 != indices2

    def test_error_on_too_many_frames(self):
        """Should raise ValueError if n_frames > dataset_length."""
        with pytest.raises(ValueError, match="Cannot select 1000 frames"):
            select_validation_frames(
                dataset_length=805, n_frames=1000, seed=42
            )


class TestApplyPerturbation:
    """Tests for spatial perturbation functions."""

    def test_mask_left_lane_single_frame(self):
        """Left lane mask should zero columns 0-74."""
        image = torch.ones(3, 224, 224)
        perturbed = apply_perturbation(image, "mask_left_lane")

        assert perturbed.shape == (3, 224, 224)
        assert perturbed[..., 0:75].sum() == 0  # Left columns zeroed
        assert perturbed[..., 75:224].sum() > 0  # Rest untouched

    def test_mask_right_lane_single_frame(self):
        """Right lane mask should zero columns 150-224."""
        image = torch.ones(3, 224, 224)
        perturbed = apply_perturbation(image, "mask_right_lane")

        assert perturbed.shape == (3, 224, 224)
        assert perturbed[..., 150:224].sum() == 0  # Right columns zeroed
        assert perturbed[..., 0:150].sum() > 0  # Rest untouched

    def test_mask_left_lane_clip(self):
        """Left lane mask should work on V-JEPA2 clips."""
        image = torch.ones(16, 3, 224, 224)
        perturbed = apply_perturbation(image, "mask_left_lane")

        assert perturbed.shape == (16, 3, 224, 224)
        assert perturbed[..., 0:75].sum() == 0  # All frames masked
        assert perturbed[..., 75:224].sum() > 0

    def test_mask_right_lane_clip(self):
        """Right lane mask should work on V-JEPA2 clips."""
        image = torch.ones(16, 3, 224, 224)
        perturbed = apply_perturbation(image, "mask_right_lane")

        assert perturbed.shape == (16, 3, 224, 224)
        assert perturbed[..., 150:224].sum() == 0
        assert perturbed[..., 0:150].sum() > 0

    def test_does_not_modify_original(self):
        """Perturbation should not modify original tensor."""
        image = torch.ones(3, 224, 224)
        original_sum = image.sum()
        perturbed = apply_perturbation(image, "mask_left_lane")

        assert image.sum() == original_sum  # Original unchanged
        assert perturbed.sum() < original_sum  # Perturbed has zeros

    def test_mask_lead_vehicle_returns_none(self):
        """Lead vehicle mask should return None (not yet implemented)."""
        image = torch.ones(3, 224, 224)
        result = apply_perturbation(image, "mask_lead_vehicle")
        assert result is None

    def test_unknown_perturbation_raises_error(self):
        """Unknown perturbation type should raise ValueError."""
        image = torch.ones(3, 224, 224)
        with pytest.raises(ValueError, match="Unknown perturbation_type"):
            apply_perturbation(image, "invalid_perturbation")

    def test_preserves_non_masked_values(self):
        """Non-masked regions should retain original values."""
        image = torch.rand(3, 224, 224)
        perturbed = apply_perturbation(image, "mask_left_lane")

        # Check middle region (not masked by left lane)
        assert torch.allclose(perturbed[..., 100:120], image[..., 100:120])

    def test_masks_are_complete_zeros(self):
        """Masked regions should be exactly zero, not just near zero."""
        image = torch.rand(3, 224, 224) + 0.5  # Ensure non-zero values
        perturbed = apply_perturbation(image, "mask_right_lane")

        # Masked region should be exactly zero
        assert (perturbed[..., 150:224] == 0).all()


class TestLoadEncoderAndProbe:
    """Tests for encoder and probe loading from checkpoint."""

    def test_error_on_missing_checkpoint(self, tmp_path):
        """Should raise FileNotFoundError if checkpoint.pt missing."""
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_encoder_and_probe(
                "vits16",
                tmp_path / "nonexistent",
                torch.device("cpu"),
            )

    def test_error_on_invalid_encoder_name(self, tmp_path):
        """Should raise ValueError for unknown encoder."""
        with pytest.raises(ValueError, match="Unknown encoder"):
            load_encoder_and_probe(
                "invalid_encoder",
                tmp_path,
                torch.device("cpu"),
            )

    def test_error_on_malformed_checkpoint(self, tmp_path):
        """Should raise RuntimeError if checkpoint missing probe_state_dict."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save({"encoder_name": "vits16"}, checkpoint_path)

        with pytest.raises(RuntimeError, match="missing 'probe_state_dict'"):
            load_encoder_and_probe("vits16", tmp_path, torch.device("cpu"))

    @mock.patch("evaluation.perturbation.importlib.import_module")
    @mock.patch("evaluation.perturbation.ActionProbe.from_canonical")
    @mock.patch("evaluation.perturbation.load_canonical")
    def test_loads_probe_with_no_adapter(
        self, mock_load_canonical, mock_probe_from_canonical, mock_import, tmp_path
    ):
        """Should load probe correctly when no adapter in checkpoint."""
        # Mock encoder with Identity adapter (ViT-S/16, DINOv2)
        mock_encoder = mock.Mock()
        mock_encoder.adapter = nn.Identity()
        mock_encoder.to.return_value = mock_encoder
        mock_encoder.eval.return_value = mock_encoder

        mock_encoder_cls = mock.Mock(return_value=mock_encoder)
        mock_module = mock.Mock()
        mock_module.ViTS16Wrapper = mock_encoder_cls
        mock_import.return_value = mock_module

        # Mock probe
        mock_probe = mock.Mock(spec=ActionProbe)
        mock_probe.to.return_value = mock_probe
        mock_probe.eval.return_value = mock_probe
        mock_probe.parameters.return_value = [torch.randn(10)]
        mock_probe_from_canonical.return_value = mock_probe

        # Create checkpoint without adapter_state_dict
        probe_state = {"net.0.weight": torch.randn(256, 384)}
        checkpoint = {
            "probe_state_dict": probe_state,
            "encoder_name": "vits16",
            "pilot_name": "vit_s16",
        }
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Load
        encoder, adapter, probe = load_encoder_and_probe(
            "vits16", tmp_path, torch.device("cpu")
        )

        # Verify
        assert encoder is mock_encoder
        assert isinstance(adapter, nn.Identity)
        assert probe is mock_probe
        mock_probe.load_state_dict.assert_called_once()  # Called with probe_state

    @mock.patch("evaluation.perturbation.importlib.import_module")
    @mock.patch("evaluation.perturbation.ActionProbe.from_canonical")
    @mock.patch("evaluation.perturbation.load_canonical")
    def test_loads_probe_with_adapter(
        self, mock_load_canonical, mock_probe_from_canonical, mock_import, tmp_path
    ):
        """Should load trained adapter when present in checkpoint."""
        # Mock encoder with trainable adapter (CLIP, VQ-VAE, V-JEPA2)
        mock_encoder = mock.Mock()
        mock_encoder.adapter = nn.Linear(512, 384, bias=False)
        mock_encoder.to.return_value = mock_encoder
        mock_encoder.eval.return_value = mock_encoder

        mock_encoder_cls = mock.Mock(return_value=mock_encoder)
        mock_module = mock.Mock()
        mock_module.CLIPB32Wrapper = mock_encoder_cls
        mock_import.return_value = mock_module

        # Mock probe
        mock_probe = mock.Mock(spec=ActionProbe)
        mock_probe.to.return_value = mock_probe
        mock_probe.eval.return_value = mock_probe
        mock_probe.parameters.return_value = [torch.randn(10)]
        mock_probe_from_canonical.return_value = mock_probe

        # Create checkpoint WITH adapter_state_dict
        adapter_weights = torch.randn(384, 512)
        probe_state = {"net.0.weight": torch.randn(256, 384)}
        checkpoint = {
            "probe_state_dict": probe_state,
            "adapter_state_dict": {"weight": adapter_weights},
            "encoder_name": "clip",
            "pilot_name": "clip_b32",
        }
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Load
        encoder, adapter, probe = load_encoder_and_probe(
            "clip", tmp_path, torch.device("cpu")
        )

        # Verify adapter was loaded
        assert isinstance(adapter, nn.Linear)
        assert adapter.weight.shape == (384, 512)
        assert torch.allclose(adapter.weight, adapter_weights)


class TestEvaluateFrame:
    """Tests for single frame evaluation function."""

    def test_evaluate_frame_returns_rmse_tuple(self):
        """Should return (steer_rmse, accel_rmse) tuple."""
        # Create simple encoder/adapter/probe that return predictable values
        encoder = nn.Identity()
        adapter = nn.Identity()
        probe = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 2))

        image = torch.rand(3, 224, 224)
        actions = torch.tensor([0.5, -0.3])

        steer_rmse, accel_rmse = evaluate_frame(
            encoder, adapter, probe, image, actions, torch.device("cpu")
        )

        assert isinstance(steer_rmse, float)
        assert isinstance(accel_rmse, float)
        assert steer_rmse >= 0
        assert accel_rmse >= 0

    def test_evaluate_frame_perfect_prediction(self):
        """RMSE should be near zero for perfect predictions."""
        # Create probe that always predicts the ground truth
        class PerfectProbe(nn.Module):
            def __init__(self, target_actions):
                super().__init__()
                self.target = target_actions

            def forward(self, x):
                return self.target.unsqueeze(0)

        encoder = nn.Identity()
        adapter = nn.Identity()
        actions = torch.tensor([0.5, -0.3])
        probe = PerfectProbe(actions)

        image = torch.rand(3, 224, 224)

        steer_rmse, accel_rmse = evaluate_frame(
            encoder, adapter, probe, image, actions, torch.device("cpu")
        )

        assert steer_rmse < 1e-6  # Near zero
        assert accel_rmse < 1e-6

    def test_evaluate_frame_with_adapter(self):
        """Should work with non-identity adapter."""
        # Encoder outputs 512-d, adapter projects to 384-d, probe predicts actions
        encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 512))
        adapter = nn.Linear(512, 384, bias=False)
        probe = nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 2))

        image = torch.rand(3, 224, 224)
        actions = torch.tensor([0.1, 0.2])

        steer_rmse, accel_rmse = evaluate_frame(
            encoder, adapter, probe, image, actions, torch.device("cpu")
        )

        assert isinstance(steer_rmse, float)
        assert isinstance(accel_rmse, float)
        assert steer_rmse >= 0
        assert accel_rmse >= 0

    def test_evaluate_frame_with_clip_input(self):
        """Should handle V-JEPA2 clip input (16, 3, 224, 224)."""
        encoder = nn.Sequential(nn.Flatten(), nn.Linear(16 * 3 * 224 * 224, 384))
        adapter = nn.Identity()
        probe = nn.Linear(384, 2)

        image = torch.rand(16, 3, 224, 224)  # Clip of 16 frames
        actions = torch.tensor([0.0, 0.0])

        steer_rmse, accel_rmse = evaluate_frame(
            encoder, adapter, probe, image, actions, torch.device("cpu")
        )

        assert isinstance(steer_rmse, float)
        assert isinstance(accel_rmse, float)


class TestComputeFrameDRMSE:
    """Tests for delta RMSE computation."""

    def test_drmse_positive_when_masking_increases_error(self):
        """DRMSE should be positive when masking increases error."""
        # Create probe that predicts well on unmasked but poorly on masked
        class MaskSensitiveProbe(nn.Module):
            def forward(self, x):
                # Check if input looks masked (has many zeros)
                is_masked = (x == 0).float().mean() > 0.1
                if is_masked:
                    return torch.tensor([[0.9, 0.9]])  # Poor prediction
                return torch.tensor([[0.1, 0.1]])  # Good prediction

        encoder = nn.Identity()
        adapter = nn.Identity()
        probe = MaskSensitiveProbe()

        image_unmasked = torch.ones(3, 224, 224)
        image_masked = torch.ones(3, 224, 224)
        image_masked[:, :, 0:75] = 0  # Mask left lane

        actions = torch.tensor([0.1, 0.1])

        steer_drmse, accel_drmse = compute_frame_drmse(
            encoder, adapter, probe, image_unmasked, image_masked, actions, torch.device("cpu")
        )

        assert steer_drmse > 0  # Masking increased error
        assert accel_drmse > 0

    def test_drmse_zero_when_masking_has_no_effect(self):
        """DRMSE should be near zero when masking doesn't affect prediction."""
        # Probe that ignores input
        class ConstantProbe(nn.Module):
            def forward(self, x):
                return torch.tensor([[0.5, 0.5]])

        encoder = nn.Identity()
        adapter = nn.Identity()
        probe = ConstantProbe()

        image_unmasked = torch.rand(3, 224, 224)
        image_masked = apply_perturbation(image_unmasked, "mask_left_lane")

        actions = torch.tensor([0.5, 0.5])

        steer_drmse, accel_drmse = compute_frame_drmse(
            encoder, adapter, probe, image_unmasked, image_masked, actions, torch.device("cpu")
        )

        assert abs(steer_drmse) < 1e-6  # No change
        assert abs(accel_drmse) < 1e-6

    def test_drmse_returns_float_tuple(self):
        """Should return tuple of two floats."""
        encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 2))
        adapter = nn.Identity()
        probe = nn.Identity()

        image_unmasked = torch.rand(3, 224, 224)
        image_masked = torch.rand(3, 224, 224)
        actions = torch.tensor([0.0, 0.0])

        result = compute_frame_drmse(
            encoder, adapter, probe, image_unmasked, image_masked, actions, torch.device("cpu")
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)
