"""Unit tests for evaluation.perturbation module."""

import pytest
import torch

from evaluation.perturbation import apply_perturbation, select_validation_frames


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
