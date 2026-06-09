"""Tests for spatial DiT and spatial MLP predictor models.

Covers forward-shape correctness and residual-identity-at-init,
as requested in PR #30 review.
"""

from __future__ import annotations

import torch
import pytest

from models.latent_dit_spatial import SpatialTemporalDiT, FourierActionEmbedding
from models.latent_pred_spatial import SpatialMLPPredictor


# --- SpatialTemporalDiT ---

class TestSpatialTemporalDiT:
    """Tests for the SpatialTemporalDiT model."""

    @pytest.fixture
    def dit(self):
        return SpatialTemporalDiT(
            z_dim=384, cond_dim=384, n_blocks=4, n_heads=6,
            horizon=4, n_spatial=49, mlp_ratio=4.0, dropout=0.0,
        )

    @pytest.fixture
    def fourier(self):
        return FourierActionEmbedding(action_dim=2, n_frequencies=64, base=2.0, out_dim=384)

    def test_forward_shape(self, dit, fourier):
        """Output shape must be (B, H*S, D)."""
        B, H, S, D = 2, 4, 49, 384
        x_noisy = torch.randn(B, H * S, D)
        z_t_spatial = torch.randn(B, S, D)
        actions = torch.randn(B, H, 2)
        a_embed = fourier(actions)
        timestep = torch.randint(0, 1000, (B,))

        out = dit(x_noisy, z_t_spatial, a_embed, timestep)
        assert out.shape == (B, H * S, D), f"Expected ({B}, {H*S}, {D}), got {out.shape}"

    def test_forward_shape_dino_grid(self):
        """Test with DINOv2 native grid (16x16 = 256 tokens)."""
        B, H, S, D = 2, 4, 256, 384
        dit = SpatialTemporalDiT(
            z_dim=D, cond_dim=D, n_blocks=2, n_heads=6,
            horizon=H, n_spatial=S,
        )
        fourier = FourierActionEmbedding(action_dim=2, n_frequencies=64, base=2.0, out_dim=D)

        x_noisy = torch.randn(B, H * S, D)
        z_t_spatial = torch.randn(B, S, D)
        a_embed = fourier(torch.randn(B, H, 2))
        timestep = torch.randint(0, 1000, (B,))

        out = dit(x_noisy, z_t_spatial, a_embed, timestep)
        assert out.shape == (B, H * S, D)

    def test_identity_at_init(self, dit, fourier):
        """At initialization, the DiT output should be near-zero.

        final_linear is zero-initialized, so gate * final_linear(...)
        should produce near-zero output regardless of input, making the
        model behave as an identity (residual anchor) at init.
        """
        B, H, S, D = 2, 4, 49, 384
        x_noisy = torch.randn(B, H * S, D)
        z_t_spatial = torch.randn(B, S, D)
        a_embed = fourier(torch.randn(B, H, 2))
        timestep = torch.randint(0, 1000, (B,))

        with torch.no_grad():
            out = dit(x_noisy, z_t_spatial, a_embed, timestep)

        # final_linear is zero-init, so output should be all zeros
        assert out.abs().max() < 1e-5, (
            f"Expected near-zero output at init (residual anchor), "
            f"got max abs = {out.abs().max().item():.6f}"
        )

    def test_final_linear_is_zero_init(self, dit):
        """Verify final_linear weights and bias are zero-initialized."""
        assert torch.all(dit.final_linear.weight == 0), "final_linear.weight not zero"
        assert torch.all(dit.final_linear.bias == 0), "final_linear.bias not zero"

    def test_batch_size_one(self, dit, fourier):
        """Single-sample batch should work without errors."""
        B, H, S, D = 1, 4, 49, 384
        out = dit(
            torch.randn(B, H * S, D),
            torch.randn(B, S, D),
            fourier(torch.randn(B, H, 2)),
            torch.randint(0, 1000, (B,)),
        )
        assert out.shape == (B, H * S, D)


# --- SpatialMLPPredictor ---

class TestSpatialMLPPredictor:
    """Tests for the SpatialMLPPredictor model."""

    @pytest.fixture
    def mlp(self):
        return SpatialMLPPredictor(
            z_dim=384, a_dim=384, horizon=4, n_spatial=49, hidden=512,
        )

    @pytest.fixture
    def fourier(self):
        return FourierActionEmbedding(action_dim=2, n_frequencies=64, base=2.0, out_dim=384)

    def test_forward_shape(self, mlp, fourier):
        """Output shape must be (B, H*S, D)."""
        B, H, S, D = 2, 4, 49, 384
        z_t_spatial = torch.randn(B, S, D)
        a_embed = fourier(torch.randn(B, H, 2))

        out = mlp(z_t_spatial, a_embed)
        assert out.shape == (B, H * S, D), f"Expected ({B}, {H*S}, {D}), got {out.shape}"

    def test_forward_shape_dino_grid(self):
        """Test with DINOv2 native grid (16x16 = 256 tokens)."""
        B, H, S, D = 2, 4, 256, 384
        mlp = SpatialMLPPredictor(z_dim=D, a_dim=D, horizon=H, n_spatial=S)
        fourier = FourierActionEmbedding(action_dim=2, n_frequencies=64, base=2.0, out_dim=D)

        z_t_spatial = torch.randn(B, S, D)
        a_embed = fourier(torch.randn(B, H, 2))

        out = mlp(z_t_spatial, a_embed)
        assert out.shape == (B, H * S, D)

    def test_residual_structure(self, mlp, fourier):
        """Output should be z_t + delta (residual prediction)."""
        B, H, S, D = 2, 4, 49, 384
        z_t_spatial = torch.randn(B, S, D)
        a_embed = fourier(torch.randn(B, H, 2))

        # Zero out the MLP weights to check that residual = z_t
        with torch.no_grad():
            for p in mlp.net.parameters():
                p.zero_()

        out = mlp(z_t_spatial, a_embed)
        # When net output is zero, z_hat = z_t + 0 = z_t for each horizon step
        # Output shape is (B, H*S, D), so each block of S tokens should equal z_t
        for h in range(H):
            block = out[:, h * S : (h + 1) * S, :]
            assert torch.allclose(block, z_t_spatial, atol=1e-6), (
                f"With zeroed net, horizon {h} output should equal z_t"
            )

    def test_batch_size_one(self, mlp, fourier):
        """Single-sample batch should work without errors."""
        B, H, S, D = 1, 4, 49, 384
        out = mlp(
            torch.randn(B, S, D),
            fourier(torch.randn(B, H, 2)),
        )
        assert out.shape == (B, H * S, D)
