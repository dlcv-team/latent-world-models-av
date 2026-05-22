"""Tests for :class:`models.latent_dit.LatentDiT` and sub-components."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.latent_dit import (
    DEFAULT_COND_DIM,
    DEFAULT_HORIZON,
    DEFAULT_MLP_RATIO,
    DEFAULT_N_BLOCKS,
    DEFAULT_N_HEADS,
    DEFAULT_Z_DIM,
    DiTBlock,
    LatentDiT,
    TimestepEmbedding,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B = 8
HORIZON = DEFAULT_HORIZON
Z_DIM = DEFAULT_Z_DIM


def _random_inputs(
    batch: int = B, z_dim: int = Z_DIM, horizon: int = HORIZON
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(x_noisy, z_t, a_embed, timestep)`` with random values."""
    x_noisy = torch.randn(batch, horizon, z_dim)
    z_t = torch.randn(batch, z_dim)
    a_embed = torch.randn(batch, z_dim)
    timestep = torch.randint(0, 1000, (batch,))
    return x_noisy, z_t, a_embed, timestep


# ---------------------------------------------------------------------------
# TimestepEmbedding
# ---------------------------------------------------------------------------


class TestTimestepEmbedding:
    def test_output_shape(self):
        te = TimestepEmbedding(cond_dim=DEFAULT_COND_DIM)
        t = torch.randint(0, 1000, (B,))
        out = te(t)
        assert out.shape == (B, DEFAULT_COND_DIM)

    def test_different_timesteps_different_output(self):
        te = TimestepEmbedding(cond_dim=DEFAULT_COND_DIM)
        t1 = torch.zeros(B, dtype=torch.long)
        t2 = torch.full((B,), 500, dtype=torch.long)
        out1 = te(t1)
        out2 = te(t2)
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# DiTBlock
# ---------------------------------------------------------------------------


class TestDiTBlock:
    def test_output_shape(self):
        block = DiTBlock(dim=Z_DIM, cond_dim=DEFAULT_COND_DIM)
        x = torch.randn(B, HORIZON, Z_DIM)
        cond = torch.randn(B, DEFAULT_COND_DIM)
        out = block(x, cond)
        assert out.shape == (B, HORIZON, Z_DIM)

    def test_zero_gate_init(self):
        """Fresh DiTBlock has zero-initialised adaLN linear (gates = 0)."""
        block = DiTBlock(dim=Z_DIM, cond_dim=DEFAULT_COND_DIM)
        assert torch.allclose(
            block.adaln_linear.weight, torch.zeros_like(block.adaln_linear.weight)
        )
        assert torch.allclose(
            block.adaln_linear.bias, torch.zeros_like(block.adaln_linear.bias)
        )

    def test_identity_at_init(self):
        """With zero gates a fresh DiTBlock should act near-identity."""
        block = DiTBlock(dim=Z_DIM, cond_dim=DEFAULT_COND_DIM).eval()
        x = torch.randn(B, HORIZON, Z_DIM)
        cond = torch.randn(B, DEFAULT_COND_DIM)
        out = block(x, cond)
        assert torch.allclose(out, x, atol=1e-5)


# ---------------------------------------------------------------------------
# LatentDiT – shape checks
# ---------------------------------------------------------------------------


class TestLatentDiTShape:
    def test_forward_shape(self):
        """Forward pass produces correct output shape ``(B, 4, 384)``."""
        model = LatentDiT()
        x_noisy, z_t, a_embed, timestep = _random_inputs()
        out = model(x_noisy, z_t, a_embed, timestep)
        assert out.shape == (B, HORIZON, Z_DIM)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("b", [1, 4, 16])
    def test_various_batch_sizes(self, b: int):
        model = LatentDiT().eval()
        x_noisy, z_t, a_embed, timestep = _random_inputs(batch=b)
        out = model(x_noisy, z_t, a_embed, timestep)
        assert out.shape == (b, HORIZON, Z_DIM)

    def test_output_dtype_float32(self):
        model = LatentDiT()
        x_noisy, z_t, a_embed, timestep = _random_inputs()
        out = model(x_noisy, z_t, a_embed, timestep)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# LatentDiT – gradient flow
# ---------------------------------------------------------------------------


class TestLatentDiTGradient:
    def test_all_params_have_grad(self):
        """Loss on output .backward() produces non-None gradients on all parameters."""
        model = LatentDiT()
        x_noisy, z_t, a_embed, timestep = _random_inputs(batch=4)
        out = model(x_noisy, z_t, a_embed, timestep)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"param {name!r} has None gradient"
            # Gate params are zero-init so some grads may be near-zero,
            # but they must still exist.


# ---------------------------------------------------------------------------
# LatentDiT – conditioning sensitivity
# ---------------------------------------------------------------------------


class TestLatentDiTConditioning:
    """Conditioning sensitivity tests.

    Because adaLN-Zero initialises all gates to zero, a fresh model
    outputs near-zero regardless of conditioning.  We perform one
    gradient step to break this symmetry before checking sensitivity.
    """

    @staticmethod
    def _trained_model() -> LatentDiT:
        """Return a model after one gradient step to break zero-gate symmetry."""
        torch.manual_seed(42)
        model = LatentDiT()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        x_noisy, z_t, a_embed, timestep = _random_inputs(batch=4)
        target = torch.randn_like(x_noisy)
        loss = ((model(x_noisy, z_t, a_embed, timestep) - target) ** 2).mean()
        loss.backward()
        opt.step()
        model.eval()
        return model

    def test_timestep_sensitivity(self):
        """Different timesteps produce different outputs."""
        model = self._trained_model()
        x_noisy = torch.randn(B, HORIZON, Z_DIM)
        z_t = torch.randn(B, Z_DIM)
        a_embed = torch.randn(B, Z_DIM)

        t1 = torch.zeros(B, dtype=torch.long)
        t2 = torch.full((B,), 500, dtype=torch.long)
        out1 = model(x_noisy, z_t, a_embed, t1)
        out2 = model(x_noisy, z_t, a_embed, t2)
        assert not torch.allclose(out1, out2)

    def test_z_t_sensitivity(self):
        """Different z_t embeddings produce different outputs."""
        model = self._trained_model()
        x_noisy = torch.randn(B, HORIZON, Z_DIM)
        a_embed = torch.randn(B, Z_DIM)
        timestep = torch.randint(0, 1000, (B,))

        z_t_1 = torch.randn(B, Z_DIM)
        z_t_2 = torch.randn(B, Z_DIM)
        out1 = model(x_noisy, z_t_1, a_embed, timestep)
        out2 = model(x_noisy, z_t_2, a_embed, timestep)
        assert not torch.allclose(out1, out2)

    def test_a_embed_sensitivity(self):
        """Different action embeddings produce different outputs."""
        model = self._trained_model()
        x_noisy = torch.randn(B, HORIZON, Z_DIM)
        z_t = torch.randn(B, Z_DIM)
        timestep = torch.randint(0, 1000, (B,))

        a1 = torch.randn(B, Z_DIM)
        a2 = torch.randn(B, Z_DIM)
        out1 = model(x_noisy, z_t, a1, timestep)
        out2 = model(x_noisy, z_t, a2, timestep)
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# LatentDiT – zero gate initialization (adaLN-Zero property)
# ---------------------------------------------------------------------------


class TestLatentDiTAdaLNZero:
    def test_block_gates_zero(self):
        """All DiTBlocks should have zero-initialised adaLN weights and biases."""
        model = LatentDiT()
        for i, block in enumerate(model.blocks):
            assert torch.allclose(
                block.adaln_linear.weight,
                torch.zeros_like(block.adaln_linear.weight),
            ), f"block {i} adaln weight not zero"
            assert torch.allclose(
                block.adaln_linear.bias,
                torch.zeros_like(block.adaln_linear.bias),
            ), f"block {i} adaln bias not zero"

    def test_final_adaln_zero(self):
        """Final layer's adaLN linear should also be zero-initialised."""
        model = LatentDiT()
        assert torch.allclose(
            model.final_adaln.weight,
            torch.zeros_like(model.final_adaln.weight),
        )
        assert torch.allclose(
            model.final_adaln.bias,
            torch.zeros_like(model.final_adaln.bias),
        )


# ---------------------------------------------------------------------------
# LatentDiT – config loading
# ---------------------------------------------------------------------------


class TestLatentDiTConfig:
    def test_from_dit_config_default_path(self):
        """``from_dit_config()`` loads correctly from ``configs/dit.yaml``."""
        model = LatentDiT.from_dit_config()
        assert isinstance(model, LatentDiT)
        assert model.z_dim == DEFAULT_Z_DIM
        assert model.cond_dim == DEFAULT_COND_DIM
        assert model.n_blocks == DEFAULT_N_BLOCKS
        assert model.horizon == DEFAULT_HORIZON

    def test_from_dit_config_explicit_path(self):
        """``from_dit_config`` accepts an explicit path."""
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "dit.yaml"
        model = LatentDiT.from_dit_config(cfg_path)
        assert isinstance(model, LatentDiT)
        x_noisy, z_t, a_embed, timestep = _random_inputs()
        out = model(x_noisy, z_t, a_embed, timestep)
        assert out.shape == (B, HORIZON, Z_DIM)

    def test_from_dit_config_forward_works(self):
        """Model loaded from config produces valid output."""
        model = LatentDiT.from_dit_config()
        x_noisy, z_t, a_embed, timestep = _random_inputs()
        out = model(x_noisy, z_t, a_embed, timestep)
        assert out.shape == (B, HORIZON, Z_DIM)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# LatentDiT – parameter count
# ---------------------------------------------------------------------------


class TestLatentDiTParamCount:
    def test_param_count_reasonable(self):
        """Parameter count should be in the 5-15M range for 4 blocks."""
        model = LatentDiT()
        total = sum(p.numel() for p in model.parameters())
        assert 5_000_000 <= total <= 15_000_000, (
            f"Expected 5-15M params, got {total:,}"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestLatentDiTDeterminism:
    def test_deterministic_eval(self):
        """Same inputs give identical output in eval mode."""
        model = LatentDiT().eval()
        x_noisy, z_t, a_embed, timestep = _random_inputs()
        a = model(x_noisy, z_t, a_embed, timestep).detach().clone()
        b = model(x_noisy, z_t, a_embed, timestep).detach().clone()
        assert torch.allclose(a, b)
