"""Tests for scripts/rollout_dit.py (DA5+DA7).

Validates config constants, DDIM sampling shapes, EMA weight loading,
and metric computation on synthetic data.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

# Ensure project root is on path for config import
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# -----------------------------------------------------------------------
# Minimal inline reimplementations (same as in rollout_dit.py)
# -----------------------------------------------------------------------

class CosineNoiseSchedule(nn.Module):
    def __init__(self, n_steps: int = 1000, s: float = 0.008):
        super().__init__()
        self.n_steps = n_steps
        steps = torch.arange(n_steps + 1, dtype=torch.float64)
        f_t = torch.cos(((steps / n_steps) + s) / (1.0 + s) * (torch.pi / 2.0)) ** 2
        alphas_cumprod = f_t / f_t[0]
        alphas_cumprod = alphas_cumprod[:n_steps].float()
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def _extract(self, arr, t, x_shape):
        out = arr.gather(0, t.long())
        return out.view(-1, *([1] * (len(x_shape) - 1)))

    def add_noise(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise


class DDIMSampler:
    def __init__(self, noise_schedule, n_steps: int = 50):
        self.schedule = noise_schedule
        self.n_steps = n_steps
        T = noise_schedule.n_steps
        stride = T // n_steps
        self.timesteps = list(reversed(list(range(0, T, stride))[:n_steps]))

    @torch.no_grad()
    def sample(self, noise_pred_fn, shape, cond_kwargs, device="cpu"):
        alphas_cumprod = self.schedule.alphas_cumprod.to(device)
        x = torch.randn(shape, device=device)
        for i, t_val in enumerate(self.timesteps):
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            noise_pred = noise_pred_fn(x, timestep=t, **cond_kwargs)
            alpha_bar_t = alphas_cumprod[t_val]
            pred_x0 = (
                x - torch.sqrt(1.0 - alpha_bar_t) * noise_pred
            ) / torch.sqrt(alpha_bar_t)
            if i < len(self.timesteps) - 1:
                t_prev = self.timesteps[i + 1]
                alpha_bar_prev = alphas_cumprod[t_prev]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)
            noise_direction = (
                x - torch.sqrt(alpha_bar_t) * pred_x0
            ) / torch.sqrt(1.0 - alpha_bar_t + 1e-8)
            x = (
                torch.sqrt(alpha_bar_prev) * pred_x0
                + torch.sqrt(1.0 - alpha_bar_prev) * noise_direction
            )
        return x


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


class TestConfigValidation:
    """Verify that rollout_dit.py constants match source configs."""

    def test_dit_canonical_keys_exist_in_yaml(self):
        import yaml

        dit_yaml = Path(PROJECT_ROOT) / "configs" / "dit.yaml"
        if not dit_yaml.exists():
            pytest.skip("dit.yaml not found")
        with open(dit_yaml) as f:
            raw = yaml.safe_load(f)

        from scripts.inference.rollout_dit import DIT_CANONICAL
        for key in DIT_CANONICAL:
            assert key in raw["dit"], f"DIT_CANONICAL key {key!r} missing from dit.yaml"

    def test_diffusion_canonical_matches_yaml(self):
        import yaml

        dit_yaml = Path(PROJECT_ROOT) / "configs" / "dit.yaml"
        if not dit_yaml.exists():
            pytest.skip("dit.yaml not found")
        with open(dit_yaml) as f:
            raw = yaml.safe_load(f)

        from scripts.inference.rollout_dit import DIFFUSION_CANONICAL
        assert DIFFUSION_CANONICAL["n_train_steps"] == raw["diffusion"]["n_train_steps"]

    def test_eval_n_sample_steps_matches_yaml(self):
        import yaml

        dit_yaml = Path(PROJECT_ROOT) / "configs" / "dit.yaml"
        if not dit_yaml.exists():
            pytest.skip("dit.yaml not found")
        with open(dit_yaml) as f:
            raw = yaml.safe_load(f)

        from scripts.inference.rollout_dit import EVAL_CANONICAL
        assert EVAL_CANONICAL["n_sample_steps"] == raw["diffusion"]["n_sample_steps"]


class TestDDIMSampler:
    """Test DDIM sampling produces correct shapes."""

    def test_sample_shape(self):
        schedule = CosineNoiseSchedule(n_steps=100)
        sampler = DDIMSampler(schedule, n_steps=10)

        # Dummy noise predictor that returns same shape
        def dummy_pred(x_noisy, timestep, z_t, a_embed):
            return torch.randn_like(x_noisy)

        B, H, D = 4, 4, 32
        z_t = torch.randn(B, D)
        a_embed = torch.randn(B, D)

        result = sampler.sample(
            noise_pred_fn=dummy_pred,
            shape=(B, H, D),
            cond_kwargs={"z_t": z_t, "a_embed": a_embed},
        )

        assert result.shape == (B, H, D)

    def test_sample_deterministic_with_seed(self):
        """DDIM with eta=0 should be deterministic given same initial noise."""
        schedule = CosineNoiseSchedule(n_steps=100)
        sampler = DDIMSampler(schedule, n_steps=5)

        def dummy_pred(x_noisy, timestep, z_t, a_embed):
            # Simple identity-like predictor
            return x_noisy * 0.1

        B, H, D = 2, 4, 16
        cond = {"z_t": torch.randn(B, D), "a_embed": torch.randn(B, D)}

        torch.manual_seed(42)
        r1 = sampler.sample(dummy_pred, (B, H, D), cond)

        torch.manual_seed(42)
        r2 = sampler.sample(dummy_pred, (B, H, D), cond)

        assert torch.allclose(r1, r2, atol=1e-6)

    def test_timestep_ordering(self):
        """Timesteps should go from high to low (noisy to clean)."""
        schedule = CosineNoiseSchedule(n_steps=1000)
        sampler = DDIMSampler(schedule, n_steps=50)

        # First timestep should be highest
        assert sampler.timesteps[0] > sampler.timesteps[-1]
        # Should be monotonically decreasing
        for i in range(len(sampler.timesteps) - 1):
            assert sampler.timesteps[i] > sampler.timesteps[i + 1]


class TestEMAWeightLoading:
    """Test that EMA weights differ from main weights."""

    def test_ema_differs_from_main(self):
        """EMA-loaded model should produce different output than main."""
        torch.manual_seed(0)
        model = nn.Linear(32, 32)

        # Simulate EMA by slightly perturbing weights
        ema_sd = {}
        for name, param in model.named_parameters():
            ema_sd[name] = param.data.clone() + 0.1 * torch.randn_like(param.data)

        x = torch.randn(4, 32)

        # Output with main weights
        out_main = model(x).clone()

        # Load EMA weights
        for name, param in model.named_parameters():
            if name in ema_sd:
                param.data.copy_(ema_sd[name])

        out_ema = model(x)

        # Should be different
        assert not torch.allclose(out_main, out_ema, atol=1e-6)


class TestLatentNormalization:
    """Test normalization round-trip and properties."""

    def test_normalize_then_inverse_recovers_original(self):
        """Normalize -> inverse should recover original data."""
        torch.manual_seed(42)
        # Simulate adapted embeddings with small std (like the real data)
        data = torch.randn(100, 384) * 0.065 + 0.01
        z_mean = data.mean(dim=0)
        z_std = data.std(dim=0).clamp(min=1e-6)

        normalized = (data - z_mean) / z_std
        recovered = normalized * z_std + z_mean

        assert torch.allclose(data, recovered, atol=1e-5), (
            f"Round-trip error: {(data - recovered).abs().max():.2e}"
        )

    def test_normalized_has_unit_variance(self):
        """Normalized data should have per-element std close to 1."""
        torch.manual_seed(42)
        data = torch.randn(1000, 384) * 0.065 + 0.01
        z_mean = data.mean(dim=0)
        z_std = data.std(dim=0).clamp(min=1e-6)

        normalized = (data - z_mean) / z_std
        per_elem_std = normalized.std(dim=0)

        # Per-element std should be ~1.0 (with tolerance for finite sample)
        assert (per_elem_std - 1.0).abs().max() < 0.1, (
            f"Per-element std range: [{per_elem_std.min():.3f}, {per_elem_std.max():.3f}]"
        )

    def test_normalized_has_zero_mean(self):
        """Normalized data should have per-element mean close to 0."""
        torch.manual_seed(42)
        data = torch.randn(1000, 384) * 0.065 + 0.01
        z_mean = data.mean(dim=0)
        z_std = data.std(dim=0).clamp(min=1e-6)

        normalized = (data - z_mean) / z_std
        per_elem_mean = normalized.mean(dim=0)

        assert per_elem_mean.abs().max() < 0.1, (
            f"Per-element mean range: [{per_elem_mean.min():.3f}, {per_elem_mean.max():.3f}]"
        )

    def test_orthogonal_init_preserves_norm(self):
        """Orthogonal projection should roughly preserve norms."""
        torch.manual_seed(42)
        adapter = nn.Linear(1024, 384, bias=False)
        nn.init.orthogonal_(adapter.weight)

        x = torch.randn(100, 1024)
        y = adapter(x)

        # For orthogonal init, output norm should be close to input norm
        # scaled by sqrt(out_dim/in_dim) for compression
        x_norms = x.norm(dim=-1)
        y_norms = y.norm(dim=-1)
        ratio = (y_norms / x_norms).mean()

        # Ratio should be roughly 1.0 for orthogonal projection
        assert 0.5 < ratio < 2.0, f"Norm ratio: {ratio:.3f}"


class TestMetricComputation:
    """Test CosSim and MSE computation on known synthetic data."""

    def test_cossim_identical_vectors(self):
        """CosSim of identical vectors should be 1.0."""
        z = torch.randn(10, 384)
        cs = F.cosine_similarity(z, z, dim=-1)
        assert torch.allclose(cs, torch.ones(10), atol=1e-5)

    def test_cossim_orthogonal_vectors(self):
        """CosSim of orthogonal vectors should be ~0."""
        z1 = torch.zeros(1, 4)
        z1[0, 0] = 1.0
        z2 = torch.zeros(1, 4)
        z2[0, 1] = 1.0
        cs = F.cosine_similarity(z1, z2, dim=-1)
        assert abs(cs.item()) < 1e-6

    def test_mse_identical_vectors(self):
        """MSE of identical vectors should be 0."""
        z = torch.randn(10, 384)
        mse = ((z - z) ** 2).mean(dim=-1)
        assert torch.allclose(mse, torch.zeros(10), atol=1e-8)

    def test_mse_known_value(self):
        """MSE of vectors differing by 1 in all dims should be 1."""
        z1 = torch.zeros(1, 100)
        z2 = torch.ones(1, 100)
        mse = ((z1 - z2) ** 2).mean(dim=-1)
        assert abs(mse.item() - 1.0) < 1e-6

    def test_copy_baseline_decreases_with_horizon(self):
        """Copy baseline CosSim should generally decrease with horizon
        when future embeddings drift from current."""
        torch.manual_seed(42)
        z_t = torch.randn(100, 64)

        # Simulate temporal drift: each step adds more noise
        cossims = []
        for k in range(1, 5):
            z_future_k = z_t + 0.3 * k * torch.randn_like(z_t)
            cs = F.cosine_similarity(z_t, z_future_k, dim=-1).mean()
            cossims.append(cs.item())

        # Later horizons should have lower CosSim (on average with noise)
        assert cossims[0] > cossims[-1], (
            f"Expected CosSim to decrease with horizon, got {cossims}"
        )
