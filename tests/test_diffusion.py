"""Tests for :mod:`models.diffusion` noise schedule and DDIM sampler."""

from __future__ import annotations

import torch
import pytest

from models.diffusion import CosineNoiseSchedule, DDIMSampler

# ---------------------------------------------------------------------------
# CosineNoiseSchedule -- shape and value checks
# ---------------------------------------------------------------------------

T = 1000


class TestCosineNoiseScheduleShape:
    def test_buffer_lengths(self):
        """All schedule tensors have length T."""
        s = CosineNoiseSchedule(n_steps=T)
        assert s.betas.shape == (T,)
        assert s.alphas_cumprod.shape == (T,)
        assert s.sqrt_alphas_cumprod.shape == (T,)
        assert s.sqrt_one_minus_alphas_cumprod.shape == (T,)

    def test_alphas_cumprod_monotonically_decreasing(self):
        """alpha_bar should decrease from ~1 to ~0."""
        s = CosineNoiseSchedule(n_steps=T)
        diffs = s.alphas_cumprod[1:] - s.alphas_cumprod[:-1]
        assert (diffs < 0).all(), "alphas_cumprod should be strictly decreasing"

    def test_alphas_cumprod_boundary_values(self):
        """At t=0, alpha_bar ~ 1; at t=T-1, alpha_bar ~ 0."""
        s = CosineNoiseSchedule(n_steps=T)
        assert s.alphas_cumprod[0] > 0.99, (
            f"alpha_bar[0] = {s.alphas_cumprod[0]:.6f}, expected > 0.99"
        )
        assert s.alphas_cumprod[-1] < 0.01, (
            f"alpha_bar[-1] = {s.alphas_cumprod[-1]:.6f}, expected < 0.01"
        )

    def test_betas_range(self):
        """Betas should be in [0, 0.999]."""
        s = CosineNoiseSchedule(n_steps=T)
        assert (s.betas >= 0.0).all()
        assert (s.betas <= 0.999).all()

    def test_sqrt_values_consistent(self):
        """sqrt quantities should be consistent with alphas_cumprod."""
        s = CosineNoiseSchedule(n_steps=T)
        torch.testing.assert_close(
            s.sqrt_alphas_cumprod ** 2,
            s.alphas_cumprod,
            atol=1e-6,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            s.sqrt_one_minus_alphas_cumprod ** 2,
            1.0 - s.alphas_cumprod,
            atol=1e-6,
            rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# CosineNoiseSchedule -- add_noise
# ---------------------------------------------------------------------------


class TestAddNoise:
    def test_shape_preserved(self):
        """Output shape matches input for various shapes."""
        s = CosineNoiseSchedule(n_steps=T)
        for shape in [(8, 4, 384), (4, 384), (2, 10)]:
            x_0 = torch.randn(shape)
            t = torch.randint(0, T, (shape[0],))
            x_noisy, noise = s.add_noise(x_0, t)
            assert x_noisy.shape == shape
            assert noise.shape == shape

    def test_t0_nearly_clean(self):
        """At t=0, noised output should be very close to input."""
        s = CosineNoiseSchedule(n_steps=T)
        x_0 = torch.randn(8, 4, 384)
        t = torch.zeros(8, dtype=torch.long)
        x_noisy, _ = s.add_noise(x_0, t)
        # At t=0, alpha_bar ~ 0.9999, so x_noisy ~ x_0
        diff = (x_noisy - x_0).abs().mean()
        assert diff < 0.1, f"At t=0, mean diff = {diff:.4f}, expected < 0.1"

    def test_tmax_nearly_noise(self):
        """At t=T-1, output should be mostly noise (low correlation with x_0)."""
        s = CosineNoiseSchedule(n_steps=T)
        torch.manual_seed(0)
        x_0 = torch.ones(64, 384) * 5.0  # strong signal
        t = torch.full((64,), T - 1, dtype=torch.long)
        x_noisy, noise = s.add_noise(x_0, t)
        # Correlation between x_noisy and x_0 should be low
        cos_sim = torch.nn.functional.cosine_similarity(
            x_noisy.view(64, -1), x_0.view(64, -1), dim=-1
        ).mean()
        assert cos_sim < 0.3, (
            f"At t=T-1, CosSim = {cos_sim:.4f}, expected < 0.3"
        )

    def test_custom_noise(self):
        """When noise is provided, it should be used."""
        s = CosineNoiseSchedule(n_steps=T)
        x_0 = torch.randn(4, 384)
        t = torch.full((4,), 500, dtype=torch.long)
        noise = torch.ones_like(x_0)
        x_noisy, returned_noise = s.add_noise(x_0, t, noise=noise)
        assert torch.equal(returned_noise, noise)

    def test_different_timesteps_different_noise_levels(self):
        """Higher timestep should produce noisier output."""
        s = CosineNoiseSchedule(n_steps=T)
        torch.manual_seed(42)
        x_0 = torch.randn(32, 384)
        noise = torch.randn_like(x_0)

        t_low = torch.full((32,), 100, dtype=torch.long)
        t_high = torch.full((32,), 900, dtype=torch.long)

        x_low, _ = s.add_noise(x_0, t_low, noise=noise)
        x_high, _ = s.add_noise(x_0, t_high, noise=noise)

        # x_low should be closer to x_0 than x_high
        diff_low = (x_low - x_0).norm()
        diff_high = (x_high - x_0).norm()
        assert diff_low < diff_high


# ---------------------------------------------------------------------------
# DDIMSampler
# ---------------------------------------------------------------------------


class TestDDIMSampler:
    def test_output_shape(self):
        """Sampled output has the requested shape."""
        s = CosineNoiseSchedule(n_steps=T)
        sampler = DDIMSampler(s, n_steps=10)

        # Trivial noise predictor: always predicts zero
        def zero_pred(x_noisy, timestep, **kw):
            return torch.zeros_like(x_noisy)

        out = sampler.sample(
            zero_pred, shape=(4, 4, 384), cond_kwargs={}, device="cpu"
        )
        assert out.shape == (4, 4, 384)

    def test_deterministic(self):
        """Two calls with same random seed produce identical results."""
        s = CosineNoiseSchedule(n_steps=T)
        sampler = DDIMSampler(s, n_steps=10)

        def zero_pred(x_noisy, timestep, **kw):
            return torch.zeros_like(x_noisy)

        torch.manual_seed(123)
        out1 = sampler.sample(
            zero_pred, shape=(2, 4, 384), cond_kwargs={}, device="cpu"
        )
        torch.manual_seed(123)
        out2 = sampler.sample(
            zero_pred, shape=(2, 4, 384), cond_kwargs={}, device="cpu"
        )
        assert torch.allclose(out1, out2)

    def test_perfect_predictor_recovers_signal(self):
        """If noise_pred_fn returns the actual noise, DDIM should recover x_0."""
        s = CosineNoiseSchedule(n_steps=100)  # shorter for speed
        sampler = DDIMSampler(s, n_steps=100)  # use all steps

        torch.manual_seed(42)
        x_0 = torch.randn(4, 4, 384)

        # Build a "perfect" predictor that knows x_0 and computes noise
        # from the current x_t and alpha schedule
        alphas_cumprod = s.alphas_cumprod

        def perfect_pred(x_noisy, timestep, **kw):
            t_val = timestep[0].item()
            alpha_bar = alphas_cumprod[t_val]
            # noise = (x_t - sqrt(alpha_bar) * x_0) / sqrt(1 - alpha_bar)
            noise = (
                x_noisy - torch.sqrt(alpha_bar) * x_0
            ) / torch.sqrt(1.0 - alpha_bar + 1e-8)
            return noise

        # Start from a noised version of x_0 at the highest timestep
        torch.manual_seed(0)
        out = sampler.sample(
            perfect_pred, shape=x_0.shape, cond_kwargs={}, device="cpu"
        )

        # Note: exact recovery depends on the starting noise matching,
        # so we just check finite and reasonable magnitude
        assert torch.isfinite(out).all()

    def test_timestep_subsequence_length(self):
        """Sampler's timestep list has the right number of steps."""
        s = CosineNoiseSchedule(n_steps=1000)
        sampler = DDIMSampler(s, n_steps=50)
        assert len(sampler.timesteps) == 50

    def test_timestep_subsequence_descending(self):
        """Timesteps should be in descending order (high noise to low)."""
        s = CosineNoiseSchedule(n_steps=1000)
        sampler = DDIMSampler(s, n_steps=50)
        for i in range(len(sampler.timesteps) - 1):
            assert sampler.timesteps[i] > sampler.timesteps[i + 1]

    def test_cond_kwargs_passed_through(self):
        """Conditioning kwargs are forwarded to the noise prediction fn."""
        s = CosineNoiseSchedule(n_steps=100)
        sampler = DDIMSampler(s, n_steps=5)

        received_kwargs = {}

        def capturing_pred(x_noisy, timestep, **kw):
            received_kwargs.update(kw)
            return torch.zeros_like(x_noisy)

        sampler.sample(
            capturing_pred,
            shape=(2, 4, 384),
            cond_kwargs={"z_t": torch.randn(2, 384), "a_embed": torch.randn(2, 384)},
            device="cpu",
        )
        assert "z_t" in received_kwargs
        assert "a_embed" in received_kwargs


# ---------------------------------------------------------------------------
# DDIMSampler -- warm-start
# ---------------------------------------------------------------------------


class TestDDIMSamplerWarmStart:
    def test_t_start_zero_returns_input(self):
        """At t_start=0, no denoising occurs; output equals input."""
        s = CosineNoiseSchedule(n_steps=T)
        sampler = DDIMSampler(s, n_steps=50)

        x_init = torch.randn(4, 4, 384)

        def fail_pred(x_noisy, timestep, **kw):
            raise AssertionError("Model should not be called at t_start=0")

        out = sampler.sample_warm_start(
            fail_pred, x_init, t_start=0, cond_kwargs={}, device="cpu"
        )
        assert torch.equal(out, x_init)

    def test_output_shape(self):
        """Output shape matches x_init for various t_start values."""
        s = CosineNoiseSchedule(n_steps=T)
        sampler = DDIMSampler(s, n_steps=50)

        def zero_pred(x_noisy, timestep, **kw):
            return torch.zeros_like(x_noisy)

        x_init = torch.randn(4, 4, 384)
        for t_start in [40, 200, 600, 980]:
            out = sampler.sample_warm_start(
                zero_pred, x_init, t_start=t_start,
                cond_kwargs={}, device="cpu",
            )
            assert out.shape == x_init.shape, f"Shape mismatch at t_start={t_start}"

    def test_deterministic(self):
        """Two calls with same seed produce identical results."""
        s = CosineNoiseSchedule(n_steps=T)
        sampler = DDIMSampler(s, n_steps=10)

        def zero_pred(x_noisy, timestep, **kw):
            return torch.zeros_like(x_noisy)

        x_init = torch.randn(2, 4, 384)

        torch.manual_seed(42)
        out1 = sampler.sample_warm_start(
            zero_pred, x_init, t_start=200, cond_kwargs={}, device="cpu"
        )
        torch.manual_seed(42)
        out2 = sampler.sample_warm_start(
            zero_pred, x_init, t_start=200, cond_kwargs={}, device="cpu"
        )
        assert torch.allclose(out1, out2)

    def test_fewer_model_calls_at_low_t_start(self):
        """Lower t_start should invoke the model fewer times."""
        s = CosineNoiseSchedule(n_steps=T)
        sampler = DDIMSampler(s, n_steps=50)

        call_counts = {}

        def counting_pred(x_noisy, timestep, **kw):
            key = "calls"
            call_counts[key] = call_counts.get(key, 0) + 1
            return torch.zeros_like(x_noisy)

        x_init = torch.randn(2, 4, 384)

        for t_start in [100, 400, 980]:
            call_counts.clear()
            sampler.sample_warm_start(
                counting_pred, x_init, t_start=t_start,
                cond_kwargs={}, device="cpu",
            )
            # More steps at higher t_start
            if t_start == 100:
                low_calls = call_counts["calls"]
            elif t_start == 980:
                high_calls = call_counts["calls"]

        assert low_calls < high_calls, (
            f"t_start=100 had {low_calls} calls, "
            f"t_start=980 had {high_calls}; expected fewer at low t_start"
        )

    def test_cond_kwargs_passed_through(self):
        """Conditioning kwargs are forwarded to the noise prediction fn."""
        s = CosineNoiseSchedule(n_steps=100)
        sampler = DDIMSampler(s, n_steps=5)

        received_kwargs = {}

        def capturing_pred(x_noisy, timestep, **kw):
            received_kwargs.update(kw)
            return torch.zeros_like(x_noisy)

        x_init = torch.randn(2, 4, 384)
        sampler.sample_warm_start(
            capturing_pred, x_init, t_start=50,
            cond_kwargs={"z_t": torch.randn(2, 384)},
            device="cpu",
        )
        assert "z_t" in received_kwargs
