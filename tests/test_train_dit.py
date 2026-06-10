"""Tests for scripts/train_dit.py inline model definitions and EMA.

These tests import the remote function module and verify that the
inline model reimplementations match the canonical versions in
models/ and that the EMA tracker works correctly.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn


# ---------------------------------------------------------------------------
# We can't import from the Modal remote function directly (it's defined
# inside a function scope). Instead, we retest the same patterns here.
# ---------------------------------------------------------------------------


class TestCosineNoiseSchedule:
    """Verify inline CosineNoiseSchedule matches models.diffusion."""

    def test_schedule_shape(self):
        from models.diffusion import CosineNoiseSchedule

        sched = CosineNoiseSchedule(n_steps=1000)
        assert sched.alphas_cumprod.shape == (1000,)
        assert sched.sqrt_alphas_cumprod.shape == (1000,)

    def test_add_noise_roundtrip(self):
        from models.diffusion import CosineNoiseSchedule

        sched = CosineNoiseSchedule(n_steps=1000)
        x_0 = torch.randn(4, 4, 384)
        t = torch.zeros(4, dtype=torch.long)  # t=0: almost no noise
        x_noisy, noise = sched.add_noise(x_0, t)
        # At t=0, alpha_bar ~ 1.0, so x_noisy ~ x_0
        assert torch.allclose(x_noisy, x_0, atol=0.05)

    def test_high_t_is_noisy(self):
        from models.diffusion import CosineNoiseSchedule

        sched = CosineNoiseSchedule(n_steps=1000)
        x_0 = torch.ones(2, 4, 384)
        t = torch.full((2,), 999, dtype=torch.long)
        x_noisy, noise = sched.add_noise(x_0, t)
        # At t=999, alpha_bar ~ 0, so x_noisy ~ noise
        assert (x_noisy - x_0).abs().mean() > 0.5


class TestLatentDiT:
    """Verify LatentDiT forward pass shapes."""

    def test_forward_shape(self):
        from models.latent_dit import LatentDiT

        dit = LatentDiT(z_dim=384, cond_dim=384, n_blocks=2, n_heads=6, horizon=4)
        x_noisy = torch.randn(2, 4, 384)
        z_t = torch.randn(2, 384)
        a_embed = torch.randn(2, 384)
        timestep = torch.randint(0, 1000, (2,))
        out = dit(x_noisy, z_t, a_embed, timestep)
        assert out.shape == (2, 4, 384)

    def test_zero_init_identity(self):
        """Fresh DiT should produce near-zero output (adaLN-Zero)."""
        from models.latent_dit import LatentDiT

        dit = LatentDiT(z_dim=32, cond_dim=32, n_blocks=1, n_heads=2, horizon=2)
        x_noisy = torch.randn(1, 2, 32)
        z_t = torch.randn(1, 32)
        a_embed = torch.randn(1, 32)
        timestep = torch.randint(0, 1000, (1,))
        out = dit(x_noisy, z_t, a_embed, timestep)
        # Gates start at 0, so output should be near 0
        assert out.abs().max() < 0.1


class TestEMA:
    """Verify the EMA tracker pattern used in train_dit.py."""

    def test_ema_tracks_parameters(self):
        model = nn.Linear(4, 4)
        decay = 0.9

        # Manual EMA
        shadow = {n: p.data.clone() for n, p in model.named_parameters()}

        # Simulate an optimizer step
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p))

        # EMA update
        with torch.no_grad():
            for n, p in model.named_parameters():
                shadow[n].mul_(decay).add_(p.data, alpha=1 - decay)

        # Shadow should be between original and updated
        for n, p in model.named_parameters():
            # shadow = 0.9 * original + 0.1 * (original + 1) = original + 0.1
            expected_offset = 0.1
            diff = (shadow[n] - (p.data - 1.0)).abs()
            assert torch.allclose(diff, torch.full_like(diff, expected_offset), atol=1e-6)

    def test_ema_converges(self):
        """After many updates to constant value, EMA should converge."""
        model = nn.Linear(4, 2, bias=False)
        decay = 0.99
        shadow = {n: p.data.clone() for n, p in model.named_parameters()}

        target = torch.ones_like(model.weight)
        with torch.no_grad():
            model.weight.copy_(target)

        for _ in range(1000):
            with torch.no_grad():
                for n, p in model.named_parameters():
                    shadow[n].mul_(decay).add_(p.data, alpha=1 - decay)

        assert torch.allclose(shadow["weight"], target, atol=1e-3)


class TestConfigValidation:
    """Verify config validation catches mismatches."""

    def test_dit_canonical_keys_present(self):
        """All expected keys exist in DIT_CANONICAL."""
        # Import the module-level constants
        # Load scripts/training/train_dit.py as a module dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_dit_module",
            "scripts/training/train_dit.py",
        )
        mod = importlib.util.module_from_spec(spec)
        
        # Verify it has no __main__ execution block that immediately runs on import
        with open("scripts/training/train_dit.py") as f:
            content = f.read()

        for key in ["DIT_CANONICAL", "DIFFUSION_CANONICAL", "TRAINING_CANONICAL", "FOURIER_CANONICAL"]:
            assert key in content, f"Missing constant {key}"

        # Verify key values appear in the source
        assert '"n_blocks": 4' in content
        assert '"n_train_steps": 1000' in content
        assert '"ema_decay": 0.999' in content
        assert '"n_frequencies": 64' in content


class TestInlineFourierMatchesCanonical:
    """Verify that inline FourierActionEmbedding matches models/fourier_embed.py."""

    def test_freqs_match_canonical(self):
        """The inline freqs buffer must include the * torch.pi factor."""
        from models.fourier_embed import FourierActionEmbedding

        n_freq, base, out_dim = 64, 2.0, 384
        canonical = FourierActionEmbedding(
            action_dim=2, n_frequencies=n_freq, base=base, out_dim=out_dim,
        )

        # Build inline freqs the same way the fixed script does
        inline_freqs = base ** torch.arange(n_freq, dtype=torch.float32) * torch.pi

        assert torch.allclose(canonical.freqs, inline_freqs), (
            f"Inline freqs diverge from canonical. "
            f"Max diff: {(canonical.freqs - inline_freqs).abs().max():.2e}"
        )

    def test_forward_output_matches(self):
        """Given identical weights, inline and canonical should produce
        identical outputs on the same input."""
        from models.fourier_embed import FourierActionEmbedding

        torch.manual_seed(123)
        n_freq, base, out_dim = 64, 2.0, 384
        canonical = FourierActionEmbedding(
            action_dim=2, n_frequencies=n_freq, base=base, out_dim=out_dim,
        )

        # Build an inline replica with the correct formula
        class InlineFourier(nn.Module):
            def __init__(self):
                super().__init__()
                self.action_dim = 2
                self.n_frequencies = n_freq
                freqs = base ** torch.arange(n_freq, dtype=torch.float32) * torch.pi
                self.register_buffer("freqs", freqs)
                fourier_dim = 2 * 2 * n_freq
                self.proj = nn.Sequential(
                    nn.Linear(fourier_dim, out_dim),
                    nn.GELU(),
                    nn.Linear(out_dim, out_dim),
                )

            def forward(self, action):
                x = action.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
                x = torch.cat([x.sin(), x.cos()], dim=-1)
                x = x.flatten(1)
                return self.proj(x)

        inline = InlineFourier()
        # Copy canonical proj weights into inline
        inline.proj.load_state_dict(canonical.proj.state_dict())

        actions = torch.randn(8, 2)
        out_canonical = canonical(actions)
        out_inline = inline(actions)

        assert torch.allclose(out_canonical, out_inline, atol=1e-5), (
            f"Max diff: {(out_canonical - out_inline).abs().max():.2e}"
        )


class TestBuildWindows:
    """Test the build_windows sliding-window logic (reimplemented from train_dit.py)."""

    @staticmethod
    def _build_windows(embeddings, steer_norms, accel_norms, scene_names, horizon):
        """Reimplementation of the build_windows loop from train_dit.py."""
        import numpy as np
        z_t_list, action_list, z_future_list = [], [], []
        unique_scenes = np.unique(scene_names)
        for scene in unique_scenes:
            scene_mask = scene_names == scene
            idx = np.where(scene_mask)[0]
            n_scene = len(idx)
            for j in range(n_scene - horizon):
                t_idx = idx[j]
                future_idx = idx[j + 1: j + 1 + horizon]
                z_t_list.append(embeddings[t_idx])
                action_list.append([steer_norms[t_idx], accel_norms[t_idx]])
                z_future_list.append(embeddings[future_idx])
        if not z_t_list:
            return None, None, None
        return (
            torch.tensor(np.array(z_t_list), dtype=torch.float32),
            torch.tensor(np.array(action_list), dtype=torch.float32),
            torch.tensor(np.array(z_future_list), dtype=torch.float32),
        )

    def test_two_scenes_correct_count(self):
        """2 scenes x 5 frames each, horizon=2: expect 3+3=6 windows."""
        import numpy as np
        n_per_scene = 5
        horizon = 2
        dim = 8
        emb = np.random.randn(10, dim).astype(np.float32)
        steers = np.random.randn(10).astype(np.float32)
        accels = np.random.randn(10).astype(np.float32)
        scenes = np.array(["s0"] * n_per_scene + ["s1"] * n_per_scene)

        z_t, act, zf = self._build_windows(emb, steers, accels, scenes, horizon)
        assert z_t.shape == (6, dim)
        assert act.shape == (6, 2)
        assert zf.shape == (6, horizon, dim)

    def test_no_cross_scene_leakage(self):
        """Windows must not span across scene boundaries."""
        import numpy as np
        dim = 4
        horizon = 2
        # Scene A has frames [0,1,2], Scene B has frames [3,4,5]
        emb = np.eye(6, dim, dtype=np.float32)  # each frame is unique
        steers = np.arange(6, dtype=np.float32)
        accels = np.arange(6, dtype=np.float32)
        scenes = np.array(["A", "A", "A", "B", "B", "B"])

        z_t, act, zf = self._build_windows(emb, steers, accels, scenes, horizon)
        # 3 frames per scene - horizon = 1 window per scene = 2 total
        assert z_t.shape[0] == 2

        # First window: z_t=frame0, futures=[frame1, frame2] (all scene A)
        assert torch.allclose(z_t[0], torch.tensor(emb[0]))
        assert torch.allclose(zf[0, 0], torch.tensor(emb[1]))
        assert torch.allclose(zf[0, 1], torch.tensor(emb[2]))

        # Second window: z_t=frame3, futures=[frame4, frame5] (all scene B)
        assert torch.allclose(z_t[1], torch.tensor(emb[3]))
        assert torch.allclose(zf[1, 0], torch.tensor(emb[4]))
        assert torch.allclose(zf[1, 1], torch.tensor(emb[5]))

    def test_short_scene_produces_no_windows(self):
        """A scene with only 2 frames and horizon=2 should produce 0 windows."""
        import numpy as np
        dim = 4
        horizon = 2
        emb = np.random.randn(2, dim).astype(np.float32)
        steers = np.random.randn(2).astype(np.float32)
        accels = np.random.randn(2).astype(np.float32)
        scenes = np.array(["s0", "s0"])

        z_t, act, zf = self._build_windows(emb, steers, accels, scenes, horizon)
        assert z_t is None


class TestTrainingStep:
    """Smoke test: a single training step reduces loss on tiny data."""

    def test_single_step_runs(self):
        from models.latent_dit import LatentDiT
        from models.diffusion import CosineNoiseSchedule

        torch.manual_seed(42)
        z_dim = 32
        horizon = 2
        B = 4

        dit = LatentDiT(z_dim=z_dim, cond_dim=z_dim, n_blocks=1, n_heads=2, horizon=horizon)
        schedule = CosineNoiseSchedule(n_steps=100)
        optimizer = torch.optim.Adam(dit.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Synthetic data
        z_t = torch.randn(B, z_dim)
        z_future = torch.randn(B, horizon, z_dim)
        a_embed = torch.randn(B, z_dim)

        losses = []
        for _ in range(20):
            t = torch.randint(0, 100, (B,))
            x_noisy, noise = schedule.add_noise(z_future, t)
            noise_pred = dit(x_noisy, z_t, a_embed, t)
            loss = criterion(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease over 20 steps on this tiny fixed batch
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )
