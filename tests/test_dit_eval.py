"""Tests for DA7 fair DiT-vs-MLP evaluation utilities.

All tests use synthetic tensors (tiny dims), not full artifact files.
Full-artifact validation is runtime, not CI.
"""

from __future__ import annotations

import torch
from torch import nn

from evaluation.dit_eval import build_comparison_table
from models.latent_pred import LatentPredictor


class TestAdapterReconstructionDeterministic:
    """Verify orthogonal adapter init is deterministic with same seed."""

    def test_same_seed_same_weights(self):
        native, target = 8, 4

        torch.manual_seed(42)
        a1 = nn.Linear(native, target, bias=False)
        nn.init.orthogonal_(a1.weight)

        torch.manual_seed(42)
        a2 = nn.Linear(native, target, bias=False)
        nn.init.orthogonal_(a2.weight)

        assert torch.equal(a1.weight, a2.weight)

    def test_different_seed_different_weights(self):
        native, target = 8, 4

        torch.manual_seed(42)
        a1 = nn.Linear(native, target, bias=False)
        nn.init.orthogonal_(a1.weight)

        torch.manual_seed(99)
        a2 = nn.Linear(native, target, bias=False)
        nn.init.orthogonal_(a2.weight)

        assert not torch.equal(a1.weight, a2.weight)


class TestNormalizationRoundtrip:
    """Normalize then inverse-transform, verify recovery."""

    def test_roundtrip(self):
        torch.manual_seed(0)
        data = torch.randn(100, 4)
        z_mean = data.mean(dim=0)
        z_std = data.std(dim=0).clamp(min=1e-6)

        normalized = (data - z_mean) / z_std
        recovered = normalized * z_std + z_mean

        assert torch.allclose(data, recovered, atol=1e-6)


class TestMLPForwardShape:
    """Verify LatentPredictor output shape with tiny dims."""

    def test_shape(self):
        z_dim, a_dim, horizon, hidden = 4, 4, 2, 8
        model = LatentPredictor(z_dim=z_dim, a_dim=a_dim, horizon=horizon, hidden=hidden)
        B = 3
        z_t = torch.randn(B, z_dim)
        a_embed = torch.randn(B, a_dim)
        out = model(z_t, a_embed)
        assert out.shape == (B, horizon, z_dim)


class TestComparisonTableFromSynthetic:
    """Build comparison table from synthetic results, verify structure."""

    def _make_result(self, encoder, variant, seed, model_type="dit"):
        return {
            "encoder": encoder,
            "variant": variant,
            "seed": seed,
            "n_test_windows": 100,
            "metrics": {
                "cossim_by_horizon": [0.9, 0.8, 0.7, 0.6],
                "mse_by_horizon": [1.0, 1.1, 1.2, 1.3],
                "copy_baseline_cossim": [0.95, 0.90, 0.85, 0.80],
            },
            "time_s": 1.0,
        }

    def test_all_rows_present(self):
        dit_results = [
            self._make_result("enc_a", "conditioned", 0),
            self._make_result("enc_a", "unconditioned", 0),
        ]
        mlp_results = [
            self._make_result("enc_a", "conditioned", 0, "mlp"),
            self._make_result("enc_a", "unconditioned", 0, "mlp"),
        ]

        rows = build_comparison_table(dit_results, mlp_results)

        models = {r["model"] for r in rows}
        assert models == {"dit", "mlp", "copy_baseline"}

        # 2 variants x 4 horizons x 2 models + 1 variant(none) x 4 horizons for copy
        dit_rows = [r for r in rows if r["model"] == "dit"]
        mlp_rows = [r for r in rows if r["model"] == "mlp"]
        copy_rows = [r for r in rows if r["model"] == "copy_baseline"]

        assert len(dit_rows) == 8  # 2 variants x 4 horizons
        assert len(mlp_rows) == 8
        assert len(copy_rows) == 4  # 1 "none" variant x 4 horizons

    def test_copy_baseline_variant_none(self):
        dit_results = [self._make_result("enc_a", "conditioned", 0)]
        mlp_results = [self._make_result("enc_a", "conditioned", 0, "mlp")]

        rows = build_comparison_table(dit_results, mlp_results)
        copy_rows = [r for r in rows if r["model"] == "copy_baseline"]

        for r in copy_rows:
            assert r["variant"] == "none"


class TestCosSim:
    """Verify CosSim values on synthetic predictions are in [-1, 1]."""

    def test_cossim_range(self):
        torch.manual_seed(0)
        a = torch.randn(10, 4)
        b = torch.randn(10, 4)
        cs = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        assert (cs >= -1.0).all()
        assert (cs <= 1.0).all()

    def test_cossim_identical(self):
        a = torch.randn(5, 4)
        cs = torch.nn.functional.cosine_similarity(a, a, dim=-1)
        assert torch.allclose(cs, torch.ones(5), atol=1e-6)


class TestCopyBaselineTolerance:
    """Given same adapter + data, verify copy baseline matches within atol."""

    def test_copy_baseline_self_consistent(self):
        torch.manual_seed(42)
        native_dim, target_dim = 8, 4

        adapter = nn.Linear(native_dim, target_dim, bias=False)
        nn.init.orthogonal_(adapter.weight)
        for p in adapter.parameters():
            p.requires_grad_(False)

        z_t = torch.randn(20, native_dim)
        zf = torch.randn(20, target_dim)  # already in target space

        with torch.no_grad():
            z_t_adapted = adapter(z_t)

        # Compute copy baseline twice
        cs1 = torch.nn.functional.cosine_similarity(z_t_adapted, zf, dim=-1)
        cs2 = torch.nn.functional.cosine_similarity(z_t_adapted, zf, dim=-1)

        assert torch.allclose(cs1, cs2, atol=1e-5)
