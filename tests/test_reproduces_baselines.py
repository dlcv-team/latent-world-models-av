"""Pinned reference numbers from M1's pilot run.

The substantive numerical reproduction tests (training a probe on cached
embeddings and asserting RMSE within ±1e-4) live in module-level test
files that will land alongside their modules. This file enforces that the
pinned reference fixture itself is well-formed and internally consistent
so those module tests have a stable contract to check against.

If a module's actual numbers drift outside the pinned tolerance, the
correct response is to investigate before updating the fixture.
"""

from __future__ import annotations

import math

import pytest


REQUIRED_TOP = {"version", "subset", "encoders", "paired_tests", "p1", "tolerance"}


def test_pilot_baselines_well_formed(pilot_baselines):
    missing = REQUIRED_TOP - set(pilot_baselines)
    assert not missing, f"pilot_baselines.json missing: {missing!r}"


def test_pilot_subset_matches_canonical(cfg, pilot_baselines):
    pilot_subset = pilot_baselines["subset"]
    assert pilot_subset["seed"] == cfg.global_seed
    assert pilot_subset["counts"] == cfg.expected_split_counts


def test_pilot_encoder_set_matches_canonical(cfg, pilot_baselines):
    pinned = set(pilot_baselines["encoders"])
    declared = set(cfg.raw["encoders"])
    assert pinned == declared, (
        "pilot_baselines.json must reference the same encoder set as "
        "configs/canonical.yaml; got pinned={pinned}, declared={declared}"
        .format(pinned=sorted(pinned), declared=sorted(declared))
    )


def test_pilot_rmse_values_are_plausible(pilot_baselines):
    """Quick sanity range: scene-mean steer RMSE for any encoder must lie in
    a reasonable interval. Catches accidental zeroing or unit confusion.
    """
    for name, row in pilot_baselines["encoders"].items():
        rmse = row["steer_rmse_scene_mean"]["expected"]
        assert 0.0 < rmse < 1.0, f"{name} steer RMSE {rmse} outside plausible range"
        ci = row["steer_rmse_scene_mean"]["ci_95"]
        assert ci[0] <= rmse <= ci[1], f"{name} expected RMSE {rmse} outside CI {ci}"


def test_pilot_paired_test_pvalues_are_probabilities(pilot_baselines):
    for pair in pilot_baselines["paired_tests"]:
        p = pair["p_bonferroni"]["expected"]
        assert 0.0 <= p <= 1.0, f"p_bonferroni for {pair} not a probability"


def test_pilot_tolerance_envelopes_are_positive(pilot_baselines):
    tol = pilot_baselines["tolerance"]
    assert tol["rmse_abs_atol"] > 0
    assert tol["pvalue_rel_rtol"] > 0


def test_pilot_expected_best_encoder_is_declared(cfg, pilot_baselines):
    best = pilot_baselines["expected_best_encoder"]
    assert best in cfg.raw["encoders"], (
        f"expected_best_encoder {best!r} not declared in canonical.yaml"
    )


@pytest.mark.skipif(
    not pytest.importorskip("scipy", reason="scipy not installed"),
    reason="scipy not installed",
)
def test_pilot_p1_delta_cossim_within_pinned_envelope(pilot_baselines):
    """P1 deltas should remain small (~|delta| < 0.01); a regression that
    flips this would invalidate the negative-result framing.
    """
    for horizon, expected in pilot_baselines["p1"]["delta_cossim_per_horizon"].items():
        assert math.isfinite(expected), f"P1 horizon {horizon} delta is non-finite"
        assert abs(expected) < 0.01, (
            f"P1 horizon {horizon} expected delta {expected} is unexpectedly "
            "large; either the fixture is wrong or P1 substance has changed."
        )
