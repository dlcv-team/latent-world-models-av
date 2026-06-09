"""Unit tests for :mod:`evaluation.latent_eval` (C4).

Covers:

* Mathematical correctness against ``F.cosine_similarity`` on random tensors.
* Edge cases: identical tensors (CosSim=1), anti-aligned (-1), orthogonal (0),
  variable horizon, single-sample input, single-horizon, ``float16`` inputs.
* Shape / dtype validation errors.
* JSON + CSV artifact schema, formatting, and round-trip equality.
* ``DeltaCosSim`` composition and horizon-mismatch errors.
* End-to-end ``run_latent_eval`` pipeline.
* CLI smoke test via ``main([...])``.

These tests intentionally do **not** depend on a trained latent predictor;
the export pipeline is exercised in ``tests/test_export_z_hat.py`` and the
loader cascade in ``tests/test_z_hat_loader.py``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from evaluation.latent_eval import (
    COSSIM_CSV_FILENAME,
    COSSIM_JSON_FILENAME,
    CSV_COLUMNS,
    CSV_COLUMNS_PERTURBED,
    compute_delta_cossim,
    compute_perturbation_delta_cossim,
    evaluate_cossim,
    export_cossim_results,
    main,
    run_latent_eval,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Make every test deterministic without leaking RNG state across tests."""
    torch.manual_seed(20260524)
    yield


def _save(t: torch.Tensor, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t, path)
    return path


def _reference_per_horizon_cossim(
    z_hat: torch.Tensor, z_real: torch.Tensor
) -> dict[int, float]:
    """Independent reference implementation of the formula in the task spec."""
    out: dict[int, float] = {}
    for k in range(1, z_hat.shape[1] + 1):
        sim = F.cosine_similarity(
            z_hat[:, k - 1, :].float(), z_real[:, k - 1, :].float(), dim=-1
        ).mean()
        out[k] = float(sim.item())
    return out


# ---------------------------------------------------------------------------
# Core math: identity, anti-alignment, orthogonality
# ---------------------------------------------------------------------------


def test_evaluate_cossim_identical_tensors_is_one(tmp_path):
    """CosSim(x, x) = 1 for every horizon when z_hat == z_real."""
    n, horizon, z_dim = 16, 4, 384
    z = torch.randn(n, horizon, z_dim)
    p_hat = _save(z, tmp_path / "z_hat.pt")
    p_real = _save(z, tmp_path / "z_real.pt")

    result = evaluate_cossim(p_hat, p_real)
    assert sorted(result) == [1, 2, 3, 4]
    for k, v in result.items():
        assert v == pytest.approx(1.0, abs=1e-6), f"k={k} expected 1.0, got {v}"


def test_evaluate_cossim_anti_aligned_is_negative_one(tmp_path):
    """CosSim(x, -x) = -1 for every horizon (modulo float epsilon)."""
    n, horizon, z_dim = 8, 4, 384
    z = torch.randn(n, horizon, z_dim)
    p_hat = _save(z, tmp_path / "z_hat.pt")
    p_real = _save(-z, tmp_path / "z_real.pt")

    result = evaluate_cossim(p_hat, p_real)
    for k, v in result.items():
        assert v == pytest.approx(-1.0, abs=1e-6), f"k={k} expected -1.0, got {v}"


def test_evaluate_cossim_orthogonal_is_zero(tmp_path):
    """Hand-built orthogonal vectors should give CosSim=0 exactly."""
    # Pick z_dim large enough to embed both basis vectors comfortably.
    n, horizon, z_dim = 4, 4, 8
    z_hat = torch.zeros(n, horizon, z_dim)
    z_real = torch.zeros(n, horizon, z_dim)
    z_hat[..., 0] = 1.0  # along e_0
    z_real[..., 1] = 1.0  # along e_1 (orthogonal to e_0)

    p_hat = _save(z_hat, tmp_path / "z_hat.pt")
    p_real = _save(z_real, tmp_path / "z_real.pt")

    result = evaluate_cossim(p_hat, p_real)
    for v in result.values():
        assert v == pytest.approx(0.0, abs=1e-7)


def test_evaluate_cossim_matches_reference_implementation(tmp_path):
    """Random tensors: evaluate_cossim must match a direct F.cosine_similarity call."""
    n, horizon, z_dim = 64, 4, 384
    z_hat = torch.randn(n, horizon, z_dim)
    z_real = torch.randn(n, horizon, z_dim)

    p_hat = _save(z_hat, tmp_path / "z_hat.pt")
    p_real = _save(z_real, tmp_path / "z_real.pt")

    got = evaluate_cossim(p_hat, p_real)
    expected = _reference_per_horizon_cossim(z_hat, z_real)
    assert got.keys() == expected.keys()
    for k in expected:
        assert got[k] == pytest.approx(expected[k], abs=1e-6), (
            f"k={k}: got {got[k]}, expected {expected[k]}"
        )


def test_evaluate_cossim_per_horizon_independence(tmp_path):
    """Each horizon's CosSim must depend ONLY on that horizon's slice."""
    n, horizon, z_dim = 8, 4, 16
    # Construct so that horizon 1 is identical and horizon 2 is anti-aligned;
    # horizons 3 / 4 are independent random tensors.
    z_hat = torch.randn(n, horizon, z_dim)
    z_real = z_hat.clone()
    z_real[:, 1, :] = -z_hat[:, 1, :]  # k=2: anti-aligned
    z_real[:, 2, :] = torch.randn(n, z_dim)  # k=3: random
    z_real[:, 3, :] = torch.randn(n, z_dim)  # k=4: random

    p_hat = _save(z_hat, tmp_path / "z_hat.pt")
    p_real = _save(z_real, tmp_path / "z_real.pt")

    result = evaluate_cossim(p_hat, p_real)
    assert result[1] == pytest.approx(1.0, abs=1e-6)
    assert result[2] == pytest.approx(-1.0, abs=1e-6)
    # Horizons 3 and 4 are independent draws -- means should differ.
    assert result[3] != result[4]


def test_evaluate_cossim_horizon_one(tmp_path):
    """Single-horizon input still works (degenerate but legal)."""
    z = torch.randn(5, 1, 32)
    p_hat = _save(z, tmp_path / "z_hat.pt")
    p_real = _save(z, tmp_path / "z_real.pt")

    result = evaluate_cossim(p_hat, p_real)
    assert sorted(result) == [1]
    assert result[1] == pytest.approx(1.0, abs=1e-6)


def test_evaluate_cossim_horizon_eight(tmp_path):
    """Non-canonical horizon length works -- horizon is inferred from shape."""
    horizon = 8
    z = torch.randn(3, horizon, 16)
    p_hat = _save(z, tmp_path / "z_hat.pt")
    p_real = _save(z, tmp_path / "z_real.pt")

    result = evaluate_cossim(p_hat, p_real)
    assert sorted(result) == list(range(1, horizon + 1))


def test_evaluate_cossim_single_sample(tmp_path):
    """N=1 input is supported -- cosine_similarity reduces to a scalar per k."""
    z_hat = torch.randn(1, 4, 64)
    z_real = z_hat.clone()
    p_hat = _save(z_hat, tmp_path / "z_hat.pt")
    p_real = _save(z_real, tmp_path / "z_real.pt")
    result = evaluate_cossim(p_hat, p_real)
    for v in result.values():
        assert v == pytest.approx(1.0, abs=1e-6)


def test_evaluate_cossim_float16_input_does_not_overflow(tmp_path):
    """fp16 inputs must be upcast internally so the mean stays stable."""
    n, horizon, z_dim = 32, 4, 384
    z_hat = torch.randn(n, horizon, z_dim).half()
    z_real = z_hat.clone()
    p_hat = _save(z_hat, tmp_path / "z_hat.pt")
    p_real = _save(z_real, tmp_path / "z_real.pt")

    result = evaluate_cossim(p_hat, p_real)
    for v in result.values():
        # fp16 norm noise is larger than fp32, hence a looser tol.
        assert v == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_evaluate_cossim_shape_mismatch_raises(tmp_path):
    z_hat = torch.randn(10, 4, 384)
    z_real = torch.randn(10, 4, 256)  # different z_dim
    p_hat = _save(z_hat, tmp_path / "z_hat.pt")
    p_real = _save(z_real, tmp_path / "z_real.pt")
    with pytest.raises(ValueError, match="identical shapes"):
        evaluate_cossim(p_hat, p_real)


def test_evaluate_cossim_horizon_mismatch_raises(tmp_path):
    z_hat = torch.randn(10, 4, 384)
    z_real = torch.randn(10, 3, 384)  # different horizon
    p_hat = _save(z_hat, tmp_path / "z_hat.pt")
    p_real = _save(z_real, tmp_path / "z_real.pt")
    with pytest.raises(ValueError, match="identical shapes"):
        evaluate_cossim(p_hat, p_real)


def test_evaluate_cossim_n_mismatch_raises(tmp_path):
    z_hat = torch.randn(10, 4, 384)
    z_real = torch.randn(9, 4, 384)
    p_hat = _save(z_hat, tmp_path / "z_hat.pt")
    p_real = _save(z_real, tmp_path / "z_real.pt")
    with pytest.raises(ValueError, match="identical shapes"):
        evaluate_cossim(p_hat, p_real)


def test_evaluate_cossim_wrong_dim_raises(tmp_path):
    """2D tensors are rejected -- API explicitly requires (N, H, D)."""
    z = torch.randn(10, 384)
    p_hat = _save(z, tmp_path / "z_hat.pt")
    p_real = _save(z, tmp_path / "z_real.pt")
    with pytest.raises(ValueError, match="3D"):
        evaluate_cossim(p_hat, p_real)


def test_evaluate_cossim_empty_n_raises(tmp_path):
    z = torch.empty(0, 4, 384)
    p_hat = _save(z, tmp_path / "z_hat.pt")
    p_real = _save(z, tmp_path / "z_real.pt")
    with pytest.raises(ValueError, match="empty"):
        evaluate_cossim(p_hat, p_real)


def test_evaluate_cossim_zero_z_dim_raises(tmp_path):
    z = torch.empty(4, 4, 0)
    p_hat = _save(z, tmp_path / "z_hat.pt")
    p_real = _save(z, tmp_path / "z_real.pt")
    with pytest.raises(ValueError, match="zero embedding dimension"):
        evaluate_cossim(p_hat, p_real)


def test_evaluate_cossim_missing_file_raises(tmp_path):
    z = torch.randn(4, 4, 16)
    p_real = _save(z, tmp_path / "z_real.pt")
    with pytest.raises(FileNotFoundError, match="z_hat tensor not found"):
        evaluate_cossim(tmp_path / "does_not_exist.pt", p_real)


def test_evaluate_cossim_missing_real_raises(tmp_path):
    z = torch.randn(4, 4, 16)
    p_hat = _save(z, tmp_path / "z_hat.pt")
    with pytest.raises(FileNotFoundError, match="z_real tensor not found"):
        evaluate_cossim(p_hat, tmp_path / "does_not_exist.pt")


def test_evaluate_cossim_non_tensor_raises(tmp_path):
    p_hat = tmp_path / "z_hat.pt"
    torch.save({"not": "a tensor"}, p_hat)
    z = torch.randn(4, 4, 16)
    p_real = _save(z, tmp_path / "z_real.pt")
    with pytest.raises(ValueError, match="torch.Tensor"):
        evaluate_cossim(p_hat, p_real)


# ---------------------------------------------------------------------------
# DeltaCosSim
# ---------------------------------------------------------------------------


def test_compute_delta_cossim_simple():
    cond = {1: 0.5, 2: 0.4, 3: 0.3, 4: 0.2}
    uncond = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1}
    delta = compute_delta_cossim(cond, uncond)
    assert delta == {
        1: 0.4,
        2: pytest.approx(0.3),
        3: pytest.approx(0.2),
        4: pytest.approx(0.1),
    }


def test_compute_delta_cossim_negative_when_uncond_wins():
    """Delta is allowed to be negative -- regression check."""
    cond = {1: 0.1}
    uncond = {1: 0.4}
    assert compute_delta_cossim(cond, uncond) == {1: pytest.approx(-0.3)}


def test_compute_delta_cossim_horizon_mismatch_raises():
    with pytest.raises(ValueError, match="horizon mismatch"):
        compute_delta_cossim({1: 0.1, 2: 0.2}, {1: 0.05})


def test_compute_delta_cossim_perfect_conditioning_yields_one(tmp_path):
    """End-to-end: identical cond pair + orthogonal uncond pair -> Delta ≈ 1.0."""
    n, horizon, z_dim = 32, 4, 64

    # Conditioned: perfect prediction (CosSim = 1.0 everywhere).
    z = torch.randn(n, horizon, z_dim)
    _save(z, tmp_path / "z_hat_cond.pt")
    _save(z, tmp_path / "z_real_cond.pt")

    # Unconditional: orthogonal one-hot vectors (CosSim = 0.0 everywhere).
    z_hat_uncond = torch.zeros(n, horizon, z_dim)
    z_real_uncond = torch.zeros(n, horizon, z_dim)
    z_hat_uncond[..., 0] = 1.0
    z_real_uncond[..., 1] = 1.0
    _save(z_hat_uncond, tmp_path / "z_hat_uncond.pt")
    _save(z_real_uncond, tmp_path / "z_real_uncond.pt")

    cond = evaluate_cossim(tmp_path / "z_hat_cond.pt", tmp_path / "z_real_cond.pt")
    uncond = evaluate_cossim(
        tmp_path / "z_hat_uncond.pt", tmp_path / "z_real_uncond.pt"
    )
    delta = compute_delta_cossim(cond, uncond)
    for v in delta.values():
        assert v == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Export: JSON + CSV
# ---------------------------------------------------------------------------


def test_export_cossim_results_writes_both_files(tmp_path):
    cond = {1: 0.5, 2: 0.4}
    uncond = {1: 0.1, 2: 0.1}
    delta = compute_delta_cossim(cond, uncond)

    json_path, csv_path = export_cossim_results(cond, uncond, delta, tmp_path)
    assert json_path == tmp_path / COSSIM_JSON_FILENAME
    assert csv_path == tmp_path / COSSIM_CSV_FILENAME
    assert json_path.exists()
    assert csv_path.exists()


def test_export_cossim_json_schema(tmp_path):
    cond = {1: 0.5, 2: 0.4, 3: 0.3, 4: 0.2}
    uncond = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1}
    delta = compute_delta_cossim(cond, uncond)
    metadata = {"encoder": "vjepa2_rep64", "seed": 0}

    json_path, _ = export_cossim_results(
        cond, uncond, delta, tmp_path, metadata=metadata
    )
    payload = json.loads(json_path.read_text())

    assert payload["schema_version"] == "1.0"
    assert payload["horizon"] == 4
    assert set(payload["per_horizon"]) == {"1", "2", "3", "4"}
    for k_str in ("1", "2", "3", "4"):
        row = payload["per_horizon"][k_str]
        assert set(row) == {
            "cossim_conditioned",
            "cossim_unconditioned",
            "delta_cossim",
        }
        assert row["delta_cossim"] == pytest.approx(
            row["cossim_conditioned"] - row["cossim_unconditioned"], abs=1e-9
        )
    assert payload["mean_over_horizons"]["delta_cossim"] == pytest.approx(
        sum(delta.values()) / len(delta)
    )
    assert payload["metadata"] == metadata


def test_export_cossim_csv_schema_and_round_trip(tmp_path):
    cond = {1: 0.50, 2: 0.40, 3: 0.30, 4: 0.20}
    uncond = {1: 0.10, 2: 0.05, 3: 0.02, 4: 0.01}
    delta = compute_delta_cossim(cond, uncond)

    _, csv_path = export_cossim_results(cond, uncond, delta, tmp_path)

    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames is not None
        assert tuple(reader.fieldnames) == CSV_COLUMNS
        rows = list(reader)

    assert len(rows) == 4
    for row in rows:
        k = int(row["k"])
        assert float(row["cossim_conditioned"]) == pytest.approx(cond[k], abs=1e-9)
        assert float(row["cossim_unconditioned"]) == pytest.approx(uncond[k], abs=1e-9)
        assert float(row["delta_cossim"]) == pytest.approx(delta[k], abs=1e-9)


def test_export_cossim_creates_missing_output_dir(tmp_path):
    deep = tmp_path / "nested" / "dir" / "cossim"
    cond = {1: 0.1}
    uncond = {1: 0.05}
    delta = compute_delta_cossim(cond, uncond)

    json_path, csv_path = export_cossim_results(cond, uncond, delta, deep)
    assert json_path.exists() and csv_path.exists()
    assert deep.is_dir()


def test_export_cossim_no_temp_files_left_behind(tmp_path):
    """Atomic-ish writes must not leave .tmp siblings on disk."""
    cond = {1: 0.5}
    uncond = {1: 0.1}
    delta = compute_delta_cossim(cond, uncond)
    export_cossim_results(cond, uncond, delta, tmp_path)
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == [], f"unexpected tmp files: {leftover}"


def test_export_cossim_overwrites_existing_files(tmp_path):
    """Second export call replaces both artifacts cleanly."""
    cond1 = {1: 0.1}
    uncond1 = {1: 0.05}
    export_cossim_results(
        cond1, uncond1, compute_delta_cossim(cond1, uncond1), tmp_path
    )

    cond2 = {1: 0.9, 2: 0.8}
    uncond2 = {1: 0.1, 2: 0.05}
    json_path, csv_path = export_cossim_results(
        cond2, uncond2, compute_delta_cossim(cond2, uncond2), tmp_path
    )

    payload = json.loads(json_path.read_text())
    assert payload["horizon"] == 2

    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def _seed_four_pt_files(tmp_path: Path) -> dict[str, Path]:
    """Create the four canonical .pt files with known shapes."""
    n, horizon, z_dim = 16, 4, 384
    z_hat_cond = torch.randn(n, horizon, z_dim)
    z_real_cond = z_hat_cond + 0.01 * torch.randn_like(z_hat_cond)  # near 1.0
    z_hat_uncond = torch.randn(n, horizon, z_dim)
    z_real_uncond = torch.randn(n, horizon, z_dim)  # near 0.0

    paths = {
        "z_hat_cond": tmp_path / "z_hat_conditioned.pt",
        "z_real_cond": tmp_path / "z_real_conditioned.pt",
        "z_hat_uncond": tmp_path / "z_hat_unconditioned.pt",
        "z_real_uncond": tmp_path / "z_real_unconditioned.pt",
    }
    _save(z_hat_cond, paths["z_hat_cond"])
    _save(z_real_cond, paths["z_real_cond"])
    _save(z_hat_uncond, paths["z_hat_uncond"])
    _save(z_real_uncond, paths["z_real_uncond"])
    return paths


def test_run_latent_eval_end_to_end(tmp_path):
    paths = _seed_four_pt_files(tmp_path)
    out_dir = tmp_path / "out"

    payload = run_latent_eval(
        z_hat_conditioned_path=paths["z_hat_cond"],
        z_real_conditioned_path=paths["z_real_cond"],
        z_hat_unconditioned_path=paths["z_hat_uncond"],
        z_real_unconditioned_path=paths["z_real_uncond"],
        output_dir=out_dir,
        extra_metadata={"encoder": "synthetic"},
    )
    assert payload["horizon"] == 4
    assert payload["metadata"]["encoder"] == "synthetic"
    assert payload["metadata"]["n_samples"] == 16
    assert payload["metadata"]["z_dim"] == 384
    assert payload["metadata"]["horizon"] == 4
    assert set(payload["metadata"]["source_paths"]) == {
        "z_hat_conditioned",
        "z_real_conditioned",
        "z_hat_unconditioned",
        "z_real_unconditioned",
    }

    # Files were written and round-trip to the same numbers.
    json_path = out_dir / COSSIM_JSON_FILENAME
    csv_path = out_dir / COSSIM_CSV_FILENAME
    assert json_path.exists() and csv_path.exists()

    on_disk = json.loads(json_path.read_text())
    assert on_disk["horizon"] == payload["horizon"]
    for k_str, row in on_disk["per_horizon"].items():
        assert row == pytest.approx(payload["per_horizon"][k_str])


def test_run_latent_eval_signal_is_positive_when_cond_predicts_well(tmp_path):
    """Sanity: with the synthetic fixture the conditioned model should win.

    The fixture makes the conditioned z_hat nearly equal to z_real
    (~CosSim ≈ 1) and the unconditioned z_hat random (~CosSim ≈ 0),
    so Delta should be strongly positive on every horizon.
    """
    paths = _seed_four_pt_files(tmp_path)
    out_dir = tmp_path / "out"

    payload = run_latent_eval(
        z_hat_conditioned_path=paths["z_hat_cond"],
        z_real_conditioned_path=paths["z_real_cond"],
        z_hat_unconditioned_path=paths["z_hat_uncond"],
        z_real_unconditioned_path=paths["z_real_uncond"],
        output_dir=out_dir,
    )
    for k_str in ("1", "2", "3", "4"):
        row = payload["per_horizon"][k_str]
        assert row["cossim_conditioned"] > 0.95
        assert abs(row["cossim_unconditioned"]) < 0.15
        assert row["delta_cossim"] > 0.5


def test_run_latent_eval_pipeline_matches_direct_evaluation(tmp_path):
    """run_latent_eval must agree with composing evaluate_cossim + compute_delta_cossim."""
    paths = _seed_four_pt_files(tmp_path)
    out_dir = tmp_path / "out"

    payload = run_latent_eval(
        z_hat_conditioned_path=paths["z_hat_cond"],
        z_real_conditioned_path=paths["z_real_cond"],
        z_hat_unconditioned_path=paths["z_hat_uncond"],
        z_real_unconditioned_path=paths["z_real_uncond"],
        output_dir=out_dir,
    )

    cond = evaluate_cossim(paths["z_hat_cond"], paths["z_real_cond"])
    uncond = evaluate_cossim(paths["z_hat_uncond"], paths["z_real_uncond"])
    delta = compute_delta_cossim(cond, uncond)
    for k, expected in delta.items():
        row = payload["per_horizon"][str(k)]
        assert row["cossim_conditioned"] == pytest.approx(cond[k], abs=1e-9)
        assert row["cossim_unconditioned"] == pytest.approx(uncond[k], abs=1e-9)
        assert row["delta_cossim"] == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Cross-variant shape validation
# ---------------------------------------------------------------------------


def test_run_latent_eval_cross_variant_n_mismatch_raises(tmp_path):
    """Different N between conditioned and unconditioned tensors must raise."""
    n_cond, n_uncond, horizon, z_dim = 16, 12, 4, 384
    _save(torch.randn(n_cond, horizon, z_dim), tmp_path / "z_hat_cond.pt")
    _save(torch.randn(n_cond, horizon, z_dim), tmp_path / "z_real_cond.pt")
    _save(torch.randn(n_uncond, horizon, z_dim), tmp_path / "z_hat_uncond.pt")
    _save(torch.randn(n_uncond, horizon, z_dim), tmp_path / "z_real_uncond.pt")

    with pytest.raises(ValueError, match="Shape mismatch.*conditioned.*unconditioned"):
        run_latent_eval(
            z_hat_conditioned_path=tmp_path / "z_hat_cond.pt",
            z_real_conditioned_path=tmp_path / "z_real_cond.pt",
            z_hat_unconditioned_path=tmp_path / "z_hat_uncond.pt",
            z_real_unconditioned_path=tmp_path / "z_real_uncond.pt",
            output_dir=tmp_path / "out",
        )


def test_run_latent_eval_cross_variant_z_dim_mismatch_raises(tmp_path):
    """Different z_dim between conditioned and unconditioned tensors must raise."""
    n, horizon = 16, 4
    _save(torch.randn(n, horizon, 384), tmp_path / "z_hat_cond.pt")
    _save(torch.randn(n, horizon, 384), tmp_path / "z_real_cond.pt")
    _save(torch.randn(n, horizon, 256), tmp_path / "z_hat_uncond.pt")
    _save(torch.randn(n, horizon, 256), tmp_path / "z_real_uncond.pt")

    with pytest.raises(ValueError, match="Shape mismatch.*conditioned.*unconditioned"):
        run_latent_eval(
            z_hat_conditioned_path=tmp_path / "z_hat_cond.pt",
            z_real_conditioned_path=tmp_path / "z_real_cond.pt",
            z_hat_unconditioned_path=tmp_path / "z_hat_uncond.pt",
            z_real_unconditioned_path=tmp_path / "z_real_uncond.pt",
            output_dir=tmp_path / "out",
        )


# ---------------------------------------------------------------------------
# Payload key validation
# ---------------------------------------------------------------------------


def test_build_results_payload_key_mismatch_raises():
    """_build_results_payload must reject misaligned key sets."""
    from evaluation.latent_eval import _build_results_payload

    cond = {1: 0.5, 2: 0.4}
    uncond = {1: 0.1}
    delta = {1: 0.4, 2: 0.3}

    with pytest.raises(ValueError, match="Key mismatch"):
        _build_results_payload(cond, uncond, delta, None)


# ---------------------------------------------------------------------------
# CLI: encoder namespacing + seed metadata
# ---------------------------------------------------------------------------


def test_cli_encoder_namespaces_output_dir(tmp_path, capsys):
    """When --encoder is provided, output is written to a subdirectory."""
    paths = _seed_four_pt_files(tmp_path)
    out_dir = tmp_path / "cli_out"

    rc = main(
        [
            "--z-hat-conditioned",
            str(paths["z_hat_cond"]),
            "--z-real-conditioned",
            str(paths["z_real_cond"]),
            "--z-hat-unconditioned",
            str(paths["z_hat_uncond"]),
            "--z-real-unconditioned",
            str(paths["z_real_uncond"]),
            "--output-dir",
            str(out_dir),
            "--encoder",
            "vjepa2_rep64",
        ]
    )
    assert rc == 0
    assert (out_dir / "vjepa2_rep64" / COSSIM_JSON_FILENAME).exists()
    assert (out_dir / "vjepa2_rep64" / COSSIM_CSV_FILENAME).exists()
    assert not (out_dir / COSSIM_JSON_FILENAME).exists()


def test_cli_seed_recorded_in_metadata(tmp_path):
    """--seed value should appear in JSON metadata."""
    paths = _seed_four_pt_files(tmp_path)
    out_dir = tmp_path / "cli_out"

    main(
        [
            "--z-hat-conditioned",
            str(paths["z_hat_cond"]),
            "--z-real-conditioned",
            str(paths["z_real_cond"]),
            "--z-hat-unconditioned",
            str(paths["z_hat_uncond"]),
            "--z-real-unconditioned",
            str(paths["z_real_uncond"]),
            "--output-dir",
            str(out_dir),
            "--seed",
            "42",
        ]
    )
    payload = json.loads((out_dir / COSSIM_JSON_FILENAME).read_text())
    assert payload["metadata"]["seed"] == 42


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_main_writes_artifacts(tmp_path, capsys):
    paths = _seed_four_pt_files(tmp_path)
    out_dir = tmp_path / "cli_out"

    rc = main(
        [
            "--z-hat-conditioned",
            str(paths["z_hat_cond"]),
            "--z-real-conditioned",
            str(paths["z_real_cond"]),
            "--z-hat-unconditioned",
            str(paths["z_hat_uncond"]),
            "--z-real-unconditioned",
            str(paths["z_real_uncond"]),
            "--output-dir",
            str(out_dir),
            "--encoder",
            "synthetic",
        ]
    )
    assert rc == 0
    namespaced = out_dir / "synthetic"
    json_path = namespaced / COSSIM_JSON_FILENAME
    csv_path = namespaced / COSSIM_CSV_FILENAME
    assert json_path.exists() and csv_path.exists()

    payload = json.loads(json_path.read_text())
    assert payload["metadata"]["encoder"] == "synthetic"

    stdout = capsys.readouterr().out
    assert "cossim_results.json" in stdout
    assert "k=1" in stdout


def test_cli_main_missing_file_raises(tmp_path):
    """CLI propagates FileNotFoundError when an explicit path is bogus."""
    paths = _seed_four_pt_files(tmp_path)
    out_dir = tmp_path / "cli_out"

    with pytest.raises(FileNotFoundError):
        main(
            [
                "--z-hat-conditioned",
                str(tmp_path / "missing.pt"),
                "--z-real-conditioned",
                str(paths["z_real_cond"]),
                "--z-hat-unconditioned",
                str(paths["z_hat_uncond"]),
                "--z-real-unconditioned",
                str(paths["z_real_uncond"]),
                "--output-dir",
                str(out_dir),
            ]
        )


# ---------------------------------------------------------------------------
# Documentation-by-test: per-variant z_real policy
# ---------------------------------------------------------------------------


def test_evaluate_cossim_treats_paths_as_first_class(tmp_path):
    """The function accepts either str or Path -- no type-juggling required."""
    z = torch.randn(4, 4, 16)
    p_hat = _save(z, tmp_path / "z_hat.pt")
    p_real = _save(z, tmp_path / "z_real.pt")

    via_path = evaluate_cossim(p_hat, p_real)
    via_str = evaluate_cossim(str(p_hat), str(p_real))
    assert via_path == via_str


# ---------------------------------------------------------------------------
# Perturbation analysis tests (B10 extension)
# ---------------------------------------------------------------------------


def test_compute_perturbation_delta_cossim_math():
    """Per-horizon perturbation delta = masked - unmasked."""
    unmasked = {1: 0.95, 2: 0.90, 3: 0.85, 4: 0.80}
    masked = {1: 0.90, 2: 0.88, 3: 0.86, 4: 0.82}

    delta = compute_perturbation_delta_cossim(unmasked, masked)

    assert sorted(delta) == [1, 2, 3, 4]
    assert delta[1] == pytest.approx(-0.05)  # 0.90 - 0.95
    assert delta[2] == pytest.approx(-0.02)  # 0.88 - 0.90
    assert delta[3] == pytest.approx(0.01)  # 0.86 - 0.85
    assert delta[4] == pytest.approx(0.02)  # 0.82 - 0.80


def test_compute_perturbation_delta_cossim_horizon_mismatch():
    """Mismatched horizons between unmasked and masked raises ValueError."""
    unmasked = {1: 0.95, 2: 0.90}
    masked = {1: 0.90, 2: 0.88, 3: 0.86}

    with pytest.raises(ValueError, match="horizon mismatch"):
        compute_perturbation_delta_cossim(unmasked, masked)


def test_export_cossim_results_with_perturbation(tmp_path):
    """Perturbation columns appear in CSV and JSON when provided."""
    cossim_cond = {1: 0.95, 2: 0.90}
    cossim_uncond = {1: 0.94, 2: 0.89}
    delta = {1: 0.01, 2: 0.01}
    cossim_masked = {1: 0.90, 2: 0.88}
    perturb_delta = {1: -0.05, 2: -0.02}

    json_path, csv_path = export_cossim_results(
        cossim_cond,
        cossim_uncond,
        delta,
        tmp_path,
        cossim_masked=cossim_masked,
        perturbation_delta=perturb_delta,
    )

    # Check CSV has extended columns
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        assert reader.fieldnames == list(CSV_COLUMNS_PERTURBED)
        assert len(rows) == 2
        assert rows[0]["k"] == "1"
        assert float(rows[0]["cossim_masked"]) == pytest.approx(0.90)
        assert float(rows[0]["perturbation_delta_cossim"]) == pytest.approx(-0.05)

    # Check JSON has perturbation fields
    payload = json.loads(json_path.read_text())
    assert "cossim_masked" in payload["per_horizon"]["1"]
    assert "perturbation_delta_cossim" in payload["per_horizon"]["1"]
    assert payload["per_horizon"]["1"]["cossim_masked"] == pytest.approx(0.90)
    assert payload["per_horizon"]["1"]["perturbation_delta_cossim"] == pytest.approx(-0.05)


def test_run_latent_eval_with_masked_input(tmp_path):
    """run_latent_eval computes perturbation delta when z_hat_masked_path is provided."""
    n, horizon, z_dim = 16, 4, 384

    # Unmasked baseline
    z_hat_cond = torch.randn(n, horizon, z_dim)
    z_real_cond = torch.randn(n, horizon, z_dim)
    z_hat_uncond = torch.randn(n, horizon, z_dim)
    z_real_uncond = torch.randn(n, horizon, z_dim)

    # Masked variant (slightly different from unmasked)
    z_hat_masked = z_hat_cond + torch.randn(n, horizon, z_dim) * 0.1

    paths = {
        "z_hat_cond": _save(z_hat_cond, tmp_path / "z_hat_conditioned.pt"),
        "z_real_cond": _save(z_real_cond, tmp_path / "z_real_conditioned.pt"),
        "z_hat_uncond": _save(z_hat_uncond, tmp_path / "z_hat_unconditioned.pt"),
        "z_real_uncond": _save(z_real_uncond, tmp_path / "z_real_unconditioned.pt"),
        "z_hat_masked": _save(z_hat_masked, tmp_path / "z_hat_masked.pt"),
    }

    out_dir = tmp_path / "out"
    payload = run_latent_eval(
        z_hat_conditioned_path=paths["z_hat_cond"],
        z_real_conditioned_path=paths["z_real_cond"],
        z_hat_unconditioned_path=paths["z_hat_uncond"],
        z_real_unconditioned_path=paths["z_real_uncond"],
        output_dir=out_dir,
        z_hat_masked_path=paths["z_hat_masked"],
    )

    # Check payload has perturbation fields
    assert "cossim_masked" in payload["per_horizon"]["1"]
    assert "perturbation_delta_cossim" in payload["per_horizon"]["1"]

    # Check CSV written with extended columns
    csv_path = out_dir / COSSIM_CSV_FILENAME
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == list(CSV_COLUMNS_PERTURBED)


def test_cli_main_with_perturbation(tmp_path, capsys):
    """CLI --z-hat-masked triggers perturbation analysis."""
    n, horizon, z_dim = 16, 4, 384
    paths = _seed_four_pt_files(tmp_path)
    z_hat_masked = torch.randn(n, horizon, z_dim)
    p_masked = _save(z_hat_masked, tmp_path / "z_hat_masked.pt")
    out_dir = tmp_path / "cli_out"

    rc = main(
        [
            "--z-hat-conditioned",
            str(paths["z_hat_cond"]),
            "--z-real-conditioned",
            str(paths["z_real_cond"]),
            "--z-hat-unconditioned",
            str(paths["z_hat_uncond"]),
            "--z-real-unconditioned",
            str(paths["z_real_uncond"]),
            "--z-hat-masked",
            str(p_masked),
            "--perturbation-type",
            "mask_left_lane",
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    stdout = capsys.readouterr().out
    assert "CosSim_masked" in stdout
    assert "PerturbDelta" in stdout

    # Check JSON metadata includes perturbation_type
    json_path = out_dir / COSSIM_JSON_FILENAME
    payload = json.loads(json_path.read_text())
    assert payload["metadata"]["perturbation_type"] == "mask_left_lane"


def test_cli_main_perturbation_args_must_be_paired(tmp_path, capsys):
    """CLI requires both --z-hat-masked and --perturbation-type or neither."""
    paths = _seed_four_pt_files(tmp_path)
    out_dir = tmp_path / "cli_out"

    # Only --z-hat-masked without --perturbation-type
    rc = main(
        [
            "--z-hat-conditioned",
            str(paths["z_hat_cond"]),
            "--z-real-conditioned",
            str(paths["z_real_cond"]),
            "--z-hat-unconditioned",
            str(paths["z_hat_uncond"]),
            "--z-real-unconditioned",
            str(paths["z_real_uncond"]),
            "--z-hat-masked",
            str(paths["z_hat_cond"]),  # reuse existing file
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 1
    stderr = capsys.readouterr().err
    assert "must both be provided" in stderr
