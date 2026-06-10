"""Unit + integration tests for :mod:`evaluation.unified_cossim` (DC1).

DC1 extends the C4 CosSim evaluation to the Tier-2 fair comparison:
DiT (DDIM) and fair-MLP per-horizon CosSim merged into one artifact with
per-seed **paired** DeltaCosSim = CosSim_conditioned - CosSim_unconditioned.

Covers:

* The C4-faithful per-horizon CosSim math (identity / anti-aligned /
  orthogonal / reference implementation / fp16 upcast / validation).
* Aggregate ingestion from the Modal rollout artifacts (both top-level
  shapes, error-entry skipping, missing-variant and shape-drift errors).
* The z_hat tensor path ("load DiT z_hat tensors from HuggingFace (or
  local)" -- same math, file-based).
* ``unify``: per-seed paired delta statistics (the key property: constant
  per-seed delta => delta_std == 0 even when cond_std > 0), ddof=1,
  single-seed handling, grouping, n_test_windows consistency.
* CSV/JSON schema + round-trip + CLI smoke.
* Integration against the REAL committed artifacts on main-tier2, pinned
  to the values in ``artifacts/full/dit_vs_mlp_comparison.csv`` (DA7.5).
* Battle-test: the tensor path reproduces the C4 artifact
  (``main:artifacts/cossim_eval/cossim_results.csv``) exactly from the
  local P1-MLP z_hat tensors, when those tensors are present.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from evaluation.unified_cossim import (
    CSV_COLUMNS,
    UNIFIED_CSV_FILENAME,
    UNIFIED_JSON_FILENAME,
    _per_horizon_cossim,
    build_payload,
    main,
    record_from_tensors,
    records_from_rollout,
    unify,
    write_unified_csv,
    write_unified_json,
)

H, D = 4, 16


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(20260604)
    yield


# ---------------------------------------------------------------------------
# C4-faithful per-horizon CosSim
# ---------------------------------------------------------------------------


def test_cossim_identical_is_one():
    z = torch.randn(8, H, D)
    out = _per_horizon_cossim(z, z.clone())
    assert sorted(out) == [1, 2, 3, 4]
    for v in out.values():
        assert v == pytest.approx(1.0, abs=1e-6)


def test_cossim_anti_aligned_is_minus_one():
    z = torch.randn(8, H, D)
    out = _per_horizon_cossim(z, -z)
    for v in out.values():
        assert v == pytest.approx(-1.0, abs=1e-6)


def test_cossim_orthogonal_is_zero():
    a = torch.zeros(4, H, 8)
    b = torch.zeros(4, H, 8)
    a[..., 0] = 1.0
    b[..., 1] = 1.0
    out = _per_horizon_cossim(a, b)
    for v in out.values():
        assert v == pytest.approx(0.0, abs=1e-7)


def test_cossim_matches_reference():
    z_hat = torch.randn(32, H, D)
    z_real = torch.randn(32, H, D)
    got = _per_horizon_cossim(z_hat, z_real)
    for k in range(1, H + 1):
        expected = F.cosine_similarity(
            z_hat[:, k - 1, :].float(), z_real[:, k - 1, :].float(), dim=-1
        ).mean()
        assert got[k] == pytest.approx(float(expected), abs=1e-6)


def test_cossim_fp16_upcast():
    z = torch.randn(16, H, D).half()
    out = _per_horizon_cossim(z, z.clone())
    for v in out.values():
        assert v == pytest.approx(1.0, abs=1e-3)


def test_cossim_shape_validation():
    with pytest.raises(ValueError, match="identical shapes"):
        _per_horizon_cossim(torch.randn(4, H, D), torch.randn(4, H, D + 1))
    with pytest.raises(ValueError, match="3D"):
        _per_horizon_cossim(torch.randn(4, D), torch.randn(4, D))
    with pytest.raises(ValueError, match="empty"):
        _per_horizon_cossim(torch.randn(0, H, D), torch.randn(0, H, D))


# ---------------------------------------------------------------------------
# Aggregate ingestion (Modal rollout artifacts)
# ---------------------------------------------------------------------------


def _entry(encoder, variant, seed, cossim, n=100):
    return {
        "encoder": encoder,
        "variant": variant,
        "seed": seed,
        "n_test_windows": n,
        "metrics": {"cossim_by_horizon": list(cossim)},
    }


def test_records_from_rollout_pairs_variants():
    payload = {
        "results": [
            _entry("encA", "conditioned", 0, [0.9, 0.8, 0.7, 0.6]),
            _entry("encA", "unconditioned", 0, [0.5, 0.5, 0.5, 0.5]),
        ]
    }
    records = records_from_rollout(payload, model="dit")
    assert len(records) == 1
    rec = records[0]
    assert rec["model"] == "dit"
    assert rec["encoder"] == "encA"
    assert rec["seed"] == 0
    assert rec["n_test_windows"] == 100
    assert rec["cossim_conditioned"] == [0.9, 0.8, 0.7, 0.6]
    assert rec["cossim_unconditioned"] == [0.5, 0.5, 0.5, 0.5]


def test_records_from_rollout_accepts_bare_list():
    payload = [
        _entry("encA", "conditioned", 0, [0.9] * 4),
        _entry("encA", "unconditioned", 0, [0.8] * 4),
    ]
    assert len(records_from_rollout(payload, model="mlp")) == 1


def test_records_from_rollout_missing_variant_raises():
    payload = [_entry("encA", "conditioned", 0, [0.9] * 4)]
    with pytest.raises(ValueError, match="encA.*seed 0|seed 0.*encA"):
        records_from_rollout(payload, model="dit")


def test_records_from_rollout_horizon_mismatch_raises():
    payload = [
        _entry("encA", "conditioned", 0, [0.9] * 4),
        _entry("encA", "unconditioned", 0, [0.8] * 3),
    ]
    with pytest.raises(ValueError, match="horizon"):
        records_from_rollout(payload, model="dit")


def test_records_from_rollout_window_mismatch_raises():
    payload = [
        _entry("encA", "conditioned", 0, [0.9] * 4, n=100),
        _entry("encA", "unconditioned", 0, [0.8] * 4, n=99),
    ]
    with pytest.raises(ValueError, match="n_test_windows"):
        records_from_rollout(payload, model="dit")


def test_records_from_rollout_skips_error_entries():
    payload = [
        {"encoder": "encB", "variant": "conditioned", "seed": 1, "error": "boom"},
        {"encoder": "encB", "variant": "unconditioned", "seed": 1, "error": "boom"},
        _entry("encA", "conditioned", 0, [0.9] * 4),
        _entry("encA", "unconditioned", 0, [0.8] * 4),
    ]
    records = records_from_rollout(payload, model="dit")
    assert [r["encoder"] for r in records] == ["encA"]


# ---------------------------------------------------------------------------
# Tensor path (the C4 "load z_hat .pt" contract)
# ---------------------------------------------------------------------------


def _save(t: torch.Tensor, path: Path) -> Path:
    torch.save(t, path)
    return path


def test_record_from_tensors(tmp_path):
    z_real = torch.randn(10, H, D)
    rec = record_from_tensors(
        model="mlp_p1",
        encoder="encA",
        z_hat_conditioned=_save(z_real.clone(), tmp_path / "hc.pt"),
        z_real_conditioned=_save(z_real, tmp_path / "rc.pt"),
        z_hat_unconditioned=_save(-z_real.clone(), tmp_path / "hu.pt"),
        z_real_unconditioned=_save(z_real.clone(), tmp_path / "ru.pt"),
        seed=0,
    )
    assert rec["model"] == "mlp_p1"
    assert rec["n_test_windows"] == 10
    for k in range(H):
        assert rec["cossim_conditioned"][k] == pytest.approx(1.0, abs=1e-6)
        assert rec["cossim_unconditioned"][k] == pytest.approx(-1.0, abs=1e-6)
    assert rec["source"] == "z_hat_tensors"


def test_record_from_tensors_shape_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match="identical shapes"):
        record_from_tensors(
            model="m",
            encoder="e",
            z_hat_conditioned=_save(torch.randn(4, H, D), tmp_path / "a.pt"),
            z_real_conditioned=_save(torch.randn(4, H, D + 1), tmp_path / "b.pt"),
            z_hat_unconditioned=_save(torch.randn(4, H, D), tmp_path / "c.pt"),
            z_real_unconditioned=_save(torch.randn(4, H, D), tmp_path / "d.pt"),
        )


# ---------------------------------------------------------------------------
# unify: per-seed PAIRED delta statistics
# ---------------------------------------------------------------------------


def _rec(model, encoder, seed, cond, uncond, n=100):
    return {
        "model": model,
        "encoder": encoder,
        "seed": seed,
        "n_test_windows": n,
        "cossim_conditioned": list(cond),
        "cossim_unconditioned": list(uncond),
        "source": "test",
    }


def test_unify_paired_delta_has_zero_std_when_delta_constant():
    # Seeds differ a lot in absolute CosSim but the *paired* delta is a
    # constant +0.1 -- delta_std must be exactly 0 while cond_std > 0.
    # (An unpaired computation would report a large spurious delta spread.)
    records = [
        _rec("dit", "encA", 0, [0.90] * 4, [0.80] * 4),
        _rec("dit", "encA", 1, [0.70] * 4, [0.60] * 4),
        _rec("dit", "encA", 2, [0.50] * 4, [0.40] * 4),
    ]
    unified = unify(records)
    row = [r for r in unified["rows"] if r["k"] == 1][0]
    assert row["delta_cossim_mean"] == pytest.approx(0.1, abs=1e-12)
    assert row["delta_cossim_std"] == pytest.approx(0.0, abs=1e-12)
    assert row["cossim_conditioned_std"] == pytest.approx(0.2, abs=1e-12)  # ddof=1
    assert row["n_seeds"] == 3


def test_unify_stats_match_numpy_ddof1():
    import numpy as np

    cond = {0: 0.91, 1: 0.87, 2: 0.95}
    uncond = {0: 0.90, 1: 0.80, 2: 0.85}
    records = [
        _rec("mlp", "encA", s, [cond[s]] * 4, [uncond[s]] * 4) for s in (0, 1, 2)
    ]
    row = [r for r in unify(records)["rows"] if r["k"] == 2][0]
    deltas = [cond[s] - uncond[s] for s in (0, 1, 2)]
    assert row["cossim_conditioned_mean"] == pytest.approx(np.mean(list(cond.values())))
    assert row["cossim_conditioned_std"] == pytest.approx(
        np.std(list(cond.values()), ddof=1)
    )
    assert row["delta_cossim_mean"] == pytest.approx(np.mean(deltas))
    assert row["delta_cossim_std"] == pytest.approx(np.std(deltas, ddof=1))


def test_unify_single_seed_has_null_std():
    records = [_rec("dit", "encA", 0, [0.9] * 4, [0.8] * 4)]
    row = unify(records)["rows"][0]
    assert row["n_seeds"] == 1
    assert row["cossim_conditioned_std"] is None
    assert row["delta_cossim_std"] is None
    assert row["delta_cossim_mean"] == pytest.approx(0.1)


def test_unify_groups_models_and_encoders():
    records = [
        _rec("dit", "encA", 0, [0.9] * 4, [0.8] * 4),
        _rec("dit", "encB", 0, [0.9] * 4, [0.8] * 4),
        _rec("mlp", "encA", 0, [0.9] * 4, [0.8] * 4),
    ]
    unified = unify(records)
    assert unified["models"] == ["dit", "mlp"]
    assert unified["encoders"] == ["encA", "encB"]
    assert unified["horizon"] == 4
    assert len(unified["rows"]) == 3 * 4  # (dit,encA) (dit,encB) (mlp,encA) x k
    keys = {(r["model"], r["encoder"], r["k"]) for r in unified["rows"]}
    assert ("dit", "encB", 4) in keys and ("mlp", "encA", 1) in keys


def test_unify_handles_multiple_seedless_records():
    # Tensor-sourced records may carry seed=None; two of them in one group
    # must aggregate instead of crashing on a None<None sort comparison.
    records = [
        _rec("mlp_p1", "encA", None, [0.9] * 4, [0.8] * 4),
        _rec("mlp_p1", "encA", None, [0.7] * 4, [0.6] * 4),
    ]
    row = unify(records)["rows"][0]
    assert row["n_seeds"] == 2
    assert row["delta_cossim_mean"] == pytest.approx(0.1)
    assert row["delta_cossim_std"] == pytest.approx(0.0, abs=1e-12)


def test_unify_inconsistent_windows_raises():
    records = [
        _rec("dit", "encA", 0, [0.9] * 4, [0.8] * 4, n=100),
        _rec("dit", "encA", 1, [0.9] * 4, [0.8] * 4, n=101),
    ]
    with pytest.raises(ValueError, match="n_test_windows"):
        unify(records)


def test_unify_inconsistent_horizon_raises():
    records = [
        _rec("dit", "encA", 0, [0.9] * 4, [0.8] * 4),
        _rec("mlp", "encA", 0, [0.9] * 3, [0.8] * 3),
    ]
    with pytest.raises(ValueError, match="horizon"):
        unify(records)


# ---------------------------------------------------------------------------
# Export: CSV + JSON schema
# ---------------------------------------------------------------------------


def test_csv_schema_and_roundtrip(tmp_path):
    records = [
        _rec("dit", "encA", s, [0.9 - 0.01 * s] * 4, [0.8] * 4) for s in (0, 1, 2)
    ]
    unified = unify(records)
    path = tmp_path / UNIFIED_CSV_FILENAME
    write_unified_csv(unified["rows"], path)

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0].keys()) == CSV_COLUMNS
    assert len(rows) == 4
    assert rows[0]["model"] == "dit"
    assert rows[0]["n_seeds"] == "3"
    assert float(rows[0]["delta_cossim_mean"]) == pytest.approx(0.09, abs=1e-9)


def test_csv_single_seed_std_cells_are_empty(tmp_path):
    unified = unify([_rec("dit", "encA", 0, [0.9] * 4, [0.8] * 4)])
    path = tmp_path / "u.csv"
    write_unified_csv(unified["rows"], path)
    with path.open() as fh:
        row = next(csv.DictReader(fh))
    assert row["cossim_conditioned_std"] == ""
    assert row["delta_cossim_std"] == ""


def test_json_payload_schema_and_roundtrip(tmp_path):
    records = [_rec("dit", "encA", 0, [0.9] * 4, [0.8] * 4)]
    payload = build_payload(unify(records), inputs={"dit_results": "x.json"})
    assert payload["schema_version"] == "1.0"
    assert "conditioned_minus_unconditioned" in payload["method"]
    assert "proxy" not in payload["method"].lower()
    assert payload["inputs"] == {"dit_results": "x.json"}
    assert payload["per_seed"][0]["model"] == "dit"

    path = tmp_path / UNIFIED_JSON_FILENAME
    write_unified_json(payload, path)
    assert json.loads(path.read_text()) == payload


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_relativize_repo_paths_only():
    from config import repo_root

    from evaluation.unified_cossim import _relativize

    inside = repo_root() / "artifacts" / "full" / "rollout_results.json"
    assert _relativize(inside) == "artifacts/full/rollout_results.json"
    outside = Path("/tmp/somewhere/else.json")
    assert _relativize(outside) == str(outside)


def test_cli_main_smoke(tmp_path):
    dit = {
        "results": [
            _entry("encA", "conditioned", s, [0.6, 0.59, 0.58, 0.57]) for s in (0, 1)
        ]
        + [_entry("encA", "unconditioned", s, [0.6, 0.6, 0.6, 0.6]) for s in (0, 1)]
    }
    mlp = {
        "results": [
            _entry("encA", "conditioned", s, [0.9, 0.89, 0.88, 0.87]) for s in (0, 1)
        ]
        + [_entry("encA", "unconditioned", s, [0.8] * 4) for s in (0, 1)]
    }
    (tmp_path / "dit.json").write_text(json.dumps(dit))
    (tmp_path / "mlp.json").write_text(json.dumps(mlp))
    out = tmp_path / "out"

    rc = main(
        [
            "--dit-results",
            str(tmp_path / "dit.json"),
            "--mlp-results",
            str(tmp_path / "mlp.json"),
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads((out / UNIFIED_JSON_FILENAME).read_text())
    assert payload["models"] == ["dit", "mlp"]
    assert len(payload["rows"]) == 2 * 1 * 4
    assert payload["inputs"]["dit_results"].endswith("dit.json")
    with (out / UNIFIED_CSV_FILENAME).open() as fh:
        assert len(list(csv.DictReader(fh))) == 8


def test_cli_tensor_block_appends_model(tmp_path):
    dit = {
        "results": [
            _entry("encA", "conditioned", 0, [0.6] * 4),
            _entry("encA", "unconditioned", 0, [0.6] * 4),
        ]
    }
    (tmp_path / "dit.json").write_text(json.dumps(dit))
    (tmp_path / "mlp.json").write_text(json.dumps(dit))
    z_real = torch.randn(6, H, D)
    paths = {
        "hc": _save(z_real.clone(), tmp_path / "hc.pt"),
        "rc": _save(z_real, tmp_path / "rc.pt"),
        "hu": _save(-z_real.clone(), tmp_path / "hu.pt"),
        "ru": _save(z_real.clone(), tmp_path / "ru.pt"),
    }
    out = tmp_path / "out"
    rc = main(
        [
            "--dit-results",
            str(tmp_path / "dit.json"),
            "--mlp-results",
            str(tmp_path / "mlp.json"),
            "--output-dir",
            str(out),
            "--tensor-model",
            "mlp_p1",
            "--tensor-encoder",
            "encZ",
            "--z-hat-conditioned",
            str(paths["hc"]),
            "--z-real-conditioned",
            str(paths["rc"]),
            "--z-hat-unconditioned",
            str(paths["hu"]),
            "--z-real-unconditioned",
            str(paths["ru"]),
        ]
    )
    assert rc == 0
    payload = json.loads((out / UNIFIED_JSON_FILENAME).read_text())
    assert payload["models"] == ["dit", "mlp", "mlp_p1"]
    row = [r for r in payload["rows"] if r["model"] == "mlp_p1" and r["k"] == 1][0]
    assert row["delta_cossim_mean"] == pytest.approx(2.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Integration: REAL committed tier-2 aggregates, pinned to DA7.5's table
# ---------------------------------------------------------------------------


REPO = Path(__file__).resolve().parent.parent
DIT_RESULTS = REPO / "artifacts" / "full" / "rollout_results.json"
MLP_RESULTS = REPO / "artifacts" / "full" / "mlp_rollout_results.json"


@pytest.mark.skipif(
    not (DIT_RESULTS.exists() and MLP_RESULTS.exists()),
    reason="committed tier-2 rollout artifacts not present",
)
def test_real_aggregates_match_da75_comparison_table():
    records = records_from_rollout(
        json.loads(DIT_RESULTS.read_text()), model="dit"
    ) + records_from_rollout(json.loads(MLP_RESULTS.read_text()), model="mlp")
    unified = unify(records)

    assert unified["models"] == ["dit", "mlp"]
    assert len(unified["encoders"]) == 6
    assert len(unified["rows"]) == 2 * 6 * 4
    assert len(unified["per_seed"]) == 36  # 2 models x 6 encoders x 3 seeds
    assert all(r["n_seeds"] == 3 for r in unified["rows"])
    assert all(r["n_test_windows"] == 5419 for r in unified["rows"])

    def row(model, encoder, k):
        return [
            r
            for r in unified["rows"]
            if r["model"] == model and r["encoder"] == encoder and r["k"] == k
        ][0]

    # Pinned from artifacts/full/dit_vs_mlp_comparison.csv (DA7.5):
    # dit,clip_b32,conditioned,1 -> 0.6213091295158657
    # dit,clip_b32,unconditioned,1 -> 0.6188727347930141
    # mlp,clip_b32,conditioned,1 -> 0.9358048238107227
    r = row("dit", "clip_b32", 1)
    assert r["cossim_conditioned_mean"] == pytest.approx(0.6213091295158657, abs=1e-9)
    assert r["cossim_unconditioned_mean"] == pytest.approx(0.6188727347930141, abs=1e-9)
    # mean of per-seed paired deltas == difference of means (linearity)
    assert r["delta_cossim_mean"] == pytest.approx(
        0.6213091295158657 - 0.6188727347930141, abs=1e-7
    )
    assert row("mlp", "clip_b32", 1)["cossim_conditioned_mean"] == pytest.approx(
        0.9358048238107227, abs=1e-9
    )


# ---------------------------------------------------------------------------
# Battle-test: tensor path reproduces the C4 artifact from local tensors
# ---------------------------------------------------------------------------

LOCAL_TENSORS = {
    "z_hat_conditioned": REPO / "z_hat_conditioned.pt",
    "z_real_conditioned": REPO / "z_real_conditioned.pt",
    "z_hat_unconditioned": REPO / "z_hat_unconditioned.pt",
    "z_real_unconditioned": REPO / "z_real_unconditioned.pt",
}

# Pinned from main:artifacts/cossim_eval/cossim_results.json (C4, seed-0
# vjepa2_rep64 P1 predictor, n_samples=5419).
C4_EXPECTED = {
    1: (0.9955065250396729, 0.999889075756073, -0.0043825507164001465),
    2: (0.99550861120224, 0.999889075756073, -0.004380464553833008),
    3: (0.9955352544784546, 0.9998899698257446, -0.004354715347290039),
    4: (0.995522141456604, 0.9998902678489685, -0.004368126392364502),
}


@pytest.mark.skipif(
    not all(p.exists() for p in LOCAL_TENSORS.values()),
    reason="local C4 z_hat tensors not present (gitignored .pt files)",
)
def test_tensor_path_reproduces_c4_artifact():
    rec = record_from_tensors(
        model="mlp_p1",
        encoder="vjepa2_rep64",
        seed=0,
        **{k: str(v) for k, v in LOCAL_TENSORS.items()},
    )
    assert rec["n_test_windows"] == 5419
    for k, (cond, uncond, delta) in C4_EXPECTED.items():
        assert rec["cossim_conditioned"][k - 1] == pytest.approx(cond, abs=1e-6)
        assert rec["cossim_unconditioned"][k - 1] == pytest.approx(uncond, abs=1e-6)
        paired = rec["cossim_conditioned"][k - 1] - rec["cossim_unconditioned"][k - 1]
        assert paired == pytest.approx(delta, abs=1e-6)
        assert math.isclose(paired, delta, abs_tol=1e-6)
