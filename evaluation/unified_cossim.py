"""Unified per-horizon CosSim + DeltaCosSim across predictor families (DC1).

Extends the C4 CosSim evaluation pipeline (``evaluation/latent_eval.py`` on
``main``; not present on this branch, which forked earlier) to the Tier-2
fair comparison: **DiT (DDIM)** and **fair-MLP** results are merged into one
artifact, per ``(model x encoder x horizon k)``, with

    DeltaCosSim(k) = CosSim_conditioned(k) - CosSim_unconditioned(k)

computed **per seed (paired)** and then aggregated across seeds
(mean +- sample std, ddof=1).  Pairing within a seed matters: conditioned
and unconditioned runs of the same seed share training noise, so the paired
delta isolates the action-conditioning signal instead of burying it in
seed-to-seed variance.

Two ingestion paths
-------------------
1. **Rollout aggregates** (default): the committed Modal artifacts
   ``artifacts/full/rollout_results.json`` (DiT DDIM, DA5/DA7) and
   ``artifacts/full/mlp_rollout_results.json`` (fair MLP, DA7) -- a complete
   6-encoder x 3-seed x {conditioned, unconditioned} grid evaluated on the
   same 5419 test windows.  The Modal pipeline intentionally returns only
   these aggregates and never serialises z_hat tensors.
2. **z_hat tensors** ("load DiT z_hat tensors from HuggingFace (or local)"):
   per-variant ``z_hat``/``z_real`` ``.pt`` pairs evaluated with the same
   C4 per-horizon CosSim.  ``tests/test_unified_cossim.py`` proves this path
   reproduces the C4 artifact (``main:artifacts/cossim_eval/``) exactly from
   the exported P1-MLP tensors.

Outputs
-------
``dc1_unified_cossim.json`` + ``dc1_unified_cossim.csv`` (long format, one
row per model x encoder x k) under ``outputs/cossim_eval/``; the canonical
copies are vendored at ``artifacts/full/dc1_unified_cossim.{json,csv}``
following the tier-2 task-prefix convention (``da9_*``, ``da10_*``, ...).

CLI
---
    python -m evaluation.unified_cossim
    python -m evaluation.unified_cossim \\
        --dit-results artifacts/full/rollout_results.json \\
        --mlp-results artifacts/full/mlp_rollout_results.json \\
        --output-dir  outputs/cossim_eval
    # optionally append one tensor-sourced model (e.g. the P1 MLP):
    python -m evaluation.unified_cossim --tensor-model mlp_p1 \\
        --tensor-encoder vjepa2_rep64 \\
        --z-hat-conditioned z_hat_conditioned.pt \\
        --z-real-conditioned z_real_conditioned.pt \\
        --z-hat-unconditioned z_hat_unconditioned.pt \\
        --z-real-unconditioned z_real_unconditioned.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

UNIFIED_JSON_FILENAME = "dc1_unified_cossim.json"
UNIFIED_CSV_FILENAME = "dc1_unified_cossim.csv"

CSV_COLUMNS: tuple[str, ...] = (
    "model",
    "encoder",
    "k",
    "cossim_conditioned_mean",
    "cossim_conditioned_std",
    "cossim_unconditioned_mean",
    "cossim_unconditioned_std",
    "delta_cossim_mean",
    "delta_cossim_std",
    "n_seeds",
    "n_test_windows",
    "source",
)

_VARIANTS = ("conditioned", "unconditioned")

_METHOD = "unified_per_horizon_cossim_delta(conditioned_minus_unconditioned)"
_METRIC = (
    "CosSim(k) = mean_n cos(z_hat[n, k-1, :], z_real[n, k-1, :]) per C4; "
    "DeltaCosSim(k) = CosSim_conditioned(k) - CosSim_unconditioned(k), "
    "paired per seed, aggregated across seeds with ddof=1"
)


# ---------------------------------------------------------------------------
# Core math (mirrors evaluation/latent_eval.py::_per_horizon_cossim on main)
# ---------------------------------------------------------------------------


def _per_horizon_cossim(z_hat: torch.Tensor, z_real: torch.Tensor) -> dict[int, float]:
    """Per-horizon mean cosine similarity between two ``(N, H, D)`` tensors.

    Faithful copy of the C4 implementation (``evaluation/latent_eval.py`` on
    the ``main`` branch, which this tier-2 branch predates): float32
    reduction, 1-indexed horizons, identical validation.

    Keep in sync with that C4 original -- if either side's cosine math
    changes (reduction dtype, horizon indexing, or validation), update both
    so the two pipelines stay comparable.  ``test_cossim_matches_reference``
    pins this copy to the canonical per-horizon formula, so drift here fails
    that test.  Once ``latent_eval`` is available on this branch's base,
    prefer importing it (as ``lang_scene_eval`` does) over keeping the copy.
    """
    if z_hat.shape != z_real.shape:
        raise ValueError(
            f"z_hat and z_real must have identical shapes; "
            f"got z_hat={tuple(z_hat.shape)}, z_real={tuple(z_real.shape)}"
        )
    if z_hat.dim() != 3:
        raise ValueError(
            f"z_hat / z_real must be 3D (N, horizon, z_dim); "
            f"got {z_hat.dim()}D with shape {tuple(z_hat.shape)}"
        )
    n_samples, horizon, z_dim = z_hat.shape
    if n_samples == 0:
        raise ValueError("z_hat / z_real are empty along the sample dimension")
    if horizon == 0 or z_dim == 0:
        raise ValueError("z_hat / z_real have a zero horizon or embedding dim")

    z_hat_f = z_hat.to(torch.float32)
    z_real_f = z_real.to(torch.float32)
    return {
        k: float(
            F.cosine_similarity(
                z_hat_f[:, k - 1, :], z_real_f[:, k - 1, :], dim=-1
            ).mean()
        )
        for k in range(1, horizon + 1)
    }


def _cossim_list(z_hat: torch.Tensor, z_real: torch.Tensor) -> list[float]:
    cossim = _per_horizon_cossim(z_hat, z_real)
    return [cossim[k] for k in sorted(cossim)]


# ---------------------------------------------------------------------------
# Ingestion: z_hat tensors (the C4 export contract)
# ---------------------------------------------------------------------------


def _load_tensor(path: str | Path, role: str) -> torch.Tensor:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{role} tensor not found: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, torch.Tensor):
        raise ValueError(f"{path}: expected a torch.Tensor, got {type(obj).__name__}")
    return obj


def record_from_tensors(
    model: str,
    encoder: str,
    z_hat_conditioned: str | Path,
    z_real_conditioned: str | Path,
    z_hat_unconditioned: str | Path,
    z_real_unconditioned: str | Path,
    seed: Optional[int] = None,
    source: str = "z_hat_tensors",
) -> dict[str, Any]:
    """One per-seed record computed from exported ``z_hat``/``z_real`` pairs.

    Each variant is evaluated against its own ``z_real`` (the per-variant
    adapter policy from ``scripts/export_z_hat.py``).
    """
    hat_cond = _load_tensor(z_hat_conditioned, "z_hat_conditioned")
    real_cond = _load_tensor(z_real_conditioned, "z_real_conditioned")
    hat_uncond = _load_tensor(z_hat_unconditioned, "z_hat_unconditioned")
    real_uncond = _load_tensor(z_real_unconditioned, "z_real_unconditioned")

    cossim_cond = _cossim_list(hat_cond, real_cond)
    cossim_uncond = _cossim_list(hat_uncond, real_uncond)

    if hat_cond.shape[0] != hat_uncond.shape[0]:
        raise ValueError(
            f"n_test_windows mismatch between variants: "
            f"conditioned={hat_cond.shape[0]}, unconditioned={hat_uncond.shape[0]}"
        )
    if len(cossim_cond) != len(cossim_uncond):
        raise ValueError(
            f"horizon mismatch between variants: "
            f"conditioned={len(cossim_cond)}, unconditioned={len(cossim_uncond)}"
        )

    return {
        "model": model,
        "encoder": encoder,
        "seed": seed,
        "n_test_windows": int(hat_cond.shape[0]),
        "cossim_conditioned": cossim_cond,
        "cossim_unconditioned": cossim_uncond,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Ingestion: Modal rollout aggregates
# ---------------------------------------------------------------------------


def records_from_rollout(
    payload: Any, model: str, source: Optional[str] = None
) -> list[dict[str, Any]]:
    """Parse a rollout-results payload into per-seed records.

    Accepts either a bare list of entries or ``{"results": [...]}`` (both
    shapes exist among the committed artifacts).  Entries carrying an
    ``"error"`` key (failed Modal jobs) are skipped.  Every surviving
    ``(encoder, seed)`` must have BOTH variants -- a half-present pair is an
    artifact-corruption signal and raises rather than silently dropping.
    """
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("results"), list):
        entries = payload["results"]
    else:
        raise ValueError(
            "rollout payload must be a list of entries or {'results': [...]}"
        )

    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for entry in entries:
        if "error" in entry:
            continue
        try:
            encoder = entry["encoder"]
            variant = entry["variant"]
            seed = int(entry["seed"])
            n_windows = int(entry["n_test_windows"])
            cossim = list(entry["metrics"]["cossim_by_horizon"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed rollout entry {entry!r}: {exc}") from exc
        if variant not in _VARIANTS:
            raise ValueError(
                f"unknown variant {variant!r} for encoder {encoder!r} seed {seed}"
            )
        bucket = grouped.setdefault((encoder, seed), {})
        if variant in bucket:
            raise ValueError(
                f"duplicate {variant!r} entry for encoder {encoder!r} seed {seed}"
            )
        bucket[variant] = {"n_test_windows": n_windows, "cossim": cossim}

    records = []
    for (encoder, seed), variants in sorted(grouped.items()):
        missing = [v for v in _VARIANTS if v not in variants]
        if missing:
            raise ValueError(
                f"model {model!r}: missing {missing!r} entry for "
                f"encoder {encoder!r} seed {seed}"
            )
        cond, uncond = variants["conditioned"], variants["unconditioned"]
        if len(cond["cossim"]) != len(uncond["cossim"]):
            raise ValueError(
                f"horizon mismatch for encoder {encoder!r} seed {seed}: "
                f"conditioned has {len(cond['cossim'])} steps, "
                f"unconditioned has {len(uncond['cossim'])}"
            )
        if cond["n_test_windows"] != uncond["n_test_windows"]:
            raise ValueError(
                f"n_test_windows mismatch for encoder {encoder!r} seed {seed}: "
                f"{cond['n_test_windows']} vs {uncond['n_test_windows']}"
            )
        records.append(
            {
                "model": model,
                "encoder": encoder,
                "seed": seed,
                "n_test_windows": cond["n_test_windows"],
                "cossim_conditioned": cond["cossim"],
                "cossim_unconditioned": uncond["cossim"],
                "source": source or "rollout_aggregates",
            }
        )
    return records


# ---------------------------------------------------------------------------
# Unification: per-seed paired deltas -> per (model x encoder x k) stats
# ---------------------------------------------------------------------------


def _mean_std(values: Sequence[float]) -> tuple[float, Optional[float]]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else None  # ddof=1
    return mean, std


def unify(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed records into per ``(model, encoder, k)`` rows.

    DeltaCosSim is computed **within each seed** (paired) before averaging,
    so seed-level offsets shared by both variants cancel instead of
    inflating the delta spread.
    """
    if not records:
        raise ValueError("no records to unify")

    horizons = {len(r["cossim_conditioned"]) for r in records}
    if len(horizons) != 1:
        raise ValueError(f"records disagree on horizon length: {sorted(horizons)}")
    horizon = horizons.pop()

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["model"], record["encoder"]), []).append(record)

    rows = []
    for (model, encoder), group in sorted(grouped.items()):
        group = sorted(group, key=lambda r: (r["seed"] is None, r["seed"]))
        n_windows = {r["n_test_windows"] for r in group}
        if len(n_windows) != 1:
            raise ValueError(
                f"{model}/{encoder}: seeds disagree on n_test_windows: "
                f"{sorted(n_windows)}"
            )
        sources = sorted({r["source"] for r in group})
        for k in range(1, horizon + 1):
            cond = [r["cossim_conditioned"][k - 1] for r in group]
            uncond = [r["cossim_unconditioned"][k - 1] for r in group]
            delta = [c - u for c, u in zip(cond, uncond)]  # paired per seed
            cond_mean, cond_std = _mean_std(cond)
            uncond_mean, uncond_std = _mean_std(uncond)
            delta_mean, delta_std = _mean_std(delta)
            rows.append(
                {
                    "model": model,
                    "encoder": encoder,
                    "k": k,
                    "cossim_conditioned_mean": cond_mean,
                    "cossim_conditioned_std": cond_std,
                    "cossim_unconditioned_mean": uncond_mean,
                    "cossim_unconditioned_std": uncond_std,
                    "delta_cossim_mean": delta_mean,
                    "delta_cossim_std": delta_std,
                    "n_seeds": len(group),
                    "n_test_windows": group[0]["n_test_windows"],
                    "source": "+".join(sources),
                }
            )

    return {
        "rows": rows,
        "models": sorted({r["model"] for r in records}),
        "encoders": sorted({r["encoder"] for r in records}),
        "horizon": horizon,
        "per_seed": [dict(r) for r in records],
    }


def build_payload(
    unified: Mapping[str, Any], inputs: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    """Final JSON payload: unified stats + method/metric/provenance."""
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "method": _METHOD,
        "metric": _METRIC,
        "generated_by": "evaluation.unified_cossim (DC1)",
        **{
            k: unified[k] for k in ("rows", "models", "encoders", "horizon", "per_seed")
        },
    }
    if inputs is not None:
        payload["inputs"] = dict(inputs)
    return payload


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10f}"
    return str(value)


def write_unified_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Long-format CSV, one row per ``(model, encoder, k)`` (atomic write)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow([_fmt(row[col]) for col in CSV_COLUMNS])
    tmp.replace(path)
    return path


def write_unified_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default(relative: str) -> Path:
    from config import repo_root

    return repo_root() / relative


def _relativize(path: str | Path) -> str:
    """Repo-relative form of ``path`` for provenance blocks.

    Committed artifacts must not embed machine-specific absolute paths, so
    anything inside the repo is recorded relative to the repo root; paths
    outside the repo are recorded as given.
    """
    from config import repo_root

    try:
        return Path(path).resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return str(path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unified_cossim",
        description=(
            "Unified per-horizon CosSim + DeltaCosSim across DiT and MLP "
            "predictors (DC1). Defaults read the committed Modal rollout "
            "aggregates on this branch."
        ),
    )
    parser.add_argument(
        "--dit-results",
        type=Path,
        default=None,
        help="DiT rollout JSON (default: artifacts/full/rollout_results.json).",
    )
    parser.add_argument(
        "--mlp-results",
        type=Path,
        default=None,
        help=(
            "Fair-MLP rollout JSON (default: artifacts/full/mlp_rollout_results.json)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/cossim_eval).",
    )
    tensor = parser.add_argument_group(
        "optional tensor-sourced model (C4 z_hat export contract)"
    )
    tensor.add_argument("--tensor-model", default=None)
    tensor.add_argument("--tensor-encoder", default=None)
    tensor.add_argument("--tensor-seed", type=int, default=None)
    tensor.add_argument("--z-hat-conditioned", type=Path, default=None)
    tensor.add_argument("--z-real-conditioned", type=Path, default=None)
    tensor.add_argument("--z-hat-unconditioned", type=Path, default=None)
    tensor.add_argument("--z-real-unconditioned", type=Path, default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    dit_path = args.dit_results or _default("artifacts/full/rollout_results.json")
    mlp_path = args.mlp_results or _default("artifacts/full/mlp_rollout_results.json")
    output_dir = args.output_dir or _default("outputs/cossim_eval")

    records = records_from_rollout(
        json.loads(Path(dit_path).read_text()), model="dit", source=Path(dit_path).name
    )
    records += records_from_rollout(
        json.loads(Path(mlp_path).read_text()), model="mlp", source=Path(mlp_path).name
    )

    inputs: dict[str, Any] = {
        "dit_results": _relativize(dit_path),
        "mlp_results": _relativize(mlp_path),
    }

    tensor_args = (
        args.z_hat_conditioned,
        args.z_real_conditioned,
        args.z_hat_unconditioned,
        args.z_real_unconditioned,
    )
    if args.tensor_model is not None:
        if args.tensor_encoder is None or any(p is None for p in tensor_args):
            parser.error(
                "--tensor-model requires --tensor-encoder and all four "
                "--z-hat-*/--z-real-* paths"
            )
        records.append(
            record_from_tensors(
                model=args.tensor_model,
                encoder=args.tensor_encoder,
                z_hat_conditioned=args.z_hat_conditioned,
                z_real_conditioned=args.z_real_conditioned,
                z_hat_unconditioned=args.z_hat_unconditioned,
                z_real_unconditioned=args.z_real_unconditioned,
                seed=args.tensor_seed,
            )
        )
        inputs["tensor_model"] = {
            "model": args.tensor_model,
            "encoder": args.tensor_encoder,
            "seed": args.tensor_seed,
            "z_hat_conditioned": _relativize(args.z_hat_conditioned),
            "z_real_conditioned": _relativize(args.z_real_conditioned),
            "z_hat_unconditioned": _relativize(args.z_hat_unconditioned),
            "z_real_unconditioned": _relativize(args.z_real_unconditioned),
        }

    unified = unify(records)
    payload = build_payload(unified, inputs=inputs)

    json_path = write_unified_json(payload, Path(output_dir) / UNIFIED_JSON_FILENAME)
    csv_path = write_unified_csv(
        unified["rows"], Path(output_dir) / UNIFIED_CSV_FILENAME
    )

    print(
        f"[unified_cossim] models={unified['models']}  "
        f"encoders={len(unified['encoders'])}  horizon={unified['horizon']}  "
        f"rows={len(unified['rows'])}"
    )
    for model in unified["models"]:
        model_rows = [r for r in unified["rows"] if r["model"] == model]
        for k in range(1, unified["horizon"] + 1):
            deltas = [r["delta_cossim_mean"] for r in model_rows if r["k"] == k]
            print(
                f"  {model:>8} k={k}: mean DeltaCosSim over encoders = "
                f"{statistics.fmean(deltas):+.6f}"
            )
    print(f"[unified_cossim] wrote {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
