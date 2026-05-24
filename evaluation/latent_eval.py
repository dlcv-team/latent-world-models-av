"""Per-horizon CosSim and DeltaCosSim evaluation for latent predictors (C4).

Consumes the ``.pt`` tensors written by ``scripts/export_z_hat.py``::

    z_hat_conditioned.pt      # (N, H, D) -- predictor output (cond)
    z_hat_unconditioned.pt    # (N, H, D) -- predictor output (uncond, a=0)
    z_real_conditioned.pt     # (N, H, D) -- adapter-projected ground truth
    z_real_unconditioned.pt   # (N, H, D) -- adapter-projected ground truth

Computes per-horizon cosine similarity::

    CosSim(k) = mean_n  cos( z_hat[n, k-1, :], z_real[n, k-1, :] )

and the action-conditioning signal::

    DeltaCosSim(k) = CosSim_conditioned(k) - CosSim_unconditioned(k)

Then exports both as JSON (machine-readable, nested) and CSV (long format
suitable for the figures pipeline).

Per-variant ``z_real`` policy
-----------------------------
Each predictor variant is trained with its own adapter projection, so the
conditioned and unconditional outputs live in **different 384-d subspaces**.
``CosSim_conditioned`` must therefore be computed against
``z_real_conditioned`` (matching adapter) and ``CosSim_unconditioned``
against ``z_real_unconditioned`` -- never cross-mixed.  This matches the
export contract documented in ``scripts/export_z_hat.py``.

CLI
---
    python -m evaluation.latent_eval \\
        --z-hat-conditioned    outputs/z_hat/z_hat_conditioned.pt \\
        --z-hat-unconditioned  outputs/z_hat/z_hat_unconditioned.pt \\
        --z-real-conditioned   outputs/z_hat/z_real_conditioned.pt \\
        --z-real-unconditioned outputs/z_hat/z_real_unconditioned.pt \\
        --output-dir           outputs/cossim_eval

If the four ``--z-{hat,real}-*`` paths are omitted, the CLI falls back to
``data.z_hat.load_z_hat`` / ``load_z_real`` so the default
``outputs/z_hat/`` layout (plus the HuggingFace cascade) is honored.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

# Public file names for the two artifacts.  Kept as module-level constants
# so downstream consumers (figures, paired tests) can import the same name.
COSSIM_JSON_FILENAME = "cossim_results.json"
COSSIM_CSV_FILENAME = "cossim_results.csv"

CSV_COLUMNS: tuple[str, ...] = (
    "k",
    "cossim_conditioned",
    "cossim_unconditioned",
    "delta_cossim",
)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


def _per_horizon_cossim(z_hat: torch.Tensor, z_real: torch.Tensor) -> dict[int, float]:
    """Per-horizon mean cosine similarity between two ``(N, H, D)`` tensors.

    Returned mapping is keyed by 1-indexed horizon ``k`` (``1..H``) and the
    values are Python floats so the result is JSON-serializable as-is.

    The computation follows the task spec exactly::

        F.cosine_similarity(z_hat[:, k-1, :], z_real[:, k-1, :], dim=-1).mean()
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
    if horizon == 0:
        raise ValueError("z_hat / z_real have zero horizon dimension")
    if z_dim == 0:
        raise ValueError("z_hat / z_real have zero embedding dimension")

    # Use float32 for the reduction so half-precision exports still produce
    # numerically stable means (matches the reference computation in
    # scripts/export_z_hat.py::_print_delta_cossim).
    z_hat_f = z_hat.to(torch.float32)
    z_real_f = z_real.to(torch.float32)

    out: dict[int, float] = {}
    for k in range(1, horizon + 1):
        sims = F.cosine_similarity(
            z_hat_f[:, k - 1, :], z_real_f[:, k - 1, :], dim=-1
        )
        out[k] = float(sims.mean().item())
    return out


def _load_tensor(path: Path, role: str) -> torch.Tensor:
    """Load a ``.pt`` file and assert it is a ``torch.Tensor``."""
    if not path.exists():
        raise FileNotFoundError(f"{role} tensor not found: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, torch.Tensor):
        raise ValueError(
            f"{path}: expected a torch.Tensor, got {type(obj).__name__}"
        )
    return obj


def _evaluate_pair(
    z_hat_path: Path, z_real_path: Path
) -> tuple[dict[int, float], tuple[int, int, int]]:
    """Internal: load both tensors once, return ``(cossim_dict, (N, H, D))``."""
    z_hat = _load_tensor(z_hat_path, "z_hat")
    z_real = _load_tensor(z_real_path, "z_real")
    cossim = _per_horizon_cossim(z_hat, z_real)
    n, h, d = z_hat.shape
    return cossim, (int(n), int(h), int(d))


def evaluate_cossim(
    z_hat_path: str | Path,
    z_real_path: str | Path,
) -> dict[int, float]:
    """Load ``z_hat`` / ``z_real`` ``.pt`` files and return per-horizon CosSim.

    Parameters
    ----------
    z_hat_path
        Path to a ``.pt`` file holding the predictor output, shape
        ``(N, horizon, z_dim)``.
    z_real_path
        Path to a ``.pt`` file holding the matching ground-truth latents,
        same shape.  Must come from the *same* adapter as ``z_hat`` --
        see the per-variant ``z_real`` policy in the module docstring.

    Returns
    -------
    dict[int, float]
        ``{k: cossim_k}`` for ``k`` in ``1..horizon``.

    Raises
    ------
    FileNotFoundError
        If either ``.pt`` file does not exist.
    ValueError
        If the loaded objects are not 3D tensors of identical shape, or
        if any dimension is zero.
    """
    cossim, _ = _evaluate_pair(Path(z_hat_path), Path(z_real_path))
    return cossim


def compute_delta_cossim(
    cossim_conditioned: Mapping[int, float],
    cossim_unconditioned: Mapping[int, float],
) -> dict[int, float]:
    """Per-horizon ``DeltaCosSim = CosSim_cond - CosSim_uncond``.

    The two inputs must cover identical horizon sets.  Mismatched keys
    raise ``ValueError`` so silent horizon drift across variants can't
    leak into the exported artifacts.
    """
    cond_keys = set(cossim_conditioned)
    uncond_keys = set(cossim_unconditioned)
    if cond_keys != uncond_keys:
        only_cond = sorted(cond_keys - uncond_keys)
        only_uncond = sorted(uncond_keys - cond_keys)
        raise ValueError(
            "horizon mismatch between conditioned and unconditioned CosSim: "
            f"only_in_conditioned={only_cond} "
            f"only_in_unconditioned={only_uncond}"
        )
    return {
        k: float(cossim_conditioned[k]) - float(cossim_unconditioned[k])
        for k in sorted(cond_keys)
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _build_results_payload(
    cossim_conditioned: Mapping[int, float],
    cossim_unconditioned: Mapping[int, float],
    delta_cossim: Mapping[int, float],
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the nested JSON payload (also used as the in-memory return)."""
    horizons = sorted(cossim_conditioned)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "horizon": len(horizons),
        "per_horizon": {
            str(k): {
                "cossim_conditioned": float(cossim_conditioned[k]),
                "cossim_unconditioned": float(cossim_unconditioned[k]),
                "delta_cossim": float(delta_cossim[k]),
            }
            for k in horizons
        },
        "mean_over_horizons": {
            "cossim_conditioned": float(
                sum(cossim_conditioned.values()) / len(horizons)
            ),
            "cossim_unconditioned": float(
                sum(cossim_unconditioned.values()) / len(horizons)
            ),
            "delta_cossim": float(sum(delta_cossim.values()) / len(horizons)),
        },
    }
    if metadata is not None:
        # Shallow copy so callers can't mutate the payload via reference.
        payload["metadata"] = dict(metadata)
    return payload


def export_cossim_results(
    cossim_conditioned: Mapping[int, float],
    cossim_unconditioned: Mapping[int, float],
    delta_cossim: Mapping[int, float],
    output_dir: str | Path,
    metadata: Mapping[str, Any] | None = None,
    json_filename: str = COSSIM_JSON_FILENAME,
    csv_filename: str = COSSIM_CSV_FILENAME,
) -> tuple[Path, Path]:
    """Write ``cossim_results.json`` and ``cossim_results.csv`` under ``output_dir``.

    The CSV is the long-form artifact consumed by the figures pipeline:
    one row per horizon ``k`` with columns
    ``(k, cossim_conditioned, cossim_unconditioned, delta_cossim)``.

    The JSON mirrors the same numbers in a nested layout and optionally
    embeds ``metadata`` (source paths, sample counts, encoder name, etc.)
    so downstream consumers don't have to thread provenance through code.

    Returns ``(json_path, csv_path)``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _build_results_payload(
        cossim_conditioned, cossim_unconditioned, delta_cossim, metadata
    )

    json_path = output_dir / json_filename
    csv_path = output_dir / csv_filename

    # Write JSON atomically-ish: write to temp then rename.  This avoids
    # leaving a half-written file on disk if the process is killed mid-write,
    # which would break the downstream figures pipeline.
    tmp_json = json_path.with_suffix(json_path.suffix + ".tmp")
    with tmp_json.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp_json.replace(json_path)

    tmp_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for k in sorted(cossim_conditioned):
            writer.writerow(
                [
                    k,
                    f"{float(cossim_conditioned[k]):.10f}",
                    f"{float(cossim_unconditioned[k]):.10f}",
                    f"{float(delta_cossim[k]):.10f}",
                ]
            )
    tmp_csv.replace(csv_path)

    return json_path, csv_path


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def run_latent_eval(
    z_hat_conditioned_path: str | Path,
    z_real_conditioned_path: str | Path,
    z_hat_unconditioned_path: str | Path,
    z_real_unconditioned_path: str | Path,
    output_dir: str | Path,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full CosSim evaluation pipeline and write JSON + CSV.

    Loads all four ``.pt`` tensors, computes per-horizon ``CosSim`` for
    each variant against its own adapter-projected ``z_real``, derives
    ``DeltaCosSim``, then writes the standard artifacts under
    ``output_dir``.

    Returns the in-memory results payload (mirrors the JSON file) so
    notebooks / orchestration scripts can consume the numbers without a
    round-trip through disk.

    ``extra_metadata`` is merged into the auto-generated provenance block
    (source paths, tensor shapes, timestamp).  Caller-supplied keys win
    on collision so encoder name / seed / git SHA can be threaded
    through without touching this module.
    """
    z_hat_cond = Path(z_hat_conditioned_path)
    z_real_cond = Path(z_real_conditioned_path)
    z_hat_uncond = Path(z_hat_unconditioned_path)
    z_real_uncond = Path(z_real_unconditioned_path)

    cossim_cond, (n_samples, horizon, z_dim) = _evaluate_pair(
        z_hat_cond, z_real_cond
    )
    cossim_uncond, _ = _evaluate_pair(z_hat_uncond, z_real_uncond)
    delta = compute_delta_cossim(cossim_cond, cossim_uncond)

    metadata: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "n_samples": n_samples,
        "horizon": horizon,
        "z_dim": z_dim,
        "source_paths": {
            "z_hat_conditioned": str(z_hat_cond),
            "z_real_conditioned": str(z_real_cond),
            "z_hat_unconditioned": str(z_hat_uncond),
            "z_real_unconditioned": str(z_real_uncond),
        },
    }
    if extra_metadata is not None:
        metadata.update(dict(extra_metadata))

    export_cossim_results(
        cossim_cond, cossim_uncond, delta, output_dir, metadata=metadata
    )
    return _build_results_payload(cossim_cond, cossim_uncond, delta, metadata)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latent_eval",
        description=(
            "Compute per-horizon CosSim and DeltaCosSim from exported "
            "z_hat / z_real .pt tensors and write JSON + CSV."
        ),
    )
    parser.add_argument(
        "--z-hat-conditioned",
        type=Path,
        default=None,
        help=(
            "Path to z_hat_conditioned.pt. If omitted, falls back to "
            "data.z_hat.load_z_hat('conditioned')."
        ),
    )
    parser.add_argument(
        "--z-hat-unconditioned",
        type=Path,
        default=None,
        help="Path to z_hat_unconditioned.pt (default: data.z_hat loader).",
    )
    parser.add_argument(
        "--z-real-conditioned",
        type=Path,
        default=None,
        help="Path to z_real_conditioned.pt (default: data.z_hat loader).",
    )
    parser.add_argument(
        "--z-real-unconditioned",
        type=Path,
        default=None,
        help="Path to z_real_unconditioned.pt (default: data.z_hat loader).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cossim_eval"),
        help="Directory to write cossim_results.{json,csv} (default: outputs/cossim_eval).",
    )
    parser.add_argument(
        "--encoder",
        default=None,
        help=(
            "Optional encoder name to record in the JSON metadata block "
            "(does not affect the numbers)."
        ),
    )
    return parser


def _resolve_paths_via_loader_fallback(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    """Materialise the four .pt paths, falling back to ``data.z_hat`` if needed.

    Any path not given on the CLI is resolved through
    ``data.z_hat.load_z_hat`` / ``load_z_real``, which honors the
    standard local -> HuggingFace cascade and caches the result under
    ``outputs/z_hat/``.  The rest of the pipeline therefore stays
    purely path-based regardless of where the tensors came from.
    """
    explicit = (
        args.z_hat_conditioned,
        args.z_real_conditioned,
        args.z_hat_unconditioned,
        args.z_real_unconditioned,
    )
    if all(p is not None for p in explicit):
        a, b, c, d = explicit
        return Path(a), Path(b), Path(c), Path(d)

    # Lazy import so the core module stays usable without data.z_hat on path.
    from data.z_hat import _DEFAULT_DIR as _Z_HAT_DEFAULT_DIR  # noqa: PLC0415
    from data.z_hat import load_z_hat, load_z_real  # noqa: PLC0415

    def _cached_path(kind: str, variant: str, loader) -> Path:
        cached = _Z_HAT_DEFAULT_DIR / f"{kind}_{variant}.pt"
        if not cached.exists():
            # Touch the loader to trigger the HF cascade; the cached file
            # is then guaranteed to exist on disk at the returned path.
            loader(variant)
        return cached

    def _resolve(explicit_path: Path | None, kind: str, variant: str, loader) -> Path:
        if explicit_path is not None:
            return Path(explicit_path)
        return _cached_path(kind, variant, loader)

    return (
        _resolve(args.z_hat_conditioned, "z_hat", "conditioned", load_z_hat),
        _resolve(args.z_real_conditioned, "z_real", "conditioned", load_z_real),
        _resolve(args.z_hat_unconditioned, "z_hat", "unconditioned", load_z_hat),
        _resolve(args.z_real_unconditioned, "z_real", "unconditioned", load_z_real),
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    (
        z_hat_cond,
        z_real_cond,
        z_hat_uncond,
        z_real_uncond,
    ) = _resolve_paths_via_loader_fallback(args)

    extra_metadata: dict[str, Any] = {}
    if args.encoder is not None:
        extra_metadata["encoder"] = args.encoder

    payload = run_latent_eval(
        z_hat_conditioned_path=z_hat_cond,
        z_real_conditioned_path=z_real_cond,
        z_hat_unconditioned_path=z_hat_uncond,
        z_real_unconditioned_path=z_real_uncond,
        output_dir=args.output_dir,
        extra_metadata=extra_metadata or None,
    )

    print(
        f"[latent_eval] CosSim evaluation written to {args.output_dir}/"
        f" ({COSSIM_JSON_FILENAME}, {COSSIM_CSV_FILENAME})"
    )
    horizon = payload["horizon"]
    print(
        f"  {'k':>3}  {'CosSim_cond':>12}  {'CosSim_uncond':>14}  {'Delta':>8}"
    )
    for k in range(1, horizon + 1):
        row = payload["per_horizon"][str(k)]
        print(
            f"  k={k}:  {row['cossim_conditioned']:>12.6f}  "
            f"{row['cossim_unconditioned']:>14.6f}  "
            f"{row['delta_cossim']:>8.6f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
