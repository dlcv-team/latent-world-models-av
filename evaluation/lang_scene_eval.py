"""Per-scene-type DeltaCosSim evaluation for the language predictor (P2).

Compares the **language-conditioned** latent predictor against the
**action-only** predictor, per encoder, per horizon, **within each
scene-type bucket**::

    DeltaCosSim(k) = CosSim(z_hat_lang, z_real)(k) - CosSim(z_hat_action, z_real)(k)

The per-bucket numbers are computed on each bucket's *own* sequences (the
samples whose scene falls in that bucket), reusing the canonical C4
``evaluation.latent_eval._per_horizon_cossim``.  This is deliberately a
**real per-bucket evaluation**, not the global-delta proxy the P2 pilot
artifact used (``artifacts/pilot/canonical_closure/p2_lang_by_scene_type.json``
has ``method == "proxy_bucketed_from_scene_rmse_plus_global_p2_delta"`` and
carries only a single steer-RMSE per bucket); see
``test_by_scene_type_is_genuinely_per_bucket``.

Inputs
------
``predictions``
    ``{encoder: {"z_hat_lang": T, "z_hat_action": T, "z_real": T}}`` where
    each ``T`` is ``(N, horizon, z_dim)`` for the SAME ``N`` test sequences
    across encoders.
``sample_scene_names``
    Length-``N`` list giving the scene name of each sequence (the ordering
    that produced the tensors -- see ``scripts/export_z_hat.py``; emitting
    this sidecar is a recommended A18 follow-up).
``scene_to_bucket``
    ``{scene_name: bucket}`` from the C6 captions (:func:`bucket_map_from_captions`).

CLI
---
    python -m evaluation.lang_scene_eval \\
        --encoder vjepa2_rep64 \\
        --z-hat-lang   z_hat_lang.pt   --z-hat-action z_hat_action.pt \\
        --z-real       z_real.pt \\
        --captions     outputs/scene_captions.json \\
        --sample-scenes outputs/z_hat/sample_scenes.json \\
        --output-dir   outputs/p2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from evaluation.latent_eval import _load_tensor, _per_horizon_cossim

BY_SCENE_TYPE_FILENAME = "p2_lang_by_scene_type.json"
GLOBAL_FILENAME = "p2_lang_global.json"

_METHOD_BY_BUCKET = "per_bucket_delta_cossim(lang_conditioned_minus_action_only)"
_METHOD_GLOBAL = "global_delta_cossim(lang_conditioned_minus_action_only)"
_METRIC = (
    "DeltaCosSim(k) = CosSim(z_hat_lang, z_real)(k) - CosSim(z_hat_action, z_real)(k)"
)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def _per_horizon_list(z_hat: torch.Tensor, z_real: torch.Tensor) -> list[float]:
    """Per-horizon CosSim as a ``k=1..H`` ordered list (reuses C4)."""
    cossim = _per_horizon_cossim(z_hat, z_real)
    return [cossim[k] for k in sorted(cossim)]


def _eval_subset(
    z_hat_lang: torch.Tensor,
    z_hat_action: torch.Tensor,
    z_real: torch.Tensor,
    indices: Sequence[int],
) -> dict[str, list[float]]:
    """DeltaCosSim (+ components) over ``indices`` of one encoder's tensors."""
    idx = torch.as_tensor(list(indices), dtype=torch.long)
    z_lang = z_hat_lang.index_select(0, idx)
    z_act = z_hat_action.index_select(0, idx)
    z_gt = z_real.index_select(0, idx)
    cossim_lang = _per_horizon_list(z_lang, z_gt)
    cossim_action = _per_horizon_list(z_act, z_gt)
    delta = [lang - act for lang, act in zip(cossim_lang, cossim_action)]
    return {
        "delta_cossim": delta,
        "cossim_lang": cossim_lang,
        "cossim_action": cossim_action,
    }


def _validate_predictions(predictions: Mapping[str, Mapping[str, torch.Tensor]]):
    """Validate shapes; return ``(n_samples, horizon)`` shared across encoders."""
    if not predictions:
        raise ValueError("predictions is empty")
    shapes: dict[str, tuple[int, int, int]] = {}
    for encoder, tensors in predictions.items():
        for key in ("z_hat_lang", "z_hat_action", "z_real"):
            if key not in tensors:
                raise ValueError(f"encoder {encoder!r}: missing {key!r}")
        shape = tuple(tensors["z_real"].shape)
        if len(shape) != 3:
            raise ValueError(
                f"encoder {encoder!r}: tensors must be 3D (N, horizon, z_dim); "
                f"got {shape}"
            )
        if not (
            tensors["z_hat_lang"].shape
            == tensors["z_hat_action"].shape
            == tensors["z_real"].shape
        ):
            raise ValueError(
                f"encoder {encoder!r}: z_hat_lang / z_hat_action / z_real shapes differ"
            )
        shapes[encoder] = shape  # type: ignore[assignment]

    n_values = {s[0] for s in shapes.values()}
    h_values = {s[1] for s in shapes.values()}
    if len(n_values) != 1 or len(h_values) != 1:
        raise ValueError(f"encoders disagree on (N, horizon): {shapes}")
    return n_values.pop(), h_values.pop()


def evaluate_global(
    predictions: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    """Flat (all-sequences) DeltaCosSim per encoder -- the ``p2_lang_global`` ref."""
    n_samples, horizon = _validate_predictions(predictions)
    all_indices = list(range(n_samples))
    per_encoder = {
        encoder: _eval_subset(
            tensors["z_hat_lang"],
            tensors["z_hat_action"],
            tensors["z_real"],
            all_indices,
        )
        for encoder, tensors in predictions.items()
    }
    return {
        "method": _METHOD_GLOBAL,
        "metric": _METRIC,
        "horizon": horizon,
        "n_samples": n_samples,
        "encoders": list(predictions.keys()),
        "per_encoder": per_encoder,
    }


def evaluate_by_scene_type(
    predictions: Mapping[str, Mapping[str, torch.Tensor]],
    sample_scene_names: Sequence[str],
    scene_to_bucket: Mapping[str, str],
    buckets: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Per-bucket DeltaCosSim, computed within each bucket's own sequences.

    Parameters
    ----------
    buckets
        Optional ordered bucket whitelist (e.g. ``scenario_buckets_p2``).
        Buckets with no sequences are omitted.  When ``None``, present
        buckets are emitted in sorted order.
    """
    n_samples, horizon = _validate_predictions(predictions)
    if len(sample_scene_names) != n_samples:
        raise ValueError(
            f"sample_scene_names length {len(sample_scene_names)} != "
            f"n_samples {n_samples}"
        )

    # Per-sequence bucket (None if the scene has no caption-derived bucket).
    sample_buckets = [scene_to_bucket.get(name) for name in sample_scene_names]
    present = [b for b in dict.fromkeys(sample_buckets) if b is not None]
    ordered = (
        sorted(present) if buckets is None else [b for b in buckets if b in present]
    )

    by_scene_type = []
    for bucket in ordered:
        indices = [i for i, b in enumerate(sample_buckets) if b == bucket]
        if not indices:
            continue
        distinct_scenes = {sample_scene_names[i] for i in indices}
        per_encoder = {
            encoder: _eval_subset(
                tensors["z_hat_lang"],
                tensors["z_hat_action"],
                tensors["z_real"],
                indices,
            )
            for encoder, tensors in predictions.items()
        }
        by_scene_type.append(
            {
                "bucket": bucket,
                "n_scenes": len(distinct_scenes),
                "n_samples": len(indices),
                "per_encoder": per_encoder,
            }
        )

    return {
        "method": _METHOD_BY_BUCKET,
        "metric": _METRIC,
        "horizon": horizon,
        "encoders": list(predictions.keys()),
        "buckets": [entry["bucket"] for entry in by_scene_type],
        "by_scene_type": by_scene_type,
    }


def bucket_map_from_captions(captions_payload: Mapping[str, Any]) -> dict[str, str]:
    """Build ``{scene_name: bucket}`` from a C6 ``scene_captions.json`` payload."""
    return {c["scene_name"]: c["bucket"] for c in captions_payload["captions"]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _canonical_p2_buckets() -> Optional[list[str]]:
    """Canonical P2 bucket order, or ``None`` if the config can't be read.

    The fallback only affects bucket *ordering* (present buckets are then
    emitted in sorted order); membership is always derived from the captions.
    """
    try:
        from config import load_canonical

        return list(load_canonical().raw["evaluation"]["scenario_buckets_p2"])
    except Exception as exc:
        print(
            f"[lang_scene_eval] warning: canonical config unavailable ({exc}); "
            f"buckets will be emitted in sorted order.",
            file=sys.stderr,
        )
        return None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lang_scene_eval",
        description=(
            "Per-scene-type DeltaCosSim (language-conditioned - action-only) "
            "for the P2 language predictor."
        ),
    )
    parser.add_argument("--encoder", default="vjepa2_rep64")
    parser.add_argument("--z-hat-lang", type=Path, required=True)
    parser.add_argument("--z-hat-action", type=Path, required=True)
    parser.add_argument("--z-real", type=Path, required=True)
    parser.add_argument(
        "--captions",
        type=Path,
        required=True,
        help="C6 scene_captions.json (provides scene -> bucket).",
    )
    parser.add_argument(
        "--sample-scenes",
        type=Path,
        required=True,
        help="JSON list of scene name per test sequence (length N).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/p2"))
    parser.add_argument(
        "--status",
        default=None,
        help="Optional status stamp (e.g. 'smoke_demo') added to both JSONs.",
    )
    parser.add_argument(
        "--warning",
        default=None,
        help="Optional warning string added to both JSONs.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    predictions = {
        args.encoder: {
            "z_hat_lang": _load_tensor(args.z_hat_lang, "z_hat_lang"),
            "z_hat_action": _load_tensor(args.z_hat_action, "z_hat_action"),
            "z_real": _load_tensor(args.z_real, "z_real"),
        }
    }
    captions_payload = json.loads(Path(args.captions).read_text())
    scene_to_bucket = bucket_map_from_captions(captions_payload)
    sample_scene_names = json.loads(Path(args.sample_scenes).read_text())

    by_type = evaluate_by_scene_type(
        predictions,
        sample_scene_names,
        scene_to_bucket,
        buckets=_canonical_p2_buckets(),
    )
    global_payload = evaluate_global(predictions)

    # Machine-readable provenance: record exactly which inputs produced these
    # numbers so an artifact can never be mistaken for a different run.
    inputs = {
        "encoder": args.encoder,
        "z_hat_lang": str(args.z_hat_lang),
        "z_hat_action": str(args.z_hat_action),
        "z_real": str(args.z_real),
        "captions": str(args.captions),
        "sample_scenes": str(args.sample_scenes),
    }
    stamp: dict[str, Any] = {"inputs": inputs}
    if args.status is not None:
        stamp["status"] = args.status
    if args.warning is not None:
        stamp["warning"] = args.warning
    by_type.update(stamp)
    global_payload.update(stamp)

    _write_json(by_type, args.output_dir / BY_SCENE_TYPE_FILENAME)
    _write_json(global_payload, args.output_dir / GLOBAL_FILENAME)

    print(
        f"[lang_scene_eval] encoder={args.encoder}  "
        f"buckets={by_type['buckets']}  -> {args.output_dir}/"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
