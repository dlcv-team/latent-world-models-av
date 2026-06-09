"""Evaluation harness for encoder benchmarking.

Submodules are imported **lazily** (PEP 562 ``__getattr__``) so that
lightweight consumers -- CosSim / DeltaCosSim evaluation
(:mod:`evaluation.latent_eval`, :mod:`evaluation.lang_scene_eval`) and
:mod:`evaluation.metrics` -- can be imported with only ``torch`` /
``numpy`` / ``pandas`` installed, without dragging in the heavy
attribution stack (``pytorch_grad_cam``, ``transformers``, ``opencv``)
that ``gradcam`` / ``attribution_grid`` need.

All previously eager top-level names remain importable
(``from evaluation import AttributionPipeline``); they are just resolved
on first access instead of at package import time, so importing one
submodule no longer fails when an unrelated submodule's dependency is
missing.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Public attribute -> submodule that defines it.
_EXPORTS: dict[str, str] = {
    "AttributionGridGenerator": "attribution_grid",
    "AttributionPipeline": "gradcam",
    "COSSIM_CSV_FILENAME": "latent_eval",
    "COSSIM_JSON_FILENAME": "latent_eval",
    "compute_delta_cossim": "latent_eval",
    "evaluate_cossim": "latent_eval",
    "export_cossim_results": "latent_eval",
    "run_latent_eval": "latent_eval",
    "bootstrap_ratio_ci": "metrics",
    "classify_scenes_by_environment": "metrics",
    "classify_scenes_by_scenario": "metrics",
    "compute_per_scenario_rmse": "metrics",
    "compute_rmse": "metrics",
    "compute_robustness_ratios": "metrics",
    "convert_steer_rmse_to_deg": "metrics",
    "write_data_quality_report": "sidecars",
    "write_per_scenario_rmse": "sidecars",
}

# Submodules that may be accessed as ``evaluation.<name>`` directly.
_SUBMODULES = {
    "attribution_grid",
    "gradcam",
    "latent_eval",
    "lang_scene_eval",
    "metrics",
    "perturbation",
    "sidecars",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public name or submodule on first access (PEP 562)."""
    if name in _EXPORTS:
        module = import_module(f"{__name__}.{_EXPORTS[name]}")
        return getattr(module, name)
    if name in _SUBMODULES:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS) | _SUBMODULES)
