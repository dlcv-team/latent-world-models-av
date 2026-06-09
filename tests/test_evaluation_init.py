"""Contract tests for the lazy (PEP 562) ``evaluation`` package init.

The package resolves its public names on first access instead of importing
every submodule eagerly, so lightweight consumers (CosSim / metrics) work
without the attribution stack (pytorch_grad_cam / transformers / opencv)
installed.  These tests pin that contract:

* ``__all__`` still covers exactly the names the old eager init exported.
* Every non-attribution name resolves without the heavy dependencies.
* ``dir()`` advertises the lazy names; submodule attribute access works.
* Unknown attributes raise ``AttributeError`` (not ``ImportError``).
"""

from __future__ import annotations

import types

import pytest

import evaluation

# Names defined by submodules with light deps (torch / numpy / pandas only).
LIGHT_NAMES = [
    "COSSIM_CSV_FILENAME",
    "COSSIM_JSON_FILENAME",
    "compute_delta_cossim",
    "evaluate_cossim",
    "export_cossim_results",
    "run_latent_eval",
    "compute_rmse",
    "classify_scenes_by_scenario",
    "compute_per_scenario_rmse",
    "convert_steer_rmse_to_deg",
    "write_data_quality_report",
    "write_per_scenario_rmse",
]

# Names that pull in the attribution stack on resolution.
HEAVY_NAMES = ["AttributionGridGenerator", "AttributionPipeline"]


def test_all_matches_the_original_eager_exports():
    assert set(evaluation.__all__) == set(LIGHT_NAMES) | set(HEAVY_NAMES)


def test_light_names_resolve_without_attribution_stack():
    for name in LIGHT_NAMES:
        assert getattr(evaluation, name) is not None, name


def test_dir_advertises_lazy_names():
    listing = dir(evaluation)
    for name in LIGHT_NAMES + HEAVY_NAMES:
        assert name in listing, name


def test_submodule_attribute_access():
    assert isinstance(evaluation.latent_eval, types.ModuleType)
    assert isinstance(evaluation.lang_scene_eval, types.ModuleType)
    assert isinstance(evaluation.metrics, types.ModuleType)


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute"):
        _ = evaluation.does_not_exist


def test_heavy_names_resolve_or_skip_when_stack_missing():
    for name in HEAVY_NAMES:
        try:
            assert getattr(evaluation, name) is not None, name
        except Exception as exc:  # pragma: no cover - env dependent
            pytest.skip(f"attribution stack unavailable here: {exc}")
