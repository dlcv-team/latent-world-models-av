"""Evaluation harness for encoder benchmarking."""

from evaluation.attribution_grid import AttributionGridGenerator
from evaluation.gradcam import AttributionPipeline
from evaluation.latent_eval import (
    COSSIM_CSV_FILENAME,
    COSSIM_JSON_FILENAME,
    compute_delta_cossim,
    evaluate_cossim,
    export_cossim_results,
    run_latent_eval,
)
from evaluation.metrics import (
    classify_scenes_by_scenario,
    compute_per_scenario_rmse,
    compute_rmse,
    convert_steer_rmse_to_deg,
)

__all__ = [
    "AttributionGridGenerator",
    "AttributionPipeline",
    "COSSIM_CSV_FILENAME",
    "COSSIM_JSON_FILENAME",
    "classify_scenes_by_scenario",
    "compute_delta_cossim",
    "compute_per_scenario_rmse",
    "compute_rmse",
    "convert_steer_rmse_to_deg",
    "evaluate_cossim",
    "export_cossim_results",
    "run_latent_eval",
]
