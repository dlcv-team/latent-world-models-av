"""Evaluation harness for encoder benchmarking."""

from evaluation.latent_eval import (
    COSSIM_CSV_FILENAME,
    COSSIM_JSON_FILENAME,
    compute_delta_cossim,
    evaluate_cossim,
    export_cossim_results,
    run_latent_eval,
)
from evaluation.metrics import (
    compute_rmse,
    classify_scenes_by_scenario,
    compute_per_scenario_rmse,
    convert_steer_rmse_to_deg,
)
from evaluation.sidecars import (
    write_data_quality_report,
    write_per_scenario_rmse,
)

__all__ = [
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
