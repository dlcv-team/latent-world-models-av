"""CLI entry points for training pipelines.

Invokable as ``python -m training.train_probe`` or
``python -m training.train_latent_predictor``.
"""

from training.train_latent_predictor import main as train_lp_main
from training.train_probe import main as train_probe_main

__all__ = ["train_lp_main", "train_probe_main"]
