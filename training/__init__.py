"""CLI entry points for training pipelines.

Re-exports :func:`train_probe.main` so the module is invokable as
``python -m training.train_probe``.
"""

from training.train_probe import main as train_probe_main

__all__ = ["train_probe_main"]
