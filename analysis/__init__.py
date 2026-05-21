"""Statistical analysis modules.

Re-exports CLI entry points so the scripts are invokable as
``python -m analysis.<module>``.
"""

from analysis.identify_best_encoder import main as identify_best_encoder_main
from analysis.paired_tests import main as paired_tests_main

__all__ = ["identify_best_encoder_main", "paired_tests_main"]
