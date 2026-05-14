"""Statistical analysis modules.

Re-exports :func:`paired_tests.main` so the module is invokable as
``python -m analysis.paired_tests``.
"""

from analysis.paired_tests import main as paired_tests_main

__all__ = ["paired_tests_main"]
