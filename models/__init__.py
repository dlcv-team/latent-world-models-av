"""Trained heads that sit on top of frozen encoders.

Re-exports :class:`ActionProbe` and :func:`train_probe`.
"""

from models.probe import ActionProbe, train_probe

__all__ = ["ActionProbe", "train_probe"]
