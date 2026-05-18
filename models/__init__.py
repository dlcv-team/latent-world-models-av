"""Trained heads that sit on top of frozen encoders.

Re-exports :class:`ActionProbe` and :func:`train_probe` (M1 task A9),
plus :class:`BCBaseline` and :func:`train_bc` (M3 task C3). The two
heads share an MLP shape (`Linear(384,256) -> GELU -> Dropout(0.1) ->
Linear(256,2)`) but differ in training regime — the probe trains for a
fixed 50 epochs while the BC baseline early-stops on validation MSE
with patience 10. See ``configs/canonical.yaml`` (``probe`` and
``bc_baseline`` blocks) for the source of truth.
"""

from models.bc_baseline import BCBaseline, train_bc
from models.probe import ActionProbe, train_probe

__all__ = ["ActionProbe", "BCBaseline", "train_bc", "train_probe"]
