"""Trained heads that sit on top of frozen encoders.

Re-exports the P0 heads — :class:`ActionProbe` and :func:`train_probe`
(M1 task A9), :class:`BCBaseline` and :func:`train_bc` (M3 task C3) —
and the P1 latent-prediction stack — :class:`FourierActionEmbedding`
(A15), :class:`LatentPredictor` and :func:`train_latent_predictor`
(A16-A17).

The two action heads share an MLP shape
(`Linear(384,256) -> GELU -> Dropout(0.1) -> Linear(256,2)`) but
differ in training regime: the probe trains for a fixed 50 epochs
while the BC baseline early-stops on validation MSE with patience 10.
See ``configs/canonical.yaml`` (``probe`` and ``bc_baseline`` blocks)
for the source of truth.

:func:`precompute_embeddings` is the recommended fast path for the BC
sweep: run the frozen encoder once, then train the head on cached
``(embedding, action)`` tensors. See :mod:`models._train_utils` for the
contract.
"""

from models._train_utils import precompute_embeddings
from models.bc_baseline import BCBaseline, train_bc
from models.fourier_embed import FourierActionEmbedding
from models.latent_pred import LatentPredictor, train_latent_predictor
from models.probe import ActionProbe, train_probe

__all__ = [
    "ActionProbe",
    "BCBaseline",
    "FourierActionEmbedding",
    "LatentPredictor",
    "precompute_embeddings",
    "train_bc",
    "train_latent_predictor",
    "train_probe",
]
