"""Trained heads that sit on top of frozen encoders.

Re-exports :class:`ActionProbe`, :func:`train_probe`,
:class:`FourierActionEmbedding`, and :class:`LatentPredictor`.
"""

from models.fourier_embed import FourierActionEmbedding
from models.latent_pred import LatentPredictor
from models.probe import ActionProbe, train_probe

__all__ = ["ActionProbe", "FourierActionEmbedding", "LatentPredictor", "train_probe"]
