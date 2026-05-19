"""Trained heads that sit on top of frozen encoders.

Re-exports :class:`ActionProbe`, :func:`train_probe`, and
:class:`FourierActionEmbedding`.
"""

from models.fourier_embed import FourierActionEmbedding
from models.probe import ActionProbe, train_probe

__all__ = ["ActionProbe", "FourierActionEmbedding", "train_probe"]
