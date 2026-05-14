"""Abstract base for the five frozen-encoder wrappers under benchmark.

A wrapper combines a frozen pretrained backbone with an OPTIONAL trainable
projection adapter that brings the backbone's native output dimension to the
project-wide standard ``target_dim`` (384; the canonical value lives in
``configs/canonical.yaml::target_embedding_dim``).

Subclasses implement only:

* :meth:`_load` — instantiate the backbone and assign it to ``self.backbone``.
* :meth:`_encode` — run the backbone forward and return ``(B, native_dim)``.

The base class handles freezing the backbone, locking it into eval mode (so
dropout / BatchNorm are deterministic), wiring up the adapter when a
projection is needed, and the no-grad boundary in :meth:`forward`.

The adapter is the ONLY part of an encoder wrapper with gradient. It is
trained jointly with the probe head; the optimizer should be constructed
from ``probe.parameters() + list(encoder.trainable_parameters())``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import torch
from torch import nn


class BaseEncoderWrapper(nn.Module, ABC):
    """Frozen pretrained encoder + optional trainable projection adapter.

    Parameters
    ----------
    native_dim
        Output dimension of the pretrained backbone, before any projection.
        Must match what :meth:`_encode` returns.
    needs_projection
        If True, attach a trainable
        ``nn.Linear(native_dim, target_dim, bias=False)`` adapter.
        If False, the backbone's output is returned directly and
        ``native_dim`` MUST equal ``target_dim`` (otherwise ``ValueError``).
    target_dim
        The project-wide standardized embedding dimension. Defaults to 384;
        the canonical value lives in
        ``configs/canonical.yaml::target_embedding_dim`` and is what every
        downstream module (probe, latent predictor, BC baseline) assumes.

    Notes
    -----
    Subclasses must call ``super().__init__(...)`` with the keyword
    arguments above. The base class will then call :meth:`_load` (which the
    subclass implements) before freezing the backbone.

    The projection adapter uses ``bias=False`` so adapter parameter counts
    are directly comparable across encoders with different native
    dimensions.
    """

    backbone: nn.Module  # populated by subclass _load(); declared for type checkers
    adapter: nn.Module

    def __init__(
        self,
        *,
        native_dim: int,
        needs_projection: bool,
        target_dim: int = 384,
    ) -> None:
        super().__init__()

        if not needs_projection and native_dim != target_dim:
            raise ValueError(
                f"needs_projection=False but native_dim ({native_dim}) "
                f"!= target_dim ({target_dim}). Set needs_projection=True, "
                "or fix the dimensions in the encoder spec."
            )

        self.native_dim = int(native_dim)
        self.target_dim = int(target_dim)
        self.needs_projection = bool(needs_projection)

        # Subclass must populate self.backbone in _load().
        self._load()
        if not hasattr(self, "backbone"):
            raise RuntimeError(
                f"{type(self).__name__}._load() must assign self.backbone "
                "before returning."
            )

        self._freeze_backbone()

        if needs_projection:
            self.adapter = nn.Linear(self.native_dim, self.target_dim, bias=False)
        else:
            self.adapter = nn.Identity()

    # ---- subclass hooks ---------------------------------------------------

    @abstractmethod
    def _load(self) -> None:
        """Instantiate the pretrained backbone and assign it to ``self.backbone``.

        Called once during ``__init__`` BEFORE the backbone is frozen, so
        subclasses may perform any one-time setup that needs grad-enabled
        parameters (e.g. loading pretrained weights, building auxiliary
        buffers). Anything assigned to ``self.backbone`` will then have its
        parameters frozen and put into eval mode.
        """

    @abstractmethod
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone forward pass.

        Parameters
        ----------
        x
            Input batch. Single-frame encoders accept ``(B, 3, H, W)``;
            clip-mode encoders accept ``(B, T, 3, H, W)``. The base class
            does not constrain the shape — it is the subclass's contract
            with its caller.

        Returns
        -------
        torch.Tensor
            Backbone embedding, shape ``(B, native_dim)``.
        """

    # ---- frozen-encoder mechanics ----------------------------------------

    def _freeze_backbone(self) -> None:
        """Set ``requires_grad=False`` on every backbone parameter and put it in eval."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True) -> "BaseEncoderWrapper":  # type: ignore[override]
        """Override so the backbone stays in eval mode regardless of wrapper mode.

        Calling ``wrapper.train()`` flips the wrapper itself (and the
        adapter) into train mode for downstream consistency, but the
        pretrained backbone is forced back into eval so its dropout /
        BatchNorm stay deterministic during probe training.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    # ---- public surface ---------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch and project to ``target_dim``.

        The backbone runs inside ``torch.no_grad()`` so its activations
        never enter the autograd graph. The result is ``detach()``-ed
        defensively (in case a subclass internally re-enables grad), and
        the adapter — whose parameters DO require grad — is then applied.

        Returns
        -------
        torch.Tensor
            Shape ``(B, target_dim)``. ``requires_grad`` is True iff
            ``needs_projection`` is True (the Identity branch returns a
            detached tensor with no gradient path).
        """
        with torch.no_grad():
            z = self._encode(x)
        z = z.detach()
        return self.adapter(z)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only the adapter parameters; the optimizer sees nothing else.

        Returns an empty iterator when no adapter is attached (i.e.
        ``needs_projection=False``).
        """
        if isinstance(self.adapter, nn.Identity):
            return iter(())
        return self.adapter.parameters()
