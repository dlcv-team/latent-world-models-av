"""Internal training-loop helpers shared by ``probe.py`` and ``bc_baseline.py``.

Both heads share the same per-epoch contract: take a frozen encoder + a
small MLP head, run the encoder forward on either ``{"image", "actions"}``
dict batches (what :class:`data.dataset.NuScenesFrameDataset` emits) or
``(image, actions)`` tuple batches, and step the optimizer in train mode
or accumulate loss only in eval mode.

This module is private. Public callers should use
:func:`models.probe.train_probe`, :func:`models.bc_baseline.train_bc`, or
:func:`models.precompute_embeddings`.

Embedding pre-computation
-------------------------
:func:`precompute_embeddings` is the recommended path when the encoder is
fully frozen and the head will be trained for many epochs (e.g. the BC
baseline's early-stopping schedule). It runs the encoder ONCE, caches
``(embedding, action)`` pairs in a :class:`torch.utils.data.TensorDataset`,
and lets subsequent epochs train on cached tensors at memory-bandwidth
speed instead of re-running the backbone. Caveats:

* The encoder must be effectively frozen for the duration of training.
  For wrappers with a trainable projection adapter (CLIP, V-JEPA2, VQ-VAE
  outside of fallback mode), pre-computing freezes the adapter too — if
  you want to train the adapter jointly with the head you must keep using
  the live-encoder path.
* The cached embeddings live in host memory (or whichever ``device`` the
  caller pinned). Sizing: 384-d float32 × N samples ≈ 1.5 KB/sample, so a
  full nuScenes-subset run (~30 K samples) is ≈ 45 MB.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import TensorDataset

_BatchLike = Union[
    Mapping[str, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]


def _extract_image_actions(
    batch: _BatchLike,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pull ``(image, actions)`` from either dict-style or tuple-style batches.

    Dict-style is what ``data.dataset.NuScenesFrameDataset`` emits
    (``{"image": ..., "actions": ..., "sample_token": ..., ...}``);
    tuple-style is what synthetic test loaders typically yield. Supporting
    both keeps the training loop decoupled from the dataset's exact return
    schema.
    """
    if isinstance(batch, Mapping):
        return batch["image"], batch["actions"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(
        f"Unsupported batch type: {type(batch).__name__}. "
        "Expected a Mapping with 'image' and 'actions' keys, or a "
        "(image, actions) tuple."
    )


def _epoch_loss(
    encoder: nn.Module,
    head: nn.Module,
    loader: Iterable[_BatchLike],
    optimizer: Optional[torch.optim.Optimizer],
    loss_fn: nn.Module,
    device: Optional[torch.device],
    train: bool,
) -> float:
    """Run one pass over ``loader``; step the optimizer if ``train`` is True.

    ``head`` is whatever sits on top of the encoder embedding — the action
    probe, the BC baseline, or anything else that maps ``(B, embed_dim)``
    to a regression target. The encoder is always called in eval mode (the
    real wrappers pin their backbone to eval inside their ``train()``
    override; this is defensive for synthetic stubs).

    Returns the sample-weighted mean loss for the epoch. ``optimizer`` may
    be ``None`` for eval passes.
    """
    if train:
        head.train()
    else:
        head.eval()

    encoder.eval()

    total_loss_sum = 0.0
    total_n = 0

    grad_ctx: Any = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            image, action = _extract_image_actions(batch)
            if device is not None:
                image = image.to(device)
                action = action.to(device)

            embedding = encoder(image)
            prediction = head(embedding)
            loss = loss_fn(prediction, action)

            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = action.shape[0]
            total_loss_sum += float(loss.detach().item()) * batch_size
            total_n += batch_size

    if total_n == 0:
        raise RuntimeError(
            "loader yielded zero samples; cannot compute mean epoch loss."
        )
    return total_loss_sum / total_n


def precompute_embeddings(
    encoder: nn.Module,
    loader: Iterable[_BatchLike],
    *,
    device: Optional[torch.device] = None,
) -> TensorDataset:
    """Run ``encoder`` once over ``loader`` and cache ``(embedding, action)`` pairs.

    The returned :class:`torch.utils.data.TensorDataset` yields
    ``(embedding, action)`` tuples, which both :func:`train_probe` and
    :func:`train_bc` already accept (the tuple-batch branch of
    :func:`_extract_image_actions`). Wrap it in a fresh
    :class:`torch.utils.data.DataLoader` and pass an
    :class:`torch.nn.Identity` (or any pass-through ``nn.Module``) as the
    encoder for the training call.

    Both ``embeddings`` and ``actions`` are detached and moved to CPU
    before being concatenated so the cache survives even if the encoder
    lived on GPU during this single forward pass. Pinning the result back
    onto a device (``ds.tensors = (ds.tensors[0].to(dev), ...)``) is left
    to the caller — for typical small-MLP heads, the per-batch
    ``.to(device)`` in the training loop is plenty fast.

    Parameters
    ----------
    encoder
        Any module that maps an image batch to ``(B, embed_dim)``. Must be
        frozen — see the module docstring for the adapter caveat.
    loader
        Iterable yielding dict-style or tuple-style batches.
    device
        If provided, batches are moved to ``device`` before the encoder
        forward.

    Returns
    -------
    torch.utils.data.TensorDataset
        Yields ``(embedding, action)`` per sample.
    """
    encoder.eval()
    embeddings: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            image, action = _extract_image_actions(batch)
            if device is not None:
                image = image.to(device)
            embedding = encoder(image)
            embeddings.append(embedding.detach().to("cpu"))
            actions.append(action.detach().to("cpu"))

    if not embeddings:
        raise RuntimeError(
            "loader yielded zero samples; cannot pre-compute embeddings."
        )

    return TensorDataset(
        torch.cat(embeddings, dim=0),
        torch.cat(actions, dim=0),
    )
