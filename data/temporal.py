"""Temporal sequence dataset for latent predictor training (P1).

Wraps pre-computed embedding arrays into ``(z_t, action_t, z_future)``
tuples suitable for training a :class:`~models.latent_pred.LatentPredictor`.
Each sample is a sliding window of ``1 + horizon`` consecutive frames
from the same scene, guaranteeing no scene-boundary crossings.

The dataset operates entirely on pre-computed embeddings (no encoder
forward passes), making training ~100x faster than running a frozen
encoder live each epoch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from data.embeddings import load_encoder_embedding

DEFAULT_HORIZON = 4


class TemporalEmbeddingDataset(Dataset):
    """Yields ``(z_t, action, z_future)`` from pre-computed embeddings.

    Each item is a dict with:

    - ``z_t``: ``(embed_dim,)`` — embedding at frame *t*
    - ``action``: ``(2,)`` — ``(steer_norm, accel_norm)`` at frame *t*
    - ``z_future``: ``(horizon, embed_dim)`` — embeddings at
      frames *t+1* through *t+horizon*

    Parameters
    ----------
    embeddings
        ``(N, embed_dim)`` array of encoder outputs.
    steer_norms, accel_norms
        ``(N,)`` arrays of normalized action labels.
    scene_names
        ``(N,)`` array of scene identifiers per frame.
    timestamps_us
        ``(N,)`` array of microsecond timestamps per frame.
    splits
        ``(N,)`` array of split labels (``'train'``, ``'val'``,
        ``'test'``).
    split
        Which split to use.
    horizon
        Number of future frames to predict. Defaults to 4.

    Notes
    -----
    The constructor groups frames by scene, sorts within each scene by
    timestamp, and builds a flat index of valid starting positions where
    ``frame_offset + horizon < scene_length``. Scene boundaries are
    never crossed.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        steer_norms: np.ndarray,
        accel_norms: np.ndarray,
        scene_names: np.ndarray,
        timestamps_us: np.ndarray,
        splits: np.ndarray,
        split: str = "train",
        horizon: int = DEFAULT_HORIZON,
    ) -> None:
        self.horizon = int(horizon)
        self.split = split

        # Filter to requested split
        split_mask = splits == split
        if not split_mask.any():
            raise ValueError(
                f"No samples found for split {split!r}. "
                f"Available splits: {np.unique(splits).tolist()}"
            )

        self._embeddings = torch.as_tensor(
            embeddings[split_mask], dtype=torch.float32
        )
        self._steer = torch.as_tensor(
            steer_norms[split_mask], dtype=torch.float32
        )
        self._accel = torch.as_tensor(
            accel_norms[split_mask], dtype=torch.float32
        )
        filtered_scenes = scene_names[split_mask]
        filtered_ts = timestamps_us[split_mask]

        # Build temporal index: list of (start_idx_in_filtered, scene_len)
        # for valid sliding-window positions.
        self._valid_indices: list[int] = []
        self._scene_starts: list[int] = []
        self._scene_lengths: list[int] = []
        self._timestamps = torch.as_tensor(filtered_ts, dtype=torch.long)

        # Group by scene, preserving temporal order
        unique_scenes = np.unique(filtered_scenes)
        for scene in unique_scenes:
            scene_mask = filtered_scenes == scene
            scene_indices = np.where(scene_mask)[0]

            # Sort by timestamp within scene (should already be sorted,
            # but verify)
            ts_order = np.argsort(filtered_ts[scene_indices])
            scene_indices = scene_indices[ts_order]

            scene_len = len(scene_indices)
            scene_start = int(scene_indices[0])

            # Verify frames are contiguous in the filtered array
            # (they should be since we filtered by split and scenes
            # don't span splits)
            expected_indices = np.arange(scene_start, scene_start + scene_len)
            if not np.array_equal(scene_indices, expected_indices):
                # Non-contiguous — store mapping for this scene
                # This shouldn't happen with well-formed data, but handle
                # it gracefully by using the scene_indices directly
                for offset in range(scene_len - self.horizon):
                    # Store the actual global index of this window start
                    self._valid_indices.append(int(scene_indices[offset]))
                self._scene_starts.append(scene_start)
                self._scene_lengths.append(scene_len)
                continue

            self._scene_starts.append(scene_start)
            self._scene_lengths.append(scene_len)

            # Each valid position: frame t such that t+horizon is still
            # within the scene
            for offset in range(scene_len - self.horizon):
                self._valid_indices.append(scene_start + offset)

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self._valid_indices[idx]
        return {
            "z_t": self._embeddings[start],
            "action": torch.stack(
                [self._steer[start], self._accel[start]]
            ),
            "z_future": self._embeddings[start + 1 : start + 1 + self.horizon],
        }

    @property
    def embed_dim(self) -> int:
        """Embedding dimensionality."""
        return int(self._embeddings.shape[1])

    @property
    def timestamps(self) -> torch.Tensor:
        """Microsecond timestamps for all frames in this split."""
        return self._timestamps

    @classmethod
    def from_encoder(
        cls,
        encoder_name: str,
        split: str = "train",
        horizon: int = DEFAULT_HORIZON,
        directory: Optional[str] = None,
    ) -> "TemporalEmbeddingDataset":
        """Construct from a single encoder's pre-computed embeddings.

        Parameters
        ----------
        encoder_name
            One of the encoder names in
            :data:`data.embeddings.ENCODER_NAMES`.
        split
            ``'train'``, ``'val'``, or ``'test'``.
        horizon
            Number of future frames to predict.
        directory
            Override embedding directory (for testing).
        """
        from pathlib import Path

        d = Path(directory) if directory else None
        data = load_encoder_embedding(encoder_name, directory=d)
        return cls(
            embeddings=data["embeddings"],
            steer_norms=data["steer_norms"],
            accel_norms=data["accel_norms"],
            scene_names=data["scene_names"],
            timestamps_us=data["timestamps_us"],
            splits=data["splits"],
            split=split,
            horizon=horizon,
        )
