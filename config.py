"""Loader and validator for ``configs/canonical.yaml``.

This module is the only sanctioned way to access shared project constants
(splits, seeds, normalization, hyperparameters). Module code MUST NOT
hardcode any of these values; call :func:`load_canonical` instead.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CANONICAL_RELPATH = "configs/canonical.yaml"

# Canonical encoder keys → human-readable display names for figures
# NOTE: "rep64" = V-JEPA2 checkpoint variant (facebook/vjepa2-vitl-fpc64-256),
# NOT our input frame count (always 16). "fpc64" = pre-trained on 64-frame clips.
ENCODER_DISPLAY = {
    # M1 full-dataset canonical keys
    "vjepa2_rep64": "V-JEPA2\n(fpc64)",
    "vjepa2_rep1": "V-JEPA2\n(fpc1)",
    "dino_vits14": "DINOv2\nViT-S/14",
    "vq_track": "VQ-VAE\nTracker",
    # P0 canonical encoder names (backward compat; remove after P0 scripts retired)
    "vjepa2": "V-JEPA2\n(fpc64)",
    "dinov2_s14": "DINOv2\nViT-S/14",
    "vqvae": "VQ-VAE\nTracker",
    # Shared names (same across both conventions)
    "clip_b32": "CLIP\nViT-B/32",
    "vit_s16": "ViT-S/16\n(supervised)",
}


def repo_root() -> Path:
    """Return the repository root, located by walking up to find ``configs/canonical.yaml``."""
    here = Path(__file__).resolve()
    for ancestor in [here, *here.parents]:
        if (ancestor / CANONICAL_RELPATH).exists():
            return ancestor
    raise RuntimeError(
        f"Could not locate {CANONICAL_RELPATH!r} above {here}; "
        "this module must be imported from inside the repo (or with the "
        "repo root on sys.path)."
    )


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CanonicalConfig:
    """Typed view over ``configs/canonical.yaml``.

    The full nested dict is preserved on :attr:`raw` so module code can
    pull less-common fields without us needing to add typed accessors for
    every leaf.
    """

    raw: dict[str, Any] = field(repr=False)
    version: str
    root: Path = field(repr=False)

    # Frequently-accessed fields surfaced for ergonomics.
    global_seed: int
    target_embedding_dim: int

    @property
    def manifest_path(self) -> Path:
        return self.root / self.raw["dataset"]["subset_manifest"]["path"]

    @property
    def manifest_sha256(self) -> str:
        return self.raw["dataset"]["subset_manifest"]["sha256"]

    @property
    def expected_split_counts(self) -> dict[str, int]:
        return dict(self.raw["dataset"]["subset_manifest"]["expected_split_counts"])

    def normalization(self, channel: str) -> dict[str, Any]:
        """Return the normalization block for ``steering`` or ``acceleration``."""
        return dict(self.raw["dataset"]["normalization"][channel])

    def encoder(self, name: str) -> dict[str, Any]:
        return dict(self.raw["encoders"][name])

    def probe(self) -> dict[str, Any]:
        return dict(self.raw["probe"])

    def bc(self) -> dict[str, Any]:
        return dict(self.raw["bc_baseline"])

    def latent_predictor(self) -> dict[str, Any]:
        return dict(self.raw["latent_predictor"])


@lru_cache(maxsize=1)
def load_canonical(config_path: Path | str | None = None) -> CanonicalConfig:
    """Load and lightly validate the canonical config.

    Parameters
    ----------
    config_path
        Override location, primarily for tests. Defaults to
        ``<repo_root>/configs/canonical.yaml``.
    """
    if config_path is None:
        root = repo_root()
        config_path = root / CANONICAL_RELPATH
    else:
        config_path = Path(config_path).resolve()
        root = _root_from_config(config_path)

    with config_path.open("r") as fh:
        raw = yaml.safe_load(fh)

    _shallow_validate(raw)

    return CanonicalConfig(
        raw=raw,
        version=str(raw["version"]),
        root=root,
        global_seed=int(raw["reproducibility"]["global_seed"]),
        target_embedding_dim=int(raw["target_embedding_dim"]),
    )


def _root_from_config(config_path: Path) -> Path:
    # configs/canonical.yaml lives one level under repo root.
    return config_path.parent.parent


def _shallow_validate(raw: dict[str, Any]) -> None:
    required_top = [
        "version",
        "dataset",
        "reproducibility",
        "probe",
        "bc_baseline",
        "latent_predictor",
        "encoders",
        "target_embedding_dim",
        "evaluation",
        "figures",
    ]
    missing = [k for k in required_top if k not in raw]
    if missing:
        raise ValueError(f"canonical.yaml missing required top-level keys: {missing!r}")

    enc = raw["encoders"]
    expected_encoders = {"vit_s16", "dinov2_s14", "clip_b32", "vqvae", "vjepa2"}
    if set(enc) != expected_encoders:
        raise ValueError(
            f"canonical.yaml encoders must be exactly {sorted(expected_encoders)!r}, "
            f"got {sorted(enc)!r}"
        )


def resolve_action_labels_path(cfg: CanonicalConfig) -> Path | None:
    """Locate the action-labels CSV.

    Resolution order:
      1. ``$NUSCENES_ACTIONS_CSV`` if set.
      2. ``<repo_root>/<relative_path>`` from canonical.yaml.

    Returns ``None`` if neither exists. Callers decide whether absence is
    fatal (training pipelines yes; the contract check tolerates it but
    warns).
    """
    env_value = os.environ.get("NUSCENES_ACTIONS_CSV")
    if env_value:
        candidate = Path(env_value).expanduser().resolve()
        if candidate.exists():
            return candidate

    rel = cfg.raw["dataset"]["action_labels"]["relative_path"]
    candidate = (cfg.root / rel).resolve()
    if candidate.exists():
        return candidate
    return None


def manifest_split(cfg: CanonicalConfig, split: str) -> list[str]:
    """Return the list of scene names for a named split."""
    with cfg.manifest_path.open("r") as fh:
        manifest = json.load(fh)
    return list(manifest["splits"][split])
