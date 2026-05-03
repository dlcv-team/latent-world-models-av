"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import load_canonical, repo_root


@pytest.fixture(scope="session")
def cfg():
    return load_canonical()


@pytest.fixture(scope="session")
def manifest(cfg):
    with cfg.manifest_path.open("r") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def pilot_baselines() -> dict:
    path = repo_root() / "tests" / "data" / "pilot_baselines.json"
    with path.open("r") as fh:
        return json.load(fh)
