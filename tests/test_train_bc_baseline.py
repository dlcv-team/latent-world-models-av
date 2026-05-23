"""Tests for :mod:`scripts.train_bc_baseline` CLI.

Exercises the end-to-end ``main()`` entry point on synthetic embeddings
(tiny dims, few samples) so no real encoder or nuScenes data is needed.
Covers: arg parsing, training loop, early stopping, checkpoint schema,
summary CSV schema, provenance JSON, and train_log.csv sidecar.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

# We import main() and mock the embedding loader so tests are self-contained.
from scripts.train_bc_baseline import main, NATIVE_DIMS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Use a small encoder that exists in NATIVE_DIMS with native_dim == 384
# (no adapter needed), keeping tests fast.
_ENCODER = "vit_s16"
_NATIVE_DIM = NATIVE_DIMS[_ENCODER]
assert _NATIVE_DIM == 384, "test assumes vit_s16 has native_dim 384"


def _make_synthetic_embeddings(
    n_train: int = 64,
    n_val: int = 16,
    n_test: int = 16,
    dim: int = _NATIVE_DIM,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create a synthetic embedding dict matching ``load_encoder_embedding``."""
    rng = np.random.RandomState(seed)
    n_total = n_train + n_val + n_test
    embeddings = rng.randn(n_total, dim).astype(np.float32)
    splits = np.array(
        ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    )
    steer_norms = rng.randn(n_total).astype(np.float32)
    accel_norms = rng.randn(n_total).astype(np.float32)
    # Create 3 distinct scenes per split for scene counting
    scene_names = np.array(
        [f"scene_train_{i % 3}" for i in range(n_train)]
        + [f"scene_val_{i % 2}" for i in range(n_val)]
        + [f"scene_test_{i % 4}" for i in range(n_test)]
    )
    return {
        "embeddings": embeddings,
        "splits": splits,
        "steer_norms": steer_norms,
        "accel_norms": accel_norms,
        "scene_names": scene_names,
    }


@pytest.fixture()
def synthetic_data():
    return _make_synthetic_embeddings()


@pytest.fixture()
def mock_loader(synthetic_data):
    """Patch ``load_encoder_embedding`` to return synthetic data."""
    with mock.patch(
        "scripts.train_bc_baseline.load_encoder_embedding",
        return_value=synthetic_data,
    ) as m:
        yield m


def _run_main(tmp_path: Path, extra_args: list[str] | None = None) -> int:
    """Run main() with minimal args pointing output to tmp_path."""
    args = [
        "--encoder", _ENCODER,
        "--output-root", str(tmp_path / "bc_out"),
        "--epochs", "3",
        "--batch-size", "16",
        "--seed", "0",
    ]
    if extra_args:
        args.extend(extra_args)
    return main(args)


def _out_dir(tmp_path: Path) -> Path:
    return tmp_path / "bc_out" / _ENCODER / "seed_0"


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------


def test_main_runs_and_returns_zero(tmp_path, mock_loader):
    """Smoke test: main() completes without error."""
    rc = _run_main(tmp_path)
    assert rc == 0


def test_main_creates_all_output_files(tmp_path, mock_loader):
    """All four output files must be created."""
    _run_main(tmp_path)
    out = _out_dir(tmp_path)
    assert (out / "checkpoint.pt").exists()
    assert (out / "provenance.json").exists()
    assert (out / "bc_baseline_row.csv").exists()
    assert (out / "train_log.csv").exists()


def test_main_no_parent_summary_csv(tmp_path, mock_loader):
    """Parent-level summary CSV should NOT exist (removed per review)."""
    _run_main(tmp_path)
    parent_csv = tmp_path / "bc_out" / "bc_baseline_row.csv"
    assert not parent_csv.exists()


# ---------------------------------------------------------------------------
# Checkpoint schema
# ---------------------------------------------------------------------------


def test_checkpoint_schema(tmp_path, mock_loader):
    """Checkpoint must contain all expected keys with correct types."""
    _run_main(tmp_path)
    ckpt = torch.load(_out_dir(tmp_path) / "checkpoint.pt", weights_only=False)

    required_keys = {
        "bc_state_dict",
        "adapter_state_dict",
        "encoder_name",
        "seed",
        "best_epoch",
        "epochs_run",
        "stopped_early",
        "best_val_loss",
        "steer_rmse",
        "accel_rmse",
        "history",
    }
    assert required_keys <= set(ckpt.keys())
    assert ckpt["encoder_name"] == _ENCODER
    assert ckpt["seed"] == 0
    assert isinstance(ckpt["steer_rmse"], float)
    assert isinstance(ckpt["accel_rmse"], float)
    assert ckpt["steer_rmse"] > 0
    assert ckpt["accel_rmse"] > 0
    assert isinstance(ckpt["history"]["train_loss"], list)
    assert isinstance(ckpt["history"]["val_loss"], list)


def test_checkpoint_adapter_state_none_when_identity(tmp_path, mock_loader):
    """vit_s16 has native_dim == 384 == target_dim, so no adapter needed."""
    _run_main(tmp_path)
    ckpt = torch.load(_out_dir(tmp_path) / "checkpoint.pt", weights_only=False)
    assert ckpt["adapter_state_dict"] is None


# ---------------------------------------------------------------------------
# Summary CSV schema
# ---------------------------------------------------------------------------


def test_summary_csv_schema(tmp_path, mock_loader):
    """bc_baseline_row.csv must have exactly the expected columns and one data row."""
    _run_main(tmp_path)
    csv_path = _out_dir(tmp_path) / "bc_baseline_row.csv"

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    expected_cols = {
        "encoder", "lr", "wd", "batch", "epochs_run",
        "early_stop_epoch", "seed", "steer_rmse", "accel_rmse",
        "n_test_scenes",
    }
    assert expected_cols == set(row.keys())
    assert row["encoder"] == _ENCODER
    assert int(row["seed"]) == 0
    assert float(row["steer_rmse"]) > 0
    assert float(row["accel_rmse"]) > 0


def test_summary_csv_n_test_scenes_is_correct(tmp_path, mock_loader, synthetic_data):
    """n_test_scenes must match the number of unique scenes in the test split."""
    _run_main(tmp_path)
    csv_path = _out_dir(tmp_path) / "bc_baseline_row.csv"

    with csv_path.open() as f:
        row = list(csv.DictReader(f))[0]

    test_mask = synthetic_data["splits"] == "test"
    expected_scenes = len(set(synthetic_data["scene_names"][test_mask]))
    assert int(row["n_test_scenes"]) == expected_scenes


# ---------------------------------------------------------------------------
# Train log CSV
# ---------------------------------------------------------------------------


def test_train_log_csv_schema(tmp_path, mock_loader):
    """train_log.csv must have header + one row per epoch."""
    _run_main(tmp_path)
    log_path = _out_dir(tmp_path) / "train_log.csv"

    with log_path.open() as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["epoch", "train_loss", "val_loss"]
    # At most 3 epochs (may be fewer if early stopping fires)
    assert 1 <= len(rows) - 1 <= 3
    for row in rows[1:]:
        assert len(row) == 3
        assert int(row[0]) >= 1
        assert float(row[1]) > 0
        assert float(row[2]) > 0


# ---------------------------------------------------------------------------
# Provenance JSON
# ---------------------------------------------------------------------------


def test_provenance_json_schema(tmp_path, mock_loader):
    """provenance.json must contain all expected fields."""
    _run_main(tmp_path)
    prov_path = _out_dir(tmp_path) / "provenance.json"
    prov = json.loads(prov_path.read_text())

    required_keys = {
        "encoder_name", "seed", "git_sha", "native_dim", "target_dim",
        "needs_adapter", "lr", "weight_decay", "batch_size",
        "epochs_max", "epochs_run", "early_stopping_patience",
        "stopped_early", "best_epoch",
    }
    assert required_keys <= set(prov.keys())
    assert prov["encoder_name"] == _ENCODER
    assert prov["seed"] == 0
    assert prov["needs_adapter"] is False
    assert prov["native_dim"] == 384
    assert prov["target_dim"] == 384


def test_provenance_json_has_trailing_newline(tmp_path, mock_loader):
    """POSIX: files should end with a newline."""
    _run_main(tmp_path)
    raw = (_out_dir(tmp_path) / "provenance.json").read_bytes()
    assert raw.endswith(b"\n")


# ---------------------------------------------------------------------------
# Adapter path (encoder with native_dim != 384)
# ---------------------------------------------------------------------------


def test_main_with_adapter_encoder(tmp_path):
    """Encoders needing an adapter (native_dim != 384) must train successfully."""
    encoder = "clip_b32"  # native_dim=512 -> needs adapter
    data = _make_synthetic_embeddings(dim=NATIVE_DIMS[encoder])

    with mock.patch(
        "scripts.train_bc_baseline.load_encoder_embedding",
        return_value=data,
    ):
        rc = main([
            "--encoder", encoder,
            "--output-root", str(tmp_path / "bc_out"),
            "--epochs", "3",
            "--batch-size", "16",
            "--seed", "0",
        ])

    assert rc == 0
    out = tmp_path / "bc_out" / encoder / "seed_0"
    ckpt = torch.load(out / "checkpoint.pt", weights_only=False)
    assert ckpt["adapter_state_dict"] is not None
    assert "weight" in ckpt["adapter_state_dict"]


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


def test_early_stopping_fires_when_loss_plateaus(tmp_path):
    """With constant embeddings, val loss plateaus immediately."""
    n = 32
    data = {
        "embeddings": np.zeros((n * 3, _NATIVE_DIM), dtype=np.float32),
        "splits": np.array(["train"] * n + ["val"] * n + ["test"] * n),
        "steer_norms": np.zeros(n * 3, dtype=np.float32),
        "accel_norms": np.zeros(n * 3, dtype=np.float32),
        "scene_names": np.array([f"s{i}" for i in range(n * 3)]),
    }
    with mock.patch(
        "scripts.train_bc_baseline.load_encoder_embedding",
        return_value=data,
    ):
        rc = main([
            "--encoder", _ENCODER,
            "--output-root", str(tmp_path / "bc_out"),
            "--epochs", "50",
            "--batch-size", "16",
            "--seed", "0",
        ])
    assert rc == 0
    ckpt = torch.load(_out_dir(tmp_path) / "checkpoint.pt", weights_only=False)
    # Should stop well before 50 epochs with patience=10
    assert ckpt["stopped_early"] is True
    assert ckpt["epochs_run"] < 50


def test_epochs_run_matches_train_log_rows(tmp_path, mock_loader):
    """epochs_run in checkpoint must match the number of rows in train_log.csv."""
    _run_main(tmp_path)
    out = _out_dir(tmp_path)
    ckpt = torch.load(out / "checkpoint.pt", weights_only=False)

    with (out / "train_log.csv").open() as f:
        n_data_rows = sum(1 for _ in csv.reader(f)) - 1  # minus header

    assert ckpt["epochs_run"] == n_data_rows
