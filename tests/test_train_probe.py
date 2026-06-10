"""Tests for ``training.train_probe`` CLI.

Fast tests use ``pretrained=False`` and an in-memory dataset stub so
nothing touches nuScenes or downloads any weights. The integration
test against the real dataset is gated by ``RUN_SLOW_TESTS=1`` + the
dataset's presence on disk.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from scripts import train_probe as tp


# ---------------------------------------------------------------------------
# In-memory dataset stub matching NuScenesFrameDataset's dict schema.
# ---------------------------------------------------------------------------


class _FakeSingleFrameDataset(torch.utils.data.Dataset):
    """Yields ``NuScenesFrameDataset``-shaped batches: image + actions + meta."""

    def __init__(
        self,
        n: int = 8,
        scenes: tuple[str, ...] = ("scene-0001", "scene-0002"),
        mode: str = "single_frame",
        clip_frames: int = 16,
        seed: int = 0,
    ) -> None:
        g = torch.Generator().manual_seed(seed)
        if mode == "single_frame":
            self.images = torch.rand(n, 3, 224, 224, generator=g)
        else:
            self.images = torch.rand(n, clip_frames, 3, 224, 224, generator=g)
        self.actions = torch.tanh(torch.randn(n, 2, generator=g))
        self.tokens = [f"tok-{i:04d}" for i in range(n)]
        self.scene_names = [scenes[i % len(scenes)] for i in range(n)]

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "image": self.images[idx],
            "actions": self.actions[idx],
            "sample_token": self.tokens[idx],
            "scene_token": f"scene-tok-{idx % len(self.scene_names):04d}",
            "scene_name": self.scene_names[idx],
            "timestamp_us": 0,
        }


@pytest.fixture
def patched_dataset(monkeypatch):
    """Patch ``NuScenesFrameDataset`` inside ``training.train_probe``.

    ``build_loaders`` lazily imports ``data.dataset.NuScenesFrameDataset``,
    so we patch the import target rather than the already-imported name.
    """
    from data import dataset as data_dataset_module

    def _factory(split, mode="single_frame", clip_frames=16, **_kwargs):
        # Different "splits" get different seeds so the test exercises
        # all three loaders end-to-end (train != val != test).
        seed = {"p0_train": 1, "p0_val": 2, "p0_test": 3}.get(split, 0)
        return _FakeSingleFrameDataset(n=8, mode=mode, clip_frames=clip_frames, seed=seed)

    monkeypatch.setattr(
        data_dataset_module, "NuScenesFrameDataset", _factory
    )
    return _factory


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_encoder_registry_covers_all_five():
    expected = {"vits16", "dinov2", "clip", "vqvae", "vjepa2"}
    assert set(tp.ENCODER_REGISTRY) == expected


def test_registry_pilot_names_match_canonical_closure():
    """Pilot names must match what A12 / pilot artifacts already use."""
    expected = {
        "vits16": "vit_s16",
        "dinov2": "dino_vits14",
        "clip": "clip_b32",
        "vqvae": "vq_track",
        "vjepa2": "vjepa2_rep64",
    }
    for cli_name, pilot_name in expected.items():
        assert tp.ENCODER_REGISTRY[cli_name].pilot_name == pilot_name


@pytest.mark.parametrize(
    "cli_name,class_name",
    [
        ("vits16", "ViTS16Wrapper"),
        ("dinov2", "DINOv2S14Wrapper"),
        ("clip", "CLIPB32Wrapper"),
        ("vqvae", "VQVAEWrapper"),
        ("vjepa2", "VJEPA2Wrapper"),
    ],
)
def test_build_encoder_returns_correct_class(cli_name, class_name):
    # vjepa2 with pretrained=False still pulls the small config file
    # from HF; if the env has no network and no cache, skip.
    try:
        enc = tp.build_encoder(cli_name, pretrained=False)
    except Exception as exc:
        pytest.skip(
            f"{cli_name} construction failed (likely no network/cache): {exc}"
        )
    assert type(enc).__name__ == class_name


def test_build_encoder_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown encoder"):
        tp.build_encoder("bogus", pretrained=False)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class _IdentityEncoderWithAdapter(nn.Module):
    """Encoder stub with a one-parameter adapter for optimizer tests."""

    def __init__(self):
        super().__init__()
        self.adapter = nn.Linear(384, 384, bias=False)

    def trainable_parameters(self):
        return iter(self.adapter.parameters())

    def forward(self, x):
        return self.adapter(x)


class _IdentityEncoderNoAdapter(nn.Module):
    """Encoder stub without an adapter."""

    def trainable_parameters(self):
        return iter(())

    def forward(self, x):
        return x


def test_build_optimizer_includes_probe_and_adapter_when_present():
    from models.probe import ActionProbe

    encoder = _IdentityEncoderWithAdapter()
    probe = ActionProbe()
    opt = tp.build_optimizer(probe, encoder, lr=1e-3, weight_decay=0.0)

    opt_param_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    probe_ids = {id(p) for p in probe.parameters()}
    adapter_ids = {id(p) for p in encoder.adapter.parameters()}

    assert probe_ids <= opt_param_ids
    assert adapter_ids <= opt_param_ids


def test_build_optimizer_only_probe_params_when_no_adapter():
    from models.probe import ActionProbe

    encoder = _IdentityEncoderNoAdapter()
    probe = ActionProbe()
    opt = tp.build_optimizer(probe, encoder, lr=1e-3, weight_decay=0.0)

    opt_param_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    probe_ids = {id(p) for p in probe.parameters()}
    assert opt_param_ids == probe_ids


# ---------------------------------------------------------------------------
# write_per_scene_rmse
# ---------------------------------------------------------------------------


def test_write_per_scene_rmse_schema_and_aggregation(tmp_path):
    """Verify CSV schema, per-scene grouping, and RMSE math."""
    from models.probe import ActionProbe
    import csv

    encoder = _IdentityEncoderNoAdapter()
    probe = ActionProbe()
    # Deterministic data: 2 scenes × 2 samples each, embedding dim 384.
    ds = _FakeSingleFrameDataset(n=4, scenes=("scene-A", "scene-B"), seed=42)
    # Replace the image tensor with a 384-d "embedding" since the stub
    # encoder is identity and expects flat features.
    ds.images = torch.randn(4, 384)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)

    out_path = tmp_path / "per_scene_rmse.csv"
    tp.write_per_scene_rmse(
        encoder=encoder,
        probe=probe,
        test_loader=loader,
        out_path=out_path,
        pilot_name="vit_s16",
    )

    with out_path.open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [
        "encoder",
        "scene_name",
        "scenario",
        "fold_id",
        "steer_rmse",
        "accel_rmse",
        "n",
    ]
    assert len(rows) == 3  # header + 2 scenes
    scene_names = {r[1] for r in rows[1:]}
    assert scene_names == {"scene-A", "scene-B"}
    for row in rows[1:]:
        assert row[0] == "vit_s16"
        assert row[2] == ""  # scenario blank — added downstream
        assert row[3] == "0"  # fold_id
        assert float(row[4]) > 0  # RMSE is positive
        assert int(row[6]) == 2  # n samples per scene


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_schema_for_vits16(tmp_path, monkeypatch):
    from config import load_canonical

    cfg = load_canonical()
    encoder = _IdentityEncoderNoAdapter()
    out_path = tmp_path / "provenance.json"
    tp.write_provenance(
        encoder=encoder,
        encoder_name="vits16",
        out_path=out_path,
        cfg=cfg,
        pretrained=True,
        seed=42,
    )
    payload = json.loads(out_path.read_text())
    expected_keys = {
        "encoder_name",
        "pilot_name",
        "wrapper_class",
        "pretrained_weights_id",
        "pretrained",
        "config_version",
        "manifest_sha256",
        "action_labels_sha256",
        "git_sha",
        "fallback_caveat",
        "torch_version",
        "seed",
        "source",
    }
    assert set(payload.keys()) == expected_keys
    assert payload["encoder_name"] == "vits16"
    assert payload["pilot_name"] == "vit_s16"
    assert payload["pretrained_weights_id"] == "vit_small_patch16_224"
    assert payload["fallback_caveat"] == ""  # no caveat for non-VQ
    assert payload["pretrained"] is True
    assert payload["seed"] == 42


def test_provenance_carries_no_vq_fallback_caveat(tmp_path):
    """With pretrained=False the vendored encoder is used, not the fallback."""
    from config import load_canonical

    cfg = load_canonical()
    try:
        encoder = tp.build_encoder("vqvae", pretrained=False)
    except Exception as exc:
        pytest.skip(f"VQ wrapper construction failed: {exc}")

    out_path = tmp_path / "provenance.json"
    tp.write_provenance(
        encoder=encoder,
        encoder_name="vqvae",
        out_path=out_path,
        cfg=cfg,
        pretrained=False,
        seed=0,
    )
    payload = json.loads(out_path.read_text())
    assert payload["fallback_caveat"] == ""


# ---------------------------------------------------------------------------
# main() smoke
# ---------------------------------------------------------------------------


def test_main_rejects_unknown_encoder():
    with pytest.raises(SystemExit):
        tp.main(["--encoder", "bogus", "--no-pretrained", "--epochs", "1"])


@pytest.mark.parametrize(
    "cli_name,pilot_name",
    [
        ("vits16", "vit_s16"),
        ("vqvae", "vq_track"),
        ("vjepa2", "vjepa2_rep64"),
    ],
)
def test_main_smoke_with_pretrained_false_synthetic_data(
    cli_name, pilot_name, tmp_path, patched_dataset
):
    """End-to-end smoke: ``--no-pretrained``, 1 epoch, synthetic data.

    Parametrized across three encoders that exercise distinct CLI paths:

    * ``vits16`` — single-frame baseline, no adapter, no fallback.
    * ``vqvae`` — may trigger FR-08 DINOv2 fallback if checkpoint unavailable; checks
      ``fallback_caveat`` persistence into provenance.
    * ``vjepa2`` — runs in ``mode="clip"`` (5-D ``(B, T, 3, H, W)``
      batches); exercises the V-JEPA-specific dataset stub path.

    ``dinov2`` and ``clip`` are intentionally excluded — they exercise
    the same CLI code path as ``vits16`` (single-frame, no fallback)
    and are already covered by the construction tests.

    Skips cleanly when the wrapper can't construct (e.g. no network or
    HF cache for V-JEPA's config).
    """
    try:
        tp.build_encoder(cli_name, pretrained=False)
    except Exception as exc:
        pytest.skip(
            f"{cli_name} construction failed (likely no network/cache): {exc}"
        )

    out_root = tmp_path / "outputs" / "probes"
    rc = tp.main(
        [
            "--encoder",
            cli_name,
            "--no-pretrained",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--output-root",
            str(out_root),
        ]
    )
    assert rc == 0

    enc_dir = out_root / pilot_name
    assert (enc_dir / "train_log.csv").exists()
    assert (enc_dir / "checkpoint.pt").exists()
    assert (enc_dir / "per_scene_rmse.csv").exists()
    assert (enc_dir / "provenance.json").exists()

    train_log = (enc_dir / "train_log.csv").read_text().splitlines()
    assert train_log[0] == "epoch,train_loss,val_loss"
    assert len(train_log) == 2  # header + 1 epoch

    ckpt = torch.load(enc_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert "probe_state_dict" in ckpt
    assert ckpt["pilot_name"] == pilot_name

    payload = json.loads((enc_dir / "provenance.json").read_text())
    assert payload["pretrained"] is False
    assert payload["seed"] == 42  # from canonical config (global_seed)
    assert payload["pilot_name"] == pilot_name
    assert (
        payload["pretrained_weights_id"]
        == tp.ENCODER_REGISTRY[cli_name].pretrained_weights_id
    )

    # No encoder uses the fallback when pretrained=False.
    assert payload["fallback_caveat"] == ""


def test_main_output_root_argument_is_honored(tmp_path, patched_dataset):
    custom_out = tmp_path / "custom_outputs"
    rc = tp.main(
        [
            "--encoder",
            "vits16",
            "--no-pretrained",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--output-root",
            str(custom_out),
        ]
    )
    assert rc == 0
    assert (custom_out / "vit_s16" / "provenance.json").exists()


# ---------------------------------------------------------------------------
# Slow integration test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason=(
        "Set RUN_SLOW_TESTS=1 to run the real-dataset integration test "
        "(requires nuScenes v1.0-trainval + action labels CSV)."
    ),
)
def test_main_one_epoch_against_real_dataset(tmp_path):
    """1 epoch on p0_val with --no-pretrained; verifies real pipeline runs."""
    out_root = tmp_path / "outputs" / "probes"
    rc = tp.main(
        [
            "--encoder",
            "vits16",
            "--no-pretrained",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--output-root",
            str(out_root),
        ]
    )
    assert rc == 0
    assert (out_root / "vit_s16" / "per_scene_rmse.csv").exists()
