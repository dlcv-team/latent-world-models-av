"""Verifies that the canonical contract has not drifted.

These tests are the floor for *every* PR in the repo. If any of them fail
on `main`, treat it as a P0 bug — the rest of the project's claims are
based on these constants.
"""

from __future__ import annotations

import pytest

from config import sha256_file


def test_config_loads(cfg):
    assert cfg.version, "canonical.yaml must declare a version"
    assert cfg.global_seed == 42
    assert cfg.target_embedding_dim == 384


def test_manifest_present_and_hash_pinned(cfg):
    path = cfg.manifest_path
    assert path.exists(), f"subset manifest missing at {path}"

    actual = sha256_file(path)
    assert actual == cfg.manifest_sha256, (
        "subset manifest sha256 has drifted from the value pinned in "
        "configs/canonical.yaml; if this change is intentional, bump the "
        "config version, update docs/CANONICAL_ARTIFACTS.md, and get joint "
        "team sign-off."
    )


def test_manifest_split_counts(cfg, manifest):
    actual = {k: len(v) for k, v in manifest["splits"].items()}
    for split, expected_n in cfg.expected_split_counts.items():
        assert actual.get(split) == expected_n, (
            f"split {split!r} expected {expected_n} scenes, got {actual.get(split)}"
        )


def test_manifest_seed_matches_config(cfg, manifest):
    assert int(manifest["seed"]) == cfg.global_seed


def test_no_scene_token_overlap_across_p0_splits(manifest):
    train = set(manifest["splits"]["p0_train"])
    val = set(manifest["splits"]["p0_val"])
    test = set(manifest["splits"]["p0_test"])
    assert not (train & val), "p0_train and p0_val share scenes"
    assert not (train & test), "p0_train and p0_test share scenes"
    assert not (val & test), "p0_val and p0_test share scenes"


def test_normalization_constants(cfg):
    steer = cfg.normalization("steering")
    accel = cfg.normalization("acceleration")
    assert steer["divisor"] == 6.0
    assert steer["clip_range"] == [-1.0, 1.0]
    assert steer["raw_unit"] == "rad"
    assert accel["divisor"] == 10.0
    assert accel["clip_range"] == [-1.0, 1.0]
    assert accel["raw_unit"] == "m_per_s2"


def test_probe_hyperparams_locked(cfg):
    probe = cfg.probe()
    assert probe["learning_rate"] == 1.0e-3
    assert probe["batch_size"] == 256
    assert probe["epochs"] == 50
    assert probe["loss"] == "mse"
    assert probe["optimizer"] == "adam"
    assert probe["early_stopping"] is False


def test_bc_baseline_hyperparams_locked(cfg):
    bc = cfg.bc()
    assert bc["learning_rate"] == 1.0e-3
    assert bc["batch_size"] == 256
    assert bc["epochs"] == 50
    assert bc["early_stopping_patience"] == 10


def test_all_five_encoders_present(cfg):
    expected = {"vit_s16", "dinov2_s14", "clip_b32", "vqvae", "vjepa2"}
    assert set(cfg.raw["encoders"]) == expected


def test_vjepa2_uses_clip_input_mode(cfg):
    enc = cfg.encoder("vjepa2")
    assert enc["input_mode"] == "clip"
    assert enc["clip_frames"] == 16


def test_vqvae_records_fr08_fallback_policy(cfg):
    enc = cfg.encoder("vqvae")
    assert "FR-08" in enc.get("fallback_policy", ""), (
        "VQ encoder must document the FR-08 DINOv2 fallback policy so "
        "Member 2's figure-rendering code can detect it."
    )


def test_evaluation_buckets(cfg):
    assert cfg.raw["evaluation"]["scenario_buckets"] == [
        "highway",
        "urban",
        "intersection",
        "other",
    ]
    p2 = cfg.raw["evaluation"]["scenario_buckets_p2"]
    for required in ("night", "rain", "highway", "urban", "intersection", "other"):
        assert required in p2, f"P2 buckets must include {required!r}"


def test_figures_dpi_locked(cfg):
    assert cfg.raw["figures"]["dpi"] == 300


def test_required_caption_strings_present(cfg):
    captions = cfg.raw["figures"]["required_captions"]
    assert "180/20/40" in captions["subset"]
    assert "FR-08" in captions["vq_fallback"]
