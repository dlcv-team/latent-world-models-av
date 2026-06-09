"""Unit tests for :mod:`evaluation.lang_scene_eval` (P2 per-bucket eval).

The headline requirement (and the explicit anti-pattern the P2 pilot
artifact fell into) is that the per-scene-type result must be a **genuine
per-bucket evaluation** -- CosSim computed within each bucket's own
sequences -- not one global DeltaCosSim copied across buckets.  The
``test_by_scene_type_is_genuinely_per_bucket`` test enforces exactly that:
it builds two buckets with deliberately different deltas and asserts they
come out different.

DeltaCosSim here is ``CosSim(z_hat_lang, z_real) - CosSim(z_hat_action,
z_real)`` per horizon (language-conditioned minus action-only), reusing the
canonical C4 ``_per_horizon_cossim`` implementation.
"""

from __future__ import annotations

import json

import pytest
import torch

from evaluation.lang_scene_eval import (
    bucket_map_from_captions,
    evaluate_by_scene_type,
    evaluate_global,
    main,
)

H, D = 4, 16


def _entry(payload: dict, bucket: str) -> dict:
    matches = [e for e in payload["by_scene_type"] if e["bucket"] == bucket]
    assert len(matches) == 1, f"bucket {bucket!r} should appear exactly once"
    return matches[0]


# ---------------------------------------------------------------------------
# Global delta
# ---------------------------------------------------------------------------


def test_global_delta_is_lang_minus_action():
    torch.manual_seed(0)
    z_real = torch.randn(6, H, D)
    predictions = {
        "encA": {
            "z_hat_lang": z_real.clone(),  # CosSim(lang, real) = 1
            "z_hat_action": -z_real.clone(),  # CosSim(action, real) = -1
            "z_real": z_real,
        }
    }
    payload = evaluate_global(predictions)
    enc = payload["per_encoder"]["encA"]
    assert payload["n_samples"] == 6
    assert payload["horizon"] == H
    for k in range(H):
        assert enc["cossim_lang"][k] == pytest.approx(1.0, abs=1e-5)
        assert enc["cossim_action"][k] == pytest.approx(-1.0, abs=1e-5)
        assert enc["delta_cossim"][k] == pytest.approx(2.0, abs=1e-5)


def test_global_method_field_is_not_a_proxy():
    torch.manual_seed(1)
    z_real = torch.randn(4, H, D)
    predictions = {
        "encA": {"z_hat_lang": z_real, "z_hat_action": z_real, "z_real": z_real}
    }
    payload = evaluate_global(predictions)
    assert "proxy" not in payload["method"].lower()
    assert "lang" in payload["method"].lower()


# ---------------------------------------------------------------------------
# Per-bucket: the anti-proxy guarantee
# ---------------------------------------------------------------------------


def _two_bucket_predictions():
    """rain sequences: delta ~ +2 ; urban sequences: delta ~ 0."""
    torch.manual_seed(7)
    # 5 sequences: 3 rain (scenes s_rain1 x2, s_rain2 x1), 2 urban (s_urban1 x2).
    sample_scene_names = ["s_rain1", "s_rain1", "s_rain2", "s_urban1", "s_urban1"]
    scene_to_bucket = {
        "s_rain1": "rain",
        "s_rain2": "rain",
        "s_urban1": "urban",
    }
    z_real = torch.randn(5, H, D)
    z_hat_action = torch.randn(5, H, D)

    z_hat_lang = z_hat_action.clone()
    rain_idx = [0, 1, 2]
    urban_idx = [3, 4]
    # rain: lang aligns with real (CosSim 1), action anti-aligns (-1) -> delta 2
    for i in rain_idx:
        z_hat_lang[i] = z_real[i]
        z_hat_action[i] = -z_real[i]
    # urban: lang == action -> delta 0
    for i in urban_idx:
        z_hat_lang[i] = z_hat_action[i]

    predictions = {
        "encA": {
            "z_hat_lang": z_hat_lang,
            "z_hat_action": z_hat_action,
            "z_real": z_real,
        }
    }
    return predictions, sample_scene_names, scene_to_bucket


def test_by_scene_type_is_genuinely_per_bucket():
    predictions, sample_scene_names, scene_to_bucket = _two_bucket_predictions()
    payload = evaluate_by_scene_type(predictions, sample_scene_names, scene_to_bucket)

    rain = _entry(payload, "rain")["per_encoder"]["encA"]["delta_cossim"]
    urban = _entry(payload, "urban")["per_encoder"]["encA"]["delta_cossim"]

    # The whole point: each bucket's delta comes from ITS OWN sequences.
    for k in range(H):
        assert rain[k] == pytest.approx(2.0, abs=1e-5)
        assert urban[k] == pytest.approx(0.0, abs=1e-5)
    # A global-delta proxy would make these identical; they must not be.
    assert rain != urban


def test_by_scene_type_n_scenes_counts_distinct_scenes_not_sequences():
    predictions, sample_scene_names, scene_to_bucket = _two_bucket_predictions()
    payload = evaluate_by_scene_type(predictions, sample_scene_names, scene_to_bucket)
    rain = _entry(payload, "rain")
    urban = _entry(payload, "urban")
    assert rain["n_scenes"] == 2  # s_rain1, s_rain2
    assert rain["n_samples"] == 3  # three sequences
    assert urban["n_scenes"] == 1  # s_urban1
    assert urban["n_samples"] == 2


def test_by_scene_type_method_marks_real_per_bucket():
    predictions, names, mapping = _two_bucket_predictions()
    payload = evaluate_by_scene_type(predictions, names, mapping)
    assert "per_bucket" in payload["method"].lower()
    assert "proxy" not in payload["method"].lower()


def test_by_scene_type_per_encoder_per_horizon_shape():
    predictions, names, mapping = _two_bucket_predictions()
    # add a second encoder so we exercise the per-encoder dict
    predictions["encB"] = {k: v.clone() for k, v in predictions["encA"].items()}
    payload = evaluate_by_scene_type(predictions, names, mapping)
    assert set(payload["encoders"]) == {"encA", "encB"}
    for entry in payload["by_scene_type"]:
        for enc in ("encA", "encB"):
            block = entry["per_encoder"][enc]
            assert len(block["delta_cossim"]) == H
            assert len(block["cossim_lang"]) == H
            assert len(block["cossim_action"]) == H


def test_global_matches_full_pooled_subset():
    predictions, names, mapping = _two_bucket_predictions()
    g = evaluate_global(predictions)["per_encoder"]["encA"]["delta_cossim"]
    # Pool all 5 sequences: mean of three +2 deltas and two 0 deltas per k.
    # Verify global is computed over ALL samples (not per-bucket).
    payload_bucket = evaluate_by_scene_type(predictions, names, mapping)
    rain = _entry(payload_bucket, "rain")["per_encoder"]["encA"]["delta_cossim"]
    urban = _entry(payload_bucket, "urban")["per_encoder"]["encA"]["delta_cossim"]
    # global delta should lie strictly between the two bucket deltas
    for k in range(H):
        assert urban[k] < g[k] < rain[k]


def test_unmapped_scene_is_skipped():
    # A sequence whose scene has no bucket mapping must not crash or land in
    # an arbitrary bucket; it is simply excluded.
    torch.manual_seed(3)
    z = torch.randn(3, H, D)
    predictions = {"encA": {"z_hat_lang": z, "z_hat_action": z, "z_real": z}}
    names = ["known", "known", "unmapped"]
    mapping = {"known": "urban"}
    payload = evaluate_by_scene_type(predictions, names, mapping)
    assert [e["bucket"] for e in payload["by_scene_type"]] == ["urban"]
    assert _entry(payload, "urban")["n_samples"] == 2


def test_empty_bucket_not_emitted():
    predictions, names, mapping = _two_bucket_predictions()
    payload = evaluate_by_scene_type(
        predictions, names, mapping, buckets=["rain", "urban", "highway"]
    )
    # "highway" was requested but has no scenes -> not emitted.
    assert {e["bucket"] for e in payload["by_scene_type"]} == {"rain", "urban"}


def test_length_mismatch_raises():
    torch.manual_seed(4)
    z = torch.randn(3, H, D)
    predictions = {"encA": {"z_hat_lang": z, "z_hat_action": z, "z_real": z}}
    with pytest.raises(ValueError, match="length"):
        evaluate_by_scene_type(predictions, ["a", "b"], {"a": "urban"})


# ---------------------------------------------------------------------------
# bucket_map_from_captions
# ---------------------------------------------------------------------------


def test_bucket_map_from_captions():
    captions_payload = {
        "captions": [
            {"scene_name": "scene-0001", "bucket": "rain"},
            {"scene_name": "scene-0002", "bucket": "highway"},
        ]
    }
    assert bucket_map_from_captions(captions_payload) == {
        "scene-0001": "rain",
        "scene-0002": "highway",
    }


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_main_smoke_writes_both_json(tmp_path):
    torch.manual_seed(5)
    z_real = torch.randn(4, H, D)
    torch.save(z_real.clone(), tmp_path / "lang.pt")
    torch.save(-z_real.clone(), tmp_path / "action.pt")
    torch.save(z_real, tmp_path / "real.pt")

    captions = {
        "captions": [
            {"scene_name": "scene-A", "bucket": "rain"},
            {"scene_name": "scene-B", "bucket": "urban"},
        ]
    }
    (tmp_path / "captions.json").write_text(json.dumps(captions))
    (tmp_path / "sample_scenes.json").write_text(
        json.dumps(["scene-A", "scene-A", "scene-B", "scene-B"])
    )

    out_dir = tmp_path / "p2"
    rc = main(
        [
            "--encoder",
            "encA",
            "--z-hat-lang",
            str(tmp_path / "lang.pt"),
            "--z-hat-action",
            str(tmp_path / "action.pt"),
            "--z-real",
            str(tmp_path / "real.pt"),
            "--captions",
            str(tmp_path / "captions.json"),
            "--sample-scenes",
            str(tmp_path / "sample_scenes.json"),
            "--output-dir",
            str(out_dir),
            "--status",
            "smoke_demo",
        ]
    )
    assert rc == 0
    by_type = json.loads((out_dir / "p2_lang_by_scene_type.json").read_text())
    glob = json.loads((out_dir / "p2_lang_global.json").read_text())

    assert by_type["status"] == "smoke_demo"
    assert "per_bucket" in by_type["method"].lower()
    assert {e["bucket"] for e in by_type["by_scene_type"]} == {"rain", "urban"}
    assert glob["per_encoder"]["encA"]["delta_cossim"]  # non-empty
    assert glob["n_samples"] == 4

    # Provenance: both JSONs must record exactly which inputs produced them,
    # so an artifact can never be mistaken for a run on different data.
    for payload in (by_type, glob):
        inputs = payload["inputs"]
        assert inputs["encoder"] == "encA"
        assert inputs["z_hat_lang"].endswith("lang.pt")
        assert inputs["z_hat_action"].endswith("action.pt")
        assert inputs["z_real"].endswith("real.pt")
        assert inputs["captions"].endswith("captions.json")
        assert inputs["sample_scenes"].endswith("sample_scenes.json")
