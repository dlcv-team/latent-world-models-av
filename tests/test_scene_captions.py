"""Unit tests for :mod:`data.scene_captions` (P2, task C6).

Covers:

* ``generate_scene_caption`` format string and the four parsed fields
  (weather / time-of-day / scenario / vehicle + pedestrian counts).
* Object counting: **unique instances** across the scene's keyframes
  (the same ``instance_token`` seen in N frames counts once), vehicle vs
  pedestrian category prefixes, and that non-actor categories are ignored.
* Scenario keyword heuristic (highway / urban / intersection / other),
  matching ``evaluation.metrics.classify_scenes_by_scenario``.
* ``classify_scene_bucket`` priority (rain > night > scenario) over the
  six P2 buckets.
* ``build_caption_record`` / ``export_scene_captions`` JSON schema.
* Edge cases: no annotations, empty / missing description, single-sample
  scene, case-insensitivity.

These tests use a tiny in-memory ``_FakeNuScenes`` stub instead of the
real devkit + dataset (mirrors the ``_DummyEncoder`` stubbing style in
``tests/test_encoders_base.py``), so they need no nuScenes download.
"""

from __future__ import annotations

import json

import pytest

from data.scene_captions import (
    build_caption_record,
    classify_scene_bucket,
    export_scene_captions,
    generate_scene_caption,
    main,
    parse_scene_fields,
)


# ---------------------------------------------------------------------------
# Fake nuScenes devkit stub + scene builder
# ---------------------------------------------------------------------------


class _FakeNuScenes:
    """Minimal stand-in exposing only ``.get(table, token)`` and ``.scene``.

    The real :class:`nuscenes.nuscenes.NuScenes` indexes records by token;
    we replicate just enough of that surface for the caption code.

    Note: this stub mimics the devkit's *post-load* state -- the real
    ``__make_reverse_index__`` decorates ``sample_annotation`` records with
    ``category_name`` and ``sample`` records with ``anns`` (verified against
    nuscenes-devkit 1.2.0 source); neither field exists in the raw JSON.
    ``data.scene_captions`` therefore requires a loaded ``NuScenes`` instance
    (or this stub), not raw table dicts.
    """

    version = "fake-v1.0"

    def __init__(self) -> None:
        self.scene: list[dict] = []
        self._tables: dict[str, dict[str, dict]] = {
            "scene": {},
            "sample": {},
            "sample_annotation": {},
        }

    def get(self, table_name: str, token: str) -> dict:
        return self._tables[table_name][token]


def _build_nusc(scenes: list[dict]) -> _FakeNuScenes:
    """Build a ``_FakeNuScenes`` from a compact scene spec.

    Each scene spec is a dict::

        {
            "name": "scene-0001",
            "description": "Rainy night ...",
            "frames": [
                [("vehicle.car", "inst_a"), ("human.pedestrian.adult", "inst_b")],
                [("vehicle.car", "inst_a")],   # same instance -> counts once
            ],
        }

    Tokens are derived deterministically from the scene name + indices so
    failures are reproducible and assertions can reference exact tokens.
    """
    nusc = _FakeNuScenes()
    for s_idx, spec in enumerate(scenes):
        name = spec.get("name", f"scene-{s_idx:04d}")
        scene_token = f"scene_tok_{s_idx}"
        frames = spec.get("frames", [])

        sample_tokens = [f"{scene_token}_sample_{i}" for i in range(len(frames))]
        for f_idx, anns in enumerate(frames):
            sample_token = sample_tokens[f_idx]
            ann_tokens = []
            for a_idx, (category_name, instance_token) in enumerate(anns):
                ann_token = f"{sample_token}_ann_{a_idx}"
                nusc._tables["sample_annotation"][ann_token] = {
                    "token": ann_token,
                    "category_name": category_name,
                    "instance_token": instance_token,
                }
                ann_tokens.append(ann_token)
            nusc._tables["sample"][sample_token] = {
                "token": sample_token,
                "anns": ann_tokens,
                "next": (sample_tokens[f_idx + 1] if f_idx + 1 < len(frames) else ""),
                "prev": sample_tokens[f_idx - 1] if f_idx > 0 else "",
            }

        scene_record = {
            "token": scene_token,
            "name": name,
            "description": spec["description"],
            "first_sample_token": sample_tokens[0] if sample_tokens else "",
            "last_sample_token": sample_tokens[-1] if sample_tokens else "",
            "nbr_samples": len(frames),
        }
        nusc._tables["scene"][scene_token] = scene_record
        nusc.scene.append(scene_record)
    return nusc


def _single(
    description: str, anns: list[tuple[str, str]] | None = None
) -> _FakeNuScenes:
    """One-scene, one-frame fake with the given description + annotations."""
    return _build_nusc(
        [{"name": "scene-0001", "description": description, "frames": [anns or []]}]
    )


# ---------------------------------------------------------------------------
# generate_scene_caption: format string + four fields
# ---------------------------------------------------------------------------


def test_caption_rain_night_urban_with_counts():
    nusc = _single(
        "Rainy night drive through urban downtown",
        [("vehicle.car", "v1"), ("human.pedestrian.adult", "p1")],
    )
    caption = generate_scene_caption("scene_tok_0", nusc)
    assert caption == "rain night urban, 1 vehicles, 1 pedestrians"


def test_caption_clear_daytime_highway_zero_actors():
    nusc = _single("Daytime highway cruising, light traffic", [])
    caption = generate_scene_caption("scene_tok_0", nusc)
    assert caption == "clear daytime highway, 0 vehicles, 0 pedestrians"


def test_caption_is_case_insensitive():
    nusc = _single("RAIN at NIGHT on the HIGHWAY", [])
    assert generate_scene_caption("scene_tok_0", nusc) == (
        "rain night highway, 0 vehicles, 0 pedestrians"
    )


def test_caption_count_words_are_always_plural():
    # The spec format string uses literal "vehicles"/"pedestrians"; counts
    # of exactly 1 are NOT specially singularized.
    nusc = _single(
        "Clear day in the city",
        [("vehicle.car", "v1"), ("human.pedestrian.adult", "p1")],
    )
    caption = generate_scene_caption("scene_tok_0", nusc)
    assert "1 vehicles" in caption and "1 pedestrians" in caption


# ---------------------------------------------------------------------------
# parse_scene_fields: weather / time / scenario
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "description, expected_weather",
    [
        ("Heavy rain on the freeway", "rain"),
        ("Light drizzle and rain showers", "rain"),
        ("Rainy night in the city", "rain"),  # suffixes still match
        ("Clear sunny afternoon", "clear"),
        ("Overcast but dry", "clear"),
        ("Driving across rough terrain", "clear"),  # 'terrain' is not rain
    ],
)
def test_weather_detection(description, expected_weather):
    fields = parse_scene_fields("scene_tok_0", _single(description))
    assert fields["weather"] == expected_weather


@pytest.mark.parametrize(
    "description, expected_time",
    [
        ("Night drive downtown", "night"),
        ("Late at night, empty roads", "night"),
        ("Nighttime drive through downtown", "night"),  # suffixes still match
        ("Bright daytime scene", "daytime"),
        ("Morning commute", "daytime"),
        ("A knight statue by the road", "daytime"),  # 'knight' is not night
    ],
)
def test_time_of_day_detection(description, expected_time):
    fields = parse_scene_fields("scene_tok_0", _single(description))
    assert fields["time_of_day"] == expected_time


@pytest.mark.parametrize(
    "description, expected_scenario",
    [
        ("Cruising on the highway", "highway"),
        ("Merging onto the freeway", "highway"),
        ("Dense urban traffic", "urban"),
        ("Driving through the city center", "urban"),
        ("Downtown crawl", "urban"),
        ("Waiting at the intersection", "intersection"),
        ("Turning at a junction", "intersection"),
        ("Parking lot maneuver", "other"),
        ("", "other"),
    ],
)
def test_scenario_keyword_heuristic(description, expected_scenario):
    fields = parse_scene_fields("scene_tok_0", _single(description))
    assert fields["scenario"] == expected_scenario


def test_scenario_precedence_highway_over_urban():
    # Mirrors metrics.classify_scenes_by_scenario ordering: highway wins.
    fields = parse_scene_fields("scene_tok_0", _single("urban highway interchange"))
    assert fields["scenario"] == "highway"


def test_scenario_precedence_intersection_over_urban():
    # metrics.classify_scenes_by_scenario checks specific before general:
    # "intersection" before "urban", since descriptions may contain both.
    fields = parse_scene_fields(
        "scene_tok_0", _single("Wait at the intersection in the city")
    )
    assert fields["scenario"] == "intersection"


# ---------------------------------------------------------------------------
# Object counting: unique instances, category prefixes, ignored categories
# ---------------------------------------------------------------------------


def test_counts_unique_instances_across_frames():
    # Same vehicle instance "v1" appears in both frames -> counts once.
    # A second vehicle "v2" appears only in frame 2.
    nusc = _build_nusc(
        [
            {
                "name": "scene-0001",
                "description": "Clear day city",
                "frames": [
                    [("vehicle.car", "v1"), ("human.pedestrian.adult", "p1")],
                    [("vehicle.car", "v1"), ("vehicle.truck", "v2")],
                ],
            }
        ]
    )
    fields = parse_scene_fields("scene_tok_0", nusc)
    assert fields["n_vehicles"] == 2
    assert fields["n_pedestrians"] == 1


def test_counts_various_vehicle_subtypes():
    anns = [
        ("vehicle.car", "a"),
        ("vehicle.truck", "b"),
        ("vehicle.bus.rigid", "c"),
        ("vehicle.motorcycle", "d"),
        ("vehicle.bicycle", "e"),
        ("vehicle.construction", "f"),
    ]
    fields = parse_scene_fields("scene_tok_0", _single("Clear day city", anns))
    assert fields["n_vehicles"] == 6
    assert fields["n_pedestrians"] == 0


def test_counts_pedestrian_subtypes():
    anns = [
        ("human.pedestrian.adult", "a"),
        ("human.pedestrian.child", "b"),
        ("human.pedestrian.construction_worker", "c"),
    ]
    fields = parse_scene_fields("scene_tok_0", _single("Clear day city", anns))
    assert fields["n_pedestrians"] == 3
    assert fields["n_vehicles"] == 0


def test_counts_ignore_non_actor_categories():
    anns = [
        ("vehicle.car", "v1"),
        ("movable_object.barrier", "b1"),
        ("movable_object.trafficcone", "c1"),
        ("static_object.bicycle_rack", "r1"),
        ("animal", "z1"),
    ]
    fields = parse_scene_fields("scene_tok_0", _single("Clear day city", anns))
    assert fields["n_vehicles"] == 1
    assert fields["n_pedestrians"] == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_description_is_clear_daytime_other():
    fields = parse_scene_fields("scene_tok_0", _single(""))
    assert fields == {
        "weather": "clear",
        "time_of_day": "daytime",
        "scenario": "other",
        "n_vehicles": 0,
        "n_pedestrians": 0,
    }


def test_missing_description_key_is_handled():
    # Some records may lack the key entirely; treat as empty.
    nusc = _single("placeholder")
    del nusc._tables["scene"]["scene_tok_0"]["description"]
    fields = parse_scene_fields("scene_tok_0", nusc)
    assert fields["weather"] == "clear"
    assert fields["time_of_day"] == "daytime"
    assert fields["scenario"] == "other"


def test_single_sample_scene_terminates():
    nusc = _single("Night rain highway", [("vehicle.car", "v1")])
    # next == "" on the only sample; must not loop forever.
    caption = generate_scene_caption("scene_tok_0", nusc)
    assert caption == "rain night highway, 1 vehicles, 0 pedestrians"


def test_scene_with_no_samples():
    nusc = _build_nusc(
        [{"name": "scene-0001", "description": "Clear day highway", "frames": []}]
    )
    fields = parse_scene_fields("scene_tok_0", nusc)
    assert fields["n_vehicles"] == 0 and fields["n_pedestrians"] == 0


# ---------------------------------------------------------------------------
# classify_scene_bucket: rain > night > scenario
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weather, time_of_day, scenario, expected_bucket",
    [
        ("rain", "night", "urban", "rain"),  # rain beats night + scenario
        ("rain", "daytime", "highway", "rain"),
        ("clear", "night", "urban", "night"),  # night beats scenario
        ("clear", "night", "highway", "night"),
        ("clear", "daytime", "highway", "highway"),
        ("clear", "daytime", "urban", "urban"),
        ("clear", "daytime", "intersection", "intersection"),
        ("clear", "daytime", "other", "other"),
    ],
)
def test_classify_scene_bucket_priority(
    weather, time_of_day, scenario, expected_bucket
):
    fields = {
        "weather": weather,
        "time_of_day": time_of_day,
        "scenario": scenario,
        "n_vehicles": 0,
        "n_pedestrians": 0,
    }
    assert classify_scene_bucket(fields) == expected_bucket


def test_bucket_values_are_within_canonical_p2_set(cfg):
    p2_buckets = set(cfg.raw["evaluation"]["scenario_buckets_p2"])
    # Every bucket the classifier can emit must be a declared P2 bucket.
    for scenario in ["highway", "urban", "intersection", "other"]:
        for weather, tod in [("clear", "daytime"), ("rain", "night")]:
            fields = {
                "weather": weather,
                "time_of_day": tod,
                "scenario": scenario,
                "n_vehicles": 0,
                "n_pedestrians": 0,
            }
            assert classify_scene_bucket(fields) in p2_buckets


# ---------------------------------------------------------------------------
# build_caption_record + export_scene_captions JSON schema
# ---------------------------------------------------------------------------


def test_build_caption_record_schema():
    nusc = _single("Rainy night urban downtown", [("vehicle.car", "v1")])
    rec = build_caption_record("scene_tok_0", nusc)
    assert rec["scene_token"] == "scene_tok_0"
    assert rec["scene_name"] == "scene-0001"
    assert rec["caption"] == "rain night urban, 1 vehicles, 0 pedestrians"
    assert rec["bucket"] == "rain"
    assert rec["fields"] == {
        "weather": "rain",
        "time_of_day": "night",
        "scenario": "urban",
        "n_vehicles": 1,
        "n_pedestrians": 0,
    }


def test_export_scene_captions_writes_json(tmp_path):
    nusc = _build_nusc(
        [
            {
                "name": "scene-0001",
                "description": "Rainy night urban",
                "frames": [[("vehicle.car", "v1")]],
            },
            {
                "name": "scene-0002",
                "description": "Clear day highway",
                "frames": [[("human.pedestrian.adult", "p1")]],
            },
        ]
    )
    out = tmp_path / "scene_captions.json"
    payload = export_scene_captions(
        nusc, ["scene-0001", "scene-0002"], out, split="p0_test"
    )

    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk == payload
    assert payload["split"] == "p0_test"
    assert payload["n_scenes"] == 2
    assert [c["scene_name"] for c in payload["captions"]] == [
        "scene-0001",
        "scene-0002",
    ]
    assert payload["captions"][0]["bucket"] == "rain"
    assert payload["captions"][1]["bucket"] == "highway"
    assert "provenance" in payload


def test_export_unknown_scene_name_raises(tmp_path):
    nusc = _build_nusc(
        [{"name": "scene-0001", "description": "Clear day", "frames": [[]]}]
    )
    with pytest.raises(KeyError, match="scene-9999"):
        export_scene_captions(
            nusc, ["scene-9999"], tmp_path / "out.json", split="p0_test"
        )


# ---------------------------------------------------------------------------
# CLI smoke test (real nuScenes is stubbed via monkeypatch)
# ---------------------------------------------------------------------------


def test_cli_main_smoke_with_explicit_scenes(tmp_path, monkeypatch):
    nusc = _build_nusc(
        [
            {
                "name": "scene-0001",
                "description": "Rainy night urban",
                "frames": [[("vehicle.car", "v1")]],
            },
            {
                "name": "scene-0002",
                "description": "Clear day highway",
                "frames": [[("human.pedestrian.adult", "p1")]],
            },
        ]
    )
    # Stub the devkit loader so the CLI needs no real dataset.
    monkeypatch.setattr(
        "data.scene_captions._load_nuscenes", lambda version, dataroot: nusc
    )
    out = tmp_path / "scene_captions.json"
    rc = main(
        [
            "--scenes",
            "scene-0001,scene-0002",
            "--output",
            str(out),
            "--split",
            "p0_test",
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["n_scenes"] == 2
    assert payload["split"] == "p0_test"
    assert [c["bucket"] for c in payload["captions"]] == ["rain", "highway"]


def test_cli_scenes_without_split_is_labelled_custom(tmp_path, monkeypatch):
    # --scenes overrides WHICH scenes are captioned; without an explicit
    # --split the payload must not claim a canonical split it didn't use.
    nusc = _build_nusc(
        [{"name": "scene-0001", "description": "Clear day", "frames": [[]]}]
    )
    monkeypatch.setattr(
        "data.scene_captions._load_nuscenes", lambda version, dataroot: nusc
    )
    out = tmp_path / "scene_captions.json"
    rc = main(["--scenes", "scene-0001", "--output", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["split"] == "custom"


def test_cli_default_split_path_uses_canonical_manifest(tmp_path, monkeypatch):
    # Without --scenes the CLI resolves scene names via
    # data.splits.get_split_from_canonical(split).
    nusc = _build_nusc(
        [{"name": "scene-0042", "description": "Rainy day city", "frames": [[]]}]
    )
    monkeypatch.setattr(
        "data.scene_captions._load_nuscenes", lambda version, dataroot: nusc
    )
    import data.splits

    monkeypatch.setattr(
        data.splits,
        "get_split_from_canonical",
        lambda split: ["scene-0042"] if split == "p0_test" else [],
    )
    out = tmp_path / "scene_captions.json"
    rc = main(["--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["split"] == "p0_test"
    assert [c["scene_name"] for c in payload["captions"]] == ["scene-0042"]


# ---------------------------------------------------------------------------
# _resolve_dataroot resolution order
# ---------------------------------------------------------------------------


def test_resolve_dataroot_explicit_wins(monkeypatch):
    from data.scene_captions import _resolve_dataroot

    monkeypatch.setenv("NUSCENES_DATAROOT", "/env/path")
    assert _resolve_dataroot("/explicit/path") == "/explicit/path"


def test_resolve_dataroot_env_var_beats_default(monkeypatch):
    from data.scene_captions import _resolve_dataroot

    monkeypatch.setenv("NUSCENES_DATAROOT", "/env/path")
    assert _resolve_dataroot(None) == "/env/path"


def test_resolve_dataroot_defaults_to_repo_data(monkeypatch):
    from config import repo_root

    from data.scene_captions import _resolve_dataroot

    monkeypatch.delenv("NUSCENES_DATAROOT", raising=False)
    assert _resolve_dataroot(None) == str(repo_root() / "data")
