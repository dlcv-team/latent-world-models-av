"""Scene caption generation for nuScenes scenes (P2, task C6).

Turns a nuScenes scene into a short natural-language caption such as::

    "rain night urban, 12 vehicles, 3 pedestrians"

The caption has four parsed fields, all derived from nuScenes metadata
(no pixels):

* **weather** -- ``rain`` if the scene ``description`` contains a word
  starting with "rain" ("rain", "rainy", "raining"), else ``clear``.
* **time-of-day** -- ``night`` if the description contains a word starting
  with "night" ("night", "nighttime"), else ``daytime``.
* **scenario** -- ``highway`` / ``urban`` / ``intersection`` / ``other`` via
  the same keyword heuristic as
  :func:`evaluation.metrics.classify_scenes_by_scenario`, so the two modules
  classify scenes identically.
* **vehicle / pedestrian counts** -- the number of *distinct object
  instances* (deduplicated by ``instance_token``) seen across the scene's
  keyframes whose ``category_name`` is under ``vehicle.*`` /
  ``human.pedestrian.*`` respectively.  Counting instances rather than
  raw annotations avoids inflating the count by the keyframe rate.

The captions feed P2's language-conditioned predictor (task C7): each
test scene is assigned one of the six P2 buckets declared in
``configs/canonical.yaml`` (``evaluation.scenario_buckets_p2``) via
:func:`classify_scene_bucket`, with priority **rain > night > scenario**
so a single scene lands in exactly one bucket.

Caveats
-------
**weather** and **time-of-day** are parsed *only* from the free-text
``description``.  When a description is empty or simply doesn't mention
rain/night, both silently fall back to ``clear`` / ``daytime`` -- so an
unlabelled rainy or night scene reads as ``clear daytime``.
:func:`parse_scene_fields` emits a :mod:`warnings` warning for each such
scene so the defaults are visible during generation rather than hidden.

This description-derived night set is deliberately *not* the same as B12's
(PR #29) **timestamp**-based night classification: B12 derives night/rain
from timestamps and treats them as overlapping subsets, whereas C6 buckets
every scene into exactly one of ``rain > night > scenario`` (a partition).
On ``p0_test`` the two disagree (B12 night=6 by timestamp; C6 night=4,
rain=9 by description) -- reconcile them deliberately, not blindly.

CLI
---
    # all p0_test scenes -> outputs/scene_captions.json
    python -m data.scene_captions
    python -m data.scene_captions --split p0_test --version v1.0-trainval

Requires the nuScenes dataset (``$NUSCENES_DATAROOT`` or ``<repo>/data``).
The pure functions (:func:`generate_scene_caption`,
:func:`parse_scene_fields`, :func:`classify_scene_bucket`,
:func:`build_caption_record`, :func:`export_scene_captions`) take any object
exposing the devkit's ``get(table, token)`` surface, so they are unit-tested
without a real download.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

# nuScenes category-name prefixes for the two actor classes we count.
VEHICLE_PREFIX = "vehicle."
PEDESTRIAN_PREFIX = "human.pedestrian"

# Scenario keyword heuristic, evaluated in order -- specific before general
# ("intersection" before "urban", since descriptions may contain both).
# Mirrors ``evaluation.metrics.classify_scenes_by_scenario`` exactly so a
# scene is never classified two different ways by the two modules.
_SCENARIO_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("highway", ("highway", "freeway")),
    ("intersection", ("intersection", "junction")),
    ("urban", ("urban", "city", "downtown")),
)

# Weather / time-of-day matchers.  The word-boundary *prefix* keeps suffixed
# forms ("rainy", "raining", "nighttime") while rejecting embedded matches
# ("terrain", "knight").  Known limitation: keywords embedded after other
# letters ("midnight", "overnight") do not match; nuScenes descriptions use
# the plain forms.
_RAIN_RE = re.compile(r"\brain")
_NIGHT_RE = re.compile(r"\bnight")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _scenario_from_description(description_lower: str) -> str:
    """First matching scenario keyword group, else ``"other"``."""
    for label, keywords in _SCENARIO_KEYWORDS:
        if any(keyword in description_lower for keyword in keywords):
            return label
    return "other"


def _count_actor_instances(scene: dict, nusc: Any) -> tuple[int, int]:
    """Count distinct vehicle / pedestrian instances across the scene.

    Walks the scene's keyframe samples via the ``first_sample_token`` ->
    ``next`` chain and deduplicates annotations by ``instance_token`` so an
    object tracked across many frames is counted once.
    """
    vehicle_instances: set[str] = set()
    pedestrian_instances: set[str] = set()

    sample_token = scene.get("first_sample_token", "")
    while sample_token:
        sample = nusc.get("sample", sample_token)
        for ann_token in sample.get("anns", []):
            ann = nusc.get("sample_annotation", ann_token)
            category = ann.get("category_name", "") or ""
            # Fall back to the annotation token if instance_token is absent,
            # so a malformed record still counts as one distinct object.
            instance = ann.get("instance_token") or ann_token
            if category.startswith(VEHICLE_PREFIX):
                vehicle_instances.add(instance)
            elif category.startswith(PEDESTRIAN_PREFIX):
                pedestrian_instances.add(instance)
        sample_token = sample.get("next", "")

    return len(vehicle_instances), len(pedestrian_instances)


def parse_scene_fields(scene_token: str, nusc: Any) -> dict[str, Any]:
    """Parse the four caption fields for one scene.

    Returns a dict with keys ``weather``, ``time_of_day``, ``scenario``,
    ``n_vehicles``, ``n_pedestrians``.

    Two field semantics worth flagging for C7 consumers:

    * ``weather`` / ``time_of_day`` are derived *solely* from the scene's
      free-text ``description``.  An empty description, or one that doesn't
      mention rain/night, silently yields ``clear`` / ``daytime`` (a
      :mod:`warnings` warning is emitted so this is visible during
      generation).  This description-based night set intentionally differs
      from B12's timestamp-based one; see the module-level *Caveats*.
    * ``n_vehicles`` / ``n_pedestrians`` are counts of *distinct actor
      instances over the entire scene* (deduplicated by ``instance_token``),
      **not** a per-frame count or actor density -- a car tracked across
      every keyframe contributes 1, not one-per-frame.
    """
    scene = nusc.get("scene", scene_token)
    description = (scene.get("description") or "").lower()
    if not description.strip():
        # weather/time-of-day are description-only, so a blank description
        # silently reads as "clear daytime".  Surface it (don't fail) so the
        # default is visible at generation time, not buried in the JSON.
        scene_label = scene.get("name") or scene_token
        warnings.warn(
            f"scene {scene_label!r} has an empty/missing 'description'; "
            "weather and time-of-day default to 'clear'/'daytime'",
            stacklevel=2,
        )
    n_vehicles, n_pedestrians = _count_actor_instances(scene, nusc)

    return {
        "weather": "rain" if _RAIN_RE.search(description) else "clear",
        "time_of_day": "night" if _NIGHT_RE.search(description) else "daytime",
        "scenario": _scenario_from_description(description),
        "n_vehicles": n_vehicles,
        "n_pedestrians": n_pedestrians,
    }


def _format_caption(fields: dict[str, Any]) -> str:
    """Render the canonical caption string from parsed fields."""
    return (
        f"{fields['weather']} {fields['time_of_day']} {fields['scenario']}, "
        f"{fields['n_vehicles']} vehicles, {fields['n_pedestrians']} pedestrians"
    )


def generate_scene_caption(scene_token: str, nusc: Any) -> str:
    """Return the caption string for ``scene_token``.

    Format: ``"{weather} {time} {scenario}, {n_veh} vehicles, {n_ped} pedestrians"``.
    """
    return _format_caption(parse_scene_fields(scene_token, nusc))


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------


def classify_scene_bucket(fields: dict[str, Any]) -> str:
    """Map parsed caption fields to one of the six P2 scene-type buckets.

    Priority is **rain > night > scenario**: weather and time-of-day are
    rarer, higher-signal conditions, so a rainy-night-urban scene is bucketed
    as ``rain`` (not ``urban``).  This makes the buckets a partition (each
    scene lands in exactly one) and reproduces the P2 pilot's bucket counts
    on ``p0_test``.
    """
    if fields["weather"] == "rain":
        return "rain"
    if fields["time_of_day"] == "night":
        return "night"
    return fields["scenario"]


def build_caption_record(
    scene_token: str, nusc: Any, scene_name: str | None = None
) -> dict[str, Any]:
    """Build the full per-scene record emitted in ``scene_captions.json``."""
    fields = parse_scene_fields(scene_token, nusc)
    if scene_name is None:
        scene_name = nusc.get("scene", scene_token).get("name")
    return {
        "scene_token": scene_token,
        "scene_name": scene_name,
        "caption": _format_caption(fields),
        "fields": fields,
        "bucket": classify_scene_bucket(fields),
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _scene_name_to_token(nusc: Any) -> dict[str, str]:
    """Map scene name -> scene token from ``nusc.scene``."""
    return {scene["name"]: scene["token"] for scene in nusc.scene}


def export_scene_captions(
    nusc: Any,
    scene_names: list[str],
    output_path: str | Path,
    split: str = "p0_test",
) -> dict[str, Any]:
    """Generate captions for ``scene_names`` and write ``scene_captions.json``.

    Returns the JSON payload (also written to ``output_path``).  Raises
    ``KeyError`` if a requested scene name is not present in ``nusc``.
    """
    name_to_token = _scene_name_to_token(nusc)

    captions = []
    for scene_name in scene_names:
        if scene_name not in name_to_token:
            raise KeyError(
                f"scene name {scene_name!r} not found in nuScenes "
                f"({len(name_to_token)} scenes available)"
            )
        captions.append(
            build_caption_record(name_to_token[scene_name], nusc, scene_name)
        )

    payload = {
        "split": split,
        "n_scenes": len(captions),
        "provenance": {
            "generator": "data.scene_captions",
            "nuscenes_version": getattr(nusc, "version", None),
            "counting": (
                "distinct instance_token per vehicle.*/human.pedestrian.* "
                "across the whole scene (unique instances, not per-frame)"
            ),
            "bucket_priority": "rain > night > scenario",
        },
        "captions": captions,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically (temp file + rename) so an interrupted run can't
    # leave a truncated JSON behind -- same pattern as evaluation.latent_eval.
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
    tmp_path.replace(output_path)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_nuscenes(version: str, dataroot: str) -> Any:
    """Instantiate the real nuScenes devkit (imported lazily)."""
    from nuscenes.nuscenes import NuScenes

    return NuScenes(version=version, dataroot=dataroot, verbose=False)


def _resolve_dataroot(explicit: str | None) -> str:
    """Resolve the nuScenes dataroot: explicit -> $NUSCENES_DATAROOT -> <repo>/data."""
    import os

    from config import repo_root

    if explicit:
        return explicit
    env = os.environ.get("NUSCENES_DATAROOT")
    if env:
        return env
    return str(repo_root() / "data")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scene_captions",
        description=(
            "Generate nuScenes scene captions and export them to JSON. "
            "Requires the nuScenes dataset ($NUSCENES_DATAROOT or <repo>/data)."
        ),
    )
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "Canonical split whose scenes to caption (default: p0_test). "
            "With --scenes the split is only a label and defaults to 'custom'."
        ),
    )
    parser.add_argument(
        "--scenes",
        default=None,
        help=(
            "Comma-separated scene names to caption, overriding --split "
            "(mainly for testing/ad-hoc runs)."
        ),
    )
    parser.add_argument(
        "--dataroot",
        default=None,
        help="nuScenes dataroot (default: $NUSCENES_DATAROOT or <repo>/data).",
    )
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        help="nuScenes version (default: v1.0-trainval).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <repo>/outputs/scene_captions.json).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.scenes:
        scene_names = [s.strip() for s in args.scenes.split(",") if s.strip()]
        # --scenes overrides WHICH scenes are captioned; don't claim a
        # canonical split unless the caller explicitly labelled one.
        split_label = args.split or "custom"
    else:
        from data.splits import get_split_from_canonical

        split_label = args.split or "p0_test"
        scene_names = get_split_from_canonical(split_label)

    if args.output is not None:
        output_path = Path(args.output)
    else:
        from config import repo_root

        output_path = repo_root() / "outputs" / "scene_captions.json"

    dataroot = _resolve_dataroot(args.dataroot)
    nusc = _load_nuscenes(args.version, dataroot)

    payload = export_scene_captions(nusc, scene_names, output_path, split=split_label)

    from collections import Counter

    bucket_counts = Counter(c["bucket"] for c in payload["captions"])
    print(
        f"[scene_captions] wrote {payload['n_scenes']} captions for "
        f"split={split_label} -> {output_path}"
    )
    print(f"[scene_captions] bucket counts: {dict(sorted(bucket_counts.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
