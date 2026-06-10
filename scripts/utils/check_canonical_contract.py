#!/usr/bin/env python3
"""CLI guard that verifies the canonical contract.

Exits non-zero if any of these drift:

* ``configs/canonical.yaml`` fails shallow validation.
* The in-repo subset manifest sha256 does not match the value pinned in
  the config.
* The manifest's split counts disagree with ``expected_split_counts``.
* (Optional) The local action-labels CSV is present but its sha256
  disagrees with the pinned value.

Run locally:

    python scripts/check_canonical_contract.py

Run in CI (see ``.github/workflows/ci.yml``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make repo modules importable when the script is invoked directly without
# `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (  # noqa: E402  (sys.path tweak above)
    load_canonical,
    resolve_action_labels_path,
    sha256_file,
)


class ContractError(RuntimeError):
    """Raised when the contract is violated."""


def check_manifest(cfg) -> None:
    path = cfg.manifest_path
    if not path.exists():
        raise ContractError(f"subset manifest missing at {path}")

    actual = sha256_file(path)
    expected = cfg.manifest_sha256
    if actual != expected:
        raise ContractError(
            f"subset manifest sha256 drift\n  expected: {expected}\n  actual:   {actual}\n  path:     {path}"
        )

    with path.open("r") as fh:
        manifest = json.load(fh)

    actual_counts = {k: len(v) for k, v in manifest["splits"].items()}
    for split, expected_n in cfg.expected_split_counts.items():
        actual_n = actual_counts.get(split)
        if actual_n != expected_n:
            raise ContractError(
                f"split count drift for {split!r}: expected {expected_n}, got {actual_n}"
            )


def check_action_labels(cfg) -> str:
    """Return a status string. Tolerates absence (returns 'absent')."""
    path = resolve_action_labels_path(cfg)
    if path is None:
        return "absent (set NUSCENES_ACTIONS_CSV or place CSV at data/raw/...)"

    actual = sha256_file(path)
    expected = cfg.raw["dataset"]["action_labels"]["sha256"]
    if actual != expected:
        raise ContractError(
            f"action labels sha256 drift\n  expected: {expected}\n  actual:   {actual}\n  path:     {path}"
        )
    return f"verified at {path}"


def main(argv: list[str] | None = None) -> int:
    cfg = load_canonical()
    print(f"canonical.yaml version: {cfg.version}")
    print(f"global seed: {cfg.global_seed}")

    try:
        check_manifest(cfg)
    except ContractError as exc:
        print(f"FAIL manifest: {exc}", file=sys.stderr)
        return 1
    print(f"OK   manifest: {cfg.manifest_path} ({cfg.manifest_sha256[:12]}...)")

    try:
        status = check_action_labels(cfg)
    except ContractError as exc:
        print(f"FAIL action_labels: {exc}", file=sys.stderr)
        return 1
    print(f"OK   action_labels: {status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
