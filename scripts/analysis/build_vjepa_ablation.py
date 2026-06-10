#!/usr/bin/env python3
"""Build V-JEPA2 multi-frame ablation table (A19).

Reads existing full-dataset results from outputs/analysis/encoder_summary_with_ci.csv
and baselines.json to produce a 3-row ablation table comparing:

  1. V-JEPA2 rep64 (fpc64 checkpoint, 16-frame temporal input)
  2. V-JEPA2 rep1  (fpc1 checkpoint, single-frame input)
  3. DINOv2-S      (single-frame, different architecture -- best non-JEPA baseline)

No new training or embedding computation is needed; this script only reshapes
existing results into the ablation-specific format with interpretation.

Outputs:
  outputs/analysis/vjepa_ablation_table.csv
  outputs/analysis/vjepa_ablation_interpretation.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BASELINES = ROOT / "configs" / "baselines.json"
ENCODER_CI = ROOT / "outputs" / "analysis" / "encoder_summary_with_ci.csv"
OUT_DIR = ROOT / "outputs" / "analysis"

# The three rows we care about, in display order.
ABLATION_ROWS = [
    {
        "encoder": "vjepa2_rep64",
        "display_name": "V-JEPA2 (fpc64, 16-frame input)",
        "input_mode": "temporal",
        "frames": 16,
    },
    {
        "encoder": "vjepa2_rep1",
        "display_name": "V-JEPA2 (1-frame)",
        "input_mode": "single-frame",
        "frames": 1,
    },
    {
        "encoder": "dino_vits14",
        "display_name": "DINOv2-S (1-frame)",
        "input_mode": "single-frame",
        "frames": 1,
    },
]


def _load_ci_data() -> dict[str, dict[str, float]]:
    """Load per-encoder bootstrap CIs from encoder_summary_with_ci.csv."""
    rows: dict[str, dict[str, float]] = {}
    with open(ENCODER_CI) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["encoder"]] = {k: float(v) for k, v in row.items() if k != "encoder"}
    return rows


def _load_baselines() -> dict[str, dict]:
    """Load combined RMSE and rank from baselines.json."""
    with open(BASELINES) as f:
        data = json.load(f)
    return data["encoders"]


def build_table() -> list[dict]:
    ci = _load_ci_data()
    bl = _load_baselines()

    # Determine rank among the 3 ablation encoders by combined RMSE.
    ablation_encoders = [r["encoder"] for r in ABLATION_ROWS]
    combined_rmses = {e: bl[e]["test_rmse_mean"] for e in ablation_encoders}
    rank_order = sorted(ablation_encoders, key=lambda e: combined_rmses[e])

    table = []
    for spec in ABLATION_ROWS:
        enc = spec["encoder"]
        ci_row = ci[enc]
        table.append(
            {
                "encoder": spec["display_name"],
                "input_mode": spec["input_mode"],
                "frames": spec["frames"],
                "steer_rmse_scene_mean": round(ci_row["steer_rmse_scene_mean"], 4),
                "steer_ci95_lo": round(ci_row["steer_ci95_lo"], 4),
                "steer_ci95_hi": round(ci_row["steer_ci95_hi"], 4),
                "accel_rmse_scene_mean": round(ci_row["accel_rmse_scene_mean"], 4),
                "accel_ci95_lo": round(ci_row["accel_ci95_lo"], 4),
                "accel_ci95_hi": round(ci_row["accel_ci95_hi"], 4),
                "combined_rmse": round(combined_rmses[enc], 4),
                "rank": rank_order.index(enc) + 1,
            }
        )
    return table


def write_csv(table: list[dict]) -> Path:
    out = OUT_DIR / "vjepa_ablation_table.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    return out


def write_interpretation(table: list[dict]) -> Path:
    rep64 = table[0]
    rep1 = table[1]
    dino = table[2]

    # Compute relative improvements.
    rep64_vs_dino_steer = (
        (dino["steer_rmse_scene_mean"] - rep64["steer_rmse_scene_mean"])
        / dino["steer_rmse_scene_mean"]
        * 100
    )
    rep64_vs_rep1_steer = (
        (rep1["steer_rmse_scene_mean"] - rep64["steer_rmse_scene_mean"])
        / rep1["steer_rmse_scene_mean"]
        * 100
    )
    rep1_vs_dino_combined_pct = abs(
        rep1["combined_rmse"] - dino["combined_rmse"]
    ) / dino["combined_rmse"] * 100

    text = f"""\
# V-JEPA2 Multi-Frame Ablation (A19)

V-JEPA2 fpc64 (pre-trained on 64-frame clips, fed 16-frame input) achieves a steer scene-mean RMSE of
{rep64['steer_rmse_scene_mean']:.4f}, which is {rep64_vs_dino_steer:.0f}% lower
than the best single-frame encoder (DINOv2-S at {dino['steer_rmse_scene_mean']:.4f})
and {rep64_vs_rep1_steer:.0f}% lower than V-JEPA2 in single-frame mode
({rep1['steer_rmse_scene_mean']:.4f}). At single-frame input, V-JEPA2 rep1
({rep1['combined_rmse']:.4f} combined RMSE) and DINOv2-S ({dino['combined_rmse']:.4f})
are within {rep1_vs_dino_combined_pct:.1f}% of each other, suggesting the
architectural differences between the two encoders matter less than the
availability of temporal context. Notably, the rep1-vs-rep64 gap was absent in
the 240-scene pilot (both ~0.121 combined RMSE) but emerges clearly at 850
scenes, likely because the larger and more diverse training set provides enough
variation for temporal features to express their advantage over single-frame
representations.
"""
    out = OUT_DIR / "vjepa_ablation_interpretation.md"
    out.write_text(text)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_table()
    csv_path = write_csv(table)
    md_path = write_interpretation(table)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")

    # Print table for quick inspection.
    print()
    header = list(table[0].keys())
    print(",".join(header))
    for row in table:
        print(",".join(str(row[k]) for k in header))


if __name__ == "__main__":
    main()
