#!/usr/bin/env python3
"""Generate attribution visualizations for all encoders.

Entry point for B7 attribution pipeline. Generates:
- 120 PNG overlays (20 frames × 6 encoders)
- 6 PDFs at 300 DPI (one per encoder)
- JSON method report documenting per-encoder attribution methods

Usage:
    python scripts/generate_attribution.py \\
        --dataroot /path/to/nuscenes \\
        --split p0_test \\
        --device cuda
"""

import argparse
from pathlib import Path

from config import ENCODER_DISPLAY
from evaluation.gradcam import AttributionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Generate attribution visualizations for encoder interpretability",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="p0_test",
        help="Dataset split to use (e.g., p0_test, smoke_test)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run attribution on",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/attribution"),
        help="Directory to save output PNGs, PDFs, and JSON report",
    )
    parser.add_argument(
        "--n-per-scenario",
        type=int,
        default=5,
        help="Number of frames to sample per scenario type",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic frame selection",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Attribution Pipeline - B7")
    print("=" * 80)
    print(f"Split:           {args.split}")
    print(f"Device:          {args.device}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Frames/scenario: {args.n_per_scenario}")
    print(f"Seed:            {args.seed}")
    print("=" * 80)

    # Initialize pipeline
    pipeline = AttributionPipeline(
        split=args.split,
        device=args.device,
        output_dir=args.output_dir,
        n_per_scenario=args.n_per_scenario,
        seed=args.seed,
    )

    # Run pipeline
    report = pipeline.run()

    # Print summary
    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)
    print(f"Total frames processed: {report['n_frames']}")
    print(f"Encoders: {len(report['encoders'])}")
    for enc_name, enc_info in report['encoders'].items():
        display_name = ENCODER_DISPLAY.get(enc_name, enc_name)
        print(f"  - {display_name}: {enc_info['method']}")
        if enc_info['fallback_used']:
            print(f"    ⚠️  Fallback active")
    print(f"\nOutputs saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
