#!/usr/bin/env python3
"""CLI script to generate attribution grid figure PDF.

Composes per-encoder attribution overlay PNGs into a 6×4 grid figure.

Usage:
    python scripts/generate_attribution_grid.py
    python scripts/generate_attribution_grid.py --output-dir outputs/my_run
    python scripts/generate_attribution_grid.py --annotations configs/attribution_annotations_example.json
    python scripts/generate_attribution_grid.py --input-dir outputs/attribution --output my_grid.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.attribution_grid import AttributionGridGenerator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate attribution grid figure PDF from per-encoder overlays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/attribution"),
        help="Base directory for inputs and outputs (default: outputs/attribution)",
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing {encoder}_{scenario}_{index:02d}.png files (default: --output-dir)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF file path (default: {output-dir}/attribution_grid.pdf)",
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Optional path to annotation JSON config file",
    )

    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index to use for each (encoder, scenario) pair (default: 0)",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for PDF export (default: 300)",
    )

    args = parser.parse_args()

    # Apply defaults: use output-dir if input-dir or output not specified
    if args.input_dir is None:
        args.input_dir = args.output_dir
    if args.output is None:
        args.output = args.output_dir / "attribution_grid.pdf"

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1

    # Validate annotation config if provided
    if args.annotations and not args.annotations.exists():
        print(f"Error: Annotation config not found: {args.annotations}")
        return 1

    # Create generator
    generator = AttributionGridGenerator(
        input_dir=args.input_dir,
        output_path=args.output,
        annotation_config=args.annotations,
        frame_index=args.frame_index,
        dpi=args.dpi,
    )

    # Generate grid figure
    try:
        output_path = generator.generate()
        print(f"\n✓ Success! Generated: {output_path}")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
