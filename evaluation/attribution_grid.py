"""Attribution grid figure generation for encoder comparison.

Creates 6×4 grid figure PDF from per-encoder attribution outputs (B7).
Composes individual overlay PNGs into publication-ready comparison figure.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from config import ENCODER_DISPLAY


class AttributionGridGenerator:
    """Generates 6×4 grid figure from per-encoder attribution outputs.

    Creates a publication-ready grid layout with:
    - 6 rows (one per encoder: ViT-S/16, DINOv2-S/14, CLIP ViT-B/32, VQ-VAE, V-JEPA2, V-JEPA2 (1-frame))
    - 4 columns (one per scenario: intersection, other, highway, urban)
    - Row labels (encoder names) on the left
    - Column headers (scenario names) on top
    - Optional annotations (arrows, boxes) highlighting driving-relevant regions
    - Caption with VQ fallback caveat if applicable

    Examples
    --------
    >>> generator = AttributionGridGenerator(
    ...     input_dir=Path("outputs/attribution"),
    ...     output_path=Path("outputs/attribution/attribution_grid.pdf"),
    ... )
    >>> output_path = generator.generate()
    >>> print(f"Generated: {output_path}")
    """

    # Encoder keys in order (M1 canonical pilot_name keys from training.train_probe.ENCODER_REGISTRY)
    ENCODERS = [
        "vit_s16",
        "dino_vits14",
        "clip_b32",
        "vq_track",
        "vjepa2_rep64",
        "vjepa2_rep1",
    ]

    # Scenario display names in order (4 columns)
    SCENARIOS = [
        ("intersection", "Intersection"),
        ("other", "Other"),
        ("highway", "Highway"),
        ("urban", "Urban"),
    ]

    def __init__(
        self,
        input_dir: Path | str,
        output_path: Path | str,
        annotation_config: Optional[Path | str] = None,
        frame_index: int = 0,
        dpi: int = 300,
    ):
        """Initialize attribution grid generator.

        Parameters
        ----------
        input_dir
            Directory containing {encoder}_{scenario}_{index:02d}.png files.
        output_path
            Output PDF file path.
        annotation_config
            Optional path to annotation JSON config file.
        frame_index
            Frame index to use for each (encoder, scenario) pair (default: 0).
        dpi
            Figure DPI for PDF export (default: 300).
        """
        self.input_dir = Path(input_dir)
        self.output_path = Path(output_path)
        self.annotation_config = Path(annotation_config) if annotation_config else None
        self.frame_index = frame_index
        self.dpi = dpi

        # Will be populated by load_images() and parse_annotation_config()
        self.images: Dict[Tuple[str, str], np.ndarray] = {}
        self.annotations: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    def load_images(self) -> Dict[Tuple[str, str], np.ndarray]:
        """Load PNG images from input directory.

        Loads images with naming pattern: {encoder}_{scenario}_{index:02d}.png
        Only loads scenarios that have data (intersection, other).

        If frame_index is specified, uses that index. Otherwise, finds the first
        available image for each (encoder, scenario) pair.

        Returns
        -------
        images
            Dict mapping (encoder_key, scenario_key) to image array (H, W, 3).

        Raises
        ------
        FileNotFoundError
            If input directory doesn't exist or expected files are missing.
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        images = {}

        # Discover scenarios with available PNG files
        available_scenarios = self._discover_available_scenarios()

        if not available_scenarios:
            print("Warning: No PNG files found matching pattern {encoder}_{scenario}_{index:02d}.png")
            self.images = {}
            return {}

        for encoder_key in self.ENCODERS:
            for scenario_key in available_scenarios:
                # Try specified frame_index first
                filename = f"{encoder_key}_{scenario_key}_{self.frame_index:02d}.png"
                filepath = self.input_dir / filename

                # If not found, search for first available image for this (encoder, scenario)
                if not filepath.exists():
                    # Search for any matching file
                    pattern = f"{encoder_key}_{scenario_key}_*.png"
                    matching_files = sorted(self.input_dir.glob(pattern))

                    if matching_files:
                        filepath = matching_files[0]
                        print(f"Note: Using {filepath.name} (frame_index {self.frame_index} not found)")
                    else:
                        print(f"Warning: No images found for {encoder_key}/{scenario_key}")
                        continue

                # Load image as numpy array
                img = Image.open(filepath)
                img_array = np.array(img)

                images[(encoder_key, scenario_key)] = img_array

        self.images = images
        return images

    def _discover_available_scenarios(self) -> List[str]:
        """Discover scenarios with available PNG files in input directory.

        Scans input directory for files matching {encoder}_{scenario}_{index:02d}.png
        and extracts unique scenario names. Returns scenarios in canonical SCENARIOS order.

        Returns
        -------
        available_scenarios
            List of scenario keys that have at least one PNG file,
            ordered according to SCENARIOS constant.

        Examples
        --------
        >>> # With files: vit_s16_intersection_00.png, vit_s16_other_00.png
        >>> gen._discover_available_scenarios()
        ['intersection', 'other']
        """
        if not self.input_dir.exists():
            return []

        # Find all PNG files
        png_files = self.input_dir.glob("*.png")

        # Build regex pattern for expected filename structure
        encoder_keys = self.ENCODERS
        # Escape encoder keys for regex (e.g., vit_s16 stays as-is since _ doesn't need escaping)
        encoder_pattern = '|'.join(re.escape(k) for k in encoder_keys)
        # Pattern: ^(encoder)_(scenario)_(index).png$
        # Use greedy .+ for scenario to handle scenario names with underscores
        # Regex will backtrack to allow _(\d+) to match at the end
        filename_pattern = re.compile(
            rf'^({encoder_pattern})_(.+)_(\d+)\.png$'
        )

        # Extract scenario names from filenames
        discovered_scenarios = set()
        for filepath in png_files:
            filename = filepath.name
            match = filename_pattern.match(filename)
            if match:
                # group(1) = encoder, group(2) = scenario, group(3) = index
                scenario_key = match.group(2)
                discovered_scenarios.add(scenario_key)

        # Preserve canonical ordering from SCENARIOS
        canonical_scenario_keys = [k for k, _ in self.SCENARIOS]
        available_scenarios = [
            scenario for scenario in canonical_scenario_keys
            if scenario in discovered_scenarios
        ]

        return available_scenarios

    def parse_annotation_config(self) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
        """Parse annotation configuration from JSON file.

        Returns
        -------
        annotations
            Dict mapping (encoder_key, scenario_key) to list of annotation dicts.
            Each annotation dict has keys: type, coords/start/end, label, color.

        Raises
        ------
        FileNotFoundError
            If annotation config file doesn't exist.
        ValueError
            If annotation config is malformed or has invalid schema.
        """
        if self.annotation_config is None or not self.annotation_config.exists():
            # No annotations provided - return empty dict
            self.annotations = {}
            return {}

        try:
            with open(self.annotation_config) as f:
                config = json.load(f)

            # Validate schema
            if "version" not in config:
                raise ValueError("Annotation config missing 'version' field")
            if "annotations" not in config:
                raise ValueError("Annotation config missing 'annotations' field")

            # Parse annotations
            annotations = {}
            for encoder_key, scenarios in config["annotations"].items():
                for scenario_key, annot_list in scenarios.items():
                    annotations[(encoder_key, scenario_key)] = annot_list

            self.annotations = annotations
            return annotations

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in annotation config: {e}")

    def create_grid_figure(self) -> Tuple[plt.Figure, np.ndarray]:
        """Create matplotlib figure with GridSpec layout.

        Layout: (len(ENCODERS) + 1) rows × 4 columns
        - Row 0: Column headers
        - Rows 1-6: Encoder rows (one per encoder)
        - Each cell shows an attribution overlay or empty placeholder

        Returns
        -------
        fig
            Matplotlib Figure object.
        axes
            2D numpy array of Axes objects, shape (7, 4).
        """
        n_encoder_rows = len(self.ENCODERS)
        n_total_rows = n_encoder_rows + 1  # Header + encoders

        # Create figure with appropriate size for grid at 300 DPI
        fig = plt.figure(figsize=(16, 24), dpi=self.dpi)

        # GridSpec: (len(ENCODERS) + 1) rows × 4 columns
        gs = gridspec.GridSpec(
            n_total_rows, 4,
            figure=fig,
            height_ratios=[0.3] + [1] * n_encoder_rows,  # Header row smaller
            width_ratios=[1, 1, 1, 1],
            hspace=0.05,  # Minimal vertical spacing
            wspace=0.05,  # Minimal horizontal spacing
            left=0.08,    # Space for row labels
            right=0.98,
            top=0.96,
            bottom=0.04,
        )

        # Create axes array
        axes = np.empty((n_total_rows, 4), dtype=object)
        for i in range(n_total_rows):
            for j in range(4):
                axes[i, j] = fig.add_subplot(gs[i, j])

        return fig, axes

    def populate_cell(
        self,
        ax: plt.Axes,
        image: Optional[np.ndarray],
        encoder_key: str,
        scenario_key: str,
    ):
        """Populate a single grid cell with image or placeholder.

        Parameters
        ----------
        ax
            Target matplotlib Axes object.
        image
            Image array (H, W, 3) to display, or None for empty cell.
        encoder_key
            Encoder identifier (e.g., "vit_s16").
        scenario_key
            Scenario identifier (e.g., "intersection").
        """
        if image is not None:
            # Display image
            ax.imshow(image)
        else:
            # Empty cell - show placeholder
            ax.set_facecolor('#f0f0f0')  # Light gray background
            ax.text(
                0.5, 0.5,
                'No data\navailable',
                ha='center',
                va='center',
                fontsize=10,
                color='#666666',
                transform=ax.transAxes,
            )

        # Remove axis ticks and labels
        ax.axis('off')

    def add_annotations(
        self,
        ax: plt.Axes,
        encoder_key: str,
        scenario_key: str,
    ):
        """Add annotations (boxes, arrows) to a populated cell.

        Parameters
        ----------
        ax
            Target matplotlib Axes object (with image already displayed).
        encoder_key
            Encoder identifier.
        scenario_key
            Scenario identifier.
        """
        # Get annotations for this (encoder, scenario) pair
        annots = self.annotations.get((encoder_key, scenario_key), [])

        # Determine image bounds from loaded images (fall back to 224 default)
        img = self.images.get((encoder_key, scenario_key))
        if img is not None:
            img_h, img_w = img.shape[0], img.shape[1]
        else:
            img_h, img_w = 224, 224
            if annots:
                print(
                    f"Warning: No loaded image for {encoder_key}/{scenario_key}; "
                    f"clipping annotations to default 224x224 bounds"
                )

        for annot in annots:
            annot_type = annot.get("type")
            label = annot.get("label", "")
            color = annot.get("color", "yellow")

            if annot_type == "box":
                # Bounding box
                coords = annot.get("coords", [])
                if len(coords) != 4:
                    print(f"Warning: Invalid box coords for {encoder_key}/{scenario_key}: {coords}")
                    continue

                x1, y1, x2, y2 = coords
                width = x2 - x1
                height = y2 - y1

                # Clip to image bounds
                x1 = max(0, min(img_w, x1))
                y1 = max(0, min(img_h, y1))
                width = max(0, min(img_w - x1, width))
                height = max(0, min(img_h - y1, height))

                rect = plt.Rectangle(
                    (x1, y1), width, height,
                    linewidth=annot.get("linewidth", 2),
                    edgecolor=color,
                    facecolor='none',
                )
                ax.add_patch(rect)

                # Add label
                if label:
                    ax.text(
                        x1, y1 - 5,
                        label,
                        color=color,
                        fontsize=8,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6),
                    )

            elif annot_type == "arrow":
                # Arrow annotation
                start = annot.get("start", [])
                end = annot.get("end", [])
                if len(start) != 2 or len(end) != 2:
                    print(f"Warning: Invalid arrow coords for {encoder_key}/{scenario_key}")
                    continue

                x_start, y_start = start
                x_end, y_end = end

                # Clip to image bounds
                x_start = max(0, min(img_w, x_start))
                y_start = max(0, min(img_h, y_start))
                x_end = max(0, min(img_w, x_end))
                y_end = max(0, min(img_h, y_end))

                ax.annotate(
                    '',  # No text at arrow
                    xy=(x_end, y_end),
                    xytext=(x_start, y_start),
                    arrowprops=dict(
                        arrowstyle='->',
                        color=color,
                        lw=annot.get("width", 3),
                    ),
                )

                # Add label near start point
                if label:
                    ax.text(
                        x_start, y_start - 10,
                        label,
                        color=color,
                        fontsize=8,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6),
                    )

    def generate_caption(self) -> str:
        """Generate figure caption with VQ fallback caveat if needed.

        Checks figures_method_report.json in input directory for VQ fallback status.

        Returns
        -------
        caption
            Figure caption text.
        """
        num_encoders = len(self.ENCODERS)
        base_caption = (
            f"Attribution maps for {num_encoders} encoders across 4 driving scenarios. "
            "Heatmaps highlight regions of high activation for each encoder. "
        )

        # Check if VQ-VAE used fallback
        report_path = self.input_dir / "figures_method_report.json"
        if report_path.exists():
            try:
                with open(report_path) as f:
                    report = json.load(f)

                vqvae_info = report.get("encoders", {}).get("vq_track", {})
                if vqvae_info.get("fallback_used", False):
                    # Add fallback caveat
                    vq_caveat = (
                        "VQ-VAE used DINOv2-S/14 fallback (VQGAN checkpoint load failed); "
                        "vq_track row does not represent independent VQ performance."
                    )
                    return base_caption + vq_caveat
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse method report: {e}")

        return base_caption

    def generate(self) -> Path:
        """Generate complete attribution grid figure PDF.

        Orchestrates the full pipeline:
        1. Load images
        2. Parse annotation config (if provided)
        3. Create grid figure layout
        4. Populate all cells
        5. Add annotations
        6. Add row labels and column headers
        7. Add caption
        8. Export to PDF at specified DPI

        Returns
        -------
        output_path
            Path to generated PDF file.
        """
        print(f"Loading images from {self.input_dir}...")
        self.load_images()

        if self.annotation_config:
            print(f"Loading annotations from {self.annotation_config}...")
            self.parse_annotation_config()

        print("Creating grid figure...")
        fig, axes = self.create_grid_figure()

        # Populate header row (row 0) with scenario names
        for col_idx, (scenario_key, scenario_name) in enumerate(self.SCENARIOS):
            ax = axes[0, col_idx]
            ax.text(
                0.5, 0.5,
                scenario_name,
                ha='center',
                va='center',
                fontsize=14,
                fontweight='bold',
                transform=ax.transAxes,
            )
            ax.axis('off')

        # Populate encoder rows (rows 1-6)
        n_total_rows = len(self.ENCODERS) + 1
        for row_idx, encoder_key in enumerate(self.ENCODERS, start=1):
            # Add encoder label on the left (use display name from central registry)
            encoder_name = ENCODER_DISPLAY.get(encoder_key, encoder_key)
            fig.text(
                0.02,  # Far left
                1 - (row_idx + 0.5) / n_total_rows,  # Vertical position (centered in row)
                encoder_name,
                ha='left',
                va='center',
                fontsize=12,
                fontweight='bold',
            )

            # Populate cells for this encoder
            for col_idx, (scenario_key, _) in enumerate(self.SCENARIOS):
                ax = axes[row_idx, col_idx]

                # Get image (None if not available)
                image = self.images.get((encoder_key, scenario_key))

                # Populate cell
                self.populate_cell(ax, image, encoder_key, scenario_key)

                # Add annotations if image exists
                if image is not None:
                    self.add_annotations(ax, encoder_key, scenario_key)

        # Add caption at bottom
        caption = self.generate_caption()
        fig.text(
            0.5, 0.01,
            caption,
            ha='center',
            va='bottom',
            fontsize=9,
            wrap=True,
        )

        # Export to PDF
        print(f"Exporting to {self.output_path}...")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(self.output_path) as pdf:
            pdf.savefig(fig, dpi=self.dpi, bbox_inches='tight')

        plt.close(fig)

        print(f"✓ Generated: {self.output_path}")
        return self.output_path


def get_encoder_display_name(encoder_key: str) -> str:
    """Map encoder key to human-readable display name.

    Parameters
    ----------
    encoder_key
        Encoder identifier using M1 canonical keys (e.g., "vit_s16", "dino_vits14").

    Returns
    -------
    str
        Display name (e.g., "ViT-S/16").
    """
    return ENCODER_DISPLAY.get(encoder_key, encoder_key)


def get_scenario_display_name(scenario_key: str) -> str:
    """Map scenario key to human-readable display name.

    Parameters
    ----------
    scenario_key
        Scenario identifier (e.g., "intersection").

    Returns
    -------
    str
        Display name (e.g., "Intersection").
    """
    mapping = {
        "intersection": "Intersection",
        "other": "Other",
        "highway": "Highway",
        "urban": "Urban",
    }
    return mapping.get(scenario_key, scenario_key.capitalize())
