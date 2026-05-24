"""Unit tests for attribution grid figure generation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from evaluation.attribution_grid import (
    AttributionGridGenerator,
    get_encoder_display_name,
    get_scenario_display_name,
)


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_encoder_display_name(self):
        """Test encoder key to display name mapping."""
        assert get_encoder_display_name("vit_s16") == "ViT-S/16"
        assert get_encoder_display_name("dinov2_s14") == "DINOv2-S/14"
        assert get_encoder_display_name("clip_b32") == "CLIP ViT-B/32"
        assert get_encoder_display_name("vqvae") == "VQ-VAE"
        assert get_encoder_display_name("vjepa2") == "V-JEPA2"
        assert get_encoder_display_name("vjepa2_rep1") == "V-JEPA2 (1-frame)"
        assert get_encoder_display_name("unknown") == "unknown"

    def test_get_scenario_display_name(self):
        """Test scenario key to display name mapping."""
        assert get_scenario_display_name("intersection") == "Intersection"
        assert get_scenario_display_name("other") == "Other"
        assert get_scenario_display_name("highway") == "Highway"
        assert get_scenario_display_name("urban") == "Urban"
        assert get_scenario_display_name("custom") == "Custom"


class TestAttributionGridGenerator:
    """Test AttributionGridGenerator class."""

    @pytest.fixture
    def temp_input_dir(self, tmp_path):
        """Create temporary input directory with mock PNG files."""
        input_dir = tmp_path / "attribution"
        input_dir.mkdir()

        # Create mock PNG files for intersection and other scenarios
        encoders = ["vit_s16", "dinov2_s14", "clip_b32", "vqvae", "vjepa2"]
        scenarios = ["intersection", "other"]

        for encoder in encoders:
            for scenario in scenarios:
                # Create a simple 224×224×3 image
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

                # Save as PNG
                from PIL import Image
                img = Image.fromarray(img_array)
                filepath = input_dir / f"{encoder}_{scenario}_00.png"
                img.save(filepath)

        return input_dir

    @pytest.fixture
    def temp_annotation_config(self, tmp_path):
        """Create temporary annotation config file."""
        config_path = tmp_path / "annotations.json"
        config = {
            "version": "1.0.0",
            "annotations": {
                "vit_s16": {
                    "intersection": [
                        {
                            "type": "box",
                            "coords": [50, 80, 150, 180],
                            "label": "Lane marking",
                            "color": "yellow",
                            "linewidth": 2,
                        }
                    ]
                }
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)

        return config_path

    def test_init(self, tmp_path):
        """Test AttributionGridGenerator initialization."""
        input_dir = tmp_path / "attribution"
        output_path = tmp_path / "grid.pdf"

        generator = AttributionGridGenerator(
            input_dir=input_dir,
            output_path=output_path,
            frame_index=0,
            dpi=300,
        )

        assert generator.input_dir == input_dir
        assert generator.output_path == output_path
        assert generator.frame_index == 0
        assert generator.dpi == 300
        assert generator.annotation_config is None
        assert generator.images == {}
        assert generator.annotations == {}

    def test_load_images_missing_dir(self, tmp_path):
        """Test load_images with non-existent directory."""
        generator = AttributionGridGenerator(
            input_dir=tmp_path / "nonexistent",
            output_path=tmp_path / "grid.pdf",
        )

        with pytest.raises(FileNotFoundError, match="Input directory not found"):
            generator.load_images()

    def test_load_images(self, temp_input_dir, tmp_path):
        """Test loading images from input directory."""
        generator = AttributionGridGenerator(
            input_dir=temp_input_dir,
            output_path=tmp_path / "grid.pdf",
        )

        images = generator.load_images()

        # Should load 10 images (5 encoders × 2 discovered scenarios)
        assert len(images) == 10
        # Verify dynamic discovery worked
        assert generator._discover_available_scenarios() == ["intersection", "other"]

        # Check specific keys
        assert ("vit_s16", "intersection") in images
        assert ("vit_s16", "other") in images
        assert ("dinov2_s14", "intersection") in images

        # Check image shapes
        for img in images.values():
            assert img.shape == (224, 224, 3)

    def test_parse_annotation_config_none(self, tmp_path):
        """Test parsing annotations when no config provided."""
        generator = AttributionGridGenerator(
            input_dir=tmp_path,
            output_path=tmp_path / "grid.pdf",
            annotation_config=None,
        )

        annotations = generator.parse_annotation_config()
        assert annotations == {}

    def test_parse_annotation_config(self, temp_annotation_config, tmp_path):
        """Test parsing valid annotation config."""
        generator = AttributionGridGenerator(
            input_dir=tmp_path,
            output_path=tmp_path / "grid.pdf",
            annotation_config=temp_annotation_config,
        )

        annotations = generator.parse_annotation_config()

        assert ("vit_s16", "intersection") in annotations
        annot_list = annotations[("vit_s16", "intersection")]
        assert len(annot_list) == 1
        assert annot_list[0]["type"] == "box"
        assert annot_list[0]["label"] == "Lane marking"

    def test_parse_annotation_config_invalid_json(self, tmp_path):
        """Test parsing malformed JSON."""
        config_path = tmp_path / "bad.json"
        with open(config_path, 'w') as f:
            f.write("{ invalid json")

        generator = AttributionGridGenerator(
            input_dir=tmp_path,
            output_path=tmp_path / "grid.pdf",
            annotation_config=config_path,
        )

        with pytest.raises(ValueError, match="Invalid JSON"):
            generator.parse_annotation_config()

    def test_create_grid_figure(self, tmp_path):
        """Test grid figure creation."""
        generator = AttributionGridGenerator(
            input_dir=tmp_path,
            output_path=tmp_path / "grid.pdf",
        )

        fig, axes = generator.create_grid_figure()

        # Check axes shape: 6 rows × 4 columns
        assert axes.shape == (6, 4)

        # Check that all axes are Axes objects
        for i in range(6):
            for j in range(4):
                assert axes[i, j] is not None

    def test_populate_cell_with_image(self, tmp_path):
        """Test populating cell with image."""
        generator = AttributionGridGenerator(
            input_dir=tmp_path,
            output_path=tmp_path / "grid.pdf",
        )

        fig, axes = generator.create_grid_figure()
        ax = axes[1, 0]

        # Create mock image
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Populate cell
        generator.populate_cell(ax, image, "vit_s16", "intersection")

        # Check that axis is turned off
        assert not ax.axison

    def test_populate_cell_empty(self, tmp_path):
        """Test populating empty cell with placeholder."""
        generator = AttributionGridGenerator(
            input_dir=tmp_path,
            output_path=tmp_path / "grid.pdf",
        )

        fig, axes = generator.create_grid_figure()
        ax = axes[1, 2]  # Highway column (empty)

        # Populate with None
        generator.populate_cell(ax, None, "vit_s16", "highway")

        # Check that axis is turned off
        assert not ax.axison

    def test_generate_caption_no_fallback(self, tmp_path):
        """Test caption generation without VQ fallback."""
        input_dir = tmp_path / "attribution"
        input_dir.mkdir()

        # Create method report with fallback_used=False
        report = {
            "encoders": {
                "vqvae": {
                    "fallback_used": False
                }
            }
        }
        report_path = input_dir / "figures_method_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f)

        generator = AttributionGridGenerator(
            input_dir=input_dir,
            output_path=tmp_path / "grid.pdf",
        )

        caption = generator.generate_caption()

        assert "Attribution maps" in caption
        assert "VQ" not in caption  # No fallback mention when VQ is real
        assert "DINOv2" not in caption

    def test_generate_caption_with_fallback(self, tmp_path):
        """Test caption generation with VQ fallback."""
        input_dir = tmp_path / "attribution"
        input_dir.mkdir()

        # Create mock figures_method_report.json with fallback
        report = {
            "encoders": {
                "vqvae": {
                    "fallback_used": True
                }
            }
        }
        report_path = input_dir / "figures_method_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f)

        generator = AttributionGridGenerator(
            input_dir=input_dir,
            output_path=tmp_path / "grid.pdf",
        )

        caption = generator.generate_caption()

        assert "Attribution maps" in caption
        assert "VQ-VAE" in caption
        assert "fallback" in caption
        assert "DINOv2" in caption  # Mentions DINOv2 as the fallback

    @patch('evaluation.attribution_grid.PdfPages')
    def test_generate_integration(self, mock_pdf, temp_input_dir, tmp_path):
        """Integration test for full grid generation."""
        output_path = tmp_path / "grid.pdf"

        generator = AttributionGridGenerator(
            input_dir=temp_input_dir,
            output_path=output_path,
        )

        # Mock PdfPages to avoid actual file I/O
        mock_pdf_instance = MagicMock()
        mock_pdf.return_value.__enter__.return_value = mock_pdf_instance

        result_path = generator.generate()

        # Check that output path is returned
        assert result_path == output_path

        # Check that PDF was saved
        mock_pdf.assert_called_once()
        mock_pdf_instance.savefig.assert_called_once()

    def test_discover_available_scenarios(self, temp_input_dir, tmp_path):
        """Test dynamic scenario discovery from PNG files."""
        generator = AttributionGridGenerator(
            input_dir=temp_input_dir,
            output_path=tmp_path / "grid.pdf",
        )

        scenarios = generator._discover_available_scenarios()

        # Should discover exactly 2 scenarios from fixture
        assert len(scenarios) == 2
        assert "intersection" in scenarios
        assert "other" in scenarios

        # Should be in canonical order
        assert scenarios == ["intersection", "other"]

    def test_discover_scenarios_preserves_canonical_order(self, tmp_path):
        """Test that discovered scenarios maintain canonical ordering."""
        input_dir = tmp_path / "attribution"
        input_dir.mkdir()

        # Create files in reverse order: urban, highway, other, intersection
        encoders = ["vit_s16"]
        scenarios = ["urban", "highway", "other", "intersection"]

        for encoder in encoders:
            for scenario in scenarios:
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                from PIL import Image
                img = Image.fromarray(img_array)
                filepath = input_dir / f"{encoder}_{scenario}_00.png"
                img.save(filepath)

        generator = AttributionGridGenerator(
            input_dir=input_dir,
            output_path=tmp_path / "grid.pdf",
        )

        discovered = generator._discover_available_scenarios()

        # Should return all 4 in canonical order (not file discovery order)
        assert discovered == ["intersection", "other", "highway", "urban"]

    def test_discover_scenarios_empty_directory(self, tmp_path):
        """Test discovery with no PNG files."""
        input_dir = tmp_path / "empty"
        input_dir.mkdir()

        generator = AttributionGridGenerator(
            input_dir=input_dir,
            output_path=tmp_path / "grid.pdf",
        )

        scenarios = generator._discover_available_scenarios()
        assert scenarios == []

    def test_discover_scenarios_ignores_unknown(self, tmp_path):
        """Test that unknown scenarios are filtered out."""
        input_dir = tmp_path / "attribution"
        input_dir.mkdir()

        # Create file with unknown scenario name
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        from PIL import Image
        img = Image.fromarray(img_array)
        filepath = input_dir / "vit_s16_unknown_scenario_00.png"
        img.save(filepath)

        generator = AttributionGridGenerator(
            input_dir=input_dir,
            output_path=tmp_path / "grid.pdf",
        )

        scenarios = generator._discover_available_scenarios()
        assert scenarios == []  # Unknown scenario filtered out


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_annotation_out_of_bounds(self, tmp_path):
        """Test that out-of-bounds annotation coords are clipped."""
        generator = AttributionGridGenerator(
            input_dir=tmp_path,
            output_path=tmp_path / "grid.pdf",
        )

        # Create figure and axes
        fig, axes = generator.create_grid_figure()
        ax = axes[1, 0]

        # Display a mock image
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        ax.imshow(image)

        # Add annotation with out-of-bounds coords
        generator.annotations = {
            ("vit_s16", "intersection"): [
                {
                    "type": "box",
                    "coords": [-10, -10, 300, 300],  # Out of bounds
                    "label": "Test",
                    "color": "red",
                }
            ]
        }

        # Should not raise error (coords get clipped)
        generator.add_annotations(ax, "vit_s16", "intersection")

    def test_missing_annotation_fields(self, tmp_path):
        """Test handling of annotation with missing fields."""
        generator = AttributionGridGenerator(
            input_dir=tmp_path,
            output_path=tmp_path / "grid.pdf",
        )

        fig, axes = generator.create_grid_figure()
        ax = axes[1, 0]

        # Annotation with missing coords
        generator.annotations = {
            ("vit_s16", "intersection"): [
                {
                    "type": "box",
                    # Missing coords
                    "label": "Test",
                }
            ]
        }

        # Should not raise error (warning printed)
        generator.add_annotations(ax, "vit_s16", "intersection")
