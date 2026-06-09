#!/usr/bin/env python3
"""Tests for figures/render_dit_architecture.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from figures.render_dit_architecture import (
    render_dit_architecture,
    main,
    Z_DIM,
    N_BLOCKS,
    N_HEADS,
    HORIZON,
    MLP_RATIO,
    COND_DIM,
    DPI,
)


def test_render_dit_architecture_creates_pdf():
    """Verify PDF is created with correct structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_diagram.pdf"
        render_dit_architecture(out_path)

        assert out_path.exists(), "PDF should be created"
        assert out_path.stat().st_size > 1000, "PDF should have content (>1KB)"

        # Verify it's a valid PDF by checking magic bytes
        with out_path.open("rb") as f:
            magic = f.read(4)
            assert magic == b"%PDF", "Output should be a valid PDF file"


def test_dit_config_matches_yaml():
    """Ensure hardcoded constants match configs/dit.yaml."""
    config_path = Path(__file__).parent.parent / "configs" / "dit.yaml"

    if not config_path.exists():
        pytest.skip(f"DiT config not found at {config_path}")

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    dit_cfg = config["dit"]

    # Verify all hardcoded constants match the config file
    assert Z_DIM == dit_cfg["z_dim"], f"Z_DIM mismatch: {Z_DIM} != {dit_cfg['z_dim']}"
    assert COND_DIM == dit_cfg["cond_dim"], f"COND_DIM mismatch: {COND_DIM} != {dit_cfg['cond_dim']}"
    assert N_BLOCKS == dit_cfg["n_blocks"], f"N_BLOCKS mismatch: {N_BLOCKS} != {dit_cfg['n_blocks']}"
    assert N_HEADS == dit_cfg["n_heads"], f"N_HEADS mismatch: {N_HEADS} != {dit_cfg['n_heads']}"
    assert HORIZON == dit_cfg["horizon"], f"HORIZON mismatch: {HORIZON} != {dit_cfg['horizon']}"
    assert MLP_RATIO == dit_cfg["mlp_ratio"], f"MLP_RATIO mismatch: {MLP_RATIO} != {dit_cfg['mlp_ratio']}"


def test_main_creates_output_in_default_dir():
    """Verify main() creates output in specified directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "figures"
        exit_code = main(["--out-dir", str(out_dir)])

        assert exit_code == 0, "main() should return 0 on success"

        expected_file = out_dir / "dit_architecture_diagram.pdf"
        assert expected_file.exists(), f"Expected output at {expected_file}"
        assert expected_file.stat().st_size > 1000, "PDF should have content"


def test_main_creates_parent_directories():
    """Verify main() creates parent directories if they don't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use a nested path that doesn't exist
        out_dir = Path(tmpdir) / "nested" / "path" / "figures"
        assert not out_dir.exists(), "Directory should not exist before test"

        exit_code = main(["--out-dir", str(out_dir)])

        assert exit_code == 0, "main() should succeed even with missing parent dirs"
        assert out_dir.exists(), "Parent directories should be created"
        assert (out_dir / "dit_architecture_diagram.pdf").exists()


def test_dpi_is_300():
    """Verify DPI constant is set to 300 per canonical contract."""
    assert DPI == 300, "DPI must be 300 for publication-ready figures"


def test_output_is_pdf_format_only():
    """Verify render_dit_architecture only creates PDF (no PNG)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.pdf"
        render_dit_architecture(out_path)

        # Should create PDF
        assert out_path.exists(), "PDF should be created"

        # Should NOT create PNG sibling
        png_path = out_path.with_suffix(".png")
        assert not png_path.exists(), "Should not auto-create PNG (PDF only)"


def test_render_different_output_paths():
    """Verify render works with different output path names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test different filenames
        for filename in ["diagram.pdf", "my_figure.pdf", "test123.pdf"]:
            out_path = tmpdir / filename
            render_dit_architecture(out_path)
            assert out_path.exists(), f"Should create {filename}"

            # Cleanup for next iteration
            out_path.unlink()
