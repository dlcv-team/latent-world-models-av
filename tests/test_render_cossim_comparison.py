#!/usr/bin/env python3
"""Tests for figures/render_cossim_comparison.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.figures.render_cossim_comparison import (
    load_cossim_csv,
    build_caption,
    render_cossim_comparison,
    main,
    DPI,
    SINGLESEED_COLUMNS,
    MULTISEED_COLUMNS,
)


def test_dpi_is_300():
    """Verify DPI constant is set to 300 per canonical contract."""
    assert DPI == 300, "DPI must be 300 for publication-ready figures"


def test_load_cossim_csv_single_seed():
    """Verify single-seed CSV parsing normalizes to multi-seed schema."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = Path(f.name)
        f.write("k,cossim_conditioned,cossim_unconditioned,delta_cossim\n")
        f.write("1,0.95,0.90,0.05\n")
        f.write("2,0.93,0.88,0.05\n")
        f.write("3,0.91,0.86,0.05\n")

    try:
        df, has_ci = load_cossim_csv(csv_path)

        assert not has_ci, "Single-seed CSV should have has_ci=False"
        assert len(df) == 3, "Should have 3 rows"

        # Check that all expected columns are present (order doesn't matter)
        expected_cols = {
            "k",
            "cond_mean",
            "cond_ci95_lo",
            "cond_ci95_hi",
            "uncond_mean",
            "uncond_ci95_lo",
            "uncond_ci95_hi",
            "delta_mean",
            "delta_ci95_lo",
            "delta_ci95_hi",
        }
        assert set(df.columns) == expected_cols, "Should normalize to multi-seed schema"

        # Verify CI bounds equal mean for single-seed
        assert (df["cond_ci95_lo"] == df["cond_mean"]).all()
        assert (df["cond_ci95_hi"] == df["cond_mean"]).all()
        assert (df["uncond_ci95_lo"] == df["uncond_mean"]).all()
        assert (df["uncond_ci95_hi"] == df["uncond_mean"]).all()

        # Verify values
        assert df.loc[0, "cond_mean"] == 0.95
        assert df.loc[0, "uncond_mean"] == 0.90
        assert df.loc[0, "delta_mean"] == 0.05

    finally:
        csv_path.unlink()


def test_load_cossim_csv_multi_seed():
    """Verify multi-seed CSV with CI columns is loaded correctly."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = Path(f.name)
        f.write(
            "k,cond_mean,cond_ci95_lo,cond_ci95_hi,"
            "uncond_mean,uncond_ci95_lo,uncond_ci95_hi,"
            "delta_mean,delta_ci95_lo,delta_ci95_hi\n"
        )
        f.write("1,0.95,0.94,0.96,0.90,0.89,0.91,0.05,0.04,0.06\n")
        f.write("2,0.93,0.92,0.94,0.88,0.87,0.89,0.05,0.04,0.06\n")

    try:
        df, has_ci = load_cossim_csv(csv_path)

        assert has_ci, "Multi-seed CSV should have has_ci=True"
        assert len(df) == 2, "Should have 2 rows"

        # Check values and CI bounds
        assert df.loc[0, "cond_mean"] == 0.95
        assert df.loc[0, "cond_ci95_lo"] == 0.94
        assert df.loc[0, "cond_ci95_hi"] == 0.96

    finally:
        csv_path.unlink()


def test_missing_csv_raises_clear_error():
    """Check error handling for missing CSV file."""
    nonexistent_path = Path("/tmp/nonexistent_12345.csv")

    with pytest.raises(FileNotFoundError) as exc_info:
        load_cossim_csv(nonexistent_path)

    assert "CosSim CSV not found" in str(exc_info.value)
    assert str(nonexistent_path) in str(exc_info.value)


def test_empty_csv_raises_error():
    """Verify empty CSV is rejected."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = Path(f.name)
        f.write("k,cossim_conditioned,cossim_unconditioned,delta_cossim\n")
        # No data rows

    try:
        with pytest.raises(ValueError) as exc_info:
            load_cossim_csv(csv_path)

        assert "no rows" in str(exc_info.value).lower()
    finally:
        csv_path.unlink()


def test_invalid_schema_raises_error():
    """Verify CSV with wrong columns is rejected."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = Path(f.name)
        f.write("wrong,columns,here\n")
        f.write("1,2,3\n")

    try:
        with pytest.raises(ValueError) as exc_info:
            load_cossim_csv(csv_path)

        assert "Unrecognized CSV schema" in str(exc_info.value)
        assert "Single-seed" in str(exc_info.value)
        assert "Multi-seed" in str(exc_info.value)
    finally:
        csv_path.unlink()


def test_non_contiguous_k_raises_error():
    """Verify k column must be 1..N contiguous."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = Path(f.name)
        f.write("k,cossim_conditioned,cossim_unconditioned,delta_cossim\n")
        f.write("1,0.95,0.90,0.05\n")
        f.write("3,0.93,0.88,0.05\n")  # Skip k=2

    try:
        with pytest.raises(ValueError) as exc_info:
            load_cossim_csv(csv_path)

        assert "'k' column must be 1..N contiguous" in str(exc_info.value)
    finally:
        csv_path.unlink()


def test_inconsistent_delta_raises_error():
    """Verify delta_mean is validated against cond_mean - uncond_mean."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = Path(f.name)
        f.write("k,cossim_conditioned,cossim_unconditioned,delta_cossim\n")
        f.write("1,0.95,0.90,0.99\n")  # Wrong delta (should be 0.05)

    try:
        with pytest.raises(ValueError) as exc_info:
            load_cossim_csv(csv_path)

        assert "delta_mean inconsistent" in str(exc_info.value)
    finally:
        csv_path.unlink()


def test_invalid_ci_bounds_raises_error():
    """Verify CI bounds are validated (lo <= mean <= hi)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = Path(f.name)
        f.write(
            "k,cond_mean,cond_ci95_lo,cond_ci95_hi,"
            "uncond_mean,uncond_ci95_lo,uncond_ci95_hi,"
            "delta_mean,delta_ci95_lo,delta_ci95_hi\n"
        )
        # cond_ci95_lo > cond_mean (invalid)
        f.write("1,0.95,0.96,0.97,0.90,0.89,0.91,0.05,0.04,0.06\n")

    try:
        with pytest.raises(ValueError) as exc_info:
            load_cossim_csv(csv_path)

        assert "cond_ci95_lo > cond_mean" in str(exc_info.value)
    finally:
        csv_path.unlink()


def test_out_of_range_cossim_raises_error():
    """Verify cosine similarity values are in [-1, 1]."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = Path(f.name)
        f.write("k,cossim_conditioned,cossim_unconditioned,delta_cossim\n")
        f.write("1,1.5,0.90,0.60\n")  # cond > 1 (invalid), delta = 1.5 - 0.90 = 0.60

    try:
        with pytest.raises(ValueError) as exc_info:
            load_cossim_csv(csv_path)

        assert "outside [-1, 1]" in str(exc_info.value)
    finally:
        csv_path.unlink()


def test_build_caption_single_seed():
    """Verify caption for single-seed data."""
    caption = build_caption(has_ci=False)

    # Should mention trainval-mirror subset
    assert "trainval-mirror" in caption.lower()
    # Should mention single-seed
    assert "single-seed" in caption.lower()
    assert "no CI" in caption or "no ci" in caption.lower()


def test_build_caption_multi_seed():
    """Verify caption for multi-seed data."""
    caption = build_caption(has_ci=True)

    # Should mention trainval-mirror subset
    assert "trainval-mirror" in caption.lower()
    # Should mention multi-seed and CI
    assert "multi-seed" in caption.lower()
    assert "95%" in caption or "CI" in caption


def test_render_cossim_comparison_creates_png_and_pdf():
    """Verify render creates both PNG and PDF outputs."""
    # Create test data
    df = pd.DataFrame(
        {
            "k": [1, 2, 3, 4],
            "cond_mean": [0.95, 0.93, 0.91, 0.89],
            "cond_ci95_lo": [0.94, 0.92, 0.90, 0.88],
            "cond_ci95_hi": [0.96, 0.94, 0.92, 0.90],
            "uncond_mean": [0.90, 0.88, 0.86, 0.84],
            "uncond_ci95_lo": [0.89, 0.87, 0.85, 0.83],
            "uncond_ci95_hi": [0.91, 0.89, 0.87, 0.85],
            "delta_mean": [0.05, 0.05, 0.05, 0.05],
            "delta_ci95_lo": [0.04, 0.04, 0.04, 0.04],
            "delta_ci95_hi": [0.06, 0.06, 0.06, 0.06],
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        render_cossim_comparison(df, has_ci=True, metadata={}, out_dir=out_dir)

        # Check both files were created
        png_path = out_dir / "cossim_comparison.png"
        pdf_path = out_dir / "cossim_comparison.pdf"

        assert png_path.exists(), "PNG should be created"
        assert pdf_path.exists(), "PDF should be created"

        assert png_path.stat().st_size > 1000, "PNG should have content"
        assert pdf_path.stat().st_size > 1000, "PDF should have content"

        # Verify PNG magic bytes
        with png_path.open("rb") as f:
            magic = f.read(8)
            assert magic == b"\x89PNG\r\n\x1a\n", "Output should be valid PNG"

        # Verify PDF magic bytes
        with pdf_path.open("rb") as f:
            magic = f.read(4)
            assert magic == b"%PDF", "Output should be valid PDF"


def test_main_with_default_paths():
    """Verify main() works with custom output directory."""
    # Create temporary CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = Path(f.name)
        f.write("k,cossim_conditioned,cossim_unconditioned,delta_cossim\n")
        f.write("1,0.95,0.90,0.05\n")
        f.write("2,0.93,0.88,0.05\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        try:
            exit_code = main(
                [
                    "--cossim-csv",
                    str(csv_path),
                    "--out-dir",
                    str(out_dir),
                ]
            )

            assert exit_code == 0, "main() should return 0 on success"
            assert (out_dir / "cossim_comparison.png").exists()
            assert (out_dir / "cossim_comparison.pdf").exists()

        finally:
            csv_path.unlink()


def test_load_cossim_csv_with_vendored_artifact():
    """Verify loading the actual vendored artifact from artifacts/cossim_eval/."""
    csv_path = Path(__file__).parent.parent / "artifacts" / "cossim_eval" / "cossim_results.csv"

    if not csv_path.exists():
        pytest.skip(f"Vendored artifact not found at {csv_path}")

    df, has_ci = load_cossim_csv(csv_path)

    # Should parse successfully
    assert len(df) > 0, "Vendored CSV should have data"
    assert "k" in df.columns
    assert "cond_mean" in df.columns
    assert "uncond_mean" in df.columns
    assert "delta_mean" in df.columns

    # Verify k is contiguous starting from 1
    assert list(df["k"]) == list(range(1, len(df) + 1))


def test_singleseed_columns_tuple():
    """Verify SINGLESEED_COLUMNS constant is correct."""
    assert SINGLESEED_COLUMNS == (
        "k",
        "cossim_conditioned",
        "cossim_unconditioned",
        "delta_cossim",
    )


def test_multiseed_columns_tuple():
    """Verify MULTISEED_COLUMNS constant is correct."""
    assert len(MULTISEED_COLUMNS) == 10
    assert MULTISEED_COLUMNS[0] == "k"
    assert "cond_mean" in MULTISEED_COLUMNS
    assert "delta_ci95_hi" in MULTISEED_COLUMNS
