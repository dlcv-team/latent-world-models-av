"""Unit tests for scripts/build_cossim_figures.py.

Covers:
- CSV loader: schema validation, k contiguity, single-seed and multi-seed format support
- Fig 3: file generation, DPI correctness, caption content (with/without error bars)
- Fig 4: file generation, DPI correctness, bar coloring (with/without error bars)
- CLI: end-to-end integration
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from PIL import Image

from scripts import build_cossim_figures as bcf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_3seed_csv(tmp_path: Path, *, k_values: list[int] | None = None) -> Path:
    """Create a mock 3-seed CosSim CSV with valid schema."""
    if k_values is None:
        k_values = [1, 2, 3, 4]

    rows = []
    for k in k_values:
        rows.append(
            {
                "k": k,
                "cond_mean": 0.9955,
                "cond_ci95_lo": 0.9954,
                "cond_ci95_hi": 0.9956,
                "uncond_mean": 0.9999,
                "uncond_ci95_lo": 0.9998,
                "uncond_ci95_hi": 0.9999,
                "delta_mean": -0.0044,
                "delta_ci95_lo": -0.0045,
                "delta_ci95_hi": -0.0043,
            }
        )
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "cossim_results.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _mock_single_run_csv(tmp_path: Path) -> Path:
    """Create a mock single-run CSV (old PR#21 schema)."""
    rows = [
        {
            "k": 1,
            "cossim_conditioned": 0.9955,
            "cossim_unconditioned": 0.9999,
            "delta_cossim": -0.0044,
        },
        {
            "k": 2,
            "cossim_conditioned": 0.9955,
            "cossim_unconditioned": 0.9999,
            "delta_cossim": -0.0044,
        },
    ]
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "cossim_single_run.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# CSV loader tests
# ---------------------------------------------------------------------------


def test_load_cossim_csv_roundtrip_multiseed(tmp_path):
    """Valid multi-seed CSV loads successfully."""
    csv_path = _mock_3seed_csv(tmp_path)
    df, has_ci = bcf.load_cossim_csv(csv_path)
    assert list(df["k"]) == [1, 2, 3, 4]
    assert "cond_mean" in df.columns
    assert "uncond_ci95_hi" in df.columns
    assert has_ci is True


def test_load_cossim_csv_roundtrip_singleseed(tmp_path):
    """Valid single-seed CSV loads and normalizes to multi-seed schema."""
    csv_path = _mock_single_run_csv(tmp_path)
    df, has_ci = bcf.load_cossim_csv(csv_path)
    assert list(df["k"]) == [1, 2]
    assert "cond_mean" in df.columns
    assert "uncond_ci95_hi" in df.columns
    # CI bounds should equal mean (no error bars)
    assert (df["cond_ci95_lo"] == df["cond_mean"]).all()
    assert (df["cond_ci95_hi"] == df["cond_mean"]).all()
    assert has_ci is False


def test_load_cossim_csv_missing_file_raises(tmp_path):
    """Missing CSV raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="CosSim CSV not found"):
        bcf.load_cossim_csv(tmp_path / "nonexistent.csv")


def test_load_cossim_csv_validates_schema(tmp_path):
    """Unrecognized CSV schema raises ValueError."""
    bad_df = pd.DataFrame(
        [
            {"k": 1, "cond_mean": 0.9955, "uncond_mean": 0.9999}
            # Missing CI columns for multi-seed, missing other columns for single-seed
        ]
    )
    csv_path = tmp_path / "bad_schema.csv"
    bad_df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Unrecognized CSV schema"):
        bcf.load_cossim_csv(csv_path)


def test_load_cossim_csv_rejects_noncontiguous_k(tmp_path):
    """k column with gaps (e.g., [1, 2, 4]) raises ValueError."""
    csv_path = _mock_3seed_csv(tmp_path, k_values=[1, 2, 4])
    with pytest.raises(ValueError, match="'k' column must be 1..N contiguous"):
        bcf.load_cossim_csv(csv_path)


def test_load_cossim_csv_empty_raises(tmp_path):
    """Empty CSV raises ValueError."""
    empty_df = pd.DataFrame(columns=list(bcf.MULTISEED_COLUMNS))
    csv_path = tmp_path / "empty.csv"
    empty_df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="has no rows"):
        bcf.load_cossim_csv(csv_path)


# ---------------------------------------------------------------------------
# Fig 3 tests
# ---------------------------------------------------------------------------


def test_fig3_generates_both_formats(tmp_path):
    """Fig 3 generates both PNG and PDF."""
    csv_path = _mock_3seed_csv(tmp_path)
    df, has_ci = bcf.load_cossim_csv(csv_path)
    out_dir = tmp_path / "figures"
    out_dir.mkdir()

    bcf.build_fig3_cossim_lines(df, has_ci, out_dir)

    assert (out_dir / "fig3_cossim_lines.png").exists()
    assert (out_dir / "fig3_cossim_lines.pdf").exists()


def test_fig3_uses_correct_dpi(tmp_path):
    """Fig 3 PNG has DPI=300 in metadata."""
    csv_path = _mock_3seed_csv(tmp_path)
    df, has_ci = bcf.load_cossim_csv(csv_path)
    out_dir = tmp_path / "figures"
    out_dir.mkdir()

    bcf.build_fig3_cossim_lines(df, has_ci, out_dir)

    img = Image.open(out_dir / "fig3_cossim_lines.png")
    dpi = img.info.get("dpi")
    # Allow floating-point tolerance (matplotlib may store 299.9994)
    assert dpi is not None, "PNG missing DPI metadata"
    assert abs(dpi[0] - 300) < 0.01, f"Expected DPI≈300, got {dpi[0]}"
    assert abs(dpi[1] - 300) < 0.01, f"Expected DPI≈300, got {dpi[1]}"


def test_fig3_caption_multiseed(tmp_path):
    """Fig 3 multi-seed caption includes multi-seed + bootstrap CI."""
    csv_path = _mock_3seed_csv(tmp_path)
    df, has_ci = bcf.load_cossim_csv(csv_path)
    out_dir = tmp_path / "figures"
    out_dir.mkdir()

    # Render figure and inspect actual text artists (not mocked)
    import matplotlib.pyplot as plt
    bcf.build_fig3_cossim_lines(df, has_ci, out_dir)

    # Re-create the figure to inspect its text artists
    # (build_fig3_cossim_lines closes the figure after saving)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.text(
        0.98, 0.02,
        bcf._build_caption(has_ci),
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        color="gray",
    )

    # Check text content
    caption_text = bcf._build_caption(has_ci)
    assert "trainval-mirror subset (180/20/40, seed 42)" in caption_text
    assert "multi-seed" in caption_text
    assert "95% bootstrap CI" in caption_text
    plt.close(fig)


def test_fig3_caption_singleseed(tmp_path):
    """Fig 3 single-seed caption notes no CI."""
    csv_path = _mock_single_run_csv(tmp_path)
    df, has_ci = bcf.load_cossim_csv(csv_path)
    out_dir = tmp_path / "figures"
    out_dir.mkdir()

    # Render figure and check caption helper directly
    bcf.build_fig3_cossim_lines(df, has_ci, out_dir)

    caption_text = bcf._build_caption(has_ci)
    assert "single-seed run (no CI)" in caption_text
    assert "multi-seed" not in caption_text


# ---------------------------------------------------------------------------
# Fig 4 tests
# ---------------------------------------------------------------------------


def test_fig4_generates_both_formats(tmp_path):
    """Fig 4 generates both PNG and PDF."""
    csv_path = _mock_3seed_csv(tmp_path)
    df, has_ci = bcf.load_cossim_csv(csv_path)
    out_dir = tmp_path / "figures"
    out_dir.mkdir()

    bcf.build_fig4_delta_bars(df, has_ci, out_dir)

    assert (out_dir / "fig4_delta_bars.png").exists()
    assert (out_dir / "fig4_delta_bars.pdf").exists()


def test_fig4_uses_correct_dpi(tmp_path):
    """Fig 4 PNG has DPI=300 in metadata."""
    csv_path = _mock_3seed_csv(tmp_path)
    df, has_ci = bcf.load_cossim_csv(csv_path)
    out_dir = tmp_path / "figures"
    out_dir.mkdir()

    bcf.build_fig4_delta_bars(df, has_ci, out_dir)

    img = Image.open(out_dir / "fig4_delta_bars.png")
    dpi = img.info.get("dpi")
    assert dpi is not None, "PNG missing DPI metadata"
    assert abs(dpi[0] - 300) < 0.01, f"Expected DPI≈300, got {dpi[0]}"
    assert abs(dpi[1] - 300) < 0.01, f"Expected DPI≈300, got {dpi[1]}"


def test_fig4_colors_bars_by_sign(tmp_path):
    """Fig 4 colors bars by delta sign (green positive, red negative)."""
    # Mock CSV with mixed positive/negative deltas
    rows = [
        {
            "k": 1,
            "cond_mean": 0.9955,
            "cond_ci95_lo": 0.9954,
            "cond_ci95_hi": 0.9956,
            "uncond_mean": 0.9950,  # cond > uncond → positive delta
            "uncond_ci95_lo": 0.9949,
            "uncond_ci95_hi": 0.9951,
            "delta_mean": 0.0005,  # positive
            "delta_ci95_lo": 0.0004,
            "delta_ci95_hi": 0.0006,
        },
        {
            "k": 2,
            "cond_mean": 0.9940,
            "cond_ci95_lo": 0.9939,
            "cond_ci95_hi": 0.9941,
            "uncond_mean": 0.9945,  # cond < uncond → negative delta
            "uncond_ci95_lo": 0.9944,
            "uncond_ci95_hi": 0.9946,
            "delta_mean": -0.0005,  # negative
            "delta_ci95_lo": -0.0006,
            "delta_ci95_hi": -0.0004,
        },
    ]
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "mixed_deltas.csv"
    df.to_csv(csv_path, index=False)

    loaded_df, has_ci = bcf.load_cossim_csv(csv_path)
    out_dir = tmp_path / "figures"
    out_dir.mkdir()

    # Mock ax.bar to capture bar colors
    with mock.patch("matplotlib.pyplot.subplots") as mock_subplots:
        mock_fig = mock.Mock()
        mock_ax = mock.Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        bcf.build_fig4_delta_bars(loaded_df, has_ci, out_dir)

        # Check ax.bar was called with color array
        assert mock_ax.bar.called
        call_kwargs = mock_ax.bar.call_args[1]
        colors = call_kwargs["color"]
        # First bar (positive delta) should be green, second (negative) red
        assert len(colors) == 2
        # Green hex: #6BAA75, Red hex: #D65F5F
        assert colors[0] == "#6BAA75", f"Expected green for positive delta, got {colors[0]}"
        assert colors[1] == "#D65F5F", f"Expected red for negative delta, got {colors[1]}"


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_cli_generates_all_four_files(tmp_path):
    """CLI generates all 4 files (Fig 3 + Fig 4, PNG + PDF)."""
    csv_path = _mock_3seed_csv(tmp_path)
    out_dir = tmp_path / "output"

    exit_code = bcf.main(["--cossim-csv", str(csv_path), "--out-dir", str(out_dir)])

    assert exit_code == 0
    assert (out_dir / "fig3_cossim_lines.png").exists()
    assert (out_dir / "fig3_cossim_lines.pdf").exists()
    assert (out_dir / "fig4_delta_bars.png").exists()
    assert (out_dir / "fig4_delta_bars.pdf").exists()


def test_cli_creates_output_dir(tmp_path):
    """CLI creates output directory if it doesn't exist."""
    csv_path = _mock_3seed_csv(tmp_path)
    out_dir = tmp_path / "nonexistent" / "output"

    assert not out_dir.exists()
    bcf.main(["--cossim-csv", str(csv_path), "--out-dir", str(out_dir)])
    assert out_dir.exists()
