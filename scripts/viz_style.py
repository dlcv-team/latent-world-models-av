"""Shared house style for all DiT-WAM report/poster figures.

Okabe-Ito colorblind-safe palette with FIXED semantics (Direct=orange,
Diffusion=blue, GT/ceiling=gray, Interp=green); CVPR-ish serif fonts; report vs
poster size profiles. Local matplotlib (trivial / seconds). Retained so every
figure is reproducible. See Visualization Plan rev 4.
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----- fixed semantic colors (Okabe-Ito) -----
C = {
    "direct":    "#E69F00",  # orange   : regression mean (blur / distortion-optimal)
    "diffusion": "#0072B2",  # blue     : diffusion (sharp / distribution-optimal)
    "calib":     "#56B4E9",  # sky blue : diffusion + train-calibration (deployable)
    "interp":    "#009E73",  # green    : interpolation operating point
    "ceiling":   "#999999",  # gray     : VAE-GT reconstruction ceiling
    "chance":    "#999999",
    "accent":    "#D55E00",  # vermillion: highlight / "worse than chance"
    "muted":     "#CC79A7",
}
ARROW_BETTER = dict(arrowstyle="-|>", color="#444444", lw=1.0)

PROFILES = {
    "report": dict(base=9,  title=10, label=9,  tick=8,  legend=8,  lw=1.6, ms=6,  awidth=0.8, dpi=300),
    "poster": dict(base=22, title=26, label=23, tick=18, legend=20, lw=3.2, ms=13, awidth=1.6, dpi=300),
}

def apply(profile="report"):
    p = PROFILES[profile]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": p["base"], "axes.titlesize": p["title"], "axes.labelsize": p["label"],
        "xtick.labelsize": p["tick"], "ytick.labelsize": p["tick"], "legend.fontsize": p["legend"],
        "axes.linewidth": p["awidth"], "lines.linewidth": p["lw"], "lines.markersize": p["ms"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "figure.dpi": p["dpi"], "savefig.dpi": p["dpi"], "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return p

# ----- paths -----
ROOT = pathlib.Path(__file__).resolve().parents[3]          # .../proj
ARTI = ROOT / "code/latent-world-models-av/artifacts/full"  # result JSONs
REPORT_FIG = ROOT / "docs-repo/final-report/latex/figures"  # committed report PDFs
CODE_FIG = ROOT / "code/latent-world-models-av/artifacts/full/figures"  # PNG mirror

def load(name):
    import json
    return json.load(open(ARTI / name))

def savefig(fig, name, profile="report"):
    REPORT_FIG.mkdir(parents=True, exist_ok=True)
    CODE_FIG.mkdir(parents=True, exist_ok=True)
    suffix = "" if profile == "report" else "_poster"
    pdf = REPORT_FIG / f"{name}{suffix}.pdf"
    fig.savefig(pdf)
    fig.savefig(CODE_FIG / f"{name}{suffix}.png", dpi=200)
    plt.close(fig)
    print(f"  saved {pdf.relative_to(ROOT)}  (+png mirror)")
    return str(pdf)

def figsize(profile, w, h):
    s = 1.0 if profile == "report" else 2.6
    return (w * s, h * s)
