"""Matplotlib style matching the existing figures of the manuscript (Liberation Sans + STIX math,
inward ticks on all sides, Okabe-Ito hues in a fixed order)."""
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

BLUE, VERM, GREEN, PURPLE = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
GREY, DARK, INK = "#7F7F7F", "#4D4D4D", "#000000"
COL1 = 3.30   # single-column width (in)
COL2 = 7.05   # double-column width (in)

def use():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        "mathtext.cal": "STIXGeneral:italic",
        "mathtext.fallback": "stix",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.0,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.borderaxespad": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.dpi": 150,
    })

def panel(ax, label, x=-0.20, y=1.04):
    ax.text(x, y, f"({label})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
