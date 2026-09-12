"""Shared publication style for every figure in the package.

Import after matplotlib but before creating figures:

    from _style import *

Provides: rcParams for APS single-column (3.375 in) and double-column (7.0 in)
figures, an Okabe--Ito colour-blind-safe palette, and small helpers for panel
labels, power-law guides and legends.  Falls back to DejaVu Sans if the
Helvetica-like TeX Gyre Heros font is not installed.
"""
from __future__ import annotations
import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
import numpy as np

# ----------------------------------------------------------------- fonts
# Preference: Liberation Sans (TrueType, Arial metrics; embeds as clean Type 42),
# then Arial/Helvetica if present, then DejaVu Sans (always available).  OpenType-CFF
# fonts (e.g. TeX Gyre Heros) are avoided because some PDF preflight tools flag them.
def _font_available(name):
    try:
        return _fm.findfont(_fm.FontProperties(family=name), fallback_to_default=False) is not None
    except Exception:
        return False
_SANS = [f for f in ['Liberation Sans', 'Arial', 'Helvetica'] if _font_available(f)] + ['DejaVu Sans']

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': _SANS,
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.8,
    'legend.frameon': False,
    'legend.handlelength': 1.6,
    'legend.handletextpad': 0.5,
    'legend.labelspacing': 0.3,
    'legend.borderaxespad': 0.3,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'lines.linewidth': 1.1,
    'lines.markersize': 4,
    'lines.markeredgewidth': 0.7,
    'axes.prop_cycle': matplotlib.cycler(color=['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000']),
    'mathtext.fontset': 'stixsans',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
    'figure.dpi': 110,
    'axes.formatter.use_mathtext': True,
})

# ----------------------------------------------------------------- palette
OI = dict(blue='#0072B2', vermilion='#D55E00', green='#009E73', purple='#CC79A7',
          orange='#E69F00', sky='#56B4E9', yellow='#F0E442', black='#000000', grey='#7F7F7F')
# three-value sequences used throughout (e.g. three pulse lengths, three loss rates)
C3 = [OI['blue'], OI['vermilion'], OI['green']]
C4 = [OI['blue'], OI['vermilion'], OI['green'], OI['purple']]
# codimension colours k = 0..3
CK = {0: OI['grey'], 1: OI['blue'], 2: OI['vermilion'], 3: OI['purple']}
MARK_D = {1: 'o', 2: 's', 3: '^'}

COL_W = 3.375    # APS single column, inches
DBL_W = 7.0      # APS double column, inches


def figure(width=COL_W, height=None, ncols=1, nrows=1, aspect=0.72, **kw):
    """Create a figure sized for one or two journal columns."""
    if height is None:
        height = width / ncols * aspect * nrows
    return plt.subplots(nrows, ncols, figsize=(width, height), **kw)


def panel_label(ax, text, x=-0.18, y=1.02, **kw):
    """Bold panel label outside the top-left corner of an axes."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9, fontweight='bold',
            ha='left', va='bottom', **kw)


def slope_guide(ax, x0, y0, exponent, dx_factor=3.0, label=None, color='0.35', above=True, lw=0.8, pad=1.25):
    """Draw a short power-law guide y = y0 (x/x0)^exponent on log-log axes."""
    x = np.array([x0, x0 * dx_factor])
    y = y0 * (x / x0) ** exponent
    ax.plot(x, y, color=color, lw=lw, solid_capstyle='round', zorder=1)
    if label is None:
        label = rf'$N^{{{exponent:g}}}$'
    xm = np.sqrt(x[0] * x[1])
    ym = y0 * (xm / x0) ** exponent
    ym = ym * pad if above else ym / pad
    ax.text(xm, ym, label, color=color, fontsize=7, ha='center', va='bottom' if above else 'top')


def tidy(ax, minor=True):
    """Common cosmetics: minor ticks on log axes, light grid off."""
    if minor:
        ax.minorticks_on()
    ax.grid(False)


def logticks_dense(ax, axis='y', subs=(2, 5)):
    """Add labelled minor ticks (2 and 5) on a log axis."""
    from matplotlib.ticker import LogLocator, NullFormatter
    a = ax.yaxis if axis == 'y' else ax.xaxis
    a.set_minor_locator(LogLocator(base=10, subs=subs, numticks=100))
    a.set_minor_formatter(NullFormatter())


def sci_label(T):
    """'10^{3}' style label for a power of ten."""
    e = int(round(np.log10(T)))
    return rf'10^{{{e}}}'
