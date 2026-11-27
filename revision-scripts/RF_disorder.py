"""New figure: robustness of the scaling laws to static disorder (perturbation theory and Monte Carlo)."""
import sys, json
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import figstyle as fs
from figstyle import plt
fs.use()
PT = json.load(open(os.path.join(HERE, 'out', 'R04_disorder_pt.json')))
MC = json.load(open(os.path.join(HERE, 'out', 'R05_disorder_mc.json')))

fig, axs = plt.subplots(1, 3, figsize=(fs.COL2, 2.45), gridspec_kw=dict(wspace=0.44))
# ---------------- (a) geometric coefficient
ax = axs[0]
spec = {"1_1": (fs.BLUE, "o", "-"), "2_1": (fs.VERM, "s", (0, (4, 2))), "2_2": (fs.VERM, "s", "-"),
        "3_1": (fs.GREEN, "^", (0, (1.2, 1.6))), "3_2": (fs.GREEN, "^", (0, (4, 2))), "3_3": (fs.GREEN, "^", "-")}
for key, (c, m, ls) in spec.items():
    r = np.array(PT[key]['rows'])
    ax.loglog(r[:, 1], r[:, 2], color=c, ls=ls, lw=1.3, marker=m, ms=3.2, mfc="white", mec=c, mew=0.9)
ax.text(4.2e2, 1.6e5, r"$d=1$:  $\propto N^{3}$", fontsize=7.0, color=fs.INK)
ax.text(1.0e2, 3.0e3, r"$d=2$:  $\propto N$", fontsize=7.0, color=fs.INK)
ax.text(3.0e2, 0.75, r"$d=3$:  $\propto N^{1/3}$", fontsize=7.0, color=fs.INK)
from matplotlib.lines import Line2D
hs = [Line2D([], [], color=fs.DARK, ls="-", lw=1.2, label=r"$k=d$ (corner)"),
      Line2D([], [], color=fs.DARK, ls=(0, (4, 2)), lw=1.2, label=r"$k=d-1$"),
      Line2D([], [], color=fs.DARK, ls=(0, (1.2, 1.6)), lw=1.2, label=r"$k=d-2$")]
ax.legend(handles=hs, loc="upper left", fontsize=6.5, labelspacing=0.3, borderaxespad=0.5)
ax.set_xlim(6, 3e4); ax.set_ylim(0.3, 1e12)
ax.set_xlabel(r"number of sites $N$")
ax.set_ylabel(r"coefficient $\mathcal{C}_{d,k}$ (on-site disorder)".replace(r"\mathcal{C}", "𝒞"))
fs.panel(ax, "a", x=-0.27)

# ---------------- (b) Monte Carlo vs perturbation theory
ax = axs[1]
groups = {"1_1": (fs.BLUE, "o", r"chain"), "2_1": (fs.VERM, "s", r"square, edge"), "2_2": (fs.GREEN, "^", r"square, corner")}
xx = np.logspace(-2.6, 1.9, 10)
ax.loglog(xx, xx, color=fs.INK, lw=0.9, zorder=1)
for g, (c, m, lb) in groups.items():
    for kind, filled in [("onsite", True), ("bond", False)]:
        xs = [v['pt'] for k, v in MC.items() if k.startswith(g + "_" + kind)]
        ys = [v['rms'] for k, v in MC.items() if k.startswith(g + "_" + kind)]
        ax.loglog(xs, ys, ls="none", marker=m, ms=3.6, mfc=c if filled else "white", mec=c, mew=0.9, zorder=3,
                  label=lb if filled else None)
hs2 = [Line2D([], [], ls="none", marker="o", ms=3.6, mfc=fs.DARK, mec=fs.DARK, label="on-site"),
       Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=fs.DARK, label="hopping")]
h, l = ax.get_legend_handles_labels()
ax.legend(handles=h + hs2, loc="upper left", fontsize=6.5, labelspacing=0.3, handletextpad=0.3, borderaxespad=0.5)
ax.set_xlim(2e-3, 80); ax.set_ylim(2e-3, 80)
ax.set_xlabel(r"first-order prediction $(\sigma/J)\,\mathcal{C}^{1/2}$".replace(r"\mathcal{C}", "𝒞"))
ax.set_ylabel(r"rms of $\ln ℱ$ (Monte Carlo)")
fs.panel(ax, "b", x=-0.27)

# ---------------- (c) median and quartiles
ax = axs[2]
ax.axhline(1.0, color=fs.DARK, lw=0.7, zorder=1)
for g, (c, m, lb) in groups.items():
    items = [v for k, v in MC.items() if k.startswith(g + "_")]
    items = sorted(items, key=lambda v: v['pt'])
    x = np.array([v['pt'] for v in items]); q = np.array([v['q'] for v in items])
    sel = x < 2.6
    ax.errorbar(x[sel], q[sel, 1], yerr=[q[sel, 1] - q[sel, 0], q[sel, 2] - q[sel, 1]], ls="none", marker=m, ms=3.2, mfc="white",
                mec=c, mew=0.9, ecolor=c, elinewidth=0.7, capsize=0, zorder=3)
xx = np.logspace(-2.6, 0.5, 100)
ax.plot(xx, np.exp(0.6745 * xx), color=fs.INK, lw=0.9, ls=(0, (4, 2)), zorder=2)
ax.plot(xx, np.exp(-0.6745 * xx), color=fs.INK, lw=0.9, ls=(0, (4, 2)), zorder=2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(2e-3, 3.2); ax.set_ylim(0.1, 10)
from matplotlib.ticker import NullFormatter
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"first-order prediction $(\sigma/J)\,\mathcal{C}^{1/2}$".replace(r"\mathcal{C}", "𝒞"))
ax.set_ylabel(r"$ℱ/ℱ_{\rm clean}$: median and quartiles")
fs.panel(ax, "c", x=-0.27)
fig.savefig(os.path.join(HERE, 'figs', 'fig_disorder.pdf'))
fig.savefig(os.path.join(HERE, 'figs', 'fig_disorder.png'), dpi=200)
