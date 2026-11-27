"""New figure: saturation of the two-level-system chain -- effective two-level law."""
import sys, json, os
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import rof, figstyle as fs
from figstyle import plt
import importlib.util
spec = importlib.util.spec_from_file_location("sat", os.path.join(HERE, 'R07_saturation_lindblad.py'))
fs.use()
S = lambda u: (4 + u) / (4 * (1 + u) ** 3)
cache = os.path.join(HERE, 'out', 'fig_saturation_data.json')
if not os.path.exists(cache):
    sat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sat)
    qfi_rate = sat.qfi_rate
    us = np.logspace(-2, 1.5, 13)
    data = {}
    for kappa in [0.05, 0.2]:
        for N in [2, 3, 4, 5, 6]:
            s = (N + 1) // 2
            Gp = kappa * rof.chain_w(N, 1)
            Flin = rof.chain_F(N, s, kappa, 0.0)
            r = []
            for u in us:
                Phi = u * Gp / 8
                Id, _, _ = qfi_rate(N, s, kappa, np.sqrt(Phi))
                r.append(Id / Phi / Flin)
            data[f"{kappa}_{N}"] = r
            print(kappa, N, flush=True)
    json.dump(dict(us=us.tolist(), data=data), open(cache, 'w'))
D = json.load(open(cache))
us = np.array(D['us'])
fig, ax = plt.subplots(figsize=(fs.COL1, 2.55))
uu = np.logspace(-2.4, 1.7, 300)
ax.loglog(uu, S(uu), color=fs.INK, lw=1.3, zorder=3, label=r"$S(u)$, effective two-level law")
mk = {2: ("o", 7.0), 3: ("s", 5.4), 4: ("^", 4.4), 5: ("D", 2.9), 6: (".", 2.6)}
for N in [2, 3, 4, 5, 6]:
    m, ms = mk[N]
    ax.loglog(us, D['data'][f"0.05_{N}"], ls="none", marker=m, ms=ms, mfc="white" if N < 6 else fs.BLUE, mec=fs.BLUE, mew=0.9,
              zorder=4 + N, label=rf"$N={N}$")
ax.loglog(us, D['data']["0.2_2"], ls="none", marker="x", ms=4.6, mec=fs.VERM, mew=1.1, zorder=12, label=r"$N=2$, $\kappa/J=0.2$")
ax.plot([0.6056], [S(0.6056)], marker="*", ms=10, mfc="white", mec=fs.INK, mew=1.0, zorder=13, ls="none")
ax.annotate("maximum\nQFI rate", xy=(0.6056, S(0.6056)), xytext=(6.0, 0.75), fontsize=6.8, ha="center", va="center", linespacing=0.95,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=fs.DARK, shrinkB=5))
ax.set_xlim(5e-3, 45); ax.set_ylim(1e-4, 2.5)
ax.set_xlabel(r"drive $u=8\Phi/\Gamma_{\rm p}$")
ax.set_ylabel(r"output QFI per photon / linear value")
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, loc="lower left", fontsize=6.5, ncol=1, handletextpad=0.4, labelspacing=0.28, borderaxespad=0.7)
fig.savefig(os.path.join(HERE, 'figs', 'fig_saturation.pdf'))
fig.savefig(os.path.join(HERE, 'figs', 'fig_saturation.png'), dpi=220)
