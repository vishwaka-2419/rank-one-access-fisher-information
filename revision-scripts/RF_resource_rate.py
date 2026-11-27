"""Redrawn Fig. 8 of the original submission (same quantity and parameters; legend moved off the data).
Lossless chain, kappa/J = 0.1, drive at E_1.  Rate per stored excitation = F_s / tau_int,
tau_int = kappa * sum_j |G_j1|^2  (exact banded resolvent)."""
import sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import rof, figstyle as fs
from figstyle import plt
fs.use()
kappa = 0.1
Ns = np.unique(np.round(np.logspace(np.log10(12), np.log10(3000), 70)).astype(int))
rc, re = [], []
for N in Ns:
    g = rof.chain_G_col1_banded(int(N), rof.chain_E1(int(N)), kappa, 0.0)
    tau = kappa * np.sum(np.abs(g) ** 2)
    rc.append(4 * kappa ** 2 * abs(g[(N + 1) // 2 - 1]) ** 4 / tau)
    re.append(4 * kappa ** 2 * abs(g[N - 1]) ** 4 / tau)
rc, re = np.array(rc), np.array(re)
h = len(Ns) // 2
pc = np.polyfit(np.log(Ns[h:]), np.log(rc[h:]), 1)[0]
pe = np.polyfit(np.log(Ns[h:]), np.log(re[h:]), 1)[0]
print("fitted exponents: centre %.3f, opposite endpoint %.3f" % (pc, pe))
fig, ax = plt.subplots(figsize=(fs.COL1, 2.55))
ax.loglog(Ns, rc, ls="none", marker="o", ms=3.4, mfc=fs.VERM, mec=fs.VERM, mew=0.8, label=rf"center: fit $N^{{{pc:.2f}}}$")
ax.loglog(Ns, re, ls="none", marker="o", ms=3.4, mfc="white", mec=fs.BLUE, mew=0.9, label=rf"opposite endpoint: fit $N^{{{pe:.2f}}}$")
xg = np.array([300, 900]); ax.loglog(xg, 6e-4 * (xg / 300.0) ** -3, color=fs.DARK, lw=0.9); ax.text(430, 6e-4, r"$N^{-3}$", fontsize=7.5, color=fs.DARK)
xg = np.array([300, 900]); ax.loglog(xg, 2.4e3 * (xg / 300.0) ** 1, color=fs.DARK, lw=0.9); ax.text(430, 1.5e3, r"$N^{1}$", fontsize=7.5, color=fs.DARK, va="top")
ax.set_xlim(9, 4000); ax.set_ylim(3e-8, 6e5)
ax.set_xlabel(r"chain length $N$")
ax.set_ylabel(r"QFI rate per stored excitation $\dot{ℐ}_\epsilon/\bar{n}_{\rm int}$")
ax.legend(loc="lower left", fontsize=7.0, handletextpad=0.3, borderaxespad=0.6)
fig.savefig(os.path.join(HERE, 'figs', 'fig8_redrawn.pdf'))
fig.savefig(os.path.join(HERE, 'figs', 'fig8_redrawn.png'), dpi=220)
