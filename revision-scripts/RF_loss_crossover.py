"""New figure: fixed intrinsic loss -- window of the N^-8 law and crossover to spatial attenuation."""
import sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import rof, figstyle as fs
from figstyle import plt
fs.use()

J, kappa = 1.0, 0.1
gams = [1e-6, 1e-5, 1e-4]
cols = [fs.BLUE, fs.VERM, fs.GREEN]
labs = [r"$\gamma/J=10^{-6}$", r"$\gamma/J=10^{-5}$", r"$\gamma/J=10^{-4}$"]

def Phi(nu):
    th = np.sqrt(np.pi ** 2 - 0.5j * nu ** 2)
    return np.abs(np.cos(th / 2)) ** (-4.0)

fig, axs = plt.subplots(1, 2, figsize=(fs.COL2, 2.75), gridspec_kw=dict(wspace=0.30))
ax = axs[0]
Ns = np.unique(np.round(np.logspace(np.log10(5), np.log10(3e4), 500)).astype(int))
Ns = Ns[Ns % 2 == 1]
# lossless reference (exact at E_1)
Fll = np.array([64 / kappa ** 2 * (rof.chain_w(N, (N + 1) // 2) / rof.chain_w(N, 1)) ** 2 for N in Ns])
ax.loglog(Ns, Fll, color=fs.GREY, ls=(0, (4, 2)), lw=1.1, label="lossless", zorder=1)
for gam, c, lab in zip(gams, cols, labs):
    Fex = np.array([rof.chain_F(N, (N + 1) // 2, kappa, gam, method="closed") for N in Ns])
    Fm = np.array([rof.mode_F(rof.chain_w(N, 1), rof.chain_w(N, (N + 1) // 2), kappa, gam) for N in Ns])
    ax.loglog(Ns, Fm, color=c, ls=(0, (1.2, 1.6)), lw=1.3, zorder=2)
    ax.loglog(Ns, Fex, color=c, lw=1.8, label=lab, zorder=3)
    Nstar = (np.pi ** 2 * kappa / gam) ** (1 / 3)
    Ngam = np.pi * np.sqrt(3 * J / gam)
    f1 = rof.chain_F(int(2 * round(Nstar / 2) + 1), int(round(Nstar / 2) + 1), kappa, gam)
    Ng = int(2 * round(Ngam / 2) + 1)
    f2 = rof.chain_F(Ng, (Ng + 1) // 2, kappa, gam, method="closed")
    ax.plot([Nstar], [f1], marker="*", ms=9, mfc="white", mec=c, mew=1.2, ls="none", zorder=5)
    ax.plot([Ngam], [f2], marker="o", ms=5.5, mfc="white", mec=c, mew=1.2, ls="none", zorder=5)
# style handles
ax.plot([], [], color=fs.DARK, ls=(0, (1.2, 1.6)), lw=1.3, label="isolated-mode law")
ax.plot([], [], marker="*", ms=9, mfc="white", mec=fs.DARK, mew=1.2, ls="none", label=r"$N_\star=(\pi^2\kappa/\gamma)^{1/3}$")
ax.plot([], [], marker="o", ms=5.5, mfc="white", mec=fs.DARK, mew=1.2, ls="none", label=r"$N_\gamma$:  $\gamma=\Delta E$")
# slope guides
xg = np.array([9, 22]); ax.loglog(xg, 1.2e7 * (xg / 9.0) ** 4, color=fs.DARK, lw=0.9)
ax.text(11.0, 1.6e8, r"$N^{4}$", fontsize=7.5, color=fs.DARK)
xg = np.array([450, 1500]); ax.loglog(xg, 3e5 * (xg / 450.0) ** -8, color=fs.DARK, lw=0.9)
ax.text(1050, 2.0e3, r"$N^{-8}$", fontsize=7.5, color=fs.DARK)
ax.set_xlim(5, 3e4); ax.set_ylim(1e-13, 1e12)
ax.set_xlabel(r"chain length $N$")
ax.set_ylabel(r"center-site QFI per photon $ℱ_{\rm bulk}$")
ax.legend(loc="lower left", fontsize=6.6, handlelength=2.0, labelspacing=0.32)
fs.panel(ax, "a")

ax = axs[1]
nu = np.logspace(-0.7, np.log10(45), 400)
ax.loglog(nu, Phi(nu), color=fs.INK, lw=1.3, label=r"crossover function $\Phi$", zorder=4)
ax.loglog(nu, 4096 * np.pi ** 4 / nu ** 8, color=fs.DARK, ls=(0, (4, 2)), lw=0.9, label=r"$N^{-8}$ law", zorder=2)
ax.loglog(nu, 16 * np.exp(-nu), color=fs.DARK, ls=(0, (1.2, 1.6)), lw=1.1, label=r"$16\,\exp[-(N+1)(\gamma/J)^{1/2}]$", zorder=2)
mk = ["o", "s", "^"]
for gam, c, lab, m in zip(gams, cols, labs, mk):
    Nsel = np.unique(np.round(np.logspace(np.log10(7), np.log10(45 / np.sqrt(gam)), 34)).astype(int))
    Nsel = Nsel[Nsel % 2 == 1]
    Fex = np.array([rof.chain_F(N, (N + 1) // 2, kappa, gam, method="closed") for N in Nsel])
    ax.loglog((Nsel + 1) * np.sqrt(gam / J), 4 * J ** 4 * Fex / kappa ** 2, ls="none", marker=m, ms=3.8, mfc="white", mec=c, mew=1.0,
              label=lab, zorder=5)
ax.axvline(np.sqrt(3) * np.pi, color=fs.GREY, lw=0.7, zorder=1)
ax.text(np.sqrt(3) * np.pi * 1.10, 3e14, r"$\gamma=\Delta E$", fontsize=7, color=fs.DARK, rotation=90, va="top")
ax.set_xlim(0.2, 45); ax.set_ylim(1e-16, 1e16)
ax.set_xlabel(r"scaled size $(N+1)(\gamma/J)^{1/2}$")
ax.set_ylabel(r"$4J^{4}ℱ_{\rm bulk}/\kappa^{2}$")
ax.legend(loc="lower left", fontsize=6.6, labelspacing=0.32)
fs.panel(ax, "b")
fig.savefig(os.path.join(HERE, 'figs', 'fig_loss_crossover.pdf'))
fig.savefig(os.path.join(HERE, 'figs', 'fig_loss_crossover.png'), dpi=200)
