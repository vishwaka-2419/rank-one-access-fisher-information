"""New figure: finite-window comparisons (small chains vs single resonator, port choice, pulse shape)."""
import sys, json, os
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.optimize import brentq
import rof, figstyle as fs
from figstyle import plt
fs.use()
D = json.load(open(os.path.join(HERE, 'out', 'fig_finite_window_data.json')))
kappa = 0.1

# ---- pulse data (cached)
cache = os.path.join(HERE, 'out', 'fig_pulse_data.json')
if not os.path.exists(cache):
    def top_singular(x, n=1200):
        xg, wg = np.polynomial.legendre.leggauss(n)
        t = 0.5 * (xg + 1); w = 0.5 * wg
        Dm = t[:, None] - t[None, :]
        K = np.where(Dm > 0, Dm * np.exp(-0.5 * x * Dm), 0.0)
        A = np.sqrt(w)[:, None] * K * np.sqrt(w)[None, :]
        U, S, Vt = np.linalg.svd(A)
        v = Vt[0] / np.sqrt(w)
        if v[0] < 0: v = -v
        return S[0], t, v
    xs = np.logspace(-2, 2, 65)
    rat = []
    for x in xs:
        s1, _, _ = top_singular(x)
        rat.append(float(x ** 4 / 16 * s1 ** 2 / rof.k_fun(x)))
    shapes = {}
    for x in [1e-4, 7.55, 30.0]:
        s1, t, v = top_singular(x, 600)
        shapes[str(x)] = dict(t=t.tolist(), v=v.tolist())
    json.dump(dict(xs=xs.tolist(), rat=rat, shapes=shapes), open(cache, 'w'))
P = json.load(open(cache))

fig, axs = plt.subplots(1, 3, figsize=(fs.COL2, 2.45), gridspec_kw=dict(wspace=0.42))

# ---------------- (a) chains vs resonator
ax = axs[0]
us = np.array(D['us']); res = np.array(D['res'])
cols = {2: fs.GREY, 3: fs.BLUE, 5: fs.VERM, 7: fs.GREEN, 9: fs.PURPLE}
ax.axhline(1.0, color=fs.DARK, lw=0.7)
# continuum estimate
xstar = brentq(lambda x: (np.log(rof.k_fun(x * 1.00001)) - np.log(rof.k_fun(x / 1.00001))) / (2 * np.log(1.00001)) - 4 / 3, 2, 20)
C = rof.k_fun(xstar) / np.pi ** 4 * (2 * np.pi ** 2 / xstar) ** (4 / 3)
uu = np.logspace(np.log10(4), np.log10(3000), 200)
ax.loglog(uu, C * uu ** (4 / 3) / np.array([rof.k_fun(u) for u in uu]), color=fs.INK, ls=(0, (4, 2)), lw=1.0, zorder=2)
for N in [2, 3, 5, 7, 9]:
    R = np.array(D[f'chain_{N}']['F']) / res
    ax.loglog(us, R, color=cols[N], lw=1.6, zorder=3)
ends = {2: r"$N=2$", 3: r"$3$", 5: r"$5$", 7: r"$7$", 9: r"$9$"}
for N, s in ends.items():
    yend = D[f'chain_{N}']['F'][-1] / res[-1]
    if N == 2:
        yend = 0.52
    ax.text(3500 if N > 2 else 2100, yend, s, color=fs.INK, fontsize=7.2, ha="left" if N > 2 else "center", va="center", zorder=6)
ax.plot([20.57], [1.0], marker="o", ms=4.5, mfc="white", mec=fs.BLUE, mew=1.1, zorder=7)
ax.annotate(r"$\kappa T=20.6$", xy=(20.57, 1.0), xytext=(4.6, 5.0), fontsize=7, color=fs.INK,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=fs.DARK, shrinkA=1, shrinkB=3))
ax.text(60, 3.2e2, "continuum\nestimate", fontsize=6.8, color=fs.INK, ha="center", va="center", linespacing=0.95)
ax.annotate("", xy=(1150, 1.5e2), xytext=(160, 3.0e2), arrowprops=dict(arrowstyle="-", lw=0.6, color=fs.DARK))
ax.set_xlim(4, 9000); ax.set_ylim(1e-3, 1e3)
ax.set_xlabel(r"deadline $\kappa T$")
ax.set_ylabel(r"QFI ratio, chain / single resonator")
fs.panel(ax, "a", x=-0.27)

# ---------------- (b) port choice on the 7x7 lattice
ax = axs[1]
ub = np.array(D['ub'])
spec = [("interior", fs.GREY, "s", r"interior, $k=0$"), ("edge", fs.BLUE, "o", r"mid-edge, $k=1$"), ("corner", fs.VERM, "^", r"corner, $k=2$")]
uf = np.logspace(np.log10(20), np.log10(2e4), 300)
for name, c, m, lb in spec:
    d = D[f'lat_{name}']
    Fm = (d['ws'] / d['wp']) ** 2 * np.array([rof.k_fun(d['wp'] * u) for u in uf])
    ax.loglog(uf, Fm, color=c, lw=1.4, zorder=3)
    ax.loglog(ub[::3], kappa ** 2 * np.array(d['F'])[::3] / 64, ls="none", marker=m, ms=3.6, mfc="white", mec=c, mew=1.0, label=lb, zorder=4)
for xc in [395.6, 1312.0]:
    ax.axvline(xc, color=fs.GREY, lw=0.6, zorder=1)
ax.text(395.6 * 0.86, 3e-3, r"$396$", fontsize=6.6, color=fs.DARK, rotation=90, ha="right", va="bottom")
ax.text(1312 * 0.86, 3e-3, r"$1312$", fontsize=6.6, color=fs.DARK, rotation=90, ha="right", va="bottom")
ax.set_xlim(20, 2e4); ax.set_ylim(1e-3, 1e4)
ax.set_xlabel(r"deadline $\kappa T$")
ax.set_ylabel(r"$\kappa^2 ℱ_T^{\rm (win)}/64$")
ax.legend(loc="upper left", fontsize=6.6, handletextpad=0.3, borderaxespad=0.4, labelspacing=0.3)
fs.panel(ax, "b", x=-0.27)

# ---------------- (c) pulse shaping
ax = axs[2]
xs = np.array(P['xs']); rat = np.array(P['rat'])
ax.semilogx(xs, rat, color=fs.BLUE, lw=1.6, zorder=3)
ax.axhline(1.0, color=fs.DARK, lw=0.7)
ax.axhline(20 / 1.875104 ** 4, color=fs.GREY, lw=0.7, ls=(0, (4, 2)))
ax.text(0.012, 1.633, r"$20/\beta_1^4=1.618$", fontsize=6.8, color=fs.DARK, va="bottom")
ax.set_xlim(1e-2, 1e2); ax.set_ylim(0.95, 1.72)
ax.set_xlabel(r"$\Gamma T$")
ax.set_ylabel(r"gain of optimal over rectangular pulse")
ins = ax.inset_axes([0.17, 0.13, 0.43, 0.40])
t = np.array(P['shapes']['0.0001']['t']); v = np.array(P['shapes']['0.0001']['v'])
ins.plot(t, v ** 2, color=fs.BLUE, lw=1.3)
t2 = np.array(P['shapes']['30.0']['t']); v2 = np.array(P['shapes']['30.0']['v'])
ins.plot(t2, v2 ** 2, color=fs.VERM, lw=1.3, ls=(0, (3, 1.5)))
ins.plot([0, 1], [1, 1], color=fs.DARK, lw=0.9, ls=(0, (1.2, 1.6)))
ins.set_xlim(0, 1); ins.set_ylim(0, 4.3)
ins.set_xticks([0, 1]); ins.set_yticks([0, 2, 4])
ins.tick_params(labelsize=6.2, length=2.2)
ins.set_xlabel(r"$t/T$", fontsize=6.6, labelpad=-6)
ins.set_ylabel(r"$|e(t)|^2T$", fontsize=6.6, labelpad=1)
ins.text(0.22, 3.25, r"$\Gamma T\to0$", fontsize=6.2, color=fs.INK)
ins.text(0.50, 1.95, r"$\Gamma T=30$", fontsize=6.2, color=fs.INK)
ins.text(0.985, 1.12, "rect.", fontsize=6.0, color=fs.DARK, ha="right", va="bottom")
fs.panel(ax, "c", x=-0.25)
fig.savefig(os.path.join(HERE, 'figs', 'fig_finite_window.pdf'))
fig.savefig(os.path.join(HERE, 'figs', 'fig_finite_window.png'), dpi=200)
