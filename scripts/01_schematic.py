from _common import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch
from _style import *

# Conceptual figure: (a) rank-one access geometry on a generic passive network,
# (b) the exact information budget F = 16 pi^2 rho_s^2 x eta_in x eta_out,
# (c) the time scales that convert per-photon scaling into a finite optimum.

fig, axes = plt.subplots(1, 3, figsize=(DBL_W, 2.35), gridspec_kw=dict(width_ratios=[1.25, 1.0, 1.05], wspace=0.28))

# ------------------------------------------------------------------ (a) geometry
ax = axes[0]
ax.set_xlim(-1.75, 5.2); ax.set_ylim(-1.7, 2.25); ax.set_aspect('equal'); ax.axis('off')
pts = {0: (0.0, 0.0), 1: (1.0, 0.55), 2: (1.0, -0.55), 3: (2.0, 0.0), 4: (2.0, 1.15), 5: (3.0, 0.6),
       6: (3.0, -0.5), 7: (4.0, 0.1), 8: (2.1, -1.1), 9: (4.0, 1.25), 10: (3.9, -1.0)}
edges = [(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (3, 5), (3, 6), (5, 7), (6, 7), (4, 5), (2, 8), (6, 8), (5, 9), (7, 9), (7, 10), (6, 10)]
for a, b in edges:
    (x1, y1), (x2, y2) = pts[a], pts[b]
    ax.plot([x1, x2], [y1, y2], color='0.55', lw=1.0, zorder=1)
sig = 7
for i, (x, y) in pts.items():
    if i in (0, sig):
        continue
    ax.add_patch(Circle((x, y), 0.17, fc='white', ec='0.25', lw=0.8, zorder=3))
for i in (4, 8, 9, 10, 2):
    x, y = pts[i]
    dy = 0.42 if y > 0 else -0.42
    ax.add_patch(FancyArrowPatch((x, y + np.sign(dy) * 0.18), (x, y + dy), arrowstyle='-|>', mutation_scale=6,
                                 color=OI['grey'], lw=0.7, zorder=2))
ax.text(2.4, 1.85, r'intrinsic loss $\Gamma_{\rm int}\succeq0$', color=OI['grey'], fontsize=7, ha='center')
x0, y0 = pts[0]
ax.add_patch(Rectangle((x0 - 0.2, y0 - 0.2), 0.4, 0.4, fc=OI['blue'], ec='none', zorder=4))
ax.text(x0 + 0.02, y0 - 0.32, r'port $\kappa$', color=OI['blue'], fontsize=7.5, ha='center', va='top')
ax.plot([-1.75, x0 - 0.22], [y0, y0], color=OI['blue'], lw=1.6, zorder=2)
ax.add_patch(FancyArrowPatch((-1.7, 0.3), (-0.55, 0.3), arrowstyle='-|>', mutation_scale=8, color=OI['blue'], lw=1.0))
ax.add_patch(FancyArrowPatch((-0.55, -0.3), (-1.7, -0.3), arrowstyle='-|>', mutation_scale=8, color=OI['blue'], lw=1.0))
ax.text(-1.15, 0.45, r'$\Phi$', color=OI['blue'], fontsize=8, ha='center', va='bottom')
ax.text(-1.2, -0.42, r'$S(\omega,\epsilon)$', color=OI['blue'], fontsize=8, ha='center', va='top')
xs, ys = pts[sig]
ax.add_patch(Circle((xs, ys), 0.2, fc=OI['vermilion'], ec='none', zorder=4))
ax.text(xs + 0.32, ys - 0.02, r'signal site $s$', color=OI['vermilion'], fontsize=7.5, ha='left', va='center')
ax.text(xs, ys + 0.36, r'$\epsilon\,|s\rangle\langle s|$', color=OI['vermilion'], fontsize=8, ha='center', va='bottom')
ax.text(1.7, -1.58, r'$\partial_\epsilon S=-i\kappa\,G_{1s}G_{s1}$', fontsize=7.6, ha='center', va='center', color='0.2')
panel_label(ax, '(a)', x=0.0, y=0.93)

# ------------------------------------------------------------------ (b) information budget
ax = axes[1]
ax.set_xlim(0, 1.0); ax.set_ylim(-0.35, 3.45); ax.axis('off')
eta_in, eta_out = 0.62, 0.55
bars = [(2.45, 1.0, OI['grey'], r'LDOS budget $16\pi^2\rho_s^{2}$', 'wavefront-shaping maximum'),
        (1.5, eta_in, OI['green'], r'$\times\,\eta_{\rm in}$: all outgoing channels', 'input accessibility'),
        (0.55, eta_in * eta_out, OI['blue'], r'$\times\,\eta_{\rm out}$: monitored port', 'collection branching ratio')]
for y, w, c, lab, sub in bars:
    ax.add_patch(FancyBboxPatch((0.0, y), w, 0.5, boxstyle='round,pad=0,rounding_size=0.04', fc=c, ec='none', alpha=0.9))
    ax.text(0.0, y + 0.56, lab, fontsize=7.4, ha='left', va='bottom', color='0.1')
    if w < 0.82:
        ax.text(w + 0.02, y + 0.25, sub, fontsize=6.4, ha='left', va='center', color='0.35', style='italic')
    else:
        ax.text(0.02, y + 0.25, sub, fontsize=6.4, ha='left', va='center', color='white', style='italic')
ax.text(0.5, 0.05, r'$\mathcal{F}_s^{\rm (port)}=16\pi^2\,\eta_{\rm in}\eta_{\rm out}\,\rho_s^{2}$', fontsize=7.6,
        ha='center', va='center', color='0.1',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=OI['blue'], lw=0.7))
ax.text(0.5, -0.3, r'lossless: $\eta_{\rm in}=\eta_{\rm out}=1$ for any graph', fontsize=6.6, ha='center', va='top', color='0.35')
panel_label(ax, '(b)', x=-0.12, y=0.93)

# ------------------------------------------------------------------ (c) time scales (cartoon)
ax = axes[2]
N = np.logspace(0.7, 3.5, 400)
F_ss = N ** 4
NT = 60.0
kc = lambda x: np.where(x > 50, 1.0, (x ** 4 / 320.0) / (1 + x ** 4 / 320.0))
F_win = F_ss * kc((NT / N) ** 3)
Nprop = 1100.0
F_prop = F_win * np.exp(-((N / Nprop) ** 6))
ax.loglog(N, F_ss, color='0.55', lw=1.0, ls='--', label=r'steady state ($T\to\infty$)')
ax.loglog(N, F_win, color=OI['vermilion'], lw=1.4, label='finite deadline $T$')
ax.loglog(N, F_prop, color=OI['blue'], lw=1.4, ls=(0, (4, 1.5)), label='+ finite propagation')
ax.set_xlim(N[0], N[-1]); ax.set_ylim(1e-9, 3e12)
ax.set_xticks([]); ax.set_yticks([])
ax.tick_params(which='both', length=0)
ax.set_xlabel(r'network size $N$', labelpad=2); ax.set_ylabel(r'QFI per photon', labelpad=2)
ax.text(14, 3e5, r'$N^{4k/d}$', color='0.4', fontsize=7.5, rotation=33, ha='center')
ax.axvline(NT * 0.66, color=OI['vermilion'], lw=0.6, ls=':')
ax.text(NT * 0.58, 1e11, r'loading $\Gamma_{\rm p}T=1$', color=OI['vermilion'], fontsize=6.4, ha='right', va='top', rotation=90)
ax.axvline(Nprop * 0.9, color=OI['blue'], lw=0.6, ls=':')
ax.text(Nprop * 0.78, 1e11, r'propagation $N\sim2JT$', color=OI['blue'], fontsize=6.4, ha='right', va='top', rotation=90)
ax.text(3.2e2, 3e3, r'$N^{-8}$', color=OI['vermilion'], fontsize=7.5, rotation=-52, ha='center')
ax.legend(loc='lower left', fontsize=6.3, handlelength=1.5)
panel_label(ax, '(c)', x=-0.1, y=0.93)

fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.14)
save_figure(fig, 'fig1_schematic')
