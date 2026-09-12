from _common import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from _style import *
from rankonefi import (hypercubic_hamiltonian_sparse, hypercubic_index, hypercubic_mode_weight,
                       hypercubic_band_edge_energy, port_site_coords, sparse_fi_port, fit_power_law)

# Port-codimension law.  For an open L^d lattice probed through one site on a boundary
# feature of codimension k (k=0 interior, k=d corner), the (1,...,1) band-edge mode has
#   w_port ~ (2/L)^d (pi/L)^{2k} ~ N^{-1-2k/d},   w_bulk ~ N^{-1},
# so that in the isolated, lossless regime
#   F_bulk = 64 (w_s/w_p)^2 / kappa^2 ~ N^{4k/d},
#   Fdot / n_int = 16 w_s^2 / (kappa w_p) ~ N^{2k/d - 1},
#   1/Gamma_p = 1/(kappa w_p) ~ N^{1+2k/d}.

kappa = 0.02
# sizes with L = 3 (mod 4) so that centre and quarter points are exact lattice sites
L1 = np.array([15, 19, 27, 35, 47, 63, 83, 111, 147, 199, 263, 351, 399])
L2 = np.array([11, 19, 27, 35, 43, 51, 59, 67, 75, 83, 91])
L3 = np.array([7, 11, 15, 19, 23, 27])
cases = [
    (1, 1, L1), (1, 0, L1),
    (2, 2, L2), (2, 1, L2), (2, 0, L2),
    (3, 3, L3), (3, 2, L3), (3, 1, L3), (3, 0, L3),
]
FULL = '--full' in sys.argv or not (DATA / 'port_codimension.csv').exists()
if FULL:
    rows = []
    for d, k, Ls in cases:
        for L in Ls:
            L = int(L)
            N = L ** d
            H = hypercubic_hamiltonian_sparse(L, d)
            pc = port_site_coords(L, d, k)
            centre = (L - 1) // 2
            if k == 0:
                sc = tuple([(L + 1) // 4 - 1] * d)   # exact quarter point, away from the interior port
            else:
                sc = tuple([centre] * d)
            port, site = hypercubic_index(L, pc), hypercubic_index(L, sc)
            omega = hypercubic_band_edge_energy(L, d)
            F, tau, Gsp = sparse_fi_port(H, omega, kappa, port, site)
            wp, ws = hypercubic_mode_weight(L, pc), hypercubic_mode_weight(L, sc)
            F_iso = 64.0 * (ws / wp) ** 2 / kappa ** 2
            rows.append(dict(d=d, codim=k, L=L, N=N, port=str(pc), site=str(sc),
                             F_exact=F, F_isolated=F_iso, tau_int=tau, rate=F / tau,
                             rate_isolated=16.0 * ws ** 2 / (kappa * wp),
                             w_port=wp, w_site=ws, Gamma_p=kappa * wp,
                             level_spacing=2 * d * (np.cos(np.pi / (L + 1)) - np.cos(2 * np.pi / (L + 1)))))
    df = pd.DataFrame(rows)
    df.to_csv(DATA / 'port_codimension.csv', index=False)
else:
    print('reusing packaged data/port_codimension.csv (pass --full to recompute)')
df = pd.read_csv(DATA / 'port_codimension.csv')


# exponents: fit on the largest half of the sizes
fits = []
for (d, k), g in df.groupby(['d', 'codim']):
    g = g.sort_values('N')
    tail = max(4, len(g) // 2)
    pF, _ = fit_power_law(g['N'], g['F_exact'], tail=tail)
    pR, _ = fit_power_law(g['N'], g['rate'], tail=tail)
    pFi, _ = fit_power_law(g['N'], g['F_isolated'], tail=tail)
    fits.append(dict(d=d, codim=k, n_sizes=len(g), N_max=int(g['N'].max()),
                     F_exponent_fit=pF, F_exponent_isolated_law=pFi, F_exponent_predicted=4 * k / d,
                     rate_exponent_fit=pR, rate_exponent_predicted=2 * k / d - 1,
                     max_rel_dev_exact_vs_isolated=float(np.max(np.abs(g['F_exact'] / g['F_isolated'] - 1))),
                     min_isolation_ratio=float(np.min(g['level_spacing'] / g['Gamma_p']))))
fits = pd.DataFrame(fits)
fits.to_csv(DATA / 'port_codimension_exponents.csv', index=False)
print(fits.to_string(index=False))

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.7), gridspec_kw=dict(wspace=0.3))
for (d, k), g in df.groupby(['d', 'codim']):
    g = g.sort_values('N')
    lab = f'$d={d}$, $k={k}$'
    axes[0].loglog(g['N'], g['F_isolated'] * kappa ** 2 / 64, '-', color=CK[k], lw=0.8, alpha=0.8, zorder=1)
    axes[0].loglog(g['N'], g['F_exact'] * kappa ** 2 / 64, MARK_D[d], color=CK[k], ms=3.6, mfc='white', mew=0.8, ls='none', label=lab, zorder=2)
    axes[1].loglog(g['N'], g['rate_isolated'] * kappa / 16, '-', color=CK[k], lw=0.8, alpha=0.8, zorder=1)
    axes[1].loglog(g['N'], g['rate'] * kappa / 16, MARK_D[d], color=CK[k], ms=3.6, mfc='white', mew=0.8, ls='none', zorder=2)
ax = axes[0]
ax.set_xlabel('number of sites $N$'); ax.set_ylabel(r'$\kappa^2\mathcal{F}_{\rm bulk}/64=(w_s/w_{\rm p})^2$')
ax.set_ylim(1e-3, 1e14)
ax.legend(loc='upper left', ncol=3, fontsize=6.2, columnspacing=0.8, handletextpad=0.3)
for (x0, y0, e, lab) in [(2.5e3, 3e7, 4, r'$N^{4}$'), (2.5e3, 1.5e4, 8/3, r'$N^{8/3}$'), (2.5e3, 2e2, 2, r'$N^{2}$'), (2.5e3, 1.2e0, 4/3, r'$N^{4/3}$')]:
    slope_guide(ax, x0, y0, e, dx_factor=3.5, label=lab, above=False, pad=1.4)
ax.text(0.97, 0.93, r'$\mathcal{F}_{\rm bulk}\sim N^{4k/d}$', transform=ax.transAxes, ha='right', va='top', fontsize=8, color='0.2')
panel_label(ax, '(a)', x=-0.2); tidy(ax)
ax = axes[1]
ax.set_xlabel('number of sites $N$'); ax.set_ylabel(r'$\kappa\,\dot{\mathcal{I}}_\epsilon/(16\,\bar n_{\rm int})=w_s^{2}/w_{\rm p}$')
ax.set_ylim(3e-6, 3e3)
ax.axhspan(3e-6, 1e-4, color='0.93', lw=0, zorder=0)
for (x0, y0, e, lab) in [(2.5e3, 4e2, 1, r'$N^{1}$'), (2.5e3, 2.5e1, 1/3, r'$N^{1/3}$'), (2.5e3, 1.2e0, 0, r'$N^{0}$'), (2.5e3, 1.5e-1, -1/3, r'$N^{-1/3}$'), (2.5e3, 1e-3, -1, r'$N^{-1}$')]:
    slope_guide(ax, x0, y0, e, dx_factor=3.5, label=lab, above=True, pad=1.4)
ax.text(0.03, 0.05, r'gain iff $k>d/2$', transform=ax.transAxes, ha='left', fontsize=8, color='0.2')
ax.text(0.03, 0.93, r'$\dot{\mathcal{I}}_\epsilon/\bar n_{\rm int}\sim N^{2k/d-1}$', transform=ax.transAxes, ha='left', va='top', fontsize=8, color='0.2')
panel_label(ax, '(b)', x=-0.2); tidy(ax)
fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.17)
save_figure(fig, 'fig9_port_codimension')
