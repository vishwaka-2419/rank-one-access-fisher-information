from _common import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from _style import *
from rankonefi import hard_window_budget_sparse_chain

# True fixed-deadline asymptotics at large distance.  The nearest-neighbour chain has
# maximum group velocity v_max=2J.  For a center signal at distance ~N/2 from the port,
# the stored internal sensitivity requires one-way propagation (scale N~4JT), whereas
# information emitted back into the port by time T requires a round trip (scale N~2JT).
# We use a sparse adaptive ODE so this regime can be reached beyond the dense Van Loan
# calculations of script 13.

J = 1.0
kappa = 0.1
Ts = [20.0, 40.0, 80.0]
Ns = np.array([10, 14, 20, 28, 40, 56, 80, 100, 120, 160, 200, 240, 320])
FULL = '--full' in sys.argv or not (DATA / 'propagation_limit.csv').exists()
if FULL:
    rows = []
    for T in Ts:
        for N in Ns:
            for label, site in [('center', int(N // 2)), ('opposite endpoint', int(N - 1))]:
                r = hard_window_budget_sparse_chain(int(N), site, T, J=J, kappa=kappa)
                rows.append(dict(N=int(N), T=T, position=label, site_1based=site+1,
                                 F_port=r['F_port'], F_stored=r['F_stored'],
                                 F_budget=r['F_budget'], nfev=r['nfev']))
            print(f'T={T:5.1f} N={N:4d}', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(DATA / 'propagation_limit.csv', index=False)
else:
    print('reusing packaged data/propagation_limit.csv (pass --full to recompute)')
df = pd.read_csv(DATA / 'propagation_limit.csv')


# Main diagnostic is the center, where the two causal scales are cleanest.
# Values below ~1e-12 are below the ODE tolerance and are clipped in the plot.
floor = 1e-12
fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.6), gridspec_kw=dict(wspace=0.3))
cols = {20.0: C3[0], 40.0: C3[1], 80.0: C3[2]}
ax = axes[0]
for T in Ts:
    g = df[(df.position == 'center') & (df['T'] == T)].sort_values('N')
    ax.semilogy(g.N, np.clip(g.F_port, floor, None), 'o-', color=cols[T], ms=3.2, mew=0, lw=0.9, label=rf'$T={T:g}\,J^{{-1}}$')
    ax.semilogy(g.N, np.clip(g.F_stored, floor, None), 's--', color=cols[T], ms=3.0, mfc='white', mew=0.7, lw=0.9)
    ax.axvline(2 * J * T, color=cols[T], lw=0.6, alpha=0.5)
    ax.axvline(4 * J * T, color=cols[T], lw=0.6, alpha=0.5, ls='--')
ax.plot([], [], 'o-', color='0.4', ms=3.2, mew=0, lw=0.9, label='emitted to port')
ax.plot([], [], 's--', color='0.4', ms=3.0, mfc='white', mew=0.7, lw=0.9, label='stored internally')
ax.set_xlabel('chain length $N$'); ax.set_ylabel('QFI per photon at the deadline')
ax.set_ylim(floor, 1e2); ax.set_xlim(0, 330)
ax.legend(loc='upper right', fontsize=6.2, ncol=1)
ax.text(0.03, 0.05, 'vertical lines: $N=2JT$ (solid), $4JT$ (dashed)', transform=ax.transAxes, fontsize=6.2, color='0.3')
panel_label(ax, '(a)', x=-0.2); tidy(ax)

ax = axes[1]
for T in Ts:
    g = df[(df.position == 'center') & (df['T'] == T)].sort_values('N')
    ax.semilogy(g.N / (2 * J * T), np.clip(g.F_port / g.F_port.max(), floor, None), 'o-', color=cols[T], ms=3.2, mew=0, lw=0.9)
    ax.semilogy(g.N / (4 * J * T), np.clip(g.F_stored / g.F_stored.max(), floor, None), 's--', color=cols[T], ms=3.0, mfc='white', mew=0.7, lw=0.9)
ax.axvline(1.0, color='k', ls=':', lw=0.8)
ax.set_xlabel(r'scaled size: port $N/(2JT)$; stored $N/(4JT)$')
ax.set_ylabel('normalized QFI component')
ax.set_xlim(0.05, 4.2); ax.set_ylim(floor, 3)
ax.plot([], [], 'o-', color='0.4', ms=3.2, mew=0, lw=0.9, label='emitted to port (round trip)')
ax.plot([], [], 's--', color='0.4', ms=3.0, mfc='white', mew=0.7, lw=0.9, label='stored internally (one way)')
ax.legend(loc='upper right', fontsize=6.2)
ax.text(1.05, 3e-3, 'ballistic\ncutoff', fontsize=6.4, color='0.3', va='center')
panel_label(ax, '(b)', x=-0.2); tidy(ax)
fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.18)
save_figure(fig, 'figS_propagation_limit')
