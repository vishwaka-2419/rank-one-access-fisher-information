from _common import *
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from _style import *
from rankonefi import (uniform_chain_hamiltonian, clean_mode_energy, clean_mode_weight,
                       fi_port_per_photon_uniform, WindowedFI,
                       h_unrestricted, k_windowed, stored_single_mode_factor, single_mode_prefactor, optimum_x_windowed,
                       fit_power_law)

# Finite-duration coherent probing. A rectangular pulse of length T drives the port at
# the clean band-edge frequency.  We distinguish:
#   (i) complete ring-down collection after the drive stops (not a hard total-time limit),
#       with single-mode factor h(Gamma T);
#   (ii) a hard deadline: drive and port detection both confined to [0,T], factor k(Gamma T);
#   (iii) the exact information budget at the deadline, port output plus QFI still stored
#       in the internal coherent state, factor k(Gamma T)+q(Gamma T) for a lossless mode.
# Per incident photon.  kappa/J = 0.1, gamma = 0.

kappa = 0.1
Ts = [1.0e3, 1.0e4, 1.0e5]
Ns_spec = np.array([4, 6, 8, 10, 12, 16, 20, 26, 32, 40, 50, 64, 80, 100, 125, 160, 200])
Ns_win = Ns_spec.copy()
Delta = 100.0

FULL = '--full' in sys.argv or not (DATA / 'fixed_time.csv').exists()
if FULL:
    rows = []
    for N in Ns_spec:
        w0 = clean_mode_energy(N, 1)
        Gp = kappa * clean_mode_weight(N, 0)
        for label, site in [('center', N // 2), ('opposite endpoint', N - 1)]:
            ws = clean_mode_weight(N, site)
            F0 = single_mode_prefactor(Gp, Gp, ws)
            Fmono = fi_port_per_photon_uniform(N, site, w0, kappa=kappa)
            H = uniform_chain_hamiltonian(N)
            win = WindowedFI(H, kappa, 0, site, w0, Delta=Delta)
            for T in Ts:
                x = Gp * T
                # Exact complete-ring-down QFI from the Van Loan/Lyapunov engine.
                # The independent spectral-average implementation is retained in the tests.
                F_spec = win.fisher_per_photon(T, None)
                rec = dict(N=N, position=label, site_1based=site + 1, T=T, GammaT=x,
                           F_mono=Fmono, F0_single_mode=F0,
                           F_unrestricted_exact=F_spec, F_unrestricted_single_mode=F0 * float(h_unrestricted(x)),
                           F_window_exact=np.nan, F_window_single_mode=F0 * float(k_windowed(x)),
                           F_stored_exact=np.nan, F_budget_exact=np.nan,
                           F_stored_single_mode=F0 * float(stored_single_mode_factor(x)),
                           F_budget_single_mode=F0 * float(k_windowed(x) + stored_single_mode_factor(x)),
                           F_halfdrive_window_exact=np.nan)
                if True:
                    budget = win.hard_window_budget(T)
                    rec['F_window_exact'] = budget['F_port_per_photon']
                    rec['F_stored_exact'] = budget['F_stored_per_photon']
                    rec['F_budget_exact'] = budget['F_budget_per_photon']
                    # protocol variation: drive for T/2, keep detecting until T
                    rec['F_halfdrive_window_exact'] = win.fisher(T / 2, T) / (T / 2)
                    rec['F_unrestricted_timedomain'] = win.fisher_per_photon(T, None)
                rows.append(rec)
            print(f'N={N:4d} done', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(DATA / 'fixed_time.csv', index=False)
else:
    print('reusing packaged data/fixed_time.csv (pass --full to recompute)')
df = pd.read_csv(DATA / 'fixed_time.csv')


# --------------------------------------------------------------- exponents
key = {}
for label in ['center', 'opposite endpoint']:
    for T in Ts:
        g = df[(df.position == label) & (df['T'] == T)].sort_values('N')
        NT = (2 * np.pi ** 2 * kappa * T) ** (1 / 3)
        m = g['N'] > 2 * NT
        pu, _ = fit_power_law(g['N'][m], g['F_unrestricted_exact'][m])
        gw = g.dropna(subset=['F_window_exact'])
        # single-mode intermediate asymptote: beyond the crossover but before the
        # multimode/propagation regime takes over (exact within a factor 2 of the single-mode law)
        mw = (gw['N'] > 2 * NT) & (gw['F_window_exact'] < 2 * gw['F_window_single_mode'])
        pw = fit_power_law(gw['N'][mw], gw['F_window_exact'][mw])[0] if mw.sum() >= 3 else np.nan
        key[f'{label}_T{T:.0e}_NT'] = NT
        key[f'{label}_T{T:.0e}_slope_unrestricted'] = pu
        key[f'{label}_T{T:.0e}_slope_window'] = pw
        print(f'{label:18s} T={T:.0e}: N_T={NT:5.1f}  slope(unrestricted, N>2N_T)={pu:+.2f}  slope(window, loading-limited regime)={pw:+.2f}')

# optimum of N^4 k(2 pi^2 kappa T / N^3): x k'(x)/k(x) = 4/3
xstar = optimum_x_windowed(4.0 / 3.0)
key['x_star_window'] = xstar
for T in Ts:
    key[f'N_T_star_window_T{T:.0e}'] = (2 * np.pi ** 2 * kappa * T / xstar) ** (1 / 3)
    g = df[(df.position == 'center') & (df['T'] == T)].dropna(subset=['F_window_exact'])
    key[f'N_T_star_window_numerical_T{T:.0e}'] = int(g.loc[g['F_window_exact'].idxmax(), 'N'])
with open(DATA / 'fixed_time_key_results.json', 'w') as f:
    json.dump(key, f, indent=2)
print(json.dumps(key, indent=1))

# ---------------------------------------------------------------- figure (main)
cols = {Ts[0]: C3[0], Ts[1]: C3[1], Ts[2]: C3[2]}
Tlab = {T: rf'$T=10^{{{int(np.log10(T))}}}J^{{-1}}$' for T in Ts}
fig, axes = plt.subplots(1, 3, figsize=(DBL_W, 2.6), gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.0], wspace=0.38))
ax = axes[0]
g0 = df[df.position == 'center']
ax.loglog(g0[g0['T'] == Ts[0]]['N'], g0[g0['T'] == Ts[0]]['F_mono'], '-', color='k', lw=1.0, label=r'steady state ($T\to\infty$)')
for T in Ts:
    g = g0[g0['T'] == T].sort_values('N')
    ax.loglog(g['N'], g['F_unrestricted_exact'], '--', color=cols[T], lw=0.9)
    gw = g.dropna(subset=['F_window_exact'])
    ax.loglog(gw['N'], gw['F_window_exact'], 'o', color=cols[T], ms=3.4, mew=0, label=Tlab[T])
    ax.loglog(g['N'], g['F_window_single_mode'], ':', color=cols[T], lw=0.8)
ax.plot([], [], '--', color='0.4', lw=0.9, label='ring-down collected')
ax.plot([], [], 'o', color='0.4', ms=3.4, mew=0, label='hard deadline (exact)')
ax.plot([], [], ':', color='0.4', lw=0.8, label=r'$\mathcal{F}_0\,k(\Gamma T)$')
ax.set_xlabel('chain length $N$'); ax.set_ylabel('center-site QFI per photon')
ax.set_ylim(1e-3, 1e14); ax.set_xlim(3.5, 260)
ax.legend(loc='upper left', fontsize=6.0, ncol=1, handlelength=1.4, handletextpad=0.4)
for T in Ts:
    Ns_ = (2 * np.pi ** 2 * kappa * T / xstar) ** (1 / 3)
    ax.axvline(Ns_, color=cols[T], lw=0.5, ls='-', alpha=0.35)
ax.text(0.97, 0.05, r'$N_\star^{(T)}=1.38(\kappa T)^{1/3}$', transform=ax.transAxes, ha='right', fontsize=7, color='0.25')
panel_label(ax, '(a)', x=-0.22); tidy(ax)

ax = axes[1]
xx = np.logspace(-2.5, 2.5, 400)
qxx = stored_single_mode_factor(xx)
bxx = k_windowed(xx) + qxx
ax.loglog(xx, h_unrestricted(xx), '--', color='k', lw=1.0, label=r'$h$: ring-down collected')
ax.loglog(xx, bxx, '-.', color='0.45', lw=1.0, label=r'$b=k+q$: emitted + stored')
ax.loglog(xx, k_windowed(xx), '-', color='k', lw=1.0, label=r'$k$: emitted by deadline')
for T in Ts:
    for label, mk in [('center', 'o'), ('opposite endpoint', 's')]:
        g = df[(df.position == label) & (df['T'] == T)]
        ax.loglog(g['GammaT'], g['F_unrestricted_exact'] / g['F0_single_mode'], mk, color=cols[T], ms=2.8, mfc='none', mew=0.6)
        gw = g.dropna(subset=['F_window_exact'])
        ax.loglog(gw['GammaT'], gw['F_budget_exact'] / gw['F0_single_mode'], mk, color='0.45', ms=2.4, mfc='none', mew=0.5)
        ax.loglog(gw['GammaT'], gw['F_window_exact'] / gw['F0_single_mode'], mk, color=cols[T], ms=2.8, mew=0)
ax.set_xlabel(r'$\Gamma T$'); ax.set_ylabel(r'$\mathcal{F}_T/\mathcal{F}_0$')
ax.set_ylim(1e-9, 3); ax.set_xlim(3e-4, 3e2)
ax.legend(loc='lower right', fontsize=6.0, handlelength=1.5)
ax.text(0.03, 0.95, 'circles: center\nsquares: far end', transform=ax.transAxes, fontsize=6.2, va='top', color='0.3')
panel_label(ax, '(b)', x=-0.26); tidy(ax)

ax = axes[2]
frac_emit = k_windowed(xx) / bxx
frac_store = qxx / bxx
ax.semilogx(xx, frac_emit, '-', color='k', lw=1.1, label='emitted by deadline')
ax.semilogx(xx, frac_store, '--', color='0.45', lw=1.1, label='still stored at deadline')
for T in Ts:
    g = df[(df.position == 'center') & (df['T'] == T)].dropna(subset=['F_budget_exact'])
    ax.semilogx(g['GammaT'], g['F_window_exact'] / g['F_budget_exact'], 'o', color=cols[T], ms=2.8, mew=0)
    ax.semilogx(g['GammaT'], g['F_stored_exact'] / g['F_budget_exact'], 'o', color=cols[T], ms=2.8, mfc='none', mew=0.6)
ax.set_xlabel(r'$\Gamma T$'); ax.set_ylabel('fraction of deadline budget')
ax.set_ylim(-0.03, 1.03); ax.set_xlim(3e-4, 3e2)
ax.set_yticks([0, 0.5, 1])
ax.legend(loc='center left', fontsize=6.2, bbox_to_anchor=(0.0, 0.45))
panel_label(ax, '(c)', x=-0.26); tidy(ax)
fig.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.19)
save_figure(fig, 'fig10_fixed_time')

# ---------------------------------------------------------------- figure (supplement): endpoint + protocol variation
fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.6), gridspec_kw=dict(wspace=0.3))
ax = axes[0]
g0 = df[df.position == 'opposite endpoint']
ax.loglog(g0[g0['T'] == Ts[0]]['N'], g0[g0['T'] == Ts[0]]['F_mono'], '-', color='k', lw=1.0, label='steady state')
for T in Ts:
    g = g0[g0['T'] == T].sort_values('N')
    ax.loglog(g['N'], g['F_unrestricted_exact'], '--', color=cols[T], lw=0.9)
    gw = g.dropna(subset=['F_window_exact'])
    ax.loglog(gw['N'], gw['F_window_exact'], 's', color=cols[T], ms=3.4, mew=0, label=Tlab[T])
ax.plot([], [], '--', color='0.4', lw=0.9, label='ring-down collected')
ax.plot([], [], 's', color='0.4', ms=3.4, mew=0, label='hard deadline')
ax.set_xlabel('chain length $N$'); ax.set_ylabel('opposite-endpoint QFI per photon')
ax.set_ylim(1e-4, 3e4)
ax.legend(loc='lower left', fontsize=6.2, ncol=2, columnspacing=0.8)
panel_label(ax, '(a)', x=-0.2); tidy(ax)
ax = axes[1]
g0 = df[df.position == 'center'].dropna(subset=['F_halfdrive_window_exact'])
for T in Ts:
    g = g0[g0['T'] == T].sort_values('N')
    ax.loglog(g['N'], g['F_window_exact'], 'o-', color=cols[T], ms=3.2, mew=0, lw=0.8, label=Tlab[T] + ', drive $T$')
    ax.loglog(g['N'], g['F_halfdrive_window_exact'], 'x--', color=cols[T], ms=3.6, lw=0.8, label=Tlab[T] + ', drive $T/2$')
ax.set_xlabel('chain length $N$'); ax.set_ylabel('center-site QFI per photon, window $[0,T]$')
ax.legend(loc='lower left', fontsize=6.0, ncol=1)
panel_label(ax, '(b)', x=-0.2); tidy(ax)
fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.18)
save_figure(fig, 'figS_fixed_time_endpoint_protocol')
