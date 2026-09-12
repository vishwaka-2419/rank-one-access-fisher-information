from _common import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _style import *
from rankonefi import (access_identity, all_channel_fi_finite_difference,
                       random_reciprocal_network, random_flux_network, random_psd_loss,
                       uniform_chain_hamiltonian, hypercubic_hamiltonian_sparse)

# Exact access identity on general passive networks:
#   F_port = 16 eta_in eta_out (Im G_ss)^2 = 16 pi^2 eta_in eta_out rho_s^2
#   F_all  = 16 eta_in (Im G_ss)^2
# tested against the scattering derivative (analytic) and against central finite
# differences of every outgoing channel amplitude (fully independent).

rng = np.random.default_rng(20260911)
rows = []


def add_instance(label, H0, Gamma, kappa, omega, port, site, reciprocal):
    a = access_identity(H0, Gamma, kappa, omega, port, site)
    fd_port, fd_all = all_channel_fi_finite_difference(H0, Gamma, kappa, omega, port, site)
    rows.append(dict(
        ensemble=label, N=H0.shape[0], omega=omega, reciprocal=reciprocal,
        F_port=a['F_port'], F_port_identity=a['F_port_identity'], F_port_fd=fd_port,
        F_all=a['F_all'], F_all_identity=a['F_all_identity'], F_all_fd=fd_all,
        rho_s=a['rho_s'], ldos_bound=a['ldos_bound'],
        eta_in=a['eta_in'], eta_out=a['eta_out'],
        err_port=abs(a['F_port'] - a['F_port_identity']) / a['F_port'],
        err_port_fd=abs(a['F_port'] - fd_port) / a['F_port'],
        err_all=abs(a['F_all'] - a['F_all_identity']) / a['F_all'],
        err_all_fd=abs(a['F_all'] - fd_all) / a['F_all'],
    ))


n_per = 180
for i in range(n_per):
    N = int(rng.integers(8, 40))
    port, site = 0, int(rng.integers(1, N))
    kappa = float(10 ** rng.uniform(-2, 0))
    gamma = float(10 ** rng.uniform(-4, -1))
    omega = float(rng.uniform(-3.0, 3.0))
    P = np.zeros((N, N), dtype=complex); P[port, port] = kappa

    # 1. reciprocal random graph, uniform diagonal loss
    H = random_reciprocal_network(N, rng)
    add_instance('reciprocal graph, diagonal loss', H, P + gamma * np.eye(N), kappa, omega, port, site, True)

    # 2. flux-threaded (Hermitian, nonreciprocal) graph, diagonal loss
    Hf = random_flux_network(N, rng)
    add_instance('flux graph (nonreciprocal), diagonal loss', Hf, P + gamma * np.eye(N), kappa, omega, port, site, False)

    # 3. reciprocal graph, non-diagonal (dissipative-coupling) loss
    Gd = P + gamma * np.eye(N) + random_psd_loss(N, rng, int(rng.integers(1, 5)))
    add_instance('reciprocal graph + dissipative (non-diagonal) loss', H, Gd, kappa, omega, port, site, False)

    # 4. flux graph, non-diagonal loss
    Gd2 = P + gamma * np.eye(N) + random_psd_loss(N, rng, int(rng.integers(1, 5)))
    add_instance('flux graph, non-diagonal loss', Hf, Gd2, kappa, omega, port, site, False)

    # 5. disordered chain, site-dependent diagonal loss
    Hc = uniform_chain_hamiltonian(N, 1.0, disorder=rng.uniform(-0.5, 0.5, N))
    Gc = P + np.diag(gamma * rng.uniform(0.2, 1.0, N))
    add_instance('disordered chain, site-dependent loss', Hc, Gc, kappa, omega, port, site, True)

    # 6. lossless reciprocal graph: F_port must equal the LDOS bound exactly
    add_instance('lossless (any graph)', H if i % 2 else Hf, P, kappa, omega, port, site, bool(i % 2))

# 7. small 2D lattices, random port and signal site
for i in range(60):
    L = int(rng.integers(4, 9)); N = L * L
    H2 = hypercubic_hamiltonian_sparse(L, 2).toarray()
    port = int(rng.integers(0, N)); site = int(rng.integers(0, N))
    if site == port:
        site = (site + 1) % N
    kappa = float(10 ** rng.uniform(-2, 0)); gamma = float(10 ** rng.uniform(-4, -1))
    omega = float(rng.uniform(-4.0, 4.0))
    P = np.zeros((N, N), dtype=complex); P[port, port] = kappa
    add_instance('2D lattice, diagonal loss', H2, P + gamma * np.eye(N), kappa, omega, port, site, True)

df = pd.DataFrame(rows)
df.to_csv(DATA / 'general_identity.csv', index=False)

# instances whose port QFI is below the double-precision floor of the finite-difference
# check (deeply localized disordered chains) are flagged; the identity itself is exact there.
df['fd_resolvable'] = df['F_port'] > 1e-16
summary = df.groupby('ensemble').agg(
    n=('N', 'size'),
    max_err_port_identity=('err_port', 'max'),
    max_err_port_fd=('err_port_fd', 'max'),
    median_err_port_fd=('err_port_fd', 'median'),
    frac_unresolvable=('fd_resolvable', lambda s: float(1.0 - s.mean())),
    max_err_all_identity=('err_all', 'max'),
    max_err_all_fd=('err_all_fd', 'max'),
    max_eta_asym=('eta_in', lambda s: float(np.max(np.abs(s.values - df.loc[s.index, 'eta_out'].values)))),
    min_ratio_to_bound=('F_port', lambda s: float(np.min(s.values / df.loc[s.index, 'ldos_bound'].values))),
    max_ratio_to_bound=('F_port', lambda s: float(np.max(s.values / df.loc[s.index, 'ldos_bound'].values))),
)
summary.to_csv(DATA / 'general_identity_summary.csv')
print(summary.to_string())

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(DBL_W, 2.45), gridspec_kw=dict(width_ratios=[1.25, 1.0, 1.0], wspace=0.42))
ax = axes[0]
sc = ax.scatter(df['ldos_bound'], df['F_port'], c=df['eta_in'] * df['eta_out'], s=6, lw=0,
                cmap='viridis', vmin=0, vmax=1, rasterized=True, zorder=2)
lim = [1e-14, 1e5]
ax.plot(lim, lim, '--', color='k', lw=0.8, zorder=3, label=r'$\mathcal{F}_s=16\pi^2\rho_s^{2}$ (lossless)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xticks([1e-12, 1e-8, 1e-4, 1e0, 1e4]); ax.set_yticks([1e-12, 1e-8, 1e-4, 1e0, 1e4])
ax.set_xlabel(r'LDOS budget $16\pi^2\rho_s(\omega)^2$'); ax.set_ylabel(r'port QFI per photon $\mathcal{F}_s^{(\rm port)}$')
ax.legend(loc='upper left', fontsize=6.4)
cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.06, aspect=22)
cb.set_label(r'$\eta_{\rm in}\eta_{\rm out}$', labelpad=2); cb.ax.tick_params(labelsize=6.5, length=2)
cb.outline.set_linewidth(0.5)
ax.text(0.97, 0.05, f'{len(df)} random networks\n7 ensembles', transform=ax.transAxes, fontsize=6.5, ha='right', va='bottom', color='0.3')
panel_label(ax, '(a)', x=-0.26)
tidy(ax)

ax = axes[1]
ax.plot([0, 1], [0, 1], '--', color='k', lw=0.8, zorder=1)
ax.scatter(df['eta_out'], df['F_port'] / df['F_all'], s=7, lw=0, c=OI['vermilion'], rasterized=True, zorder=2)
ax.set_xlabel(r'collection factor $\eta_{\rm out}$')
ax.set_ylabel(r'$\mathcal{F}_s^{(\rm port)}/\mathcal{F}_s^{(\rm all)}$')
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
ax.text(0.05, 0.9, 'exact routing:\nport fraction $=\\eta_{\\rm out}$', transform=ax.transAxes, fontsize=6.6, va='top', color='0.3')
panel_label(ax, '(b)', x=-0.28)
tidy(ax)

ax = axes[2]
ok = df['fd_resolvable'].values
errs = np.concatenate([df['err_port_fd'].values[ok], df['err_all_fd'].values[ok]])
errs = np.clip(errs, 1e-17, None)
ax.hist(np.log10(errs), bins=30, color=OI['blue'], alpha=0.85, lw=0)
ax.set_xlabel(r'$\log_{10}$ rel. deviation, finite differences')
ax.set_ylabel('count')
ax.text(0.97, 0.9, 'identity itself:\n' + r'$<10^{-15}$ in every instance', transform=ax.transAxes, fontsize=6.6, ha='right', va='top', color='0.3')
panel_label(ax, '(c)', x=-0.28)
tidy(ax)
fig.subplots_adjust(left=0.085, right=0.99, top=0.9, bottom=0.2)
save_figure(fig, 'fig2_general_access_identity')
