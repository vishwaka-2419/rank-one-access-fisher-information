"""Cross-checks of statements of the original submission with the independent revision code."""
import sys, json, time
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import rof
kappa = 0.1
out = {}
# (1) hard-deadline optimum for T = 1e5/J: original text: predicted 29.7, exact maximum between N = 26 and 32
T = 1e5
vals = {}
for N in [25, 27, 29, 31, 33]:
    s = (N + 1) // 2
    t0 = time.time()
    F, _ = rof.windowed_qfi_curve(rof.chain_Heff(N, kappa), 0, s - 1, kappa, rof.chain_E1(N), [T])
    Fm = rof.mode_F(rof.chain_w(N, 1), rof.chain_w(N, s), kappa, 0.0) * rof.k_fun(kappa * rof.chain_w(N, 1) * T)
    vals[N] = (float(F[0]), float(Fm))
    print(f"T=1e5/J N={N}: exact windowed F = {F[0]:.5e}; isolated-mode F0 k(Gamma T) = {Fm:.5e}; ratio {F[0]/Fm:.4f}  ({time.time()-t0:.0f}s)", flush=True)
out['T1e5'] = vals
# (2) hard deadline with loss: 'agree with F0(gamma) k(Gamma T) to within a few percent'
for (N, gam, T) in [(15, 1e-4, 2e4), (21, 1e-4, 5e4), (31, 3e-5, 1e5)]:
    s = (N + 1) // 2
    F, _ = rof.windowed_qfi_curve(rof.chain_Heff(N, kappa, gam), 0, s - 1, kappa, rof.chain_E1(N), [T])
    G = gam + kappa * rof.chain_w(N, 1)
    Fm = rof.mode_F(rof.chain_w(N, 1), rof.chain_w(N, s), kappa, gam) * rof.k_fun(G * T)
    print(f"lossy window N={N} gamma={gam} T={T:g}: exact {F[0]:.5e}, F0(gamma)k(Gamma T) {Fm:.5e}, ratio {F[0]/Fm:.4f}", flush=True)
# (3) detuning optimum: maximizing F over omega increases it by 0.5% at -2.5% of the linewidth (lossless)
from scipy.optimize import minimize_scalar
for N in [50, 200, 800]:
    s = (N + 1) // 2
    E1 = rof.chain_E1(N); Gp = kappa * rof.chain_w(N, 1)
    f = lambda d: -rof.chain_F(N, s, kappa, 0.0, w=E1 + d * Gp)
    r = minimize_scalar(f, bounds=(-0.5, 0.5), method='bounded', options=dict(xatol=1e-6))
    print(f"N={N}: max over omega at detuning {r.x*100:+.2f}% of Gamma_p, gain {(-r.fun/rof.chain_F(N,s,kappa,0.0)-1)*100:.2f}%")
# (4) window optima x* solving x k'(x)/k(x) = 4k/(d+2k): original Appendix J quotes 7.553, 9.714, 11.56, 8.686
from scipy.optimize import brentq
def logslope(x, h=1e-4):
    return (np.log(rof.k_fun(x * (1 + h))) - np.log(rof.k_fun(x * (1 - h)))) / (np.log(1 + h) - np.log(1 - h))
for name, d, k in [("chain / corner port", 1, 1), ("square lattice, edge port", 2, 1), ("cubic lattice, face port", 3, 1), ("cubic lattice, edge port", 3, 2)]:
    target = 4 * k / (d + 2 * k)
    xs = brentq(lambda x: logslope(x) - target, 2.0, 40.0)
    print(f"window optimum, {name}: a/b = {target:.4f}, x* = {xs:.4f}, k(x*) = {float(rof.k_fun(xs)):.4f}")
    out[f'xstar_{d}_{k}'] = xs
# (5) asymptotic exponents (Table of representative exponents in Appendix N of the original), kappa/J = 0.1, lossless
Ns = np.unique(np.round(np.geomspace(400, 3200, 13)).astype(int) // 2 * 2 + 1)
Fc = np.array([rof.chain_F(N, (N + 1) // 2, kappa, 0.0) for N in Ns])
Fe = np.array([rof.chain_F(N, N, kappa, 0.0) for N in Ns])
tau = np.array([4.0 / (kappa * rof.chain_w(N, 1)) for N in Ns])          # exact at the eigenfrequency
fit = lambda y: np.polyfit(np.log(Ns), np.log(y), 1)[0]
print(f"exponents for N in [{Ns[0]}, {Ns[-1]}]: centre F {fit(Fc):.4f} (4), endpoint F {fit(Fe):+.4f} (0), "
      f"centre rate {fit(Fc/tau):.4f} (1), endpoint rate {fit(Fe/tau):.4f} (-3)")
out['exponents'] = dict(centre=fit(Fc), endpoint=fit(Fe), centre_rate=fit(Fc / tau), endpoint_rate=fit(Fe / tau))
# (6) loss added at fixed port width never increases the windowed QFI (exact time domain, N = 9, centre signal)
N = 9; s = 5
for T in [300.0, 3000.0, 30000.0]:
    row = []
    for gam in [0.0, 1e-4, 3e-4, 1e-3, 3e-3]:
        F, _ = rof.windowed_qfi_curve(rof.chain_Heff(N, kappa, gam), 0, s - 1, kappa, rof.chain_E1(N), [T])
        row.append(float(F[0]))
    mono = all(row[i + 1] < row[i] for i in range(len(row) - 1))
    print(f"N=9 T={T:g}/J: windowed F for gamma/J = 0, 1e-4, 3e-4, 1e-3, 3e-3: " + ", ".join(f"{v:.4e}" for v in row) + f"   monotonic decrease: {mono}")
    out[f'loss_T{T:g}'] = row
# (7) optimal size with both intrinsic loss and a hard deadline (isolated mode, large-N weights):
#     maximise x^(8/3) k(a + x) / (a + x)^4 over x = kappa w_1 T at fixed a = gamma T;  N_opt = (2 pi^2 kappa T / x)^(1/3)
from scipy.optimize import minimize_scalar as _ms
worst = (1.0, None)
for a in np.geomspace(1e-3, 1e3, 121):
    f = lambda lx: -(8 / 3 * lx + np.log(float(rof.k_fun(a + np.exp(lx)))) - 4 * np.log(a + np.exp(lx)))
    r = _ms(f, bounds=(np.log(1e-2), np.log(1e5)), method='bounded', options=dict(xatol=1e-10))
    ratio = (max(2 * a, 7.5533) / np.exp(r.x)) ** (1 / 3)        # N_opt / min(N_star, N_star^(T))
    if ratio < worst[0]:
        worst = (float(ratio), float(a))
print(f"loss and deadline together: N_opt / min(N_star, N_star^(T)) is always <= 1; smallest value {worst[0]:.4f} at gamma T = {worst[1]:.2f}")
out['both_constraints'] = worst
json.dump(out, open(os.path.join(HERE, 'out', 'R10_crosscheck.json'), 'w'), indent=1)
