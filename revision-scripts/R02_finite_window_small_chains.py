"""
Referee 1, point 3: chain vs directly accessible single resonator in the SAME finite window.
Exact time-domain QFI for small chains (lossless), same bare port rate kappa, rectangular
resonant pulse, drive and detection confined to [0, T].  u = kappa T.
The single resonator is the N = 1 chain with s = 1 (exact: 64 k(u)/kappa^2).
"""
import sys, json
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.optimize import brentq
import rof

J = 1.0

def F_chain_exact(N, s, kappa, u):
    H = rof.chain_Heff(N, kappa, 0.0, J)
    w0 = rof.chain_E1(N, J) if N > 1 else 0.0
    return rof.windowed_qfi(H, 0, s - 1, kappa, w0, u / kappa)

def F_res(kappa, u):
    return 64.0 * rof.k_fun(u) / kappa ** 2

def R_mode(N, s, u):
    w1, ws = rof.chain_w(N, 1), rof.chain_w(N, s)
    return (ws / w1) ** 2 * rof.k_fun(w1 * u) / rof.k_fun(u)

# --- 0. validate the time-domain integrator against the exact single-resonator formula
print("== validation: N=1 exact time domain vs 64 k(u)/kappa^2 ==")
for kappa in [0.1]:
    for u in [0.5, 5.0, 20.0, 60.0]:
        a = F_chain_exact(1, 1, kappa, u); b = F_res(kappa, u)
        print(f"u={u:5.1f}  time-domain={a:.10e}  formula={b:.10e}  rel.diff={abs(a-b)/b:.2e}")

# --- 1. continuum estimate
from scipy.optimize import minimize_scalar
def logder(x, h=1e-5):
    return x * (rof.k_fun(x * (1 + h)) - rof.k_fun(x * (1 - h))) / (2 * h * x) / rof.k_fun(x)
xstar = brentq(lambda x: logder(x) - 4.0 / 3.0, 2.0, 20.0)
kstar = rof.k_fun(xstar)
C = kstar / np.pi ** 4 * (2 * np.pi ** 2 / xstar) ** (4.0 / 3.0)
print(f"\nx* = {xstar:.5f}, k(x*) = {kstar:.5f}, N* = {(2*np.pi**2/xstar)**(1/3):.4f} (kappa T)^(1/3), C = {C:.6f}")
Rcont = lambda u: C * u ** (4.0 / 3.0) / rof.k_fun(u)
r1 = brentq(lambda u: Rcont(u) - 1, 1.0, 8.0)
r2 = brentq(lambda u: Rcont(u) - 1, 8.0, 60.0)
print(f"continuum ratio R_cont = C u^(4/3)/k(u): roots u = {r1:.3f} (N*={ (2*np.pi**2*r1/xstar)**(1/3):.2f}), "
      f"{r2:.3f} (N*={(2*np.pi**2*r2/xstar)**(1/3):.2f});  old (steady-state reference) root: {(1/C)**0.75:.2f}")
# optimized resonator
res = minimize_scalar(lambda y: -rof.k_fun(y) / y ** 2, bounds=(1, 15), method='bounded')
print(f"optimized resonator: max k(y)/y^2 = {-res.fun:.6f} at y = {res.x:.4f};  64*max = {64*-res.fun:.5f} T^2")

# --- 2. exact small chains
kappa = 0.1
results = {}
print(f"\n== exact R_N(u) = F_chain / F_resonator, kappa/J = {kappa} ==")
us = [10, 15, 20, 25, 30, 40, 60, 100]
cases = [(2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (5, 3), (6, 3), (6, 4), (7, 4), (8, 4), (9, 5)]
for (N, s) in cases:
    row = [F_chain_exact(N, s, kappa, u) / F_res(kappa, u) for u in us]
    rowm = [R_mode(N, s, u) for u in us]
    f = lambda u: F_chain_exact(N, s, kappa, u) / F_res(kappa, u) - 1.0
    fm = lambda u: R_mode(N, s, u) - 1.0
    try:
        ucross = brentq(f, 3.0, 400.0, xtol=1e-6)
    except ValueError:
        ucross = float('nan')
    try:
        ucross_m = brentq(fm, 3.0, 400.0, xtol=1e-8)
    except ValueError:
        ucross_m = float('nan')
    ss = (rof.chain_w(N, s) / rof.chain_w(N, 1)) ** 2
    results[f"{N}_{s}"] = dict(R=row, Rmode=rowm, ucross=ucross, ucross_mode=ucross_m, steady=ss)
    print(f"N={N} s={s} steady-state ratio (ws/w1)^2={ss:8.3f}  crossing u: exact={ucross:8.3f} isolated-mode={ucross_m:8.3f}")
    print("     exact:", " ".join(f"{r:8.4f}" for r in row))
    print("     mode :", " ".join(f"{r:8.4f}" for r in rowm))
print("u grid:", us)

# --- 3. dependence on kappa/J for N = 3, 5 (crossing value)
print("\n== crossing u for N=3,s=2 and N=5,s=3 versus kappa/J ==")
for kap in [0.02, 0.05, 0.1, 0.2, 0.4]:
    out = []
    for (N, s) in [(3, 2), (5, 3)]:
        f = lambda u: F_chain_exact(N, s, kap, u) / F_res(kap, u) - 1.0
        out.append(brentq(f, 3.0, 400.0, xtol=1e-6))
    print(f"kappa/J={kap}: u_cross(N=3)={out[0]:.3f}, u_cross(N=5)={out[1]:.3f}")

# --- 4. best integer chain vs u (centre site, odd N), exact
print("\n== best odd-N chain (centre signal) at fixed u, exact; ratio to same-coupler resonator ==")
best = {}
for u in [15, 20, 21, 25, 30, 40, 60, 100, 200, 400, 1000]:
    vals = {}
    for N in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
        s = (N + 1) // 2
        vals[N] = F_chain_exact(N, s, kappa, u)
    Nb = max(vals, key=vals.get)
    Ncont = (2 * np.pi ** 2 * u / xstar) ** (1 / 3)
    best[u] = dict(Nbest=Nb, R=vals[Nb] / vals[1], Rcont=Rcont(u), Ncont=Ncont)
    print(f"u={u:6.0f}: best N={Nb:2d}  R_exact={vals[Nb]/vals[1]:9.4f}   continuum: N*={Ncont:5.2f} R_cont={Rcont(u):9.4f}")
json.dump(dict(xstar=xstar, kstar=kstar, C=C, roots=[r1, r2], results=results, best=best, us=us),
          open(os.path.join(HERE, 'out', 'R02_finite_window.json'), 'w'), indent=1)
