"""
Referee 1, point 4: port codimension under a deadline.
(a) universal single-mode rule: optimal port-induced width versus gamma*T
(b) 7x7 open square lattice, signal at the centre, ports of codimension 2 (corner), 1 (mid-edge)
    and 0 (interior): exact 49-mode time-domain windowed QFI vs isolated-mode prediction.
"""
import sys, json
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.optimize import minimize_scalar, brentq
import rof

# ---------------- (a) optimal port participation
print("== optimal x_p = kappa w_p T for given g = gamma T  (maximise x_p^2 k(x_p+g)/(x_p+g)^4) ==")
def merit(xp, g):
    x = xp + g
    return xp ** 2 * rof.k_fun(x) / x ** 4
rows = []
for g in [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0]:
    r = minimize_scalar(lambda lx: -merit(np.exp(lx), g), bounds=(np.log(1e-3), np.log(1e5)), method='bounded',
                        options=dict(xatol=1e-10))
    xp = float(np.exp(r.x))
    rows.append((g, xp, xp / g if g > 0 else np.inf, -r.fun))
    print(f"gamma T = {g:7.1f}:  x_p,opt = {xp:9.4f}   Gamma_p,opt/gamma = {xp/g if g>0 else np.inf:8.4f}   merit*64 = {64*-r.fun:.5f}")
print("limits: gamma T -> 0: x_p -> 4.7445 (Gamma_p T = 4.74);  gamma T -> inf: Gamma_p -> gamma")

# ---------------- (b) 7x7 lattice
L, d, J, kappa = 7, 2, 1.0, 0.1
H0 = rof.lattice_H0(L, d, J).toarray()
Ebe = 2 * d * J * np.cos(np.pi / (L + 1))
sig = rof.lattice_index((4, 4), L)
ports = {"corner (1,1), k=2": (1, 1), "mid-edge (1,4), k=1": (1, 4), "interior (2,2), k=0": (2, 2),
         "interior (2,4), k=0": (2, 4)}
ws = rof.lattice_w((4, 4), L)
gap = 2 * J * (np.cos(np.pi / 8) - np.cos(np.pi / 4))
print(f"\n7x7 lattice: w_s = {ws:.6f} (=1/16), band-edge energy {Ebe:.5f} J, gap to next level {gap:.4f} J")
wp = {}
for name, c in ports.items():
    wp[name] = rof.lattice_w(c, L)
    print(f"  port {name}: w_p = {wp[name]:.7f},  Gamma_p/J = {kappa*wp[name]:.3e},  steady-state F kappa^2/64 = {(ws/wp[name])**2:.2f}")

def F_exact(port, u):
    p = rof.lattice_index(port, L)
    Heff = H0.astype(complex).copy()
    Heff[p, p] -= 0.5j * kappa
    return rof.windowed_qfi(Heff, p, sig, kappa, Ebe, u / kappa)

def F_mode(w, u):
    return 64.0 * (ws / w) ** 2 / kappa ** 2 * rof.k_fun(kappa * w * u / kappa)

res = {}
print("\n kappa T |  port  | exact 49-mode F | isolated-mode F | exact/mode")
us = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
for u in us:
    res[u] = {}
    for name, c in ports.items():
        fe = F_exact(c, u)
        fm = F_mode(wp[name], u)
        res[u][name] = (fe, fm)
        print(f"{u:7d} | {name:22s} | {fe:14.6e} | {fm:14.6e} | {fe/fm:.4f}")
print("\n ratios corner/mid-edge:  kappaT : exact, isolated-mode")
for u in us:
    a = res[u]["corner (1,1), k=2"]; b = res[u]["mid-edge (1,4), k=1"]
    print(f"   {u:6d}: {a[0]/b[0]:10.4f}  {a[1]/b[1]:10.4f}")
# crossover deadlines (isolated-mode)
def cross(wa, wb):
    f = lambda lu: np.log((wb / wa) ** 2 * rof.k_fun(wa * np.exp(lu)) / rof.k_fun(wb * np.exp(lu)))
    return float(np.exp(brentq(f, np.log(1.0), np.log(1e7))))
names = list(ports)
print("\ncrossover kappa*T at which the smaller-weight port overtakes the larger-weight one (isolated-mode):")
print("  corner vs mid-edge :", cross(wp[names[0]], wp[names[1]]))
print("  mid-edge vs interior(2,2):", cross(wp[names[1]], wp[names[2]]))
print("  corner vs interior(2,2):", cross(wp[names[0]], wp[names[2]]))
print("  optimal-port condition kappa w_p T = 4.74 ->  kappa T =", {n: 4.7445 / wp[n] for n in names})
json.dump(dict(rows=rows, wp=wp, ws=ws, res={str(u): res[u] for u in us}), open(os.path.join(HERE, 'out', 'R03_port_choice.json'), 'w'), indent=1)
