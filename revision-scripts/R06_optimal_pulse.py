"""
Referee 2, point 6: optimal temporal profile under a hard deadline (isolated mode).

Windowed output derivative:  d b_out(t) = Gamma_p w_s int_0^t (t-u) exp(-Gamma (t-u)/2) e(u) du  = (K e)(t)
QFI per photon             :  4 ||K e||^2 / ||e||^2  on L^2[0, T]
=> optimum = 4 sigma_max(K)^2, attained by the top right-singular function.
In units x = Gamma T :  F_opt = F_0 k_opt(x),  F_0 = 64 Gamma_p^2 w_s^2 / Gamma^4,
    k_opt(x) = (x^4/16) * s1(x)^2,  s1 = top singular value of kernel (t-u) exp(-x (t-u)/2) on [0,1].
Checks:  x -> 0 : s1 -> 1/beta_1^2 with cos(beta) cosh(beta) = -1 (cantilever), beta_1 = 1.8751
         rectangular pulse k(x) -> x^4/320 ; ratio -> 20/beta_1^4 = 1.6178
         falling ramp -> 11/7 of rectangular.
"""
import sys, json
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.optimize import brentq
import rof

def top_singular(x, n=1600):
    # Gauss-Legendre Nystrom discretisation on [0,1]
    xg, wg = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (xg + 1); w = 0.5 * wg
    D = t[:, None] - t[None, :]
    Kmat = np.where(D > 0, D * np.exp(-0.5 * x * D), 0.0)
    A = np.sqrt(w)[:, None] * Kmat * np.sqrt(w)[None, :]
    U, S, Vt = np.linalg.svd(A)
    v = Vt[0] / np.sqrt(w)
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    return S[0], t, v

def k_opt(x, n=1200):
    s1, _, _ = top_singular(x, n)
    return x ** 4 / 16.0 * s1 ** 2

beta1 = brentq(lambda b: np.cos(b) * np.cosh(b) + 1, 1.5, 2.2)
print(f"cantilever root beta_1 = {beta1:.6f};  1/beta_1^2 = {1/beta1**2:.6f};  20/beta_1^4 = {20/beta1**4:.5f}")
s1, t, v = top_singular(1e-6)
print(f"numerical s1(x->0) = {s1:.6f};  short-window optimum F = 4 Gp^2 ws^2 T^4 s1^2 = {4*s1**2:.5f} Gp^2 ws^2 T^4 "
      f"(rectangular: 0.2; ratio {4*s1**2/0.2:.5f})")
# convergence
for n in [400, 800, 1600]:
    print("   n =", n, " s1 =", top_singular(1e-6, n)[0])
# falling ramp check: e = sqrt(3)(1-t): Ke = sqrt3 (t^2/2 - t^3/6); ||.||^2 = 3 int (t^2/2 - t^3/6)^2
tt = np.linspace(0, 1, 200001)
val = 3 * np.trapezoid((tt ** 2 / 2 - tt ** 3 / 6) ** 2, tt)
print(f"falling ramp: 4||Ke||^2 = {4*val:.6f}  ratio to rectangular 0.2: {4*val/0.2:.5f}  (11/7 = {11/7:.5f})")
# shape of the optimal envelope at small x: compare with cantilever mode  (free at t=0, clamped at t=1)
b = beta1
sig = (np.sinh(b) - np.sin(b)) / (np.cosh(b) + np.cos(b))
# standard clamped-free mode clamped at xi=0: phi(xi) = cosh(b xi) - cos(b xi) - sig (sinh(b xi) - sin(b xi)); here xi = 1 - t
xi = 1 - t
phi = np.cosh(b * xi) - np.cos(b * xi) - sig * (np.sinh(b * xi) - np.sin(b * xi))
xg, wg = np.polynomial.legendre.leggauss(len(t)); w = 0.5 * wg
phi /= np.sqrt(np.sum(w * phi ** 2)); vn = v / np.sqrt(np.sum(w * v ** 2))
print(f"max |v_num - cantilever mode| = {np.max(np.abs(vn - phi)):.2e};  peak-to-rms of optimal envelope: {np.max(np.abs(vn)):.4f} "
      f"(peak power / mean power = {np.max(vn**2):.4f}; ramp: 3)")

print("\n   x = Gamma T    k_rect(x)      k_opt(x)     ratio")
rows = []
for x in [1e-3, 0.03, 0.1, 0.3, 1, 2, 3, 5, 7.553, 10, 15, 20, 30, 50, 100, 300]:
    ko = k_opt(x); kr = rof.k_fun(x)
    rows.append((x, kr, ko, ko / kr))
    print(f"   {x:9.3f}  {kr:12.5e}  {ko:12.5e}  {ko/kr:8.4f}")

# optimal size with optimised pulses: maximise N^4 k_opt(2 pi^2 kappa T / N^3)  <=> x k'/k = 4/3
def logder(f, x, h=1e-3):
    return (np.log(f(x * (1 + h))) - np.log(f(x * (1 - h)))) / (np.log(1 + h) - np.log(1 - h))
xs = brentq(lambda x: logder(lambda y: k_opt(y, 800), x) - 4.0 / 3.0, 3.0, 20.0, xtol=1e-4)
ks = k_opt(xs)
xr = brentq(lambda x: logder(rof.k_fun, x, 1e-5) - 4.0 / 3.0, 3.0, 20.0)
kr = rof.k_fun(xr)
pre_o = (2 * np.pi ** 2 / xs) ** (1 / 3); pre_r = (2 * np.pi ** 2 / xr) ** (1 / 3)
print(f"\nrectangular: x* = {xr:.4f}, k = {kr:.4f}, N* = {pre_r:.4f} (kappa T)^(1/3), peak ~ k N*^4 -> {kr*pre_r**4:.4f}")
print(f"optimal    : x* = {xs:.4f}, k = {ks:.4f}, N* = {pre_o:.4f} (kappa T)^(1/3), peak ~ k N*^4 -> {ks*pre_o**4:.4f}")
print(f"gain in peak QFI at the respective optimal sizes: {ks*pre_o**4/(kr*pre_r**4):.4f};  shift of optimal size: {pre_o/pre_r:.4f}")
# optimised single resonator with optimal pulse: max_y k_opt(y)/y^2
from scipy.optimize import minimize_scalar
r = minimize_scalar(lambda y: -k_opt(y, 800) / y ** 2, bounds=(1, 15), method='bounded', options=dict(xatol=1e-3))
print(f"optimised resonator + optimal pulse: 64 max k_opt(y)/y^2 = {64*-r.fun:.5f} T^2 at y = {r.x:.3f}   (rectangular: 0.44530 at 4.7445);  dwell-time bound: 4 T^2")
json.dump(dict(beta1=beta1, rows=rows, xs=xs, ks=ks, xr=xr, kr=kr, res_opt=64 * -r.fun, y_opt=r.x), open(os.path.join(HERE, 'out', 'R06_optimal_pulse.json'), 'w'), indent=1)
