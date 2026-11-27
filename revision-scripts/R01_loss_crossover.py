"""
Referee 1, point 2: domain of the N^-8 law at fixed intrinsic loss and the crossover to
spatial attenuation.  Uniform chain, port on site 1, drive at E1, signal at the centre.
"""
import sys, json
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import rof

J = 1.0
kappa = 0.1
out = {}

def Phi(g):
    """loss-dominated scaling function: F_c ~ (kappa^2 / 4 J^4) |cos(theta/2)|^-4, theta^2 = pi^2 - i g"""
    th = np.sqrt(np.pi ** 2 - 1j * g)
    return np.abs(np.cos(th / 2)) ** (-4.0)

print("== closed form vs banded solve (centre site) ==")
for gam in [1e-4]:
    for N in [21, 101, 501, 1001, 2001, 4001]:
        s = (N + 1) // 2
        Fb = rof.chain_F(N, s, kappa, gam, method="banded")
        Fc = rof.chain_F(N, s, kappa, gam, method="closed")
        print(N, Fb, Fc, abs(Fb - Fc) / Fb)

print("\n== integer optimum, isolation at the optimum ==")
for gam in [1e-6, 1e-5, 1e-4, 1e-3]:
    Nstar = (np.pi ** 2 * kappa / gam) ** (1 / 3)
    Ns = np.arange(3, int(4 * Nstar) + 3)
    F = np.array([rof.chain_F(N, (N + 1) // 2, kappa, gam) for N in Ns])
    Nopt = int(Ns[np.argmax(F)])
    # odd sizes only (true centre site)
    Nodd = Ns[Ns % 2 == 1]
    Fodd = np.array([rof.chain_F(N, (N + 1) // 2, kappa, gam) for N in Nodd])
    Nopt_odd = int(Nodd[np.argmax(Fodd)])
    w1 = rof.chain_w(Nopt, 1)
    Gam = gam + kappa * w1
    ratio = Gam / rof.chain_gap(Nopt)
    pred = (kappa ** 2 * gam / (np.pi ** 2 * J ** 3)) ** (1 / 3)
    Ngam = np.pi * np.sqrt(3 * J / gam)
    print(f"gamma={gam:g} Nstar={Nstar:.2f} Nopt={Nopt} (odd-only {Nopt_odd}) kappa*w1/gamma={kappa*w1/gam:.3f} "
          f"Gamma/DeltaE={ratio:.5f} closed-form estimate={pred:.5f}  N_gamma={Ngam:.1f}  N_gamma/Nstar={Ngam/Nstar:.1f}")
    out[f"opt_{gam:g}"] = dict(Nstar=Nstar, Nopt=Nopt, Nopt_odd=Nopt_odd, ratio=ratio, pred=pred, Ngam=Ngam)

print("\n== exact / single-mode / scaling function, centre site ==")
for gam in [1e-4, 1e-5, 1e-6]:
    Ngam = np.pi * np.sqrt(3 * J / gam)
    print(f"-- gamma/J = {gam:g}, N_gamma = {Ngam:.0f}, attenuation length for F: (1/2)sqrt(J/gamma) = {0.5*np.sqrt(J/gam):.1f}")
    for N in [21, 45, 99, 101, 201, 301, 545, 1001, 1721, 2001, 4001, 5441, 10001, 20001]:
        s = (N + 1) // 2
        Fex = rof.chain_F(N, s, kappa, gam, method="closed")
        Fm = rof.mode_F(rof.chain_w(N, 1), rof.chain_w(N, s), kappa, gam)
        g = gam * (N + 1) ** 2 / (2 * J)
        Fsc = kappa ** 2 / (4 * J ** 4) * Phi(g)
        gapratio = gam / rof.chain_gap(N)
        print(f"N={N:6d} gamma/DeltaE={gapratio:9.4f} Gp/gamma={kappa*rof.chain_w(N,1)/gam:9.3e} "
              f"F_exact={Fex:10.3e} exact/mode={Fex/Fm:10.4e} exact/scaling={Fex/Fsc:8.4f}")

print("\n== local exponent d ln F / d ln N (exact), minimum and where ==")
for gam in [1e-4, 1e-5, 1e-6, 1e-8]:
    Ns = np.unique(np.round(np.logspace(np.log10(9), np.log10(60000), 900)).astype(int))
    Ns = Ns[Ns % 2 == 1]
    F = np.array([rof.chain_F(N, (N + 1) // 2, kappa, gam, method="closed") for N in Ns])
    slope = np.gradient(np.log(F), np.log(Ns))
    Nstar = (np.pi ** 2 * kappa / gam) ** (1 / 3)
    Ngam = np.pi * np.sqrt(3 * J / gam)
    # window where slope within 10% of -8  (i.e. -8.8 < slope < -7.2)
    mask = (slope < -7.2) & (slope > -8.8)
    win = Ns[mask]
    # plateau: slope has an inflection near -8; report the slope at geometric mean of Nstar, Ngam
    Nmid = np.sqrt(Nstar * Ngam)
    print(f"gamma={gam:g}: Nstar={Nstar:.1f} N_gamma={Ngam:.0f}; slope at sqrt(Nstar*Ngam)={np.interp(Nmid, Ns, slope):.3f}; "
          f"-8+-10% window: N in [{win.min() if win.size else None}, {win.max() if win.size else None}] "
          f"(ratio {win.max()/win.min() if win.size else 0:.2f})")

print("\n== attenuation: -ln(F)/N slope at large N vs sqrt(gamma/J) (centre) and endpoint 2 sqrt(gamma/J) ==")
for gam in [1e-4, 1e-5]:
    N1, N2 = 20001, 30001          # deep in the attenuation regime, and F still representable in double precision
    for label, sfun, mult in [("centre", lambda N: (N + 1) // 2, 1), ("far end", lambda N: N, 2)]:
        F1 = rof.chain_F(N1, sfun(N1), kappa, gam, method="closed")
        F2 = rof.chain_F(N2, sfun(N2), kappa, gam, method="closed")
        rate = -(np.log(F2) - np.log(F1)) / (N2 - N1)
        print(f"gamma={gam:g} {label}: decay rate per site = {rate:.6f}; prediction {mult} sqrt(gamma/J) = {mult*np.sqrt(gam):.6f}")
        out[f'decay_{gam:g}_{label}'] = (float(rate), float(mult * np.sqrt(gam)))
json.dump(out, open(os.path.join(HERE, 'out', 'R01_loss_crossover.json'), 'w'), indent=1)
