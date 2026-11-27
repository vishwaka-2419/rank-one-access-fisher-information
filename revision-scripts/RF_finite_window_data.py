"""Data for the finite-window comparison figure (exact time-domain calculations)."""
import sys, json, time
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import rof

J, kappa = 1.0, 0.1
out = {}
# (a) small chains vs single resonator
us = np.logspace(np.log10(4.0), np.log10(3000.0), 70)
out['us'] = us.tolist()
res1 = np.array([64 * rof.k_fun(u) / kappa ** 2 for u in us])
out['res'] = res1.tolist()
for N in [2, 3, 5, 7, 9]:
    s = (N + 1) // 2 if N > 2 else 2
    H = rof.chain_Heff(N, kappa)
    F, _ = rof.windowed_qfi_curve(H, 0, s - 1, kappa, rof.chain_E1(N), us / kappa)
    out[f'chain_{N}'] = dict(s=s, F=F.tolist(), w1=rof.chain_w(N, 1), ws=rof.chain_w(N, s))
    print(N, s, 'done', flush=True)
# (b) 7x7 lattice ports
L, d = 7, 2
H0 = rof.lattice_H0(L, d, J).toarray()
Ebe = 2 * d * J * np.cos(np.pi / (L + 1))
sig = rof.lattice_index((4, 4), L)
ub = np.logspace(np.log10(20.0), np.log10(2e4), 46)
out['ub'] = ub.tolist()
for name, c in {"corner": (1, 1), "edge": (1, 4), "interior": (2, 2)}.items():
    p = rof.lattice_index(c, L)
    Heff = H0.astype(complex).copy(); Heff[p, p] -= 0.5j * kappa
    t0 = time.time()
    F, _ = rof.windowed_qfi_curve(Heff, p, sig, kappa, Ebe, ub / kappa)
    out[f'lat_{name}'] = dict(F=F.tolist(), wp=rof.lattice_w(c, L), ws=rof.lattice_w((4, 4), L))
    print(name, 'done', time.time() - t0, flush=True)
json.dump(out, open(os.path.join(HERE, 'out', 'fig_finite_window_data.json'), 'w'))
