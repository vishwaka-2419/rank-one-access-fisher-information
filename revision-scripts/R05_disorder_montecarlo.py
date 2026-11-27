"""
Monte Carlo check of the disorder perturbation theory (script 04) and of its breakdown.
Lossless network, probe retuned to the disordered band-edge eigenfrequency, where
F = 64 (w_s/w_p)^2 / kappa^2 exactly; we therefore need only the eigenvector.
Mode selection: among the 4 highest eigenstates, the one with the largest overlap with the
clean band-edge mode (the calibration protocol of Appendix L).
"""
import sys, json, time
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh_tridiagonal
import rof

J = 1.0
rng = np.random.default_rng(12832)
pt = json.load(open(os.path.join(HERE, 'out', 'R04_disorder_pt.json')))

def chain_sample(N, sig, kind):
    d = np.zeros(N); e = np.full(N - 1, J)
    if kind == 'onsite':
        d = rng.normal(0, sig, N)
    else:
        e = e + rng.normal(0, sig, N - 1)
    k = min(4, N)
    w, v = eigh_tridiagonal(d, e, select='i', select_range=(N - k, N - 1))
    return w, v

def lattice_sample(L, sig, kind, H0, bonds):
    N = L * L
    if kind == 'onsite':
        H = H0 + sp.diags(rng.normal(0, sig, N))
    else:
        dj = rng.normal(0, sig, len(bonds))
        V = sp.coo_matrix((np.concatenate([dj, dj]),
                           (np.concatenate([bonds[:, 0], bonds[:, 1]]), np.concatenate([bonds[:, 1], bonds[:, 0]]))),
                          shape=(N, N))
        H = H0 + V
    w, v = spla.eigsh(H.tocsc(), k=4, sigma=4.0 * J + 0.3, which='LM')
    return w, v

def bonds_2d(L):
    idx = np.arange(L * L).reshape(L, L)
    b = np.concatenate([np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], 1),
                        np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1)])
    return b

out = {}
R = 400
# ---------------- chain
print("== chain (d=1, port k=1, centre signal) ==")
print("   N   sigma/J  kind    y=(s_eff/J)N^1.5  rms(lnF)_MC  rms_PT   median(F/Fc)  [q25, q75]   overlap_med")
for kind in ['onsite', 'bond']:
    for N in [17, 33, 65, 129, 257]:
        s = (N + 1) // 2
        psi_c = np.sqrt(2 / (N + 1)) * np.sin(np.pi * np.arange(1, N + 1) / (N + 1))
        rc = (psi_c[s - 1] / psi_c[0]) ** 4
        key = [r for r in pt['1_1']['rows'] if int(r[0]) == N][0]
        C = key[2] if kind == 'onsite' else key[3]
        for sig in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
            lnr = []; ov = []
            for _ in range(R):
                w, v = chain_sample(N, sig, kind)
                o = np.abs(psi_c @ v)
                i = int(np.argmax(o))
                psi = v[:, i]
                lnr.append(np.log((psi[s - 1] / psi[0]) ** 4 / rc)); ov.append(o[i])
            lnr = np.array(lnr)
            seff = sig if kind == 'onsite' else 2 * sig
            y = seff * N ** 1.5
            q = np.percentile(np.exp(lnr), [25, 50, 75])
            out[f"1_1_{kind}_{N}_{sig:g}"] = dict(y=y, rms=float(np.std(lnr)), pt=float(sig * np.sqrt(C)), q=q.tolist(),
                                                   ov=float(np.median(ov)), mean_ln=float(np.mean(lnr)))
            print(f" {N:4d}  {sig:7.0e}  {kind:6s}  {y:10.4f}     {np.std(lnr):9.4f}  {sig*np.sqrt(C):9.4f}   {q[1]:8.4f}  [{q[0]:.3f}, {q[2]:.3f}]   {np.median(ov):.4f}")

# ---------------- square lattice
print("\n== square lattice (d=2, centre signal) ==")
print("   L    N   k  sigma/J  kind   y=(s_eff/J)N^0.5  rms(lnF)_MC  rms_PT   median(F/Fc)  [q25, q75]   overlap_med")
R2 = 300
for L in [9, 17, 33, 65]:
    N = L * L
    H0 = rof.lattice_H0(L, 2, J)
    bonds = bonds_2d(L)
    c = (L + 1) // 2
    x = np.arange(1, L + 1)
    v1 = np.sqrt(2 / (L + 1)) * np.sin(np.pi * x / (L + 1))
    psi_c = np.multiply.outer(v1, v1).ravel()
    sidx = rof.lattice_index((c, c), L)
    ports = {2: rof.lattice_index((1, 1), L), 1: rof.lattice_index((1, c), L)}
    for kind in ['onsite', 'bond']:
        for sig in [1e-3, 3e-3, 1e-2, 3e-2]:
            lnr = {1: [], 2: []}; ov = []
            for _ in range(R2):
                w, v = lattice_sample(L, sig, kind, H0, bonds)
                o = np.abs(psi_c @ v)
                i = int(np.argmax(o)); psi = v[:, i]; ov.append(o[i])
                for k, pidx in ports.items():
                    rc = (psi_c[sidx] / psi_c[pidx]) ** 4
                    lnr[k].append(np.log((psi[sidx] / psi[pidx]) ** 4 / rc))
            for k in [1, 2]:
                key = [r for r in pt[f'2_{k}']['rows'] if int(r[0]) == L][0]
                C = key[2] if kind == 'onsite' else key[3]
                a = np.array(lnr[k])
                seff = sig if kind == 'onsite' else 2 * np.sqrt(2) * sig
                y = seff * N ** 0.5
                q = np.percentile(np.exp(a), [25, 50, 75])
                out[f"2_{k}_{kind}_{L}_{sig:g}"] = dict(y=y, rms=float(np.std(a)), pt=float(sig * np.sqrt(C)), q=q.tolist(),
                                                         ov=float(np.median(ov)), mean_ln=float(np.mean(a)))
                print(f" {L:4d} {N:5d}  {k}  {sig:7.0e}  {kind:6s} {y:10.4f}     {np.std(a):9.4f}  {sig*np.sqrt(C):9.4f}   {q[1]:8.4f}  [{q[0]:.3f}, {q[2]:.3f}]   {np.median(ov):.4f}")
json.dump(out, open(os.path.join(HERE, 'out', 'R05_disorder_mc.json'), 'w'), indent=1)
