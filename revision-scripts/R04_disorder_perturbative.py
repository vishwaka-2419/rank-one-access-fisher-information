"""
Referee 2, point 2: robustness of the geometric scaling laws to static disorder.
First-order perturbation theory for the lossless steady-state QFI at the (retuned) band-edge
eigenfrequency, where F = 64 (w_s / w_p)^2 / kappa^2 holds exactly for any Hermitian H0.

    delta ln F = 4 [ dpsi(s)/psi(s) - dpsi(p)/psi(p) ],
    dpsi(x)    = sum_{m != 1} psi_m(x) <m|V|1> / (E_1 - E_m)  = sum_j G'(x, j) (V psi_1)(j)

onsite disorder  V = sum_j dw_j |j><j|                       ->  Var = sigma^2 sum_j psi_1(j)^2 A_j^2
bond disorder    V = sum_<ij> dJ_ij (|i><j| + |j><i|)        ->  Var = sigma^2 sum_<ij> (A_i psi_1(j) + A_j psi_1(i))^2
with A_j = 4 [ G'(s,j)/psi_1(s) - G'(p,j)/psi_1(p) ].

Output: C(N) = Var(ln F) / (sigma/J)^2 for chains, squares and cubes, and fitted exponents.
"""
import sys, json, itertools
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.fft import dstn

J = 1.0

def modes_1d(L):
    j = np.arange(1, L + 1)
    S = np.sqrt(2.0 / (L + 1)) * np.sin(np.outer(j, j) * np.pi / (L + 1))  # S[x-1, m-1], orthogonal symmetric
    E = 2 * J * np.cos(j * np.pi / (L + 1))
    return S, E

def Gprime_row(L, d, x):
    """G'(x, .) on the L^d lattice as a d-dimensional array, excluding the band-edge mode."""
    S, E1 = modes_1d(L)
    # c_m = psi_m(x) / (E_1 - E_m)
    Esum = np.zeros((L,) * d)
    amp = np.ones((L,) * d)
    for ax in range(d):
        shape = [1] * d; shape[ax] = L
        Esum = Esum + E1.reshape(shape)
        amp = amp * S[x[ax] - 1, :].reshape(shape)
    Ebe = Esum.flat[0]
    den = Ebe - Esum
    den.flat[0] = np.inf
    c = amp / den
    # back-transform: G'(x, j) = sum_m c_m psi_m(j)  -> apply S along each axis
    out = c
    for ax in range(d):
        out = np.tensordot(S, out, axes=([1], [ax]))
        out = np.moveaxis(out, 0, ax)
    return out

def psi1(L, d):
    S, _ = modes_1d(L)
    v = S[:, 0]
    out = v
    for _ in range(d - 1):
        out = np.multiply.outer(out, v)
    return out

def coeffs(L, d, port, sig):
    p1 = psi1(L, d)
    Gs = Gprime_row(L, d, sig)
    Gp = Gprime_row(L, d, port)
    ps = p1[tuple(c - 1 for c in sig)]
    pp = p1[tuple(c - 1 for c in port)]
    A = 4.0 * (Gs / ps - Gp / pp)
    Aport = 4.0 * (-Gp / pp)        # port contribution alone
    Asig = 4.0 * (Gs / ps)
    C_on = np.sum(p1 ** 2 * A ** 2)
    C_on_port = np.sum(p1 ** 2 * Aport ** 2)
    C_on_sig = np.sum(p1 ** 2 * Asig ** 2)
    C_b = 0.0
    for ax in range(d):
        sl_a = [slice(None)] * d; sl_b = [slice(None)] * d
        sl_a[ax] = slice(0, L - 1); sl_b[ax] = slice(1, L)
        sa, sb = tuple(sl_a), tuple(sl_b)
        C_b += np.sum((A[sa] * p1[sb] + A[sb] * p1[sa]) ** 2)
    return C_on, C_b, C_on_port, C_on_sig

def port_site(L, d, k):
    c = (L + 1) // 2
    return tuple([1] * k + [c] * (d - k))

if __name__ == "__main__":
    res = {}
    configs = {(1, 1): [9, 17, 33, 65, 129, 257, 513, 1025],
               (2, 1): [9, 13, 17, 25, 33, 49, 65, 97, 129],
               (2, 2): [9, 13, 17, 25, 33, 49, 65, 97, 129],
               (3, 1): [5, 7, 9, 13, 17, 21, 25],
               (3, 2): [5, 7, 9, 13, 17, 21, 25],
               (3, 3): [5, 7, 9, 13, 17, 21, 25]}
    for (d, k), Ls in configs.items():
        rows = []
        for L in Ls:
            c = (L + 1) // 2
            sig = tuple([c] * d)
            port = port_site(L, d, k)
            C_on, C_b, C_on_p, C_on_s = coeffs(L, d, port, sig)
            rows.append((L, L ** d, C_on, C_b, C_on_p, C_on_s))
        rows = np.array(rows)
        half = rows[len(rows) // 2:]
        e_on = np.polyfit(np.log(half[:, 1]), np.log(half[:, 2]), 1)[0]
        e_b = np.polyfit(np.log(half[:, 1]), np.log(half[:, 3]), 1)[0]
        res[f"{d}_{k}"] = dict(rows=rows.tolist(), exp_on=e_on, exp_bond=e_b)
        print(f"\n d={d} k={k}:  Var(ln F) = (sigma/J)^2 * C(N);   fitted C ~ N^p: onsite p={e_on:.3f}, bond p={e_b:.3f}"
              f"   -> rms(ln F) ~ (sigma/J) N^{e_on/2:.3f} (onsite), N^{e_b/2:.3f} (bond)")
        print("    L       N        C_onsite      C_bond     [port part]   [signal part]   sigma_c/J (rms ln F = 0.5): onsite, bond")
        for r in rows:
            print(f"  {int(r[0]):4d} {int(r[1]):8d}  {r[2]:12.4e} {r[3]:12.4e} {r[4]:12.4e} {r[5]:12.4e}     {0.5/np.sqrt(r[2]):.3e}  {0.5/np.sqrt(r[3]):.3e}")
    json.dump(res, open(os.path.join(HERE, 'out', 'R04_disorder_pt.json'), 'w'), indent=1)
