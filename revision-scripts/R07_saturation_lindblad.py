"""
Referee 2, point 7: saturation beyond weak drive in the two-level-system chain.

Lossless driven XX chain of N two-level systems, port (drive + decay kappa) on site 1,
H = J sum (s+_j s-_{j+1} + h.c.) - w sum n_j + eps n_s + i sqrt(kappa) (beta s+_1 - beta* s-_1)
Long-time QFI rate of the emitted field (all radiation goes to the monitored port):
    Idot = 8 Re int_0^inf Tr[ dn_s e^{Lt} (dn_s rho_ss) ] dt
Claim tested:  Idot/Phi = F_linear * S(u),  S(u) = (4+u) / (4 (1+u)^3),  u = 8 Phi / Gamma_p,
i.e. the isolated band-edge mode behaves as ONE effective two-level system of linewidth
Gamma_p = kappa w_1 and signal coupling w_s eps.
"""
import sys, json
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import rof

sm1 = sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=complex))   # sigma^- : |1> -> |0>, basis (|0>,|1>) = (ground, excited)
I2 = sp.identity(2, format='csr', dtype=complex)

def op(N, j, a):
    out = None
    for i in range(N):
        x = a if i == j else I2
        out = x if out is None else sp.kron(out, x, format='csr')
    return out

def liouvillian(H, cs):
    D = H.shape[0]
    I = sp.identity(D, format='csr', dtype=complex)
    L = -1j * (sp.kron(I, H) - sp.kron(H.T, I))
    for c in cs:
        cdc = (c.getH() @ c).tocsr()
        L = L + sp.kron(c.conj(), c) - 0.5 * sp.kron(I, cdc) - 0.5 * sp.kron(cdc.T, I)
    return L.tocsc()

def qfi_rate(N, s, kappa, beta, J=1.0, gamma=0.0, w=None):
    """returns (Idot, rho_ss, n_int)."""
    sm = [op(N, j, sm1) for j in range(N)]
    n = [(a.getH() @ a).tocsr() for a in sm]
    if w is None:
        w = rof.chain_E1(N, J) if N > 1 else 0.0
    H = sp.csr_matrix((2 ** N, 2 ** N), dtype=complex)
    for j in range(N - 1):
        H = H + J * (sm[j].getH() @ sm[j + 1] + sm[j + 1].getH() @ sm[j])
    for j in range(N):
        H = H - w * n[j]
    H = H + 1j * np.sqrt(kappa) * (beta * sm[0].getH() - np.conj(beta) * sm[0])
    cs = [np.sqrt(kappa) * sm[0]] + ([np.sqrt(gamma) * a for a in sm] if gamma > 0 else [])
    L = liouvillian(H, cs)
    D = 2 ** N
    # trace row
    tr = sp.csr_matrix((np.ones(D), (np.zeros(D, dtype=int), np.arange(D) * (D + 1))), shape=(1, D * D))
    Lc = sp.vstack([tr, L.tocsr()[1:, :]]).tocsc()
    lu = spla.splu(Lc)
    b = np.zeros(D * D, dtype=complex); b[0] = 1.0
    rho = lu.solve(b).reshape(D, D, order='F')
    rho = 0.5 * (rho + rho.conj().T)
    ns = n[s - 1].toarray()
    dn = ns - np.trace(ns @ rho).real * np.eye(D)
    rhs = -0.5 * (dn @ rho + rho @ dn)
    bb = rhs.reshape(-1, order='F').copy(); bb[0] = 0.0
    X = lu.solve(bb).reshape(D, D, order='F')
    Idot = 8.0 * np.trace(dn @ X).real
    nint = sum(np.trace(a.toarray() @ rho).real for a in n)
    return Idot, rho, nint

if __name__ == "__main__":
    S = lambda u: (4 + u) / (4 * (1 + u) ** 3)

    print("== single resonant TLS: Idot/Phi vs 16 (4+u) / (kappa^2 (1+u)^3), u = 8 Phi/kappa ==")
    kappa = 0.2
    for u in [1e-3, 0.1, 0.6056, 2.0, 10.0]:
        Phi = u * kappa / 8
        Id, _, nint = qfi_rate(1, 1, kappa, np.sqrt(Phi))
        form = 16 * (4 + u) / (kappa ** 2 * (1 + u) ** 3)
        print(f" u={u:7.4f}: numeric {Id/Phi:12.6f}  formula {form:12.6f}  rel.diff {abs(Id/Phi-form)/form:.2e}   <n>={nint:.4f}  (u/(2(1+u)) = {u/(2*(1+u)):.4f})")

    res = {}
    print("\n== chains: ratio (Idot/Phi)/F_linear versus S(u), u = 8 Phi / (kappa w_1) ==")
    for kappa in [0.2, 0.05]:
        for N in [2, 3, 4, 5, 6]:
            for s in sorted(set([(N + 1) // 2, N])):
                w1 = rof.chain_w(N, 1)
                Gp = kappa * w1
                Flin = rof.chain_F(N, s, kappa, 0.0)
                gap = rof.chain_gap(N)
                row = []
                for u in [0.01, 0.1, 0.3, 0.6056, 1.0, 3.0, 10.0]:
                    Phi = u * Gp / 8
                    Id, _, nint = qfi_rate(N, s, kappa, np.sqrt(Phi))
                    row.append((u, Id / Phi / Flin, S(u), nint))
                res[f"{kappa}_{N}_{s}"] = dict(Gp=Gp, gap=gap, Flin=Flin, rows=row)
                print(f" kappa/J={kappa} N={N} s={s}  Gamma_p/DeltaE={Gp/gap:.4f}  F_lin={Flin:.4e}")
                print("    u      :", " ".join(f"{r[0]:8.4f}" for r in row))
                print("    numeric:", " ".join(f"{r[1]:8.5f}" for r in row))
                print("    S(u)   :", " ".join(f"{r[2]:8.5f}" for r in row))
                print("    n_int  :", " ".join(f"{r[3]:8.5f}" for r in row))

    print("\n== Quoted values: N=4, s=2, kappa/J=0.2, beta/sqrt(J) = 0.001, 0.02, 0.05, 0.10 ==")
    N, s, kappa = 4, 2, 0.2
    Flin = rof.chain_F(N, s, kappa, 0.0); Gp = kappa * rof.chain_w(N, 1)
    for beta in [0.001, 0.02, 0.05, 0.10]:
        Id, _, nint = qfi_rate(N, s, kappa, beta)
        u = 8 * beta ** 2 / Gp
        print(f" beta={beta}: u={u:.4f}  ratio={Id/beta**2/Flin:.4f}   S(u)={S(u):.4f}")

    # maximum of the QFI rate over drive
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda u: -u * S(u), bounds=(0.01, 5), method='bounded')
    print(f"\nmax_u u S(u) = {-r.fun:.5f} at u = {r.x:.4f}  ->  Idot_max = (Gamma_p/8) F_0 max(uS) = {8*-r.fun:.4f} w_s^2/Gamma_p "
          f"= {8*-r.fun/16:.4f} x [16 w_s^2/(kappa w_p)]")
    json.dump(res, open(os.path.join(HERE, 'out', 'R07_saturation.json'), 'w'), indent=1)
