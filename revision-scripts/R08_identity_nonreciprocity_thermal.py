"""
(a) access identity on random passive networks, with the reciprocity classification asked for by
    Referee 1 (reciprocity of H0 versus reciprocity of H_eff = H0 - i Gamma/2);
(b) exact eigenfrequency identity  F(E_m) = 64 (w_s/w_p)^2 / kappa^2  for lossless one-port networks,
    including Hermitian nonreciprocal (flux) networks;
(c) Hatano-Nelson-type dissipative nonreciprocity in a one-port chain: the round-trip product
    G_1s G_s1 is invariant under the imaginary gauge transformation;
(d) thermal inputs: output occupation and displacement QFI from the full unitary scattering matrix.
"""
import sys, json
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import rof

rng = np.random.default_rng(20261126)

def analyse(H0, Gint, kappa, p, s, w):
    n = H0.shape[0]
    Gam = Gint.copy(); Gam[p, p] += kappa
    G = np.linalg.inv(w * np.eye(n) - (H0 - 0.5j * Gam))
    rho = -G[s, s].imag / np.pi
    eta_out = kappa * abs(G[p, s]) ** 2 / (2 * np.pi * rho)
    eta_in = kappa * abs(G[s, p]) ** 2 / (2 * np.pi * rho)
    Fport = 4 * kappa ** 2 * abs(G[p, s] * G[s, p]) ** 2
    # all-channel QFI from an explicit channel decomposition of Gint
    lam, V = np.linalg.eigh(Gint)
    Fall = 4 * kappa * abs(G[s, p]) ** 2 * (kappa * abs(G[p, s]) ** 2
                                           + sum(l * abs(V[:, k].conj() @ G[:, s]) ** 2 for k, l in enumerate(lam) if l > 1e-14))
    if Fport == 0.0 or Fall == 0.0:      # port and signal site not connected: nothing to test
        return np.nan, np.nan, np.nan, eta_in, eta_out
    e1 = abs(Fport - 16 * np.pi ** 2 * eta_in * eta_out * rho ** 2) / Fport
    e2 = abs(Fall - 16 * np.pi ** 2 * eta_in * rho ** 2) / Fall
    return e1, e2, abs(eta_in - eta_out), eta_in, eta_out

def rand_graph(n, pconn, complex_phase):
    A = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < pconn:
                amp = rng.uniform(0.3, 1.5)
                ph = np.exp(1j * rng.uniform(0, 2 * np.pi)) if complex_phase else 1.0
                A[i, j] = amp * ph; A[j, i] = np.conj(A[i, j])
    A += np.diag(rng.normal(0, 0.3, n))
    return A

def rand_loss(n, kind):
    if kind == 'diag':
        return np.diag(rng.uniform(0, 0.2, n)).astype(complex)
    r = rng.integers(1, n + 1)
    if kind == 'real_sym':
        B = rng.normal(0, 0.25, (n, r))
    else:  # complex Hermitian, breaks complex symmetry of H_eff
        B = rng.normal(0, 0.25, (n, r)) + 1j * rng.normal(0, 0.25, (n, r))
    return (B @ B.conj().T).astype(complex)

print("== (a) access identity; reciprocity classification ==")
print(" ensemble                                               | H0 recip. | H_eff recip. | max err port | max err all | max|eta_in-eta_out|")
ens = [("real-symmetric H0, diagonal loss", False, 'diag', True, True),
       ("flux (complex Hermitian) H0, diagonal loss", True, 'diag', False, False),
       ("real-symmetric H0, real-symmetric non-diagonal loss", False, 'real_sym', True, True),
       ("real-symmetric H0, complex-Hermitian non-diagonal loss", False, 'complex', True, False),
       ("flux H0, complex-Hermitian non-diagonal loss", True, 'complex', False, False)]
tab = []
for name, cph, lk, r0, reff in ens:
    m1 = m2 = m3 = 0.0
    nvalid = 0
    for _ in range(180):
        n = int(rng.integers(4, 13))
        H0 = rand_graph(n, 0.5, cph)
        Gint = rand_loss(n, lk)
        p, s = rng.choice(n, 2, replace=False)
        kappa = rng.uniform(0.02, 0.5)
        w = rng.uniform(-2, 2)
        e1, e2, d, _, _ = analyse(H0, Gint, kappa, p, s, w)
        if not np.isfinite(e1):      # port and signal site in different connected components: F = 0
            continue
        m1 = max(m1, e1); m2 = max(m2, e2); m3 = max(m3, d); nvalid += 1
    tab.append((name, r0, reff, m1, m2, m3, nvalid))
    print(f" {name:54s} | {str(r0):9s} | {str(reff):12s} | {m1:.1e}      | {m2:.1e}     | {m3:.2e}   ({nvalid} connected instances of 180)")

print("\n== (b) lossless one-port network at a closed-network eigenfrequency: F = 64 (w_s/w_p)^2/kappa^2 exactly ==")
for cph in [False, True]:
    worst = 0.0
    nused = 0
    for _ in range(200):
        n = int(rng.integers(4, 13))
        H0 = rand_graph(n, 0.6, cph)
        E, V = np.linalg.eigh(H0)
        m = rng.integers(n)
        p, s = rng.choice(n, 2, replace=False)
        wp, ws = abs(V[p, m]) ** 2, abs(V[s, m]) ** 2
        if wp < 1e-6 or ws < 1e-6:
            continue
        nused += 1
        kappa = rng.uniform(0.02, 2.0)   # NOT restricted to weak coupling
        Gint = np.zeros((n, n), dtype=complex)
        Gam = Gint.copy(); Gam[p, p] += kappa
        G = np.linalg.inv(E[m] * np.eye(n) - (H0 - 0.5j * Gam))
        F = 4 * kappa ** 2 * abs(G[p, s] * G[s, p]) ** 2
        worst = max(worst, abs(F - 64 * (ws / wp) ** 2 / kappa ** 2) / F)
    print(f"  {'flux (nonreciprocal)' if cph else 'real symmetric      '} H0: max relative deviation over {nused} random graphs (of 200 drawn), 0.02 <= kappa/J <= 2: {worst:.1e}")

print("\n== (c) Hatano-Nelson chain with passive (dissipative) asymmetric hopping ==")
N, J, kappa, s = 12, 1.0, 0.1, 7
for delta in [0.0, 0.05, 0.2]:
    for gloc_extra in [0.0]:
        gloc = 4 * abs(delta) + gloc_extra          # minimal local loss that keeps Gamma >= 0 for a long chain
        Heff = np.zeros((N, N), dtype=complex)
        for j in range(N - 1):
            Heff[j + 1, j] = J + delta   # hop j -> j+1
            Heff[j, j + 1] = J - delta
        Heff -= 0.5j * gloc * np.eye(N)
        Heff[0, 0] -= 0.5j * kappa
        Gam = 1j * (Heff - Heff.conj().T)
        mineig = np.linalg.eigvalsh(Gam).min()
        # reciprocal partner: hopping sqrt(J^2 - delta^2), same on-site loss
        Jr = np.sqrt(J ** 2 - delta ** 2)
        Hrec = rof.chain_Heff(N, kappa, gloc, Jr)
        w = 2 * Jr * np.cos(np.pi / (N + 1))
        G = np.linalg.inv(w * np.eye(N) - Heff); Gr = np.linalg.inv(w * np.eye(N) - Hrec)
        prod = G[0, s - 1] * G[s - 1, 0]; prodr = Gr[0, s - 1] * Gr[s - 1, 0]
        F = 4 * kappa ** 2 * abs(prod) ** 2
        Fll = 4 * kappa ** 2 * abs(np.linalg.inv(2 * J * np.cos(np.pi / (N + 1)) * np.eye(N) - rof.chain_Heff(N, kappa, 0.0, J))[s - 1, 0]) ** 4
        rho = -G[s - 1, s - 1].imag / np.pi
        ein = kappa * abs(G[s - 1, 0]) ** 2 / (2 * np.pi * rho); eout = kappa * abs(G[0, s - 1]) ** 2 / (2 * np.pi * rho)
        print(f"  delta={delta}: local loss={gloc:.2f}, min eig(Gamma)={mineig:+.3e}; |G_1s G_s1 - reciprocal partner|/|.| = {abs(prod-prodr)/abs(prodr):.1e};"
              f" eta_in={ein:.3e}, eta_out={eout:.3e}; F/F(lossless reciprocal)={F/Fll:.3e}")

print("\n== (d) thermal inputs: full scattering matrix, output occupation at the port ==")
N, kappa, gam, s = 9, 0.1, 2e-3, 5
H0 = np.real(rof.chain_Heff(N, 0.0, 0.0))
w = rof.chain_E1(N)
Gam = gam * np.eye(N, dtype=complex); Gam[0, 0] += kappa
G = np.linalg.inv(w * np.eye(N) - (H0 - 0.5j * Gam))
# channels: port (sqrt(kappa)|1>) and N local baths (sqrt(gam)|j>)
Vch = np.zeros((N, N + 1), dtype=complex)
Vch[0, 0] = np.sqrt(kappa)
for j in range(N):
    Vch[j, j + 1] = np.sqrt(gam)
Smat = np.eye(N + 1) - 1j * Vch.conj().T @ G @ Vch
print("  unitarity ||S S^dag - 1|| =", np.linalg.norm(Smat @ Smat.conj().T - np.eye(N + 1)))
S11 = Smat[0, 0]
w1 = rof.chain_w(N, 1); Gp = kappa * w1; Gt = gam + Gp
print(f"  |S_pp|^2 = {abs(S11)**2:.6f}; isolated-mode ((gamma-Gp)/(gamma+Gp))^2 = {((gam-Gp)/Gt)**2:.6f}; absorptance A = {1-abs(S11)**2:.6f} vs 4 gamma Gp/Gamma^2 = {4*gam*Gp/Gt**2:.6f}")
for (npt, nint) in [(0.5, 0.5), (0.0, 2.0), (2.0, 0.0)]:
    nvec = np.array([npt] + [nint] * N)
    nout = np.sum(np.abs(Smat[0, :]) ** 2 * nvec)
    pred = nint + (npt - nint) * abs(S11) ** 2
    print(f"  n_port={npt}, n_int={nint}:  n_out={nout:.6f} (formula {pred:.6f});  displacement-QFI factor 1/(2 n_out+1) = {1/(2*nout+1):.4f}")
import math
hf_over_k = 6.62607015e-34 * 6e9 / 1.380649e-23
for Tk in [0.02, 0.05, 0.1, 1.0, 4.0]:
    nb = 1 / (math.exp(hf_over_k / Tk) - 1)
    print(f"  6 GHz, T={Tk} K: n_b = {nb:.3e},  1/(2 n_b+1) = {1/(2*nb+1):.4f}")
json.dump(dict(table=[(t[0], t[1], t[2], t[3], t[4], t[5], t[6]) for t in tab]), open(os.path.join(HERE, 'out', 'R08_identity.json'), 'w'), indent=1)
