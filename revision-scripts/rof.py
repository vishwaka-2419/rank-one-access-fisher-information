"""
rof.py -- independent reference implementation for the AW12832 revision checks.

Conventions (as in the manuscript, hbar = 1, rates in angular-frequency units):
    H_eff = H0 + eps |s><s| - (i/2) (kappa |p><p| + Gamma_int)
    G(w)  = (w - H_eff)^(-1)
    S     = 1 - i kappa G_pp
    F_s   = 4 kappa^2 |G_ps G_sp|^2          (QFI per incident photon, monitored port)
Site indices in this module are 0-based.
"""
import numpy as np
from scipy.linalg import solve_banded, expm
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ----------------------------------------------------------------------------
# uniform chain
# ----------------------------------------------------------------------------
def chain_modes(N, J=1.0):
    j = np.arange(1, N + 1)
    m = np.arange(1, N + 1)
    psi = np.sqrt(2.0 / (N + 1)) * np.sin(np.outer(j, m) * np.pi / (N + 1))  # psi[j-1, m-1]
    E = 2.0 * J * np.cos(m * np.pi / (N + 1))
    return E, psi


def chain_w(N, s):
    """weight of the m=1 mode at 1-based site s"""
    return 2.0 / (N + 1) * np.sin(np.pi * s / (N + 1)) ** 2


def chain_E1(N, J=1.0):
    return 2.0 * J * np.cos(np.pi / (N + 1))


def chain_gap(N, J=1.0):
    u = np.pi / (N + 1)
    return 2.0 * J * (np.cos(u) - np.cos(2 * u))


def chain_G_col1_banded(N, w, kappa, gamma, J=1.0):
    """first column of G for the uniform chain with port on site 1 (1-based), banded solve."""
    ab = np.zeros((3, N), dtype=complex)
    ab[0, 1:] = -J
    ab[2, :-1] = -J
    ab[1, :] = w + 0.5j * gamma
    ab[1, 0] += 0.5j * kappa
    b = np.zeros(N, dtype=complex)
    b[0] = 1.0
    return solve_banded((1, 1), ab, b)


def chain_Gs1_closed(N, s, w, kappa, gamma, J=1.0):
    """closed form G_{s1}; s is 1-based.  2J cos q = w + i gamma/2."""
    z = (w + 0.5j * gamma) / (2.0 * J)
    q = np.arccos(z)
    # choose the branch with Im q <= 0 (decaying away from the source); result is branch independent
    num = np.sin((N + 1 - s) * q)
    den = J * np.sin((N + 1) * q) + 0.5j * kappa * np.sin(N * q)
    return num / den


def chain_F(N, s, kappa, gamma, J=1.0, w=None, method="banded"):
    if w is None:
        w = chain_E1(N, J)
    if method == "banded":
        g = chain_G_col1_banded(N, w, kappa, gamma, J)[s - 1]
    else:
        g = chain_Gs1_closed(N, s, w, kappa, gamma, J)
    return 4.0 * kappa ** 2 * abs(g) ** 4


def mode_F(w1, ws, kappa, gamma):
    """isolated-mode expression, Eq. (16)"""
    return 64.0 * kappa ** 2 * w1 ** 2 * ws ** 2 / (gamma + kappa * w1) ** 4


# ----------------------------------------------------------------------------
# universal finite-time functions
# ----------------------------------------------------------------------------
def k_fun(x):
    """finite-window function k(x); robust for small x by direct quadrature of the positive integrand."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x)
    xg, wg = np.polynomial.legendre.leggauss(60)
    for i, xi in enumerate(x):
        X = 0.5 * xi
        if X < 1e-3:
            # two-term series: k = x^4/320 - x^5/576 + O(x^6)
            out[i] = xi ** 4 / 320.0 - xi ** 5 / 576.0
            continue
        # split [0, X] in panels of width <= 2
        npan = max(1, int(np.ceil(X / 2.0)))
        edges = np.linspace(0.0, X, npan + 1)
        tot = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            y = 0.5 * (b - a) * xg + 0.5 * (b + a)
            f = (-np.expm1(-y) - y * np.exp(-y)) ** 2
            tot += 0.5 * (b - a) * np.dot(wg, f)
        out[i] = tot / X
    return out if out.size > 1 else float(out[0])


def k_closed(x):
    X = 0.5 * np.asarray(x, dtype=float)
    return (1.0 / X) * (X - 11.0 / 4 + 2 * (2 + X) * np.exp(-X)
                        - np.exp(-2 * X) * ((1 + X) ** 2 / 2 + (1 + X) / 2 + 0.25))


def h_fun(x):
    x = np.asarray(x, dtype=float)
    return 1 - 3 / x + (3 / x + 0.5) * np.exp(-x / 2)


def q_fun(x):
    x = np.asarray(x, dtype=float)
    return (1 - (1 + x / 2) * np.exp(-x / 2)) ** 2 / x


# ----------------------------------------------------------------------------
# exact time-domain windowed QFI for an arbitrary (small, dense) network
# ----------------------------------------------------------------------------
def windowed_qfi(Heff, p, s, kappa, w0, T, envelope=None, dt=None, ngauss=8, return_parts=False):
    """
    Exact (to quadrature accuracy) QFI per incident photon collected at the port in [0, T].

    psi'   = A psi - i sqrt(kappa) e(t) |p>,          A = -i (Heff - w0)
    dpsi'  = A dpsi - i |s><s| psi
    QFI    = 4 kappa int_0^T |<p|dpsi>|^2 dt / int_0^T |e|^2 dt

    For a constant envelope (rectangular pulse, e = 1) the state is advanced with exact
    propagators; the time integral uses composite Gauss-Legendre quadrature on steps short
    compared with the fastest oscillation.  `envelope(t)` (array-valued) is supported through
    the same quadrature nodes (piecewise evaluation, source treated by Gauss quadrature of the
    Duhamel integral within each step).
    Returns QFI per photon (and optionally the stored-state QFI 4||dpsi(T)||^2 per photon).
    """
    Heff = np.asarray(Heff, dtype=complex)
    n = Heff.shape[0]
    A = -1j * (Heff - w0 * np.eye(n))
    M = np.zeros((2 * n, 2 * n), dtype=complex)
    M[:n, :n] = A
    M[n:, n:] = A
    M[n + s, s] = -1j
    B = np.zeros(2 * n, dtype=complex)
    B[p] = -1j * np.sqrt(kappa)
    if dt is None:
        # fastest rate in the rotating frame
        rate = np.max(np.abs(np.linalg.eigvals(A)))
        dt = min(0.4 / max(rate, 1e-12), T)
    nstep = max(1, int(np.ceil(T / dt)))
    dt = T / nstep
    xg, wg = np.polynomial.legendre.leggauss(ngauss)
    tau = 0.5 * dt * (xg + 1.0)
    wq = 0.5 * dt * wg
    if envelope is None:
        # augmented propagators for constant drive
        Maug = np.zeros((2 * n + 1, 2 * n + 1), dtype=complex)
        Maug[:2 * n, :2 * n] = M
        Maug[:2 * n, 2 * n] = B
        Es = [expm(Maug * t) for t in tau]
        Efull = expm(Maug * dt)
        X = np.zeros(2 * n + 1, dtype=complex)
        X[2 * n] = 1.0
        acc = 0.0
        row = n + p
        rows = np.array([E[row, :] for E in Es])  # only need the <p|dpsi> component at the nodes
        for _ in range(nstep):
            vals = rows @ X
            acc += np.dot(wq, np.abs(vals) ** 2)
            X = Efull @ X
        qfi = 4.0 * kappa * acc / T
        stored = 4.0 * np.sum(np.abs(X[n:2 * n]) ** 2) / T
    else:
        # general envelope: propagate homogeneous part exactly, source by Gauss quadrature
        Efull = expm(M * dt)
        Enode = [expm(M * t) for t in tau]            # 0 -> node
        Erest = [expm(M * (dt - t)) for t in tau]     # node -> end of step
        # inner quadrature for state at node i: X(t_n + tau_i) = E(tau_i) X_n + int_0^tau_i E(tau_i - u) B e(t_n+u) du
        xg2, wg2 = np.polynomial.legendre.leggauss(ngauss)
        inner = []
        for ti in tau:
            u = 0.5 * ti * (xg2 + 1.0)
            wu = 0.5 * ti * wg2
            inner.append((u, wu, [expm(M * (ti - uu)) @ B for uu in u]))
        X = np.zeros(2 * n, dtype=complex)
        acc = 0.0
        energy = 0.0
        row = n + p
        t0 = 0.0
        for _ in range(nstep):
            et = envelope(t0 + tau)
            energy += np.dot(wq, np.abs(et) ** 2)
            for i, ti in enumerate(tau):
                u, wu, EB = inner[i]
                eu = envelope(t0 + u)
                Xi = Enode[i] @ X + sum(wu[j] * eu[j] * EB[j] for j in range(len(u)))
                acc += wq[i] * abs(Xi[row]) ** 2
            # advance to the end of the step
            src = sum(wq[i] * et[i] * (Erest[i] @ B) for i in range(len(tau)))
            X = Efull @ X + src
            t0 += dt
        qfi = 4.0 * kappa * acc / energy
        stored = 4.0 * np.sum(np.abs(X[n:]) ** 2) / energy
    if return_parts:
        return qfi, stored
    return qfi


def chain_Heff(N, kappa, gamma=0.0, J=1.0):
    H = np.zeros((N, N), dtype=complex)
    for j in range(N - 1):
        H[j, j + 1] = H[j + 1, j] = J
    H -= 0.5j * gamma * np.eye(N)
    H[0, 0] -= 0.5j * kappa
    return H


# ----------------------------------------------------------------------------
# hypercubic lattices
# ----------------------------------------------------------------------------
def lattice_H0(L, d, J=1.0):
    """sparse real-symmetric nearest-neighbour hopping on an open L^d lattice"""
    T1 = sp.diags([np.ones(L - 1), np.ones(L - 1)], [1, -1], format="csr")
    I = sp.identity(L, format="csr")
    H = sp.csr_matrix((L ** d, L ** d))
    for ax in range(d):
        ops = [I] * d
        ops[ax] = T1
        K = ops[0]
        for o in ops[1:]:
            K = sp.kron(K, o, format="csr")
        H = H + J * K
    return H.tocsr()


def lattice_index(coords, L):
    """coords: 1-based tuple -> flat index (C order)"""
    idx = 0
    for c in coords:
        idx = idx * L + (c - 1)
    return idx


def lattice_w(coords, L):
    w = 1.0
    for c in coords:
        w *= 2.0 / (L + 1) * np.sin(np.pi * c / (L + 1)) ** 2
    return w


def windowed_qfi_curve(Heff, p, s, kappa, w0, Ts, dt=None, ngauss=8):
    """
    Rectangular resonant drive switched on at t = 0.  Returns the QFI per incident photon
    collected at the port in [0, T] for every T in the increasing array Ts, and the QFI stored
    in the network at T, using one trajectory (exact propagators, Gauss-Legendre in time).
    """
    Heff = np.asarray(Heff, dtype=complex)
    n = Heff.shape[0]
    Ts = np.asarray(Ts, dtype=float)
    A = -1j * (Heff - w0 * np.eye(n))
    M = np.zeros((2 * n + 1, 2 * n + 1), dtype=complex)
    M[:n, :n] = A
    M[n:2 * n, n:2 * n] = A
    M[n + s, s] = -1j
    M[p, 2 * n] = -1j * np.sqrt(kappa)
    if dt is None:
        rate = np.max(np.abs(np.linalg.eigvals(A)))
        dt = 0.4 / max(rate, 1e-12)
    xg, wg = np.polynomial.legendre.leggauss(ngauss)
    X = np.zeros(2 * n + 1, dtype=complex)
    X[2 * n] = 1.0
    row = n + p
    t = 0.0
    acc = 0.0
    out = np.zeros(len(Ts))
    stored = np.zeros(len(Ts))
    Efull = expm(M * dt)
    rows_full = np.array([expm(M * (0.5 * dt * (x + 1.0)))[row, :] for x in xg])
    wq_full = 0.5 * dt * wg
    for i, T in enumerate(Ts):
        nfull = int(np.floor((T - t) / dt + 1e-12))
        for _ in range(nfull):
            acc += np.dot(wq_full, np.abs(rows_full @ X) ** 2)
            X = Efull @ X
        t += nfull * dt
        rem = T - t
        if rem > 1e-14:
            rows = np.array([expm(M * (0.5 * rem * (x + 1.0)))[row, :] for x in xg])
            acc += np.dot(0.5 * rem * wg, np.abs(rows @ X) ** 2)
            X = expm(M * rem) @ X
            t = T
        out[i] = 4.0 * kappa * acc / T
        stored[i] = 4.0 * np.sum(np.abs(X[n:2 * n]) ** 2) / T
    return out, stored
