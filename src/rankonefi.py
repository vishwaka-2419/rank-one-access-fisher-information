"""Core routines for rank-one-access Fisher-information calculations.

Conventions
-----------
We consider a passive linear network with Hermitian H0 and a single monitored port
coupled to site 0 at rate kappa. Uniform internal amplitude loss gamma is represented
by local Markovian baths at every site. The effective Hamiltonian is

    H_eff = H0 + eps |s><s| - i kappa/2 |0><0| - i gamma/2 I.

For a monochromatic coherent probe at angular frequency omega, the monitored-port
reflection amplitude is

    S = 1 - i kappa G_00,
    G = (omega I - H_eff)^(-1).

For a local perturbation eps |s><s|,

    dS/deps = -i kappa G_0s G_s0.

With coherent input and ideal homodyne detection, the quantum Fisher information of
the monitored coherent output per incident photon is

    F_port = 4 |dS/deps|^2.

For reciprocal networks, G_0s = G_s0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_banded, eigvals


@dataclass(frozen=True)
class ChainParams:
    N: int
    J: float = 1.0
    kappa: float = 0.1
    gamma: float = 0.0


def uniform_chain_hamiltonian(N: int, J: float = 1.0,
                              eps: float = 0.0,
                              site: Optional[int] = None,
                              disorder: Optional[ArrayLike] = None) -> np.ndarray:
    """Return the N x N tight-binding Hamiltonian for an open uniform chain.

    Parameters use zero-based site indices. ``disorder`` is an optional length-N
    onsite-energy array.
    """
    if N < 1:
        raise ValueError("N must be positive")
    H = np.zeros((N, N), dtype=complex)
    if N > 1:
        idx = np.arange(N - 1)
        H[idx, idx + 1] = J
        H[idx + 1, idx] = J
    if disorder is not None:
        d = np.asarray(disorder, dtype=float)
        if d.shape != (N,):
            raise ValueError("disorder must have length N")
        H += np.diag(d)
    if site is not None:
        H[site, site] += eps
    return H


def effective_hamiltonian(H0: np.ndarray, kappa: float,
                          gamma: float = 0.0, port: int = 0) -> np.ndarray:
    """Add rank-one monitored-port loss and uniform intrinsic amplitude loss."""
    H0 = np.asarray(H0, dtype=complex)
    N = H0.shape[0]
    if H0.shape != (N, N):
        raise ValueError("H0 must be square")
    Heff = H0.copy()
    Heff[port, port] -= 0.5j * kappa
    if gamma:
        Heff -= 0.5j * gamma * np.eye(N)
    return Heff


def green_dense(H0: np.ndarray, omega: float, kappa: float,
                gamma: float = 0.0, port: int = 0) -> np.ndarray:
    """Full resolvent G=(omega I-H_eff)^(-1) for an arbitrary network."""
    Heff = effective_hamiltonian(H0, kappa=kappa, gamma=gamma, port=port)
    N = H0.shape[0]
    return np.linalg.inv(omega * np.eye(N, dtype=complex) - Heff)


def uniform_green_column(N: int, site: int, omega: float, J: float = 1.0,
                         kappa: float = 0.1, gamma: float = 0.0,
                         disorder: Optional[ArrayLike] = None) -> np.ndarray:
    """Return column ``site`` of G for a tridiagonal uniform chain in O(N).

    The monitored port is fixed at the left boundary (site 0). Onsite disorder is
    supported without destroying tridiagonality.
    """
    if not (0 <= site < N):
        raise IndexError("site outside chain")
    ab = np.zeros((3, N), dtype=complex)
    if N > 1:
        ab[0, 1:] = -J
        ab[2, :-1] = -J
    diag = np.full(N, omega + 0.5j * gamma, dtype=complex)
    diag[0] += 0.5j * kappa
    if disorder is not None:
        d = np.asarray(disorder, dtype=float)
        if d.shape != (N,):
            raise ValueError("disorder must have length N")
        diag -= d
    ab[1, :] = diag
    rhs = np.zeros(N, dtype=complex)
    rhs[site] = 1.0
    return solve_banded((1, 1), ab, rhs, overwrite_ab=False, overwrite_b=False,
                        check_finite=False)


def scattering_amplitude_uniform(N: int, omega: float, J: float = 1.0,
                                 kappa: float = 0.1, gamma: float = 0.0,
                                 disorder: Optional[ArrayLike] = None) -> complex:
    """Single-port reflection amplitude S=1-i kappa G_00."""
    col0 = uniform_green_column(N, 0, omega, J, kappa, gamma, disorder)
    return 1.0 - 1j * kappa * col0[0]


def dS_deps_uniform(N: int, site: int, omega: float, J: float = 1.0,
                    kappa: float = 0.1, gamma: float = 0.0,
                    disorder: Optional[ArrayLike] = None) -> complex:
    """Exact derivative dS/deps at eps=0 for a local onsite perturbation.

    Reciprocity of the tridiagonal chain implies G_0s=G_s0, hence only one Green
    column is required.
    """
    col_s = uniform_green_column(N, site, omega, J, kappa, gamma, disorder)
    G0s = col_s[0]
    return -1j * kappa * G0s * G0s


def fi_port_per_photon_uniform(N: int, site: int, omega: float,
                               J: float = 1.0, kappa: float = 0.1,
                               gamma: float = 0.0,
                               disorder: Optional[ArrayLike] = None,
                               eta: float = 1.0) -> float:
    """Monitored-port coherent-output QFI (and optimal-homodyne FI) per photon."""
    dS = dS_deps_uniform(N, site, omega, J, kappa, gamma, disorder)
    return float(4.0 * eta * abs(dS) ** 2)


def fi_total_per_photon_uniform(N: int, site: int, omega: float,
                                J: float = 1.0, kappa: float = 0.1,
                                gamma: float = 0.0,
                                disorder: Optional[ArrayLike] = None) -> Tuple[float, float, float]:
    """Return (F_port, F_all_outputs, accessible_fraction) per incident photon.

    Uniform intrinsic loss is resolved into N independent, in-principle observable
    Markovian output baths. For a coherent input the full outgoing state is a product
    of coherent states, so QFIs add over channels.
    """
    col_s = uniform_green_column(N, site, omega, J, kappa, gamma, disorder)
    G0s = col_s[0]
    dS = -1j * kappa * G0s * G0s
    F_port = 4.0 * abs(dS) ** 2
    if gamma == 0.0:
        return float(F_port), float(F_port), 1.0
    # t_j = -i sqrt(kappa gamma) G_{j0}; derivative = -i sqrt(kappa gamma) G_{js} G_{s0}
    loss_deriv_sq = kappa * gamma * abs(G0s) ** 2 * float(np.vdot(col_s, col_s).real)
    F_all = 4.0 * (abs(dS) ** 2 + loss_deriv_sq)
    frac = F_port / F_all if F_all > 0 else np.nan
    return float(F_port), float(F_all), float(frac)


def clean_mode_energy(N: int, m: int = 1, J: float = 1.0) -> float:
    """Open-chain eigenenergy E_m=2J cos[m pi/(N+1)], m=1,...,N."""
    if not (1 <= m <= N):
        raise ValueError("m must satisfy 1 <= m <= N")
    return float(2.0 * J * np.cos(m * np.pi / (N + 1)))


def clean_mode_weight(N: int, site: int, m: int = 1) -> float:
    """Local weight |psi_m(site)|^2 of a normalized open-chain mode."""
    j = site + 1
    q = m * np.pi / (N + 1)
    return float(2.0 / (N + 1) * np.sin(q * j) ** 2)


def isolated_mode_fi(N: int, site: int, m: int = 1, kappa: float = 0.1,
                     gamma: float = 0.0, eta: float = 1.0) -> float:
    """Single-isolated-mode on-resonance approximation to monitored-port FI.

    F = 64 eta kappa^2 w_1^2 w_s^2 / (gamma + kappa w_1)^4.
    """
    w1 = clean_mode_weight(N, 0, m)
    ws = clean_mode_weight(N, site, m)
    Gamma = gamma + kappa * w1
    return float(64.0 * eta * kappa ** 2 * w1 ** 2 * ws ** 2 / Gamma ** 4)


def isolated_mode_total_fi(N: int, site: int, m: int = 1,
                           kappa: float = 0.1, gamma: float = 0.0) -> Tuple[float, float, float]:
    """Single-mode approximations to monitored and all-output QFI.

    F_port = 64 kappa^2 w1^2 ws^2 / Gamma^4
    F_all  = 64 kappa w1 ws^2 / Gamma^3
    fraction = kappa w1 / Gamma.
    """
    w1 = clean_mode_weight(N, 0, m)
    ws = clean_mode_weight(N, site, m)
    Gamma = gamma + kappa * w1
    Fp = 64.0 * kappa ** 2 * w1 ** 2 * ws ** 2 / Gamma ** 4
    Fa = 64.0 * kappa * w1 * ws ** 2 / Gamma ** 3
    frac = (kappa * w1 / Gamma) if Gamma > 0 else np.nan
    return float(Fp), float(Fa), float(frac)


def resonance_pole_uniform(N: int, J: float = 1.0, kappa: float = 0.1,
                           gamma: float = 0.0, m: int = 1) -> complex:
    """Dense finite-N pole continuously connected to clean mode m.

    Intended for modest N (typically <= 500) to validate the asymptotic theory.
    """
    H0 = uniform_chain_hamiltonian(N, J)
    z = eigvals(effective_hamiltonian(H0, kappa=kappa, gamma=gamma, port=0))
    E0 = clean_mode_energy(N, m, J)
    return complex(z[np.argmin(abs(z - E0))])



def internal_occupation_per_flux_uniform(N: int, omega: float, J: float = 1.0,
                                         kappa: float = 0.1, gamma: float = 0.0,
                                         disorder: Optional[ArrayLike] = None) -> float:
    """Steady-state internal excitation per unit incident coherent photon flux.

    In the linear weak-excitation model the internal amplitude vector is proportional
    to sqrt(kappa) G e_1, so n_int/Phi = kappa ||G e_1||^2. This quantity has units
    of time when rates/energies are angular frequencies.
    """
    col0 = uniform_green_column(N, 0, omega, J, kappa, gamma, disorder)
    return float(kappa * np.vdot(col0, col0).real)

def fit_power_law(x: ArrayLike, y: ArrayLike, tail: Optional[int] = None) -> Tuple[float, float]:
    """Fit y ~= A x^p in log-log space; return (p, A)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if tail is not None and len(x) > tail:
        x, y = x[-tail:], y[-tail:]
    p, logA = np.polyfit(np.log(x), np.log(y), 1)
    return float(p), float(np.exp(logA))


def predicted_bulk_nopt(kappa: float, gamma: float) -> float:
    """Asymptotic center-site optimum N* ~= (pi^2 kappa/gamma)^(1/3)."""
    if gamma <= 0:
        return np.inf
    return float((np.pi ** 2 * kappa / gamma) ** (1.0 / 3.0))


def predicted_alpha_nopt(alpha: float, kappa: float, gamma: float) -> float:
    """Asymptotic N* for a signal site s~N^alpha (0<alpha<=1) near a boundary.

    Uses w_1~2 pi^2 N^-3 and w_s~2 pi^2 N^(2 alpha-3), giving
      kappa w_1 / gamma = (3-alpha)/alpha
    at the monitored-FI optimum.
    """
    if not (0 < alpha <= 1):
        return np.inf
    if gamma <= 0:
        return np.inf
    K = 2.0 * np.pi ** 2 * kappa
    return float((K * alpha / (gamma * (3.0 - alpha))) ** (1.0 / 3.0))


# ===========================================================================
# PART II: general passive networks, exact access identity, port codimension,
#          and finite-time (windowed) Fisher information.
#
# Conventions (same as Part I): H_eff = H0 - i Gamma/2 with Gamma Hermitian
# positive semidefinite; the monitored port is the channel kappa |p><p| contained
# in Gamma.  G = (omega - H_eff)^{-1}.
#
# Sum rule (exact):  G - G^dag = -i G^dag Gamma G = -i G Gamma G^dag, hence
#     -2 Im G_ss = <s| G^dag Gamma G |s> = kappa|G_ps|^2 + (other channels) >= 0.
# Defining the collection factor and input-accessibility factor
#     eta_out = kappa |G_ps|^2 / (-2 Im G_ss),   eta_in = kappa |G_sp|^2 / (-2 Im G_ss),
# the monitored-port and all-channel QFI per incident photon obey exactly
#     F_port = 16 eta_in eta_out (Im G_ss)^2 = 16 pi^2 eta_in eta_out rho_s^2,
#     F_all  = 16 eta_in (Im G_ss)^2         = 16 pi^2 eta_in rho_s^2,
# with rho_s = -(1/pi) Im G_ss the local density of states at the signal site.
# ===========================================================================
from scipy import sparse as _sp
from scipy.sparse.linalg import splu as _splu
from scipy.linalg import expm as _expm, solve_continuous_lyapunov as _lyap


def green_from_loss(H0: np.ndarray, Gamma: np.ndarray, omega: float) -> np.ndarray:
    """Dense resolvent G=(omega I - H0 + i Gamma/2)^{-1}."""
    H0 = np.asarray(H0, dtype=complex)
    N = H0.shape[0]
    return np.linalg.inv(omega * np.eye(N) - H0 + 0.5j * np.asarray(Gamma, dtype=complex))


def access_identity(H0: np.ndarray, Gamma: np.ndarray, kappa: float, omega: float,
                    port: int, site: int) -> dict:
    """Exact one-port access identity for a general passive network.

    ``Gamma`` must contain the monitored channel ``kappa |port><port|``; any other
    positive-semidefinite loss (diagonal or not) may be added.  Returns the
    monitored-port QFI per incident photon computed from the scattering derivative,
    the all-channel QFI, the LDOS, and the collection and input-accessibility factors, together with the
    identity-based predictions for both QFIs.
    """
    G = green_from_loss(H0, Gamma, omega)
    Gps = G[port, site]
    Gsp = G[site, port]
    ImGss = float(G[site, site].imag)
    emit = -2.0 * ImGss                        # total emission weight from site s
    rho_s = -ImGss / np.pi
    dS = -1j * kappa * Gps * Gsp
    F_port = 4.0 * abs(dS) ** 2
    F_all = 4.0 * kappa * abs(Gsp) ** 2 * emit
    eta_out = kappa * abs(Gps) ** 2 / emit
    eta_in = kappa * abs(Gsp) ** 2 / emit
    return dict(F_port=float(F_port), F_all=float(F_all), rho_s=float(rho_s),
                ImGss=ImGss, eta_in=float(eta_in), eta_out=float(eta_out),
                F_port_identity=float(16.0 * eta_in * eta_out * ImGss ** 2),
                F_all_identity=float(16.0 * eta_in * ImGss ** 2),
                ldos_bound=float(16.0 * np.pi ** 2 * rho_s ** 2))


def channel_decomposition(Gamma: np.ndarray, kappa: float, port: int,
                          tol: float = 1e-12):
    """Split Gamma into the port channel and an orthogonal set of loss channels.

    Returns (rates, vectors) with Gamma - kappa|p><p| = sum_k rates[k] |v_k><v_k|.
    """
    N = Gamma.shape[0]
    P = np.zeros((N, N), dtype=complex)
    P[port, port] = kappa
    lam, V = np.linalg.eigh(np.asarray(Gamma, dtype=complex) - P)
    keep = lam > tol * max(1.0, abs(lam).max())
    if (lam < -1e-9 * max(1.0, abs(lam).max())).any():
        raise ValueError("loss matrix minus port channel is not positive semidefinite")
    return lam[keep], V[:, keep]


def all_channel_fi_finite_difference(H0: np.ndarray, Gamma: np.ndarray, kappa: float,
                                     omega: float, port: int, site: int,
                                     h: float = 1e-6):
    """Independent check: QFI in every outgoing channel from finite differences.

    Output amplitudes for unit incident amplitude at the port are
        t_port = 1 - i kappa G_pp,     t_k = -i sqrt(kappa lam_k) <v_k|G|p>.
    Returns (F_port, F_all) with F = 4 sum_channels |d t / d eps|^2.
    """
    rates, V = channel_decomposition(Gamma, kappa, port)

    def amplitudes(eps):
        Hp = np.asarray(H0, dtype=complex).copy()
        Hp[site, site] += eps
        G = green_from_loss(Hp, Gamma, omega)
        t0 = 1.0 - 1j * kappa * G[port, port]
        tk = -1j * np.sqrt(kappa * rates) * (V.conj().T @ G[:, port])
        return t0, tk

    # step adapted to the local spectral scale: the response varies over
    # eps ~ 1/|G_ss|, so use a step well inside that scale, then a 4th-order stencil
    Gss = green_from_loss(H0, Gamma, omega)[site, site]
    h = min(h, 1e-3 / max(abs(Gss), 1.0))
    t0p, tkp = amplitudes(+h)
    t0m, tkm = amplitudes(-h)
    t0P, tkP = amplitudes(+2 * h)
    t0M, tkM = amplitudes(-2 * h)
    d0 = (8 * (t0p - t0m) - (t0P - t0M)) / (12 * h)
    dk = (8 * (tkp - tkm) - (tkP - tkM)) / (12 * h)
    F_port = 4.0 * abs(d0) ** 2
    F_all = F_port + 4.0 * float(np.sum(abs(dk) ** 2))
    return float(F_port), float(F_all)


# ---------------------------------------------------------------------------
# Random network ensembles used in the identity test
# ---------------------------------------------------------------------------

def random_reciprocal_network(N: int, rng: np.random.Generator, p_edge: float = 0.35,
                              J: float = 1.0, W: float = 0.5) -> np.ndarray:
    """Erdos-Renyi graph with real symmetric couplings and onsite disorder."""
    A = rng.random((N, N)) < p_edge
    A = np.triu(A, 1)
    H = np.zeros((N, N), dtype=complex)
    vals = J * rng.normal(size=(N, N))
    H[A] = vals[A]
    H = H + H.T
    H += np.diag(W * rng.uniform(-1, 1, N))
    # keep the graph connected enough: add a backbone chain
    idx = np.arange(N - 1)
    H[idx, idx + 1] += J
    H[idx + 1, idx] += J
    return H


def random_flux_network(N: int, rng: np.random.Generator, **kw) -> np.ndarray:
    """Hermitian but nonreciprocal (complex hoppings = synthetic gauge field)."""
    H = random_reciprocal_network(N, rng, **kw)
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, N)))
    U = np.triu(H, 1) * np.triu(phases, 1)
    return U + U.conj().T + np.diag(np.diag(H).real)


def random_psd_loss(N: int, rng: np.random.Generator, n_channels: int,
                    strength: float = 0.05) -> np.ndarray:
    """Non-diagonal positive-semidefinite loss (dissipative couplings)."""
    Gamma = np.zeros((N, N), dtype=complex)
    for _ in range(n_channels):
        v = rng.normal(size=N) + 1j * rng.normal(size=N)
        v /= np.linalg.norm(v)
        Gamma += strength * rng.random() * np.outer(v, v.conj())
    return Gamma


# ---------------------------------------------------------------------------
# Hypercubic lattices: port codimension law
# ---------------------------------------------------------------------------

def hypercubic_hamiltonian_sparse(L: int, d: int, J: float = 1.0) -> _sp.csr_matrix:
    """Open L^d hypercubic tight-binding Hamiltonian (nearest-neighbour hopping J)."""
    chain = _sp.diags([np.full(L - 1, J), np.full(L - 1, J)], [1, -1], format='csr')
    I = _sp.identity(L, format='csr')
    H = _sp.csr_matrix((L ** d, L ** d), dtype=complex)
    for axis in range(d):
        ops = [I] * d
        ops[axis] = chain
        term = ops[0]
        for op in ops[1:]:
            term = _sp.kron(term, op, format='csr')
        H = H + term
    return H.tocsr()


def hypercubic_index(L: int, coords) -> int:
    """Row-major linear index of a lattice site given zero-based coordinates."""
    idx = 0
    for c in coords:
        idx = idx * L + int(c)
    return idx


def hypercubic_mode_weight(L: int, coords) -> float:
    """|psi(x)|^2 of the (1,...,1) band-edge mode at zero-based coordinates."""
    w = 1.0
    for c in coords:
        w *= 2.0 / (L + 1) * np.sin(np.pi * (int(c) + 1) / (L + 1)) ** 2
    return float(w)


def hypercubic_band_edge_energy(L: int, d: int, J: float = 1.0) -> float:
    return float(2.0 * d * J * np.cos(np.pi / (L + 1)))


def port_site_coords(L: int, d: int, codim: int):
    """Coordinates of a port located on a boundary feature of the given codimension.

    ``codim`` coordinates are put on the boundary (index 0); the remaining
    coordinates are at the lattice centre.
    """
    centre = (L - 1) // 2
    return tuple([0] * codim + [centre] * (d - codim))


def sparse_fi_port(H0: _sp.spmatrix, omega: float, kappa: float, port: int, site: int,
                   gamma: float = 0.0):
    """Monitored-port QFI per photon and internal dwell time for a sparse network.

    Returns (F_port, tau_int, G_ps) using one sparse LU factorisation.
    """
    N = H0.shape[0]
    diag = np.full(N, omega + 0.5j * gamma, dtype=complex)
    diag[port] += 0.5j * kappa
    A = (_sp.diags(diag) - H0).tocsc()
    lu = _splu(A)
    e_p = np.zeros(N, dtype=complex)
    e_p[port] = 1.0
    col_p = lu.solve(e_p)            # G|p>
    Gsp = col_p[site]
    # reciprocity of a real symmetric H0: G_ps = G_sp
    dS = -1j * kappa * Gsp * Gsp
    F = 4.0 * abs(dS) ** 2
    tau = kappa * float(np.vdot(col_p, col_p).real)
    return float(F), float(tau), complex(Gsp)


# ---------------------------------------------------------------------------
# Finite-time (windowed) Fisher information
# ---------------------------------------------------------------------------

def h_unrestricted(x):
    """Single-mode QFI factor for a rectangular pulse of length T=x/Gamma with
    unrestricted (ring-down included) detection:  F_T = F_0 h(Gamma T),
        h(x) = 1 - 3/x + (3/x + 1/2) exp(-x/2),   h -> x/8 (x<<1), h -> 1 (x>>1)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 1e-3
    xs = x[~small]
    out[~small] = 1.0 - 3.0 / xs + (3.0 / xs + 0.5) * np.exp(-0.5 * xs)
    xt = x[small]
    out[small] = xt / 8.0 - xt ** 2 / 48.0 + xt ** 3 / 384.0
    return out


def k_windowed(x):
    """Single-mode QFI factor when both drive and detection are confined to [0,T]:
        F_T^win = F_0 k(Gamma T),   k(x) = (2/x) int_0^{x/2} [1 - (1+y) e^{-y}]^2 dy
        = (2/x) [X - 11/4 + 2(2+X)e^{-X} - e^{-2X}((1+X)^2/2 + (1+X)/2 + 1/4)],  X = x/2,
    with k -> x^4/320 (x<<1) and k -> 1 (x>>1).  The closed form cancels
    catastrophically for small X, so the positive integrand is integrated directly there."""
    from scipy.integrate import quad
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x)
    big = x > 2.0
    X = 0.5 * x[big]
    out[big] = (2.0 / x[big]) * (X - 2.75 + 2.0 * (2.0 + X) * np.exp(-X)
                                 - np.exp(-2.0 * X) * ((1 + X) ** 2 / 2 + (1 + X) / 2 + 0.25))
    f = lambda y: (1.0 - (1.0 + y) * np.exp(-y)) ** 2
    for i in np.where(~big)[0]:
        out[i] = (2.0 / x[i]) * quad(f, 0.0, 0.5 * x[i], epsabs=0, epsrel=1e-10, limit=200)[0]
    return out if out.size > 1 else out[0]


def stored_single_mode_factor(x):
    """Single-mode QFI remaining in the internal coherent state at a hard deadline.

    For a lossless one-port mode driven by a rectangular pulse on [0,T], the QFI
    of the internal coherent state at t=T, divided by the monochromatic steady-state
    value F_0, is

        q(x) = [1 - (1+x/2) exp(-x/2)]^2 / x,   x = Gamma T.

    The complete coherent-state information budget available at the deadline is
    F_port[0,T] + F_stored(T) = F_0 [k(x)+q(x)].  Ring-down detection is different:
    if the parameter remains present after T, additional information is generated while
    the stored excitation decays.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 1e-4
    xs = x[~small]
    A = 1.0 - (1.0 + 0.5 * xs) * np.exp(-0.5 * xs)
    out[~small] = A * A / xs
    xt = x[small]
    # A = x^2/8 - x^3/24 + x^4/128 + O(x^5)
    out[small] = xt**3 / 64.0 - xt**4 / 96.0 + 7.0 * xt**5 / 2304.0
    return out


def k_windowed_logderiv(x: float) -> float:
    """x k'(x)/k(x) computed from the integrand (no finite differences)."""
    from scipy.integrate import quad
    f = lambda y: (1.0 - (1.0 + y) * np.exp(-y)) ** 2
    X = 0.5 * x
    I = quad(f, 0.0, X, epsabs=0, epsrel=1e-10, limit=200)[0]
    # k = (2/x) I(x/2);  x k'/k = -1 + (x/2) f(x/2) / I
    return float(-1.0 + X * f(X) / I)


def single_mode_prefactor(Gamma_p: float, Gamma: float, w_s: float) -> float:
    """Steady-state single-mode QFI per photon F_0 = 64 Gamma_p^2 w_s^2 / Gamma^4."""
    return float(64.0 * Gamma_p ** 2 * w_s ** 2 / Gamma ** 4)


def optimum_x_windowed(a_over_b: float) -> float:
    """Solve x k'(x)/k(x) = a/b for the fixed-time optimum of N^a k(c N^-b)."""
    from scipy.optimize import brentq
    return float(brentq(lambda x: k_windowed_logderiv(x) - a_over_b, 1e-2, 1e3))


def spectral_fi_pulse_uniform(N: int, site: int, T: float, J: float = 1.0,
                              kappa: float = 0.1, gamma: float = 0.0,
                              n_fine: int = 1601, n_coarse: int = 3001,
                              n_modes_resolved: int = 60) -> float:
    """QFI per photon for a rectangular pulse of length T at the clean band-edge
    frequency with unrestricted detection: sinc^2-weighted average of F_s(omega)."""
    w0 = clean_mode_energy(N, 1, J)
    G = gamma + kappa * clean_mode_weight(N, 0)
    # grid: fine around the resonance, medium over the main sinc lobes, coarse over the
    # tails; the normalisation int sinc^2(dT/2) dd = 2 pi / T is used analytically.
    fine = np.linspace(-40 * G, 40 * G, n_fine)
    medium = np.linspace(-30.0 / T, 30.0 / T, n_coarse)
    tail = np.linspace(-4000.0 / T, 4000.0 / T, 4 * n_coarse)
    # resolved windows around the other chain resonances excited by a short pulse
    others = []
    for m in range(2, min(N, n_modes_resolved) + 1):
        Gm = gamma + kappa * clean_mode_weight(N, 0, m)
        dm = clean_mode_energy(N, m, J) - w0
        others.append(dm + np.linspace(-30 * Gm, 30 * Gm, 241))
    d = np.unique(np.concatenate([fine, medium, tail] + others))
    wts = np.sinc(d * T / (2 * np.pi)) ** 2
    F = np.array([fi_port_per_photon_uniform(N, site, w0 + x, J, kappa, gamma) for x in d])
    return float(np.trapezoid(F * wts, d) / (2.0 * np.pi / T))


class WindowedFI:
    """Exact finite-window Fisher information for a coherent rectangular pulse.

    The linear network is driven through the port by a constant amplitude (in the
    frame rotating at omega0) during [0, T_drive] and the output is detected on
    [0, T_detect].  The state X=(psi, d psi/d eps) obeys dX/dt = M X + B with
        M = [[A, 0], [-i P_s, A]],   A = -i(H_eff - omega0),   B = (-i sqrt(kappa) e_p, 0).
    The QFI is 4 kappa int_0^T |<p| d psi/d eps|^2 dt.  Exact chunked propagation:
    e^{M Delta}, the driven increment phi(Delta), and the chunk Gramians
    Q = int_0^Delta e^{M^dag t} C e^{M t} dt (with C = kappa |c><c|) are obtained from
    one Van Loan block exponential, so no time-step error is incurred.
    """

    def __init__(self, H0: np.ndarray, kappa: float, port: int, site: int,
                 omega0: float, gamma: float = 0.0, Delta: float = 100.0):
        H0 = np.asarray(H0, dtype=complex)
        N = H0.shape[0]
        Heff = H0.copy()
        Heff[port, port] -= 0.5j * kappa
        if gamma:
            Heff -= 0.5j * gamma * np.eye(N)
        A = -1j * (Heff - omega0 * np.eye(N))
        P = np.zeros((N, N), dtype=complex)
        P[site, site] = 1.0
        M = np.block([[A, np.zeros((N, N))], [-1j * P, A]])
        B = np.zeros(2 * N, dtype=complex)
        B[port] = -1j * np.sqrt(kappa)
        c = np.zeros(2 * N, dtype=complex)
        c[N + port] = np.sqrt(kappa)          # output derivative = <c|X>
        self.N, self.M, self.B, self.c, self.Delta = N, M, B, c, Delta
        # augmented generator for the driven increment: exp([[M,B],[0,0]] t)
        Mt = np.zeros((2 * N + 1, 2 * N + 1), dtype=complex)
        Mt[:2 * N, :2 * N] = M
        Mt[:2 * N, 2 * N] = B
        Ct = np.zeros_like(Mt)
        Ct[:2 * N + 1, :2 * N + 1] = np.outer(np.append(c, 0), np.append(c, 0).conj())
        # Van Loan: exp([[-Mt^dag, Ct],[0, Mt]] Delta) -> (1,2) block = e^{-Mt^dag D} Q~
        n = 2 * N + 1
        VL = np.zeros((2 * n, 2 * n), dtype=complex)
        VL[:n, :n] = -Mt.conj().T
        VL[:n, n:] = Ct
        VL[n:, n:] = Mt
        E = _expm(VL * Delta)
        self.E_aug = E[n:, n:]                              # e^{Mt Delta}
        Qt = self.E_aug.conj().T @ E[:n, n:]                # e^{Mt^dag D}(e^{-Mt^dag D} Q~) = Q~
        Qt = 0.5 * (Qt + Qt.conj().T)
        self.Q = Qt[:2 * N, :2 * N]        # int |<c|e^{Mt}X|^2
        self.r = Qt[:2 * N, 2 * N]         # cross term with driven increment
        self.s = float(Qt[2 * N, 2 * N].real)   # int |<c|phi(t)|^2
        self.EM = self.E_aug[:2 * N, :2 * N]    # e^{M Delta}
        self.phi = self.E_aug[:2 * N, 2 * N]    # driven increment over one chunk
        # ring-down Gramian Q_inf: M^dag Q + Q M = -C
        C = np.outer(c, c.conj())
        self.Qinf = _lyap(M.conj().T, -C)

    def fisher(self, T_drive: float, T_detect: float | None = None) -> float:
        """Total QFI (not per photon) for drive on [0,T_drive], detection on
        [0,T_detect]; T_detect=None means unrestricted detection."""
        D = self.Delta
        nd = int(round(T_drive / D))
        if abs(nd * D - T_drive) > 1e-9 * max(1.0, T_drive):
            raise ValueError("T_drive must be an integer multiple of Delta")
        X = np.zeros(2 * self.N, dtype=complex)
        total = 0.0
        # driven chunks
        for _ in range(nd):
            total += float((X.conj() @ self.Q @ X).real) + 2.0 * float((X.conj() @ self.r).real) + self.s
            X = self.EM @ X + self.phi
        if T_detect is None:
            return 4.0 * (total + float((X.conj() @ self.Qinf @ X).real))
        nf = int(round((T_detect - T_drive) / D))
        for _ in range(nf):
            total += float((X.conj() @ self.Q @ X).real)
            X = self.EM @ X
        return 4.0 * total

    def hard_window_budget(self, T: float) -> dict:
        """Exact coherent-state information budget at a hard deadline T.

        The drive and monitored-port detection both run on [0,T].  Returns QFI in the
        emitted monitored field, QFI still encoded in the internal coherent state at T,
        and their sum.  The stored term is 4||d psi(T)/d eps||^2.  For a lossless
        one-port network this sum is the QFI of the complete system+emitted-field state
        at the deadline; it does not include any subsequent parameter imprinting.
        """
        D = self.Delta
        nd = int(round(T / D))
        if abs(nd * D - T) > 1e-9 * max(1.0, T):
            raise ValueError("T must be an integer multiple of Delta")
        X = np.zeros(2 * self.N, dtype=complex)
        emitted = 0.0
        for _ in range(nd):
            emitted += float((X.conj() @ self.Q @ X).real) + 2.0 * float((X.conj() @ self.r).real) + self.s
            X = self.EM @ X + self.phi
        chi = X[self.N:]
        F_port = 4.0 * emitted
        F_stored = 4.0 * float(np.vdot(chi, chi).real)
        return dict(F_port=F_port, F_stored=F_stored, F_budget=F_port + F_stored,
                    F_port_per_photon=F_port / T, F_stored_per_photon=F_stored / T,
                    F_budget_per_photon=(F_port + F_stored) / T)

    def fisher_per_photon(self, T_drive: float, T_detect: float | None = None) -> float:
        return self.fisher(T_drive, T_detect) / T_drive


def hard_window_budget_sparse_chain(N: int, site: int, T: float, J: float = 1.0,
                                    kappa: float = 0.1, gamma: float = 0.0,
                                    omega0: float | None = None,
                                    rtol: float = 2e-8, atol: float = 1e-10) -> dict:
    """Sparse time-domain coherent-state information budget for a long chain.

    This routine is intended for propagation-limited regimes where the dense Van Loan
    engine would be unnecessarily expensive.  A unit-amplitude coherent drive is applied
    through site 0 during [0,T].  The ODE simultaneously propagates the coherent internal
    amplitude psi, its local-parameter derivative chi=d psi/d eps, and the QFI emitted
    into the monitored port and uniform intrinsic-loss channels.

    Returned QFIs are normalized per incident photon (the incident photon number is T
    for unit flux).  The stored term is 4||chi(T)||^2/T.  For gamma=0,
    F_port + F_stored is the complete coherent-state QFI available at the hard deadline.
    """
    from scipy.integrate import solve_ivp
    if omega0 is None:
        omega0 = clean_mode_energy(N, 1, J)
    if not (0 <= site < N):
        raise IndexError("site outside chain")

    def rhs(t, y):
        psi = y[:N]
        chi = y[N:2*N]
        dpsi = (1j * omega0 - 0.5 * gamma) * psi
        dchi = (1j * omega0 - 0.5 * gamma) * chi
        dpsi = dpsi.copy(); dchi = dchi.copy()
        if N > 1:
            dpsi[:-1] += -1j * J * psi[1:]
            dpsi[1:] += -1j * J * psi[:-1]
            dchi[:-1] += -1j * J * chi[1:]
            dchi[1:] += -1j * J * chi[:-1]
        dpsi[0] += -0.5 * kappa * psi[0] - 1j * np.sqrt(kappa)
        dchi[0] += -0.5 * kappa * chi[0]
        dchi[site] += -1j * psi[site]
        q_port = 4.0 * kappa * abs(chi[0]) ** 2
        q_loss = 4.0 * gamma * float(np.vdot(chi, chi).real) if gamma else 0.0
        return np.concatenate([dpsi, dchi, np.array([q_port, q_loss], dtype=complex)])

    y0 = np.zeros(2 * N + 2, dtype=complex)
    sol = solve_ivp(rhs, (0.0, T), y0, method='DOP853', rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    y = sol.y[:, -1]
    chi = y[N:2*N]
    F_port = float(y[-2].real)
    F_loss = float(y[-1].real)
    F_stored = 4.0 * float(np.vdot(chi, chi).real)
    inv_n = 1.0 / T
    return dict(F_port=F_port * inv_n, F_loss=F_loss * inv_n,
                F_stored=F_stored * inv_n,
                F_budget=(F_port + F_loss + F_stored) * inv_n,
                nfev=int(sol.nfev), omega0=float(omega0))
