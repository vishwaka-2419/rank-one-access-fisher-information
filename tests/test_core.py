import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from rankonefi import (
    scattering_amplitude_uniform, dS_deps_uniform,
    fi_port_per_photon_uniform, clean_mode_energy,
    fit_power_law
)


def finite_difference_dS(N, site, omega, J, kappa, gamma, h=1e-7):
    # Dense helper with explicit epsilon perturbation.
    from rankonefi import uniform_chain_hamiltonian, effective_hamiltonian
    def S(eps):
        H=uniform_chain_hamiltonian(N,J,eps=eps,site=site)
        G=np.linalg.inv(omega*np.eye(N)-effective_hamiltonian(H,kappa,gamma))
        return 1-1j*kappa*G[0,0]
    return (S(h)-S(-h))/(2*h)


def test_lossless_unitarity():
    for N in [2,5,9]:
        for w in np.linspace(-1.7,1.7,7):
            S=scattering_amplitude_uniform(N,w,kappa=0.2,gamma=0)
            assert abs(abs(S)-1)<1e-10


def test_green_derivative():
    N=7; site=4; w=0.37; k=0.17; g=0.013
    ana=dS_deps_uniform(N,site,w,kappa=k,gamma=g)
    num=finite_difference_dS(N,site,w,1.0,k,g)
    assert abs(ana-num)/max(abs(ana),1e-12)<2e-7


def test_endpoint_vs_center_scaling():
    Ns=np.array([80,120,180,260,380,560])
    end=[]; mid=[]
    for N in Ns:
        omega=clean_mode_energy(int(N),1)
        end.append(fi_port_per_photon_uniform(int(N),int(N)-1,omega,kappa=0.1))
        mid.append(fi_port_per_photon_uniform(int(N),int(N)//2,omega,kappa=0.1))
    pe,_=fit_power_law(Ns,end)
    pm,_=fit_power_law(Ns,mid)
    assert abs(pe)<0.05
    assert abs(pm-4)<0.08


# ---------------------------------------------------------------------------
# Part II tests: access identity, codimension law, finite-time functions
# ---------------------------------------------------------------------------
from rankonefi import (access_identity, all_channel_fi_finite_difference,
                       random_flux_network, random_psd_loss,
                       hypercubic_hamiltonian_sparse, hypercubic_index, hypercubic_mode_weight,
                       hypercubic_band_edge_energy, port_site_coords, sparse_fi_port,
                       h_unrestricted, k_windowed, stored_single_mode_factor, single_mode_prefactor, WindowedFI,
                       spectral_fi_pulse_uniform, clean_mode_weight, uniform_chain_hamiltonian)


def test_access_identity_general_network():
    rng = np.random.default_rng(7)
    N = 13; kappa = 0.3; port, site = 0, 8
    H = random_flux_network(N, rng)                       # Hermitian, nonreciprocal
    Gam = np.zeros((N, N), dtype=complex); Gam[port, port] = kappa
    Gam += 0.02 * np.eye(N) + random_psd_loss(N, rng, 3)  # non-diagonal PSD loss
    for w in (-1.1, 0.3, 1.4):
        a = access_identity(H, Gam, kappa, w, port, site)
        assert abs(a['F_port'] - a['F_port_identity']) < 1e-12 * a['F_port']
        assert abs(a['F_all'] - a['F_all_identity']) < 1e-12 * a['F_all']
        assert 0 <= a['eta_in'] <= 1 and 0 <= a['eta_out'] <= 1
        assert a['F_port'] <= a['ldos_bound'] * (1 + 1e-12)
        fd_port, fd_all = all_channel_fi_finite_difference(H, Gam, kappa, w, port, site)
        assert abs(fd_port - a['F_port']) < 1e-6 * a['F_port']
        assert abs(fd_all - a['F_all']) < 1e-6 * a['F_all']
    # lossless: exact saturation of the LDOS bound, |G_1s| = |G_s1| despite nonreciprocity
    Gam0 = np.zeros((N, N), dtype=complex); Gam0[port, port] = kappa
    a0 = access_identity(H, Gam0, kappa, 0.3, port, site)
    assert abs(a0['F_port'] - a0['ldos_bound']) < 1e-12 * a0['ldos_bound']
    assert abs(a0['eta_in'] - 1) < 1e-12 and abs(a0['eta_out'] - 1) < 1e-12


def test_codimension_law_2d():
    kappa = 0.02; L = 27; d = 2
    H = hypercubic_hamiltonian_sparse(L, d)
    omega = hypercubic_band_edge_energy(L, d)
    centre = tuple([(L - 1) // 2] * d)
    for k in (1, 2):
        pc = port_site_coords(L, d, k)
        F, tau, _ = sparse_fi_port(H, omega, kappa, hypercubic_index(L, pc), hypercubic_index(L, centre))
        F_iso = 64.0 * (hypercubic_mode_weight(L, centre) / hypercubic_mode_weight(L, pc)) ** 2 / kappa ** 2
        assert abs(F / F_iso - 1) < 1e-5


def test_finite_time_functions_and_engine():
    x = np.array([1e-3, 1e-2])
    assert np.allclose(h_unrestricted(x), x / 8, rtol=2e-3)
    assert np.allclose(k_windowed(x), x ** 4 / 320, rtol=2e-3)
    assert np.allclose(stored_single_mode_factor(x), x ** 3 / 64, rtol=3e-3)
    assert abs(float(h_unrestricted(500.0)) - 1) < 1e-2 and abs(float(k_windowed(500.0)) - 1) < 1.2e-2
    N = 20; kappa = 0.1; site = N // 2
    w0 = clean_mode_energy(N, 1); Gp = kappa * clean_mode_weight(N, 0)
    F0 = single_mode_prefactor(Gp, Gp, clean_mode_weight(N, site))
    W = WindowedFI(uniform_chain_hamiltonian(N), kappa, 0, site, w0, Delta=50.0)
    T = 1000.0
    assert abs(W.fisher_per_photon(T, T) / (F0 * float(k_windowed(Gp * T))) - 1) < 2e-2
    b = W.hard_window_budget(T)
    q = float(stored_single_mode_factor(Gp * T))
    assert abs(b['F_stored_per_photon'] / (F0 * q) - 1) < 2e-2
    assert abs(b['F_budget_per_photon'] - b['F_port_per_photon'] - b['F_stored_per_photon']) < 1e-10 * b['F_budget_per_photon']
    assert abs(W.fisher_per_photon(T, None) / spectral_fi_pulse_uniform(N, site, T, kappa=kappa) - 1) < 1e-3


def test_deadline_law_with_intrinsic_loss():
    """Single-mode hard-deadline law with Gamma = gamma + kappa w1 and F0(gamma)."""
    N, kappa, gamma, T = 40, 0.1, 1e-4, 1e4
    w0 = clean_mode_energy(N, 1); site = N // 2
    Gp = kappa * clean_mode_weight(N, 0); G = gamma + Gp
    W = WindowedFI(uniform_chain_hamiltonian(N), kappa, 0, site, w0, gamma=gamma, Delta=100.0)
    Fsm = single_mode_prefactor(Gp, G, clean_mode_weight(N, site)) * float(k_windowed(G * T))
    assert abs(W.fisher_per_photon(T, T) / Fsm - 1) < 2e-2
