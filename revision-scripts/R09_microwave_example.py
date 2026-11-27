"""
Referee 2, point 8 and Referee 1 (minor, calibration/range): numbers for the proposed
coupled-resonator implementation.  Frequencies quoted as f = omega / 2 pi.
"""
import sys, json
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import rof

Jf, kf, gf = 31e6, 3.1e6, 10e3       # Hz
J = 1.0; kappa = kf / Jf; gam = gf / Jf
tw = 2 * np.pi * Jf                    # 1/J in seconds -> t[J^-1] = t[s] * tw
print(f"kappa/J = {kappa:.4f}, gamma/J = {gam:.3e}, Q_i at 6 GHz = {6e9/gf:.2e}")
print(f"N_star = (pi^2 kappa/gamma)^(1/3) = {(np.pi**2*kappa/gam)**(1/3):.2f};  N_gamma = pi sqrt(3J/gamma) = {np.pi*np.sqrt(3/gam):.0f}")

def single_res_F(kappa, gam):
    return 64 * kappa ** 2 / (kappa + gam) ** 4

res = {}
print("\n  N  s   w1       ws      Gamma_p[kHz] Gamma[kHz]  gap[MHz]  Gamma/gap   F_exact/F_res  mode est.  lossless (ws/w1)^2  eta_out   tau_load=2/Gamma [us]")
for N in [3, 5, 7, 9, 11, 13, 15, 17, 21]:
    s = (N + 1) // 2
    w1, ws = rof.chain_w(N, 1), rof.chain_w(N, s)
    Gp, G = kappa * w1, gam + kappa * w1
    gap = rof.chain_gap(N)
    Fex = rof.chain_F(N, s, kappa, gam)
    Fm = rof.mode_F(w1, ws, kappa, gam)
    Fr = single_res_F(kappa, gam)
    res[N] = dict(w1=w1, ws=ws, Gp_kHz=Gp * Jf / 1e3, G_kHz=G * Jf / 1e3, gap_MHz=gap * Jf / 1e6, ratio=Fex / Fr, ratio_mode=Fm / Fr)
    print(f" {N:2d} {s:2d}  {w1:.5f}  {ws:.5f}  {Gp*Jf/1e3:9.1f}  {G*Jf/1e3:9.1f}  {gap*Jf/1e6:8.2f}  {G/gap:.2e}   {Fex/Fr:10.3f}   {Fm/Fr:8.3f}   {(ws/w1)**2:10.2f}     {Gp/G:.3f}   {2/(G*tw)*1e6:8.2f}")

# spatial profile for N = 5 (what a scanned local perturbation would see)
print("\nN = 5 spatial profile F_s / F_res (exact, with loss):")
for s in range(1, 6):
    print(f"   s={s}: {rof.chain_F(5, s, kappa, gam)/single_res_F(kappa, gam):8.3f}   (lossless mode law (ws/w1)^2 = {(rof.chain_w(5,s)/rof.chain_w(5,1))**2:.2f})")

# finite windows, N = 5 centre, with loss, vs single resonator in the same window
print("\nfinite window (hard deadline), exact time domain, with gamma:")
for N, s in [(5, 3), (7, 4), (9, 5)]:
    H = rof.chain_Heff(N, kappa, gam)
    Hr = np.array([[-0.5j * (kappa + gam)]])
    for Tus in [1.0, 3.0, 10.0, 30.0]:
        T = Tus * 1e-6 * tw
        Fc = rof.windowed_qfi(H, 0, s - 1, kappa, rof.chain_E1(N), T)
        Fr = rof.windowed_qfi(Hr, 0, 0, kappa, 0.0, T)
        Gt = gam + kappa * rof.chain_w(N, 1)
        Fmode = rof.mode_F(rof.chain_w(N, 1), rof.chain_w(N, s), kappa, gam) * rof.k_fun(Gt * T)
        Frm = 64 * kappa ** 2 / (kappa + gam) ** 4 * rof.k_fun((kappa + gam) * T)
        print(f"  N={N} T={Tus:5.1f} us: Gamma T = {Gt*T:7.2f}, kappa T = {kappa*T:7.1f};  F_chain/F_res exact = {Fc/Fr:8.3f}  (isolated-mode {Fmode/Frm:8.3f});  k(Gamma T) = {rof.k_fun(Gt*T):.4f}")

# absolute numbers, N = 5
N, s = 5, 3
kap_si = 2 * np.pi * kf; gam_si = 2 * np.pi * gf
w1, ws = rof.chain_w(N, 1), rof.chain_w(N, s)
Gp = kap_si * w1; G = gam_si + Gp
F = 64 * Gp ** 2 * ws ** 2 / G ** 4            # s^2
tau_int = 4 * Gp / G ** 2                      # s, stored excitation per unit incident flux (isolated mode)
print(f"\nN=5: F_0 = {F:.3e} s^2 per photon -> delta(eps/2pi) >= {1/np.sqrt(F)/(2*np.pi)/1e3:.1f} kHz / sqrt(n_photons)")
print(f"     dwell (stored excitation per incident flux) tau_int = {tau_int*1e6:.2f} us; flux for nbar_int = 100: {100/tau_int:.2e} photons/s "
      f"-> {1/np.sqrt(F*100/tau_int)/(2*np.pi):.1f} Hz/sqrt(Hz)")
print(f"     calibration: QFI loss <1% needs |delta| < {0.5*np.sqrt(0.99**-0.5-1)*G/(2*np.pi)/1e3:.1f} kHz;  <10% needs |delta| < {0.5*np.sqrt(0.9**-0.5-1)*G/(2*np.pi)/1e3:.1f} kHz")
print(f"     local range: mode shift w_s*eps; 10% QFI loss at eps/2pi = {0.5*np.sqrt(0.9**-0.5-1)*G/ws/(2*np.pi)/1e3:.1f} kHz;  Gamma/(2 w_s) = {G/(2*ws)/(2*np.pi)/1e3:.0f} kHz; 4/sqrt(F0)*(Gp/G)= {4/np.sqrt(F)*(Gp/G)/(2*np.pi)/1e3:.0f} kHz")

# disorder: first-order perturbative log-scatter of F for small chains (script 04)
import importlib.util
spec = importlib.util.spec_from_file_location("pt", os.path.join(HERE, 'R04_disorder_perturbative.py'))
pt = importlib.util.module_from_spec(spec); spec.loader.exec_module(pt)
sig_f = 2e-4 * 6e9          # "two parts in 10^4" at 6 GHz  [Underwood et al. 2012]
print(f"\ndisorder: sigma_f = {sig_f/1e6:.2f} MHz  ->  sigma/J = {sig_f/Jf:.4f}")
for N in [3, 5, 7, 9, 11, 13]:
    c = (N + 1) // 2
    C_on, C_b, _, _ = pt.coeffs(N, 1, (1,), (c,))
    print(f"   N={N:2d}: C_onsite = {C_on:9.2f} (C/N^3 = {C_on/N**3:.4f}), C_bond = {C_b:9.2f};  rms(ln F) for onsite sigma/J={sig_f/Jf:.3f}: {sig_f/Jf*np.sqrt(C_on):.3f}")
