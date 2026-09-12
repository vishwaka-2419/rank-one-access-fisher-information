from _common import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _style import *
from scipy.linalg import solve_banded
from rankonefi import clean_mode_energy, fi_port_per_photon_uniform, fit_power_law

# Explicit readout resonator (finite port bandwidth). The resonator is retuned to the
# target clean-chain mode for each N. In the bad-cavity limit it produces an effective
# boundary rate kappa_eff = 4 g^2/kappa_c.

def cavity_Gcs(N, chain_site, omega, Ec, J, g, kappa_c, gamma):
    M=N+1
    ab=np.zeros((3,M),complex)
    # off-diagonal sequence: cavity-chain g, then chain J
    offs=np.r_[g, np.full(N-1,J)]
    ab[0,1:]=-offs
    ab[2,:-1]=-offs
    diag=np.empty(M,complex)
    diag[0]=omega-Ec+0.5j*kappa_c
    diag[1:]=omega+0.5j*gamma
    ab[1]=diag
    rhs=np.zeros(M,complex); rhs[chain_site+1]=1.0
    x=solve_banded((1,1),ab,rhs,check_finite=False)
    return x[0]

def fi_cavity(N,site,omega,Ec,J,g,kappa_c,gamma=0):
    Gcs=cavity_Gcs(N,site,omega,Ec,J,g,kappa_c,gamma)
    return 4*kappa_c**2*abs(Gcs)**4

J=1.0
kappa_eff=0.1
bandwidths=[8.0,2.0,0.5]
Ns=np.unique(np.logspace(np.log10(15),np.log10(1200),65).astype(int))
rows=[]
fig,ax=figure(COL_W, 2.45)
colmap={8.0:C3[0],2.0:C3[1],0.5:C3[2]}
# Markov reference
Fref=[]
for N in Ns:
    w=clean_mode_energy(N,1,J)
    Fref.append(fi_port_per_photon_uniform(N,N//2,w,J,kappa_eff,0))
ax.loglog(Ns,Fref,'--',color='k',lw=0.8,zorder=5,label='Markov boundary port')
for kc in bandwidths:
    g=np.sqrt(kappa_eff*kc/4)
    vals=[]
    for N in Ns:
        Ec=clean_mode_energy(N,1,J)
        F=fi_cavity(N,N//2,Ec,Ec,J,g,kc,0)
        vals.append(F)
        rows.append((N,kc,g,F))
    p,_=fit_power_law(Ns,vals,tail=18)
    print('kappa_c',kc,'g',g,'tail exponent',p)
    ax.loglog(Ns,vals,color=colmap[kc],lw={8.0:3.0,2.0:2.0,0.5:1.1}[kc],alpha=0.85,label=rf'resonator, $\kappa_c/J={kc:g}$')
ax.set_xlabel('chain length $N$')
ax.set_ylabel('center-site QFI per photon')
ax.legend(loc='upper left')
tidy(ax)
fig.subplots_adjust(left=0.19,right=0.97,top=0.97,bottom=0.18)
save_figure(fig,'figS_finite_bandwidth_port')
pd.DataFrame(rows,columns=['N','kappa_c','g','F_center']).to_csv(DATA/'finite_bandwidth.csv',index=False)
