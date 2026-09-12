from _common import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _style import *
from rankonefi import clean_mode_energy, fi_total_per_photon_uniform, clean_mode_weight

kappa=0.1
gammas=[1e-6,1e-5,1e-4]
Ns=np.unique(np.logspace(np.log10(8),np.log10(800),90).astype(int))
rows=[]
fig,ax=figure(COL_W, 2.45)
colmap={1e-6:C3[0],1e-5:C3[1],1e-4:C3[2]}
for gamma in gammas:
    frac=[]; approx=[]
    for N in Ns:
        site=N//2
        omega=clean_mode_energy(N,1)
        Fp,Fa,p=fi_total_per_photon_uniform(N,site,omega,kappa=kappa,gamma=gamma)
        w1=clean_mode_weight(N,0,1)
        pa=kappa*w1/(kappa*w1+gamma)
        frac.append(p); approx.append(pa)
        rows.append((N,gamma,Fp,Fa,p,pa))
    ax.semilogx(Ns,frac,color=colmap[gamma],lw=1.3,label=rf'$\gamma/J=10^{{{int(np.log10(gamma))}}}$')
    ax.semilogx(Ns,approx,'--',color='k',lw=0.7,alpha=0.8,label='isolated-mode law' if gamma==1e-4 else None)
ax.set_xlabel('chain length $N$')
ax.set_ylabel(r'$\mathcal{F}_s^{(\rm port)}/\mathcal{F}_s^{(\rm all)}=\eta_{\rm out}$')
ax.set_ylim(-0.03,1.03)
ax.axhline(2/3,color='0.6',lw=0.6,ls=':')
ax.text(8.5,0.69,r'$\eta_{\rm out}=2/3$ at $N_\star$',fontsize=6.6,color='0.35',va='bottom')
ax.legend(loc='lower left')
tidy(ax)
fig.subplots_adjust(left=0.17,right=0.97,top=0.97,bottom=0.18)
save_figure(fig,'fig7_information_branching')
pd.DataFrame(rows,columns=['N','gamma','F_port','F_all','accessible_fraction','single_mode_branching']).to_csv(DATA/'information_branching.csv',index=False)
