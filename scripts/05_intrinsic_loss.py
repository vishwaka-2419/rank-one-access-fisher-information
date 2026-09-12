from _common import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _style import *
from rankonefi import clean_mode_energy, fi_port_per_photon_uniform, predicted_bulk_nopt

kappa=0.1
gammas=[0.0,1e-6,1e-5,1e-4]
Ns=np.unique(np.logspace(np.log10(5),np.log10(800),100).astype(int))
rows=[]
fig,ax=figure(COL_W, 2.6)
colmap={0.0:'0.45',1e-6:C3[0],1e-5:C3[1],1e-4:C3[2]}
for gamma in gammas:
    vals=[]
    for N in Ns:
        site=N//2
        omega=clean_mode_energy(N,1)
        F=fi_port_per_photon_uniform(N,site,omega,kappa=kappa,gamma=gamma)
        vals.append(F)
        rows.append((N,gamma,F))
    vals=np.array(vals)
    label=r'lossless' if gamma==0 else rf'$\gamma/J=10^{{{int(np.log10(gamma))}}}$'
    ax.loglog(Ns,vals,color=colmap[gamma],lw=1.3 if gamma else 0.9,ls='-' if gamma else '--',label=label)
    if gamma>0:
        nstar=predicted_bulk_nopt(kappa,gamma)
        # marker at nearest simulated point
        i=int(np.argmin(abs(Ns-nstar)))
        ax.plot([Ns[i]],[vals[i]],marker='*',ms=9,mfc='white',mec=colmap[gamma],mew=1.0,ls='none',zorder=5)
        print('gamma',gamma,'pred N*',nstar,'numeric nearest maximum',Ns[np.argmax(vals)])
ax.set_xlabel('chain length $N$')
ax.set_ylabel(r'center-site QFI per photon $\mathcal{F}_{\rm bulk}$')
ax.set_ylim(1e-3,3e13)
ax.plot([],[],marker='*',ms=9,mfc='white',mec='0.3',mew=1.0,ls='none',label=r'$N_\star=(\pi^2\kappa/\gamma)^{1/3}$')
slope_guide(ax,8,1.5e5,4,dx_factor=2.5,label=r'$N^{4}$',above=True,pad=1.5)
slope_guide(ax,2.6e2,1e3,-8,dx_factor=2.0,label=r'$N^{-8}$',above=True,pad=1.5)
ax.legend(loc='lower left',ncol=1)
tidy(ax)
fig.subplots_adjust(left=0.19,right=0.97,top=0.97,bottom=0.17)
save_figure(fig,'fig6_intrinsic_loss_optimum')
pd.DataFrame(rows,columns=['N','gamma','F_center']).to_csv(DATA/'intrinsic_loss_scan.csv',index=False)
