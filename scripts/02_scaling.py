from _common import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _style import *
from rankonefi import clean_mode_energy, fi_port_per_photon_uniform, isolated_mode_fi, fit_power_law

kappa=0.1
Ns=np.unique(np.logspace(np.log10(12), np.log10(3000), 70).astype(int))
rows=[]
for N in Ns:
    omega=clean_mode_energy(N,1)
    for label,site in [('opposite endpoint',N-1),('center',N//2)]:
        F=fi_port_per_photon_uniform(N,site,omega,kappa=kappa)
        Fa=isolated_mode_fi(N,site,kappa=kappa)
        rows.append((N,label,site+1,F,Fa))
df=pd.DataFrame(rows,columns=['N','position','site_1based','F_exact','F_single_mode'])
df.to_csv(DATA/'scaling_endpoint_center.csv',index=False)

fig,ax=figure(COL_W, 2.55)
style={'opposite endpoint':dict(marker='o', mfc='white', color=OI['blue']),
       'center':dict(marker='o', mfc=OI['vermilion'], color=OI['vermilion'])}
fits={}
for label in ['opposite endpoint','center']:
    d=df[df.position==label]
    p,A=fit_power_law(d.N.values,d.F_exact.values,tail=20); fits[label]=p
    print(label,'slope',p,'prefactor',A)
    ax.loglog(d.N,d.F_exact,ls='none',ms=3.2,mew=0.7,**style[label],label=f'{label} (exact)')
    ax.loglog(d.N,d.F_single_mode,'-',color='k',lw=0.7,alpha=0.8,zorder=0,label='isolated-mode law' if label=='center' else None)
slope_guide(ax,2.5e2,1.5e10,4,dx_factor=3,label=r'$N^{4}$',above=False,pad=8)
ax.text(1.5e2,1.5e4,r'$N^{0}$  ($64q^{4}/\kappa^{2}$)',fontsize=7,color='0.35',ha='center',va='bottom')
ax.set_xlabel('chain length $N$')
ax.set_ylabel(r'QFI per incident photon $\mathcal{F}_s$')
ax.set_ylim(3e2,3e16)
ax.legend(loc='upper left')
tidy(ax)
fig.subplots_adjust(left=0.19,right=0.97,top=0.97,bottom=0.17)
save_figure(fig,'fig3_endpoint_center_scaling')

