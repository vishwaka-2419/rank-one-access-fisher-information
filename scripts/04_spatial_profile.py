from _common import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _style import *
from rankonefi import clean_mode_energy, fi_port_per_photon_uniform

kappa=0.1
rows=[]
fig,ax=figure(COL_W, 2.4)
cols=iter(C3); widths=iter([1.7,1.2,0.8])
for N in [51,101,201]:
    omega=clean_mode_energy(N,1)
    F=np.array([fi_port_per_photon_uniform(N,s,omega,kappa=kappa) for s in range(N)])
    x=(np.arange(N)+1)/(N+1)
    norm=F/F.max()
    ax.plot(x,norm,color=next(cols),lw=next(widths),label=f'$N={N}$')
    for s in range(N): rows.append((N,s+1,x[s],F[s],norm[s]))
# isolated-mode limiting envelope
xx=np.linspace(0,1,600)
ax.plot(xx,np.sin(np.pi*xx)**4,'--',color='k',lw=0.8,label=r'$\sin^4(\pi x)$')
ax.set_xlabel(r'normalized signal position $x=s/(N+1)$')
ax.set_ylabel(r'$\mathcal{F}_s\,/\,\max_s\mathcal{F}_s$')
ax.set_xlim(0,1); ax.set_ylim(0,1.05)
ax.text(0.5,0.35,r'$\propto\rho_s^{2}\propto w_s^{2}$',ha='center',fontsize=7.5,color='0.3')
ax.legend(loc='upper left')
tidy(ax)
fig.subplots_adjust(left=0.16,right=0.97,top=0.97,bottom=0.18)
save_figure(fig,'fig5_spatial_profile')
pd.DataFrame(rows,columns=['N','site_1based','x','F','F_normalized']).to_csv(DATA/'spatial_profile.csv',index=False)
