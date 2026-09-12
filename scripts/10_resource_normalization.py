from _common import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _style import *
from rankonefi import clean_mode_energy, fi_port_per_photon_uniform, internal_occupation_per_flux_uniform, fit_power_law

kappa=0.1
Ns=np.unique(np.logspace(np.log10(12),np.log10(3000),70).astype(int))
rows=[]
fig,ax=figure(COL_W, 2.5)
style={'opposite endpoint':dict(color=OI['blue'],mfc='white'),'center':dict(color=OI['vermilion'],mfc=OI['vermilion'])}
for label,posfun in [('opposite endpoint',lambda N:N-1),('center',lambda N:N//2)]:
    vals=[]
    for N in Ns:
        w=clean_mode_energy(N,1)
        F=fi_port_per_photon_uniform(N,posfun(N),w,kappa=kappa)
        tau=internal_occupation_per_flux_uniform(N,w,kappa=kappa)
        R=F/tau
        vals.append(R)
        rows.append((N,label,F,tau,R))
    p,_=fit_power_law(Ns,vals,tail=20)
    print(label,'resource-normalized exponent',p)
    ax.loglog(Ns,vals,'o',ms=3,mew=0.7,label=f'{label}: fit $N^{{{p:.2f}}}$',**style[label])
ax.set_xlabel('chain length $N$')
ax.set_ylabel(r'QFI rate per stored excitation $\dot{\mathcal{I}}_\epsilon/\bar{n}_{\rm int}$')
slope_guide(ax,3e2,1.2e4,1,dx_factor=3,label=r'$N^{1}$',above=False,pad=2.5)
slope_guide(ax,3e2,6e-4,-3,dx_factor=3,label=r'$N^{-3}$',above=True,pad=2.5)
ax.legend(loc='center left',bbox_to_anchor=(0.0,0.5))
ax.set_ylim(3e-8,3e5)
tidy(ax)
fig.subplots_adjust(left=0.19,right=0.97,top=0.97,bottom=0.17)
save_figure(fig,'fig8_resource_normalized_scaling')
pd.DataFrame(rows,columns=['N','position','F_per_incident_photon','internal_occupation_per_flux','FI_rate_per_internal_excitation']).to_csv(DATA/'resource_normalized_scaling.csv',index=False)
