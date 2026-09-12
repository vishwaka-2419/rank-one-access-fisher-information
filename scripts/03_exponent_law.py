from _common import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _style import *
from rankonefi import clean_mode_energy, fi_port_per_photon_uniform, fit_power_law

kappa=0.1
alphas=np.linspace(0,1,11)
Ns=np.unique(np.logspace(2.2,4.7,24).astype(int))  # 160 to ~50k
rows=[]
for alpha in alphas:
    Fs=[]
    sites=[]
    for N in Ns:
        if alpha==0:
            site=0
        elif alpha==1:
            site=N//2
        else:
            site=max(0,min(N-1,int(round(N**alpha))-1))
        omega=clean_mode_energy(N,1)
        F=fi_port_per_photon_uniform(N,site,omega,kappa=kappa)
        Fs.append(F); sites.append(site+1)
    p,A=fit_power_law(Ns,Fs,tail=8)
    rows.append((alpha,p,A,sites[-1]))
    print('alpha',alpha,'fit p',p)
pd.DataFrame(rows,columns=['alpha','fitted_exponent','prefactor','largest_site']).to_csv(DATA/'spatial_exponent_law.csv',index=False)

fig,ax=figure(COL_W, 2.45)
arr=np.array(rows,float)
ax.plot(alphas,4*alphas,'-',color='0.55',lw=1.0,label=r'asymptotic law $p=4\alpha$')
ax.plot(arr[:,0],arr[:,1],'o',color=OI['blue'],ms=4,mfc='white',mew=0.9,label='exact-resolvent fit')
ax.set_xlabel(r'signal-position exponent $\alpha$  ($s\sim N^{\alpha}$)')
ax.set_ylabel(r'QFI exponent $p$  ($\mathcal{F}_s\sim N^{p}$)')
ax.set_xlim(-0.03,1.03); ax.set_ylim(-0.15,4.3)
ax.set_xticks([0,0.25,0.5,0.75,1.0]); ax.set_yticks([0,1,2,3,4])
ax.annotate('opposite\nendpoint',xy=(0,0),xytext=(0.14,0.75),fontsize=6.8,color='0.3',ha='center',arrowprops=dict(arrowstyle='-',lw=0.5,color='0.5'))
ax.annotate('chain\ncenter',xy=(1,4),xytext=(0.84,3.3),fontsize=6.8,color='0.3',ha='center',arrowprops=dict(arrowstyle='-',lw=0.5,color='0.5'))
ax.legend(loc='upper left')
tidy(ax)
fig.subplots_adjust(left=0.16,right=0.97,top=0.97,bottom=0.18)
save_figure(fig,'fig4_spatial_exponent_law')

