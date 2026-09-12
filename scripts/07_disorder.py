from _common import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _style import *
from scipy.linalg import eigh_tridiagonal
from rankonefi import fi_port_per_photon_uniform

# Static onsite-disorder stress test. For each realization we track the Hermitian
# band-edge mode by overlap with the clean m=1 eigenvector and retune the probe to
# that realization's closed-chain eigenfrequency. The open-system response is then
# evaluated exactly with the O(N) tridiagonal Green-function solver.

rng=np.random.default_rng(20260911)
J=1.0; kappa=0.1; gamma=0.0
Ns=np.array([20,30,40,50,60,80,100,120])
Ws=[0.0,1e-4,3e-4,1e-3]
R=30
rows=[]
fig,ax=figure(COL_W, 2.45)
colmap={0.0:'k',1e-4:C3[0],3e-4:C3[1],1e-3:C3[2]}
for W in Ws:
    meds=[]; qlo=[]; qhi=[]
    for N in Ns:
        q=np.pi/(N+1)
        psi=np.sqrt(2/(N+1))*np.sin(q*np.arange(1,N+1))
        vals=[]
        nreal=1 if W==0 else R
        offdiag=np.full(N-1,J,float)
        for r in range(nreal):
            disorder=np.zeros(N) if W==0 else rng.normal(0,W,N)
            ev,V=eigh_tridiagonal(disorder,offdiag,check_finite=False)
            im=int(np.argmax(np.abs(V.T@psi)))
            omega=float(ev[im])
            vals.append(fi_port_per_photon_uniform(
                N,N//2,omega,J,kappa,gamma,disorder=disorder))
        vals=np.asarray(vals)
        med=float(np.median(vals)); lo=float(np.quantile(vals,.25)); hi=float(np.quantile(vals,.75))
        meds.append(med); qlo.append(lo); qhi.append(hi)
        for v in vals: rows.append((N,W,v))
    meds=np.asarray(meds); qlo=np.asarray(qlo); qhi=np.asarray(qhi)
    ax.loglog(Ns,meds,'o-' if W>0 else '--',color=colmap[W],ms=3,lw=1.0 if W>0 else 0.8,label=r'clean' if W==0 else rf'$W/J={W:g}$')
    if W>0:
        ax.fill_between(Ns,qlo,qhi,color=colmap[W],alpha=.15,lw=0)
ax.set_xlabel('chain length $N$')
ax.set_ylabel('center-site QFI per photon (retuned)')
ax.legend(loc='upper left')
tidy(ax)
fig.subplots_adjust(left=0.19,right=0.97,top=0.97,bottom=0.18)
save_figure(fig,'figS_static_disorder')
pd.DataFrame(rows,columns=['N','W','F_center']).to_csv(DATA/'static_disorder.csv',index=False)
