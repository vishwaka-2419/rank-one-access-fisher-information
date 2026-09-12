from _common import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _style import *
from scipy import sparse
from scipy.sparse.linalg import spsolve
from rankonefi import clean_mode_energy, dS_deps_uniform

# Exact finite-Hilbert-space Lindblad validation of the weak-excitation scattering law.
# No QuTiP dependency is required.

def kron_all(ops):
    out=sparse.csr_matrix([[1.0+0j]])
    for op in ops:
        out=sparse.kron(out,op,format='csr')
    return out

I2=sparse.eye(2,dtype=complex,format='csr')
sm1=sparse.csr_matrix(np.array([[0,1],[0,0]],complex))
sp1=sm1.getH()
n1=sp1@sm1


def local_op(N, op, j):
    ops=[I2]*N
    ops[j]=op
    return kron_all(ops)


def build_ops(N):
    sm=[local_op(N,sm1,j) for j in range(N)]
    sp=[x.getH() for x in sm]
    n=[sp[j]@sm[j] for j in range(N)]
    return sm,sp,n


def liouvillian(H, jumps):
    d=H.shape[0]
    I=sparse.eye(d,dtype=complex,format='csr')
    L=-1j*(sparse.kron(I,H,format='csr')-sparse.kron(H.T,I,format='csr'))
    for c in jumps:
        cdgc=c.getH()@c
        L += sparse.kron(c.conjugate(),c,format='csr')
        L += -0.5*sparse.kron(I,cdgc,format='csr')
        L += -0.5*sparse.kron(cdgc.T,I,format='csr')
    return L.tocsr()


def steady_state(L,d):
    A=L.tolil(copy=True)
    tr=np.zeros(d*d,complex)
    tr[np.arange(d)*(d+1)]=1.0
    A[0,:]=tr
    b=np.zeros(d*d,complex); b[0]=1.0
    x=spsolve(A.tocsr(),b)
    rho=x.reshape((d,d),order='F')
    rho=(rho+rho.conj().T)/2
    rho/=np.trace(rho)
    return rho


def output_amplitude(N,site,eps,beta,J,kappa,gamma,omega,ops_cache=None):
    sm,sp,n=ops_cache if ops_cache is not None else build_ops(N)
    d=2**N
    H=sparse.csr_matrix((d,d),dtype=complex)
    # H_chain - omega * total excitation number + eps n_site
    for j in range(N-1):
        H += J*(sp[j]@sm[j+1]+sp[j+1]@sm[j])
    for j in range(N):
        H += -omega*n[j]
    H += eps*n[site]
    H += 1j*np.sqrt(kappa)*(beta*sp[0]-beta*sm[0])
    jumps=[np.sqrt(kappa)*sm[0]]
    if gamma>0:
        jumps += [np.sqrt(gamma)*x for x in sm]
    L=liouvillian(H,jumps)
    rho=steady_state(L,d)
    sigma=np.trace(rho@sm[0].toarray())
    r=1.0-np.sqrt(kappa)*sigma/beta
    nexc=sum(float(np.trace(rho@n[j].toarray()).real) for j in range(N))
    return r,nexc

N=4
J=1.0
kappa=0.2
gamma=0.0
omega=clean_mode_energy(N,1,J)
ops=build_ops(N)
betas=np.logspace(-5,-0.55,12)
h=1e-5
rows=[]
fig,ax=figure(COL_W, 2.4)
style={'opposite endpoint':dict(color=OI['blue'],mfc='white'),'center':dict(color=OI['vermilion'],mfc=OI['vermilion'])}
for label,site in [('opposite endpoint',N-1),('center',N//2)]:
    dlin=dS_deps_uniform(N,site,omega,J,kappa,gamma)
    ratios=[]
    for beta in betas:
        rp,npop=output_amplitude(N,site,+h,beta,J,kappa,gamma,omega,ops)
        rm,nmom=output_amplitude(N,site,-h,beta,J,kappa,gamma,omega,ops)
        dnum=(rp-rm)/(2*h)
        ratio=abs(dnum)/abs(dlin)
        ratios.append(ratio)
        rows.append((beta,label,site+1,abs(dnum),abs(dlin),ratio,0.5*(npop+nmom)))
    ax.semilogx(betas,ratios,'o-',ms=3.5,lw=1.0,label=label,**style[label])
ax.axhline(1.0,ls='--',lw=0.7,color='0.4')
ax.text(1.2e-5,1.02,'linear Green-function limit',fontsize=6.6,color='0.35',va='bottom')
ax.set_xlabel(r'coherent input amplitude $|\beta_{\rm in}|$')
ax.set_ylabel(r'$|\partial_\epsilon S|_{\rm Lindblad}/|\partial_\epsilon S|_{\rm linear}$')
ax.set_ylim(0,1.12)
ax.legend(loc='lower left')
tidy(ax)
fig.subplots_adjust(left=0.17,right=0.97,top=0.97,bottom=0.18)
save_figure(fig,'figS_lindblad_weak_drive_validation')
pd.DataFrame(rows,columns=['beta','position','site_1based','dS_lindblad_abs','dS_linear_abs','ratio','mean_excitation']).to_csv(DATA/'lindblad_validation.csv',index=False)
