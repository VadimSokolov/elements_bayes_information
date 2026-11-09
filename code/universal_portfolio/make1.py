import numpy as np, sys, pathlib
HERE=pathlib.Path(__file__).resolve().parent; FIGS=HERE.parents[1]/"figs"; DATA=HERE/"data"
sys.path.insert(0,str(HERE))
from style import *
import matplotlib.pyplot as plt

def universal(X,B,q):
    w=q.copy(); bh=np.empty((len(X),B.shape[1])); lr=np.empty(len(X)); sd=np.empty(len(X))
    for t in range(len(X)):
        m=w@B; bh[t]=m; sd[t]=np.sqrt(max(w@(B[:,0]**2)-m[0]**2,0))
        r=B@X[t]; w=w*r; Z=w.sum(); w/=Z; lr[t]=np.log(Z)
    return bh,np.cumsum(lr),sd

# ---------------- Parrondo ----------------
T,f,p=5000,0.05,0.51
X=np.where(np.random.default_rng(51).random((T,2))<p,1+f,1-f)
N=4001; w=(np.arange(N)+0.5)/N
B=np.column_stack([w,1-w]); q=(w*(1-w))**-0.5; q/=q.sum()
bh,lS,sd=universal(X,B,q)
grid=np.array([np.log(o*X[:,0]+(1-o)*X[:,1]).sum() for o in w]); k=grid.argmax()
t=np.arange(1,T+1)

fig,ax=plt.subplots(figsize=(W,3.1))
ax.plot(t,np.exp(np.cumsum(np.log(X[:,0]))),lw=.8,color=C["blue"],ls=(0,(1,1.5)),label="asset 1")
ax.plot(t,np.exp(np.cumsum(np.log(X[:,1]))),lw=.9,color=C["red"],ls=(0,(3,1,1,1)),label="asset 2")
ax.plot(t,np.exp(np.log(0.5*X[:,0]+0.5*X[:,1]).cumsum()),lw=1.1,color=C["purple"],ls=(0,(6,2)),label="equal weight CRP")
ax.plot(t,np.exp(np.cumsum(np.log(w[k]*X[:,0]+(1-w[k])*X[:,1]))),lw=1.0,ls=(0,(5,2)),color=C["grey"],label="best ex post CRP")
ax.plot(t,np.exp(lS),lw=1.8,color="k",label="universal (posterior mean)")
ax.set_yscale("log"); ax.set_xlabel("period $t$"); ax.set_ylabel("wealth $S_t$"); ax.set_xlim(0,T)
finish(ax,ncol=5,above=True); fig.savefig(FIGS/"up_fig_wealth.pdf")

fig,axs=plt.subplots(1,2,figsize=(W,2.7))
a=axs[0]; a.plot(w,np.exp(grid),color=C["blue"],lw=1.4)
a.axhline(np.exp(lS[-1]),color=C["green"],ls="--",lw=1.1)
a.axvline(w[k],color=C["grey"],ls=":",lw=.9)
a.annotate("universal",(0.03,np.exp(lS[-1])),xytext=(0.03,np.exp(lS[-1])+0.9),color=C["green"],fontsize=9)
a.set_xlabel(r"weight $\omega$ on asset 1"); a.set_ylabel(r"terminal wealth $S_T(\omega)$"); finish(a,legend=False)
b=axs[1]
b.plot(t,grid.max()*0+np.maximum.accumulate(np.zeros(T))+ (np.array([np.log(w[k]*X[:i+1,0]+(1-w[k])*X[:i+1,1]).sum() for i in range(0)]) if False else 0),alpha=0)
reg=np.maximum.accumulate(np.zeros(T))
logS_all=np.cumsum(np.log(np.outer(X[:,0],w)+np.outer(X[:,1],1-w)),axis=1*0+1) if False else None
LS=np.cumsum(np.log(np.outer(X[:,0],w)+np.outer(X[:,1],1-w)),axis=0)
reg=LS.max(axis=1)-lS
b.plot(t,reg,color=C["blue"],lw=1.2,label="realised regret")
b.plot(t,np.log(2)+0.5*np.log(t+1),color=C["red"],ls="--",lw=1.1,label=r"$\log 2+\frac{1}{2}\log(T+1)$")
b.plot(t,0.5*np.log(t),color=C["grey"],ls=":",lw=1.0,label=r"$\frac{1}{2}\log T$")
b.set_xscale("log"); b.set_xlabel("period $t$"); b.set_ylabel("regret"); finish(b,loc="upper left")
fig.tight_layout(pad=0.4); fig.savefig(FIGS/"up_fig_regret.pdf")

fig,ax=plt.subplots(figsize=(W,2.5))
ax.fill_between(t,bh[:,0]-2*sd,bh[:,0]+2*sd,color=C["light"],alpha=.45,lw=0,label=r"posterior mean $\pm$ 2 sd")
ax.plot(t,bh[:,0],color=C["blue"],lw=1.3,label=r"$\hat\omega_{t+1}=\mathbb{E}[\omega\mid\mathcal{F}_t]$")
ax.axhline(0.5,color=C["red"],ls="--",lw=1.0,label=r"$\omega^\star=1/2$")
ax.set_xscale("log"); ax.set_ylim(0,1); ax.set_xlabel("period $t$"); ax.set_ylabel(r"$\hat\omega$")
finish(ax,ncol=3,above=True); fig.savefig(FIGS/"up_fig_posterior.pdf")
print("parrondo done")

# ---------------- real pair ----------------
d=np.load(DATA/"real_pair.npz")
Y,lS2,grid2,w2=d["Y"],d["lS"],d["grid"],d["w"]
tt=np.arange(1,len(Y)+1)/252
fig,axs=plt.subplots(1,2,figsize=(W,2.7))
a=axs[0]
a.plot(tt,np.exp(np.cumsum(np.log(Y[:,0]))),lw=.9,color=C["blue"],ls=(0,(1,1.5)),label="Newmont Mining")
a.plot(tt,np.exp(np.cumsum(np.log(Y[:,1]))),lw=1.0,color=C["red"],ls=(0,(3,1,1,1)),label="Williams Companies")
a.plot(tt,np.exp(np.log(0.5*Y[:,0]+0.5*Y[:,1]).cumsum()),lw=1.4,color=C["purple"],ls=(0,(6,2)),label="equal weight CRP")
a.plot(tt,np.exp(lS2),lw=1.8,color="k",label="universal")
a.axhline(1,color="k",lw=.6)
a.set_yscale("log"); a.set_xlabel("years from February 2013"); a.set_ylabel("wealth"); finish(a,loc="lower left")
b=axs[1]; b.plot(w2,np.exp(grid2),color=C["blue"],lw=1.4)
b.axhline(np.exp(lS2[-1]),color=C["green"],ls="--",lw=1.1)
b.axvline(w2[grid2.argmax()],color=C["grey"],ls=":",lw=.9); b.axhline(1,color="k",lw=.6)
b.set_xlabel("weight on Newmont Mining"); b.set_ylabel("terminal wealth")
b.annotate("universal",(0.02,np.exp(lS2[-1])-0.035),color=C["green"],fontsize=9)
finish(b,legend=False)
fig.tight_layout(pad=0.4); fig.savefig(FIGS/"up_fig_realpair.pdf")
print("realpair done")
