import numpy as np, sys, pathlib
HERE=pathlib.Path(__file__).resolve().parent; FIGS=HERE.parents[1]/"figs"
sys.path.insert(0,str(HERE))
from style import *
import matplotlib.pyplot as plt
def universal(X,B,q):
    w=q.copy(); bh=np.empty((len(X),B.shape[1])); lr=np.empty(len(X))
    for t in range(len(X)):
        bh[t]=w@B; r=B@X[t]; w=w*r; Z=w.sum(); w/=Z; lr[t]=np.log(Z)
    return bh,np.cumsum(lr)
T,pay,pfav=10000,36.0,1/30; prest=(1-2*pfav)/36
spins=np.random.default_rng(7).choice(38,size=T,p=np.array([pfav,pfav]+[prest]*36))
X=np.zeros((T,3)); X[:,0]=1.0; X[spins==0]=[1,pay,0]; X[spins==1]=[1,0,pay]
b0=(1-2*pfav)/(1-2/pay); bstar=np.array([b0,pfav-b0/pay,pfav-b0/pay])
N=300; e=(np.arange(N+1)/N)**3; mid=(e[:-1]+e[1:])/2; wid=np.diff(e)
M1,M2=np.meshgrid(mid,mid,indexing="ij"); W1,W2=np.meshgrid(wid,wid,indexing="ij")
k=(M1+M2)<1; b1,b2,cell=M1[k],M2[k],(W1*W2)[k]; B=np.column_stack([1-b1-b2,b1,b2])
runs={}
for nm,a in [("half",(0.5,.5,.5)),("cash",(20,.5,.5))]:
    q=cell*B[:,0]**(a[0]-1)*B[:,1]**(a[1]-1)*B[:,2]**(a[2]-1); q/=q.sum()
    runs[nm]=universal(X,B,q)
tt=np.arange(1,T+1)
fig,axs=plt.subplots(1,2,figsize=(W,2.8))
a=axs[0]
a.plot(tt,np.exp(np.cumsum(np.log(X@bstar))),color=C["grey"],ls=(0,(5,2)),lw=1.1,label="Kelly rule, bias known")
a.plot(tt,np.exp(runs["cash"][1]),color=C["blue"],lw=1.2,label=r"universal, $\mathrm{Dir}(20,\frac{1}{2},\frac{1}{2})$")
a.plot(tt,np.exp(runs["half"][1]),color=C["green"],lw=1.2,label=r"universal, $\mathrm{Dir}(\frac{1}{2},\frac{1}{2},\frac{1}{2})$")
a.axhline(1,color=C["red"],lw=.9,ls=":",label="cash")
a.set_yscale("log"); a.set_xlabel("spin $t$"); a.set_ylabel("wealth"); finish(a,loc="upper left")
b=axs[1]; bh=runs["half"][0]
b.plot(tt,bh[:,1],color=C["blue"],lw=1.0,label=r"$\hat b_{t,1}$")
b.plot(tt,bh[:,2],color=C["orange"],lw=1.0,label=r"$\hat b_{t,2}$")
b.axhline(bstar[1],color=C["grey"],ls=(0,(5,2)),lw=1.1,label="Kelly stake, 0.00588")
b.set_xscale("log"); b.set_yscale("log"); b.set_ylim(1e-3,0.6)
b.set_xlabel("spin $t$"); b.set_ylabel("stake on each favoured pocket"); finish(b,loc="lower left")
fig.tight_layout(pad=0.5); fig.savefig(FIGS/"up_fig_roulette.pdf"); print("roulette done")
