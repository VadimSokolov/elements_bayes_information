import numpy as np, json, pathlib
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
exec(open(pathlib.Path(__file__).resolve().parent/'make3.py').read().split('runs={}')[0].replace('from style import *','').replace('import matplotlib.pyplot as plt',''))
out={}
for nm,a in [("half",(0.5,.5,.5)),("d4",(4,.5,.5)),("d20",(20,.5,.5))]:
    q=cell*B[:,0]**(a[0]-1)*B[:,1]**(a[1]-1)*B[:,2]**(a[2]-1); q/=q.sum()
    bh,ls=universal(X,B,q)
    lam=(bh[:,1]+bh[:,2])/(bstar[1]+bstar[2])
    out[nm]=dict(logS=ls[-1],logS100=ls[99],frac={t:round(float(lam[t-1]),2) for t in [1,100,500,1000,3000,5000,10000]})
counts=np.zeros((T,38)); counts[np.arange(T),spins]=1; counts=counts.cumsum(0); tt=np.arange(1,T+1)
fire=counts.max(1)>tt/36+0.49*np.sqrt(tt)
f=np.where(fire&(tt>=100))[0]; out['first_fire_ge100']=int(f[0]+1) if len(f) else None
out['first_fire_pocket']=int(counts[f[0]].argmax()) if len(f) else None
nf=np.where(~fire)[0]; out['fires_every_n_from']=int(nf[-1]+2)
out['crit_T']=T/36+0.49*np.sqrt(T); out['max_other']=int(counts[-1,2:].max())
# regret constants
for m,TT in [(2,5000),(38,10000)]:
    out[f'asym_m{m}']=(m-1)/2*np.log(TT/(2*np.pi))+m/2*np.log(np.pi)-gammaln(m/2)
    out[f'bound_m{m}']=np.log(2)+(m-1)/2*np.log(TT+1)
# gamma 2 on wheel, symmetric stakes s on two pockets, reserve 1-2s
p=np.array([pfav,pfav,1-2*pfav])
def rel(s): return np.array([1-2*s+pay*s,1-2*s+pay*s,1-2*s])
for g,nm in [(1,'kelly'),(2,'g2')]:
    obj=(lambda s:-(p@np.log(rel(s)))) if g==1 else (lambda s:(p@(1/rel(s))))
    s=minimize_scalar(obj,bounds=(0,0.05),method='bounded',options={'xatol':1e-12}).x
    out[nm]=dict(s=s,res=1-2*s,g=p@np.log(rel(s)),sd=np.sqrt(p@np.log(rel(s))**2-(p@np.log(rel(s)))**2)*np.sqrt(1))
s=out['kelly']['s']/2; out['half']=dict(s=s,g=p@np.log(rel(s)),sd=np.sqrt(p@np.log(rel(s))**2-(p@np.log(rel(s)))**2))
for c in [1,.5,.25]:
    x,y=.5,2; e=2/c-1; out[f'thorp{c}']=(x**e,(1-x**e)/(1-(x/y)**e),c*(2-c))
print(json.dumps(out,indent=1,default=float))
