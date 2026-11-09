import numpy as np, json, pathlib
DATA=pathlib.Path(__file__).resolve().parent/"data"
X=np.load(DATA/"X_sp.npy"); nm=list(np.load(DATA/"names_sp.npy",allow_pickle=True))
i,j=nm.index("NEM"), nm.index("WMB")
Y=np.column_stack([X[:,i],X[:,j]]); T=len(Y)
N=4001; w=(np.arange(N)+0.5)/N
q=(w*(1-w))**-0.5; q/=q.sum()
B=np.column_stack([w,1-w])
def universal(Y,B,q):
    wt=q.copy(); bh=np.empty(len(Y)); lr=np.empty(len(Y)); sd=np.empty(len(Y))
    for t in range(len(Y)):
        m=float(wt@B[:,0]); bh[t]=m; sd[t]=np.sqrt(max(wt@(B[:,0]**2)-m*m,0))
        r=B@Y[t]; wt=wt*r; Z=wt.sum(); wt/=Z; lr[t]=np.log(Z)
    return bh, np.cumsum(lr), sd
bh,lS,sd=universal(Y,B,q)
grid=np.array([np.log(o*Y[:,0]+(1-o)*Y[:,1]).sum() for o in w])
k=grid.argmax()
out=dict(T=T, NEM=float(np.exp(np.log(Y[:,0]).sum())), WMB=float(np.exp(np.log(Y[:,1]).sum())),
  half=float(np.exp(np.log(0.5*Y[:,0]+0.5*Y[:,1]).sum())), wstar=float(w[k]), best=float(np.exp(grid[k])),
  univ=float(np.exp(lS[-1])), logbest=float(grid[k]), loguniv=float(lS[-1]),
  regret=float(grid[k]-lS[-1]), bound=float(np.log(2)+0.5*np.log(T+1)),
  bhat=float(bh[-1]), postsd=float(sd[-1]), sqrtT_sd=float(sd[-1]*np.sqrt(T)))
# market information at the optimum
o=w[k]; J=np.mean((Y[:,0]-Y[:,1])**2/(o*Y[:,0]+(1-o)*Y[:,1])**2)
out["J"]=float(J); out["Jinv_sqrt"]=float(J**-0.5)
print(json.dumps(out,indent=1))
np.savez(DATA/"real_pair.npz", Y=Y, bh=bh, lS=lS, sd=sd, grid=grid, w=w)
