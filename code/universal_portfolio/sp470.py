import numpy as np, json, pathlib
DATA=pathlib.Path(__file__).resolve().parent/"data"
X=np.load(DATA/"X_sp.npy"); T,n=X.shape
def logopt(Z, iters=400, b=None):
    if b is None: b=np.full(Z.shape[1],1.0/Z.shape[1])
    for _ in range(iters):
        d=Z@b; b=b*((Z/d[:,None]).mean(axis=0))
        b=np.maximum(b,0); b/=b.sum()
    return b
bstar=logopt(X)
lstar=np.log(X@bstar).sum()
lw=np.log(X).sum(axis=0)
res=dict(T=T, n=n,
  best_CRP=float(np.exp(lstar)), log_best=float(lstar),
  n_active=int((bstar>1e-4).sum()), top=[(int(i),float(bstar[i])) for i in np.argsort(-bstar)[:5]],
  equal_weight=float(np.exp(np.log(X.mean(axis=1)).sum())),
  best_stock=float(np.exp(lw.max())), median_stock=float(np.exp(np.median(lw))),
  bound=float(np.log(2)+(n-1)/2*np.log(T+1)))
# walk forward shrinkage toward equal weights, empirical optimum refreshed monthly
bu=np.full(n,1.0/n)
for a0 in [0, 250, 1000, 5000, 1e9]:
    b=bu.copy(); logS=0.0; bemp=bu.copy()
    for t in range(T):
        if t>=252 and t%21==0:
            bemp=logopt(X[:t], iters=60, b=bemp)
        lam = 0.0 if a0>=1e9 else (t/(t+a0) if t>=252 else 0.0)
        b=lam*bemp+(1-lam)*bu
        logS+=np.log(b@X[t])
    res[f"shrink_a0_{int(a0)}"]=dict(logS=float(logS), wealth=float(np.exp(logS)))
    print(a0, np.exp(logS))
# plug in mean variance leverage
Rt=X-1.0; mu=Rt.mean(axis=0); C=np.cov(Rt.T)
wmv=np.linalg.solve(C, mu)
res["mv_gross_leverage"]=float(np.abs(wmv).sum()); res["mv_net"]=float(wmv.sum())
res["cond_number"]=float(np.linalg.cond(C))
print(json.dumps(res,indent=1))
json.dump(res, open(DATA/"sp470.json","w"), indent=1)
np.save(DATA/"bstar470.npy", bstar)
