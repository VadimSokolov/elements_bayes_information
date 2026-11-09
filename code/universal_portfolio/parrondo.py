import numpy as np, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent

T, f, p = 5000, 0.05, 0.51
N = 4001
w = (np.arange(N)+0.5)/N
logprior = -0.5*np.log(w) - 0.5*np.log(1-w)
logprior -= np.log(np.exp(logprior).sum())

def run(X, w=w, logprior=logprior):
    T = X.shape[0]
    logR = np.log(np.outer(X[:,0], w) + np.outer(X[:,1], 1-w))
    logS = np.cumsum(logR, axis=0)
    lp = logprior[None,:] + np.vstack([np.zeros((1,len(w))), logS[:-1]])
    mx = lp.max(axis=1, keepdims=True)
    P = np.exp(lp-mx); Z = P.sum(axis=1, keepdims=True); P /= Z
    what = P @ w
    wsd = np.sqrt(np.maximum(P @ (w**2) - what**2, 0))
    logSuniv = np.cumsum(np.log(what*X[:,0] + (1-what)*X[:,1]))
    le = logprior[None,:] + logS; m2 = le.max(axis=1, keepdims=True)
    logev = (m2 + np.log(np.exp(le-m2).sum(axis=1, keepdims=True))).ravel()
    return logS, logSuniv, logev, what, wsd

# pick an illustrative path where both single asset bets lose money
rng = np.random.default_rng(0)
for seed in range(1, 200):
    r = np.random.default_rng(seed)
    X = np.where(r.random((T,2)) < p, 1+f, 1-f)
    if seed == 51:
        break
print("seed", seed)
logS, logSuniv, logev, what, wsd = run(X)
istar = int(np.argmax(logS[-1])); wstar = w[istar]
regret = logS.max(axis=1) - logSuniv
bound = np.log(2) + 0.5*np.log(np.arange(1,T+1)+1)
ihalf = int(np.argmin(abs(w-0.5)))
out = dict(seed=seed, growth_pure=float(p*np.log(1+f)+(1-p)*np.log(1-f)),
 growth_half=float(p*p*np.log(1+f)+(1-p)**2*np.log(1-f)),
 S1=float(np.exp(np.log(X[:,0]).sum())), S2=float(np.exp(np.log(X[:,1]).sum())),
 S_half=float(np.exp(logS[-1,ihalf])), wstar=float(wstar), Sstar=float(np.exp(logS[-1,istar])),
 Suniv=float(np.exp(logSuniv[-1])), logSstar=float(logS[-1,istar]), logSuniv=float(logSuniv[-1]),
 logev=float(logev[-1]), identity_gap=float(abs(logSuniv[-1]-logev[-1])),
 regret=float(regret[-1]), bound=float(bound[-1]),
 what_final=float(what[-1]), sd_scaled=float(wsd[-1]*np.sqrt(T)))
print(json.dumps(out, indent=1))

# Monte Carlo over paths
M, Nm = 400, 1201
wm = (np.arange(Nm)+0.5)/Nm
lpm = -0.5*np.log(wm)-0.5*np.log(1-wm); lpm -= np.log(np.exp(lpm).sum())
res = np.zeros((M,5))
for m in range(M):
    r = np.random.default_rng(10_000+m)
    Xm = np.where(r.random((T,2)) < p, 1+f, 1-f)
    lS, lU, _, wh, _ = run(Xm, wm, lpm)
    res[m] = [np.log(Xm[:,0]).sum(), lS[-1, np.argmin(abs(wm-0.5))], lS[-1].max(), lU[-1], wh[-1]]
mc = dict(M=M, median_logS_asset=float(np.median(res[:,0])), median_logS_half=float(np.median(res[:,1])),
 median_logS_star=float(np.median(res[:,2])), median_logS_univ=float(np.median(res[:,3])),
 mean_growth_asset=float(res[:,0].mean()/T), mean_growth_half=float(res[:,1].mean()/T),
 mean_growth_star=float(res[:,2].mean()/T), mean_growth_univ=float(res[:,3].mean()/T),
 mean_regret=float((res[:,2]-res[:,3]).mean()), max_regret=float((res[:,2]-res[:,3]).max()),
 frac_asset_loses=float((res[:,0]<0).mean()), frac_univ_wins=float((res[:,3]>0).mean()),
 bound=float(np.log(2)+0.5*np.log(T+1)), mean_what=float(res[:,4].mean()))
print(json.dumps(mc, indent=1))
json.dump(dict(path=out, mc=mc), open(HERE/"results.json","w"), indent=1)
