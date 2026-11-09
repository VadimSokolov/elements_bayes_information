import numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


M,Nm=200,1201
wm=(np.arange(Nm)+0.5)/Nm
lpm=-0.5*np.log(wm)-0.5*np.log(1-wm); lpm-=np.log(np.exp(lpm).sum())
sds=[];means=[]
for m in range(M):
    r=np.random.default_rng(20_000+m); Xm=np.where(r.random((T,2))<p,1+f,1-f)
    _,_,_,wh,ws=run(Xm,wm,lpm); sds.append(ws[-1]); means.append(wh[-1])
J=2*p*(1-p)*(2*f)**2
print("avg post sd*sqrtT",np.mean(sds)*np.sqrt(T),"sd of means*sqrtT",np.std(means)*np.sqrt(T),"Jinvsqrt",J**-.5,"mean",np.mean(means))
