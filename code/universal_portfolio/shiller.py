import pandas as pd, numpy as np, pathlib
DATA=pathlib.Path(__file__).resolve().parent/"data"
d=pd.read_csv(DATA/"shiller.csv")
d=d[(d.Dividend>0)&(d.SP500>0)&(d["Long Interest Rate"]>0)].reset_index(drop=True)
P=d.SP500.values; D=d.Dividend.values; rr=d["Long Interest Rate"].values/100
R=(P[1:]+D[:-1]/12)/P[:-1]-1; rf=(1+rr[:-1])**(1/12)-1; x=R-rf
dates=pd.to_datetime(d.Date.values[:len(x)])
T=len(x); burn=120; cap=2.0
m0,tau=0.05/12, 0.02/12          # prior on the monthly excess drift
w_plug=np.full(T,np.nan); w_bayes=np.full(T,np.nan); sd_plug=np.full(T,np.nan)
for t in range(burn,T):
    y=x[:t]; s2=y.var(ddof=1); mhat=y.mean(); v=s2/t
    w_plug[t]=mhat/s2
    sd_plug[t]=np.sqrt(v)/s2
    pv=1/(1/v+1/tau**2); pm=pv*(mhat/v+m0/tau**2)
    w_bayes[t]=pm/(s2+pv)
def wealth(w):
    g=np.zeros(T)
    for t in range(burn,T):
        a=np.clip(w[t],0,cap) if not np.isnan(w[t]) else 0.0
        g[t]=np.log1p(rf[t]+a*x[t])
    return np.exp(np.cumsum(g))
res={}
res["index"]=wealth(np.ones(T)); res["plug in Kelly"]=wealth(w_plug); res["Bayes"]=wealth(w_bayes)
res["cash"]=np.exp(np.cumsum(np.where(np.arange(T)>=burn,np.log1p(rf),0)))
res["hindsight Kelly"]=wealth(np.full(T, x.mean()/x.var(ddof=1)))
for k,v in res.items(): print(f"{k:16s} terminal {v[-1]:12.1f}   log {np.log(v[-1]):6.2f}   ann {np.log(v[-1])/((T-burn)/12)*100:5.2f}%")
print("\nfinal allocations: plug in %.2f (se %.2f), Bayes %.2f" % (w_plug[-1], sd_plug[-1], w_bayes[-1]))
for yr in (1900,1930,1960,1990,2020):
    i=np.argmax(dates.year>=yr)
    print(f"{yr}: plug in {w_plug[i]:5.2f} +/- {sd_plug[i]:4.2f}   Bayes {w_bayes[i]:5.2f}")
np.savez(DATA/"shiller_out.npz", x=x, rf=rf, w_plug=w_plug, w_bayes=w_bayes, sd=sd_plug,
         dates=dates.values, **{k.replace(" ","_"):v for k,v in res.items()})
