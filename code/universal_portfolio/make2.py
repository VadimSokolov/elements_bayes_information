import numpy as np, sys, pathlib
HERE=pathlib.Path(__file__).resolve().parent; FIGS=HERE.parents[1]/"figs"; DATA=HERE/"data"
sys.path.insert(0,str(HERE))
from style import *
import matplotlib.pyplot as plt

# ---------- Shiller ----------
import pandas as pd
z=np.load(DATA/"shiller_out.npz", allow_pickle=True)
dates=pd.to_datetime(z["dates"]); yr=dates.year+dates.month/12
wp,wb,sd=z["w_plug"],z["w_bayes"],z["sd"]; m=~np.isnan(wp)
fig,axs=plt.subplots(1,2,figsize=(W,3.1))
a=axs[0]
a.fill_between(yr[m],wp[m]-2*sd[m],wp[m]+2*sd[m],color=C["light"],alpha=.45,lw=0,label="plug in, two standard errors")
a.plot(yr[m],wp[m],color=C["blue"],lw=1.2,ls=(0,(4,2)),label="plug in Kelly")
a.plot(yr[m],wb[m],color="k",lw=1.5,label="Bayes")
a.axhline(2.72,color=C["grey"],ls=(0,(5,2)),lw=1.0,label="full sample $f=2.72$")
a.axhline(2.0,color=C["red"],ls=":",lw=1.0,label="leverage cap")
a.set_ylim(0,7); a.set_xlim(1880,2025); a.set_xlabel("year"); a.set_ylabel("allocation to the index")
finish(a,ncol=2,above=True)
b=axs[1]
for k,c,l,s_ in [("cash",C["grey"],"cash","-"),("index",C["blue"],"index","-"),
                 ("plug_in_Kelly",C["orange"],"plug in Kelly",(0,(4,2))),("Bayes",C["green"],"Bayes","-"),
                 ("hindsight_Kelly",C["red"],"hindsight Kelly",(0,(5,2)))]:
    b.plot(yr,z[k],color=c,lw=1.3 if k in("Bayes",) else 1.0,ls=s_,label=l)
b.set_yscale("log"); b.set_xlim(1880,2025); b.set_xlabel("year"); b.set_ylabel("wealth from one dollar in 1881")
finish(b,loc="upper left")
fig.tight_layout(pad=0.5); fig.savefig(FIGS/"up_fig_shiller.pdf"); print("shiller")

# ---------- SP470 ----------
X=np.load(DATA/"X_sp.npy"); bstar=np.load(DATA/"bstar470.npy")
T,n=X.shape; lw=np.log(X).sum(axis=0); tt=np.arange(1,T+1)/252
fig,axs=plt.subplots(1,2,figsize=(W,2.8))
a=axs[0]
a.hist(np.exp(lw),bins=70,color=C["blue"],alpha=.7,edgecolor="white",linewidth=.3)
a.axvline(2.107,color=C["red"],lw=1.3,ls=(0,(5,2)),label="best CRP in hindsight, 2.11")
a.axvline(1.915,color=C["green"],lw=1.3,ls=":",label="equal weight CRP, 1.92")
a.set_xlim(0,8); a.set_xlabel("terminal wealth of a single stock"); a.set_ylabel("number of stocks")
finish(a,loc="upper right")
b=axs[1]
b.plot(tt,np.exp(np.cumsum(np.log(X[:,int(np.argmax(lw))]))),color=C["grey"],lw=.9,label="best single stock (NVDA)")
b.plot(tt,np.exp(np.cumsum(np.log(X.mean(axis=1)))),color="k",lw=1.8,label="equal weight CRP")
b.plot(tt,np.exp(np.cumsum(np.log(X@bstar))),color=C["red"],lw=1.3,ls=(0,(1,1.5)),label="best CRP in hindsight")
b.set_yscale("log"); b.set_xlabel("years from February 2013"); b.set_ylabel("wealth")
finish(b,loc="upper left")
fig.tight_layout(pad=0.5); fig.savefig(FIGS/"up_fig_sp470.pdf"); print("sp470")

# ---------- Thorp comparison ----------
s_,Tn,r=0.2016,34,0.06; mhat,vlik=0.18,0.2016**2/34
def post(m0,tau):
    pv=1/(1/vlik+1/tau**2); return pv*(mhat/vlik+m0/tau**2), pv
def dens(x,m,v): return np.exp(-(x-m)**2/(2*v))/np.sqrt(2*np.pi*v)
fig,axs=plt.subplots(1,2,figsize=(W,2.8))
a=axs[0]; f=np.linspace(-1,10,900)
for nm,m0,tau,col in [("plug in, no prior",None,None,C["blue"]),("prior $N(5\\%,5\\%^2)$",.05,.05,C["orange"]),
                      ("prior $N(5\\%,3\\%^2)$",.05,.03,C["green"]),("prior $N(5\\%,2\\%^2)$",.05,.02,C["purple"])]:
    m,v=(mhat,vlik) if m0 is None else post(m0,tau)
    a.plot(f,dens(f*s_**2,m,v)*s_**2,color=col,lw=1.3,label=nm)
a.axvline(6.22,color=C["red"],lw=1.1,ls=(0,(5,2))); a.axvline(2.0,color=C["grey"],lw=1.0,ls=":")
a.annotate("Thorp $f^\\star=6.22$",(6.40,0.55),color=C["red"],fontsize=8.5,rotation=90,va="center")
a.annotate("Thorp's choice, $f=2$",(2.18,0.42),color=C["grey"],fontsize=8.5,rotation=90,va="center")
a.set_xlim(-1,10); a.set_ylim(0,1.18); a.set_xlabel(r"Kelly fraction $f=(\mu-r)/\sigma^2$"); a.set_ylabel("posterior density")
finish(a,ncol=2,above=True)
b=axs[1]; taus=np.linspace(0.004,0.15,500)
b.plot(taus*100,[post(0.05,t)[0]/(s_**2+post(0.05,t)[1]) for t in taus],color=C["blue"],lw=1.5,label="Bayes rule, prior mean 5 percent")
b.axhline(6.22,color=C["red"],lw=1.1,ls=(0,(5,2)),label=r"Thorp $f^\star=6.22$")
b.axhline(4.43,color=C["green"],lw=1.0,ls="-.",label="plug in, $f=4.43$")
b.axhline(2.0,color=C["grey"],lw=1.0,ls=":",label="Thorp's choice, $f=2$")
b.set_xscale("log"); b.set_xlabel(r"prior standard deviation $\tau$ (percent)"); b.set_ylabel("allocation $f$")
finish(b,loc="lower right")
fig.tight_layout(pad=0.5); fig.savefig(FIGS/"up_fig_thorp.pdf"); print("thorp")

# ---------- growth and risk ----------
fig,axs=plt.subplots(1,2,figsize=(W,2.8))
a=axs[0]; fs=np.linspace(0,9,500)
for exc,col,lab in [(0.08,C["blue"],"excess drift 8 percent"),(0.14,C["green"],"14 percent (Thorp)"),(0.18,C["red"],"18 percent (realised)")]:
    a.plot(fs,exc*fs-0.5*s_**2*fs**2,color=col,lw=1.3,label=lab)
a.axhline(0,color="k",lw=.7)
a.axvline(6.22,color=C["red"],lw=1.0,ls=(0,(5,2))); a.axvline(2.02,color=C["blue"],lw=1.0,ls=":")
a.annotate("Thorp $f^\\star$",(6.35,-0.17),color=C["red"],fontsize=8.5)
a.annotate("Bayes",(1.3,-0.17),color=C["blue"],fontsize=8.5)
a.set_ylim(-0.22,0.52); a.set_xlabel("allocation $f$"); a.set_ylabel(r"growth rate $g_\infty(f)-r$")
finish(a,loc="upper right")
b=axs[1]; c=np.linspace(0.05,1.6,500)
b.plot(c,c*(2-c),color=C["blue"],lw=1.4,label="growth relative to the maximum")
b.plot(c,0.5**(2/c-1),color=C["red"],lw=1.4,label="probability of ever halving")
for x,lab in ((0.5,"half Kelly"),(1.0,"full Kelly")):
    b.axvline(x,color=C["grey"],lw=.8,ls=":"); b.annotate(lab,(x+0.02,0.03),color=C["grey"],fontsize=8.5)
b.set_xlabel("fraction of Kelly, $c$"); b.set_ylabel("probability or ratio")
finish(b,loc="upper left")
fig.tight_layout(pad=0.5); fig.savefig(FIGS/"up_fig_growth.pdf"); print("growth")

# ---------- cost of learning ----------
fig,ax=plt.subplots(figsize=(0.62*W,2.6))
Ts=np.logspace(0,2.3,400)
for cc,col,lab in [(0.05**2,C["blue"],r"$\tau=5\%$"),(0.03**2,C["green"],r"$\tau=3\%$"),(0.02**2,C["purple"],r"$\tau=2\%$")]:
    ax.plot(Ts,0.5*np.log(1+cc*Ts/s_**2),color=col,lw=1.3,label="cost of learning, "+lab)
ax.plot(Ts,np.log(2)+0.5*np.log(Ts+1),color=C["red"],lw=1.2,ls=(0,(5,2)),label=r"Cover bound")
ax.plot(Ts,0.5*np.log(Ts),color=C["grey"],lw=1.0,ls=":",label=r"$\frac{1}{2}\log T$")
ax.set_xscale("log"); ax.set_xlabel("horizon $T$ (years)"); ax.set_ylabel("log wealth")
finish(ax,loc="upper left")
fig.tight_layout(pad=0.4); fig.savefig(FIGS/"up_fig_learning.pdf"); print("learning")
