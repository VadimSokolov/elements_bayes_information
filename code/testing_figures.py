import pathlib
HERE = pathlib.Path(__file__).resolve().parent
FIGS = pathlib.Path(__file__).resolve().parents[1] / "figs"
import numpy as np, matplotlib, math, json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
NAVY,RUST,TEAL,GOLD,GRAY,PLUM="#26456e","#c0562f","#2a8f82","#d4a11e","#6b6b6b","#7b5ea7"
plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"serif","font.serif":["CMU Serif","DejaVu Serif"],
 "mathtext.fontset":"cm","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
out={}
# ---------- Figure A: three scales for sums of Exp(1)
ns=np.unique(np.logspace(1,6,60).astype(int))
import mpmath
mpmath.mp.dps=40
def glogsf(n,x):  # log P(Gamma(n,1) > n + x sqrt n), exact via mpmath
    return float(mpmath.log(mpmath.gammainc(n, n+x*math.sqrt(n), mpmath.inf, regularized=True)))
def nlogsf(x): return float(mpmath.log(mpmath.erfc(x/mpmath.sqrt(2))/2))
def ratio(n,x):
    return glogsf(n,x)-nlogsf(x)
regimes=[("CLT scale: $x=2$",lambda n:2.0,NAVY),("moderate: $x=\\sqrt{2\\log n}$",lambda n:math.sqrt(2*math.log(n)),TEAL),("large: $x=0.3\\sqrt{n}$",lambda n:0.3*math.sqrt(n),RUST)]
fig,ax=plt.subplots(1,2,figsize=(9.2,3.4))
for lab,xf,c in regimes:
    r=[ratio(n,xf(n)) for n in ns]; ax[0].plot(ns,np.exp(np.clip(r,-700,700)),color=c,lw=2,label=lab)
    out.setdefault('ratio',{})[lab]={int(n):float(np.exp(ratio(n,xf(n)))) for n in (100,10**4,10**6)}
ax[0].axhline(1,color=GRAY,lw=.8,ls=":"); ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_ylim(0.5,1e3)
ax[0].set_xlabel("$n$"); ax[0].set_ylabel("exact tail / Gaussian tail"); ax[0].set_title("(a) relative error of the normal tail",fontsize=10)
ax[0].legend(frameon=False,fontsize=7.5,loc="upper left")
for lab,xf,c in regimes[1:]:
    q=[glogsf(n,xf(n))/nlogsf(xf(n)) for n in ns]
    ax[1].plot(ns,q,color=c,lw=2,label=lab)
    out.setdefault('lograte',{})[lab]={int(n):glogsf(n,xf(n))/nlogsf(xf(n)) for n in (100,10**4,10**6)}
I=lambda e: e-math.log(1+e)
ax[1].axhline(1,color=GRAY,lw=.8,ls=":"); ax[1].axhline(I(.3)/(.3**2/2),color=RUST,lw=.8,ls="--")
ax[1].set_xscale('log'); ax[1].set_ylim(0.6,1.2); ax[1].set_xlabel("$n$"); ax[1].set_ylabel("$\\log P_{\\mathrm{exact}}\\,/\\,\\log \\bar\\Phi(x)$")
ax[1].set_title("(b) ratio of exponents",fontsize=10); ax[1].legend(frameon=False,fontsize=7.5,loc="lower right")
out['cramer_ratio_LD']=I(.3)/(.3**2/2)
fig.tight_layout(); fig.savefig(FIGS / "tst_three_scales.pdf")
# ---------- Figure B: sequential evidence
rng=np.random.default_rng(7)
p0,p1=.5,.65; up,dn=math.log(p1/p0),math.log((1-p1)/(1-p0)); mu=p1*up+(1-p1)*dn; s2=p1*(1-p1)*(up-dn)**2; lb=math.log(20)
x=rng.random((500,200))<p1; S=np.cumsum(np.where(x,up,dn),1); t=np.arange(1,201)
fig,ax=plt.subplots(1,3,figsize=(10,3.2))
for i in range(60): ax[0].plot(t,S[i],color=NAVY,lw=.5,alpha=.35)
ax[0].plot(t,mu*t,color=GOLD,lw=2,label="$t\\,D(P_1\\|P_0)$"); ax[0].axhline(lb,color=RUST,lw=1.5,label="$\\log 20$")
ax[0].set_xlabel("observation $t$"); ax[0].set_ylabel("$\\log E_t$"); ax[0].set_title("(a) evidence under $H_1$",fontsize=10); ax[0].legend(frameon=False,fontsize=8)
# stopping time distribution b=20 from 200k reps
R=200_000; Sx=np.zeros(R); tau=np.full(R,-1)
for tt in range(1,3000):
    Sx+=np.where(rng.random(R)<p1,up,dn); m=(tau<0)&(Sx>=lb); tau[m]=tt
    if (tau>=0).all(): break
ax[1].hist(tau,bins=np.arange(0,301,6),density=True,color=TEAL,alpha=.6,label="simulated $\\tau_{20}$")
g=np.linspace(1,300,400); ax[1].plot(g,stats.norm.pdf(g,lb/mu,math.sqrt(s2*lb/mu**3)),color=NAVY,lw=1.6,label="Gaussian, MDP scale")
# inverse Gaussian first passage approximation (Brownian drift mu var s2)
lam=lb**2/s2; mig=lb/mu
ax[1].plot(g,stats.invgauss.pdf(g,mig/lam,scale=lam),color=RUST,lw=1.6,ls="--",label="inverse Gaussian")
ax[1].axvline(lb/mu,color=GRAY,lw=.8,ls=":"); ax[1].set_xlabel("stopping time"); ax[1].set_title("(b) when the test stops, $b=20$",fontsize=10); ax[1].legend(frameon=False,fontsize=7.5)
out['tau20']=dict(mean=float(tau.mean()),median=float(np.median(tau)),pred=lb/mu,sd=float(tau.std()))
J=json.load(open(HERE/'testing_sim_numbers.json'))
vals=[J['typeI']['LR'],J['typeI']['KT'],J['typeI']['ML']]
ax[2].bar([0,1,2],vals,color=[NAVY,TEAL,RUST],width=.6); ax[2].axhline(.05,color=GRAY,ls="--",lw=1)
ax[2].set_xticks([0,1,2]); ax[2].set_xticklabels(["likelihood\nratio","prequential\nmixture","maximum\nlikelihood"],fontsize=8)
for i,v in enumerate(vals): ax[2].text(i,v+.006,f"{100*v:.1f}%",ha='center',fontsize=8)
ax[2].set_ylabel("false rejection rate"); ax[2].set_title("(c) optional stopping under $H_0$",fontsize=10)
fig.tight_layout(); fig.savefig(FIGS / "tst_sequential.pdf")
# ---------- Figure C: misspecification
pt,pa=.55,.8; up2,dn2=math.log(pa/p0),math.log((1-pa)/(1-p0)); d=pt*up2+(1-pt)*dn2
x=rng.random((500,300))<pt; Sm=np.cumsum(np.where(x,up2,dn2),1); t3=np.arange(1,301)
cr=(Sm>=lb).any(1)
fig,ax=plt.subplots(figsize=(6,3.2))
for i in range(80):
    if not cr[i]: ax.plot(t3,Sm[i],color="#bbbbbb",lw=.6,alpha=.8,zorder=1)
for i in range(80):
    if cr[i]: ax.plot(t3,Sm[i],color="#111111",lw=1.1,alpha=.9,zorder=2)
ax.plot(t3,d*t3,color=GOLD,lw=2,label="drift $t\\,[D(P\\|P_0)-D(P\\|P_1)]$"); ax.axhline(lb,color=GRAY,lw=1.2,ls="--",label="$\\log 20$")
ax.set_xlabel("observation $t$"); ax.set_ylabel("$\\log E_t$"); ax.legend(frameon=True,facecolor="white",edgecolor="none",framealpha=0.95,fontsize=8,loc="lower left"); ax.set_ylim(-60,8)
fig.tight_layout(); fig.savefig(FIGS / "tst_misspec.pdf"); out['misspec_cross_fig']=int(cr.sum())
# ---------- Figure D: Rubin Sethuraman
R2=json.load(open(HERE/'testing_rs_numbers.json'))
fig,ax=plt.subplots(1,3,figsize=(10,3.2))
cols={(1,0):NAVY,(1,2):RUST,(2,0):TEAL,(3,0):PLUM}
for (k,l),c in cols.items():
    rows=[r for r in R2['boundary'] if r['k']==k and r['lam']==l]; ln=[math.log(r['n']) for r in rows]
    ax[0].plot(ln,[r['tsq'] for r in rows],'o',color=c,ms=4)
    L=np.linspace(4,14,50); ax[0].plot(L,(k+l)*L,color=c,lw=1,ls=':')
    tt=[]; 
    for Lv in L:
        z=(k+l)*Lv
        for _ in range(50): z=(k+l)*Lv-k*math.log(2*math.pi)-l*math.log(z)
        tt.append(z)
    ax[0].plot(L,tt,color=c,lw=1.5,label=f"$k={k},\\ \\lambda={l}$")
ax[0].set_xlabel("$\\log n$"); ax[0].set_ylabel("$n\\|\\hat\\theta\\|^2$ at the boundary"); ax[0].set_title("(a) the Bayes boundary",fontsize=10); ax[0].legend(frameon=False,fontsize=7.5)
rk=R2['risk']; ln=[math.log(r['n']) for r in rk]
ax[1].plot(ln,[r['typeII'] for r in rk],'o-',color=RUST,label="missed signal")
ax[1].plot(ln,[r['typeI'] for r in rk],'s-',color=NAVY,label="false alarm")
L=np.linspace(4.6,13.8,40); ax[1].plot(L,0.798*np.sqrt(L/np.exp(L)),color=GRAY,ls=':',label="$0.798\\sqrt{\\log n/n}$")
ax[1].set_yscale('log'); ax[1].set_xlabel("$\\log n$"); ax[1].set_title("(b) the two parts of the risk",fontsize=10); ax[1].legend(frameon=False,fontsize=7.5)
ef=R2['efficiency']; ax[2].plot([e['n_sign'] for e in ef],[e['ratio'] for e in ef],'o-',color=TEAL,label="sign test vs mean test")
ax[2].axhline(math.pi/2,color=GRAY,ls='--',label="$\\pi/2$"); ax[2].set_xscale('log'); ax[2].set_ylim(1.5,1.65)
ax[2].set_xlabel("sign test sample size"); ax[2].set_ylabel("sample size ratio at equal risk"); ax[2].set_title("(c) Bayes risk efficiency",fontsize=10); ax[2].legend(frameon=False,fontsize=7.5)
fig.tight_layout(); fig.savefig(FIGS / "tst_rubin_sethuraman.pdf")
print(json.dumps(out,indent=1))
