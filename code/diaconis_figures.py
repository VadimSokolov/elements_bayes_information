import pathlib
HERE = pathlib.Path(__file__).resolve().parent
FIGS = pathlib.Path(__file__).resolve().parents[1] / "figs"
import numpy as np, matplotlib, math
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import hypergeom, binom, beta
from scipy.special import betaln, gammaln
from scipy.optimize import brentq
NAVY,RUST,TEAL,GOLD,GRAY,PLUM="#26456e","#c0562f","#2a8f82","#d4a11e","#6b6b6b","#7b5ea7"
plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"serif","font.serif":["CMU Serif","DejaVu Serif"],
 "mathtext.fontset":"cm","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
# Figure 1 finite de Finetti
fig,ax=plt.subplots(figsize=(6.2,3.5)); cols={100:NAVY,1000:RUST}
for n in (100,1000):
    ks=np.arange(1,(60 if n==100 else 400)+1); tv=[];bd=[]
    for k in ks:
        x=np.arange(k+1); tv.append(0.5*np.abs(hypergeom.pmf(x,n,n//2,k)-binom.pmf(x,k,0.5)).sum())
        bd.append(1-np.prod((n-np.arange(k))/n))
    ax.plot(ks/n,tv,color=cols[n],lw=2,label=f"exact distance, $n={n}$")
    ax.plot(ks/n,bd,color=cols[n],lw=1,ls=":",label=f"birthday bound, $n={n}$")
r=np.linspace(0,.6,100); ax.plot(r,np.minimum(2*r,1),color=GRAY,lw=1,ls="--",label="$ck/n$ with $c=2$")
ax.plot(r,r/4,color=TEAL,lw=1,ls="-.",label="$k/(4n)$")
ax.set_xlabel("$k/n$"); ax.set_ylabel("variation distance"); ax.set_ylim(0,1.02); ax.set_xlim(0,.6)
ax.legend(frameon=False,fontsize=7.5); fig.tight_layout(); fig.savefig(FIGS / "dia_finite_definetti.pdf")
n=1000
print("TV n=1000:",[round(0.5*np.abs(hypergeom.pmf(np.arange(k+1),n,n//2,k)-binom.pmf(np.arange(k+1),k,.5)).sum(),4) for k in (10,50,100)])
# Figure 2 conjugate linearity spun coin
n=20; ks=np.arange(n+1); comps=[(10,20),(20,10)]; w0=np.array([.5,.5])
def mixp(k,comps=comps,w0=w0,n=n):
    lw=np.log(w0)+np.array([betaln(a+k,b+n-k)-betaln(a,b) for a,b in comps]); w=np.exp(lw-lw.max()); w/=w.sum()
    m=np.array([(a+k)/(a+b+n) for a,b in comps]); return w,(w*m).sum()
conj=(5+ks)/(10+n); mix=np.array([mixp(k)[1] for k in ks])
fig,ax=plt.subplots(1,2,figsize=(7.2,3.1))
ax[0].plot(ks/n,ks/n,color=GRAY,lw=1,ls=":",label="sample proportion")
ax[0].plot(ks/n,conj,color=NAVY,lw=2,label="conjugate Beta(5,5)")
ax[0].plot(ks/n,mix,color=RUST,lw=2,label="mixture of two Betas")
ax[0].set_xlabel("$k/n$ ($n=20$)"); ax[0].set_ylabel("posterior mean of $p$"); ax[0].legend(frameon=False,fontsize=8)
ax[0].set_title("(a) linear only if conjugate",fontsize=10)
p=np.linspace(0,1,400); w,_=mixp(14)
ax[1].plot(p,sum(wi*beta.pdf(p,a,b) for wi,(a,b) in zip(w0,comps)),color=GRAY,lw=1.5,ls="--",label="mixture prior")
ax[1].plot(p,sum(wi*beta.pdf(p,a+14,b+6) for wi,(a,b) in zip(w,comps)),color=RUST,lw=2,label="posterior, 14 of 20")
ax[1].set_xlabel("$p$"); ax[1].set_title("(b) spun coin prior",fontsize=10); ax[1].legend(frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig(FIGS / "dia_conjugate_linearity.pdf")
print("spun: w at 14",w.round(3),"mix mean 14",round(mix[14],3),"conj",round(conj[14],3))
# Figure 3 sports
comps=[(65,35),(84,16)]
pm=sum(.5*a/(a+b) for a,b in comps); ex2=sum(.5*a*(a+1)/((a+b)*(a+b+1)) for a,b in comps); v=ex2-pm**2; s=pm*(1-pm)/v-1; A,B=pm*s,(1-pm)*s
print("single beta",round(A,2),round(B,2),"n0",round(s,2),"mean",round(pm,4))
def post(k,n=20): return mixp(k,comps,np.array([.5,.5]),n)
for k in (12,14,15,16,18): print("FT",k,k/20,round(post(k)[0][1],3),round(post(k)[1],3),round((A+k)/(A+B+20),3))
fig,ax=plt.subplots(1,3,figsize=(9.6,3.1)); p=np.linspace(.4,1,400)
ax[0].plot(p,sum(.5*beta.pdf(p,a,b) for a,b in comps),color=GRAY,lw=1.6,ls=":",label="mixture prior")
ax[0].plot(p,beta.pdf(p,A,B),color=NAVY,lw=1,ls="--",label="single Beta prior")
wk,_=post(15)
ax[0].plot(p,sum(wi*beta.pdf(p,a+15,b+5) for wi,(a,b) in zip(wk,comps)),color=RUST,lw=2,label="mixture posterior, 15 of 20")
ax[0].plot(p,beta.pdf(p,A+15,B+5),color=NAVY,lw=2,label="single Beta posterior")
ax[0].set_xlabel("free throw rate $p$"); ax[0].set_title("(a) prior and posterior",fontsize=10); ax[0].set_ylim(0,8.5); ax[0].legend(frameon=False,fontsize=6.6,loc="upper left")
kk=np.arange(21)
ax[1].plot(kk/20,kk/20,color=GRAY,ls=":",lw=1,label="raw $k/n$")
ax[1].plot(kk/20,(A+kk)/(A+B+20),color=NAVY,lw=2,label="single conjugate")
ax[1].plot(kk/20,[post(k)[1] for k in kk],color=RUST,lw=2,label="mixture")
ax[1].set_xlim(.4,1); ax[1].set_ylim(.4,1); ax[1].set_xlabel("$k/n$ ($n=20$)"); ax[1].set_ylabel("posterior mean"); ax[1].set_title("(b) shrinkage",fontsize=10); ax[1].legend(frameon=False,fontsize=7.5)
for nn,ls,c in ((5,":",GOLD),(20,"-",RUST),(100,"--",PLUM)):
    k2=np.arange(nn+1); ax[2].plot(k2/nn,[post(k,nn)[0][1] for k in k2],color=c,ls=ls,lw=1.6,label=f"$n={nn}$")
ax[2].set_xlim(.4,1); ax[2].set_xlabel("$k/n$"); ax[2].set_ylabel("P(guard type | data)"); ax[2].set_title("(c) type identified",fontsize=10); ax[2].legend(frameon=False,fontsize=7.5)
fig.tight_layout(); fig.savefig(FIGS / "dia_sports_mixture.pdf")
F=lambda x:sum(wi*beta.cdf(x,a+15,b+5) for wi,(a,b) in zip(wk,comps))
print("mix90",round(brentq(lambda x:F(x)-.05,.01,.99),3),round(brentq(lambda x:F(x)-.95,.01,.99),3),"single90",beta.ppf([.05,.95],A+15,B+5).round(3))
g=[(12,10),(18,10)]; S=11; m=5
lw=np.log([.5,.5])+np.array([gammaln(a+S)-gammaln(a)+a*np.log(b)-(a+S)*np.log(b+m) for a,b in g]); w=np.exp(lw-lw.max()); w/=w.sum()
means=[(a+S)/(b+m) for a,b in g]; print("soccer w",w.round(3),"means",np.round(means,3),"mix",round(float((w*np.array(means)).sum()),3))
pm=1.5; ex2=sum(.5*a*(a+1)/b**2 for a,b in g); v=ex2-pm**2; bb=pm/v; aa=pm*bb; print("single gamma",round(aa,2),round(bb,2),"post",round((aa+S)/(bb+m),3))
