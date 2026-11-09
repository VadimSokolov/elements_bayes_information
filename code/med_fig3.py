import pathlib
HERE = pathlib.Path(__file__).resolve().parent
FIGS = pathlib.Path(__file__).resolve().parents[1] / "figs"
import numpy as np, matplotlib, math, json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import betaln, gammaln
src=open(HERE/'med_numbers.py').read()
exec(src[src.index('# 5 single arm'):src.index("out['design']['null']")].replace("out['design']=dict(final_needed=xs_final)",""))
NAVY,RUST,TEAL,GOLD,GRAY,PLUM="#26456e","#c0562f","#2a8f82","#d4a11e","#6b6b6b","#7b5ea7"
plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"serif","font.serif":["CMU Serif","DejaVu Serif"],
 "mathtext.fontset":"cm","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
ps=np.round(np.arange(0.1,0.56,0.05),2); oc=[sim(p,R=40000,seed=int(p*100)) for p in ps]
fig,ax=plt.subplots(1,2,figsize=(9.4,3.4))
ax[0].plot(ps,[o['reject'] for o in oc],'o-',color=NAVY,label="P(declare promising)")
ax2=ax[0].twinx(); ax2.plot(ps,[o['EN'] for o in oc],'s--',color=RUST,label="expected patients"); ax2.set_ylabel("expected patients",color=RUST); ax2.set_ylim(0,42)
ax2.spines['top'].set_visible(False)
ax[0].axvline(.2,color=GRAY,ls=':'); ax[0].axvline(.4,color=GRAY,ls=':')
ax[0].set_xlabel("true response rate"); ax[0].set_ylabel("probability",color=NAVY); ax[0].set_title("(a) adaptive single arm trial",fontsize=10)
h1,l1=ax[0].get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax[0].legend(h1+h2,l1+l2,frameon=False,fontsize=8,loc='center right')
lam0=2.0; a,bb=1.0,1.0
def eproc(C):
    S=np.cumsum(C,1); T=np.arange(1,C.shape[1]+1)*lam0
    return gammaln(a+S)-gammaln(a)+a*np.log(bb)-(a+S)*np.log(bb+T)+S*np.log(lam0)-(-T+S*np.log(lam0))
r=np.random.default_rng(9)
for rr,col,lab in [(1.0,GRAY,"no excess risk"),(1.5,GOLD,"relative risk 1.5"),(2.0,RUST,"relative risk 2")]:
    E=eproc(r.poisson(rr*lam0,size=(25,104)))
    for i in range(25): ax[1].plot(np.arange(1,105),E[i],color=col,lw=.7,alpha=.6,label=lab if i==0 else None)
ax[1].axhline(math.log(20),color='k',lw=1,ls='--',label="$\\log 20$")
ax[1].set_ylim(-8,25); ax[1].set_xlabel("week"); ax[1].set_ylabel("log e-value"); ax[1].set_title("(b) safety surveillance",fontsize=10); ax[1].legend(frameon=True,facecolor='white',edgecolor='none',framealpha=0.95,fontsize=7.5,loc='upper left')
fig.tight_layout(); fig.savefig(FIGS / "med_trial_surveillance.pdf")
print([(p,round(o['reject'],3),round(o['EN'],1)) for p,o in zip(ps,oc)])
