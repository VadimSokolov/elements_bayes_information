import pathlib
HERE = pathlib.Path(__file__).resolve().parent
FIGS = pathlib.Path(__file__).resolve().parents[1] / "figs"
import numpy as np, matplotlib, math, json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
exec(open(HERE/'med_numbers.py').read().split('# 6 safety')[0].split('out={}')[0])
NAVY,RUST,TEAL,GOLD,GRAY,PLUM="#26456e","#c0562f","#2a8f82","#d4a11e","#6b6b6b","#7b5ea7"
plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"serif","font.serif":["CMU Serif","DejaVu Serif"],
 "mathtext.fontset":"cm","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
J=json.load(open(HERE/'med_numbers.json'))
def ppv(prev,se,sp): return se*prev/(se*prev+(1-sp)*(1-prev))
# Fig 1
fig,ax=plt.subplots(1,2,figsize=(9.4,3.4))
pr=np.logspace(-3,-0.3,200)
ax[0].plot(pr,ppv(pr,.9,.91),color=NAVY,lw=2,label="one positive test")
ax[0].plot(pr,ppv(ppv(pr,.9,.91),.9,.91),color=RUST,lw=2,label="two independent positives")
ax[0].axvline(.01,color=GRAY,ls=':',lw=1); ax[0].plot([.01],[J['mammo']['ppv']],'o',color=NAVY); ax[0].plot([.01],[J['mammo']['ppv2']],'o',color=RUST)
ax[0].set_xscale('log'); ax[0].set_xlabel("prevalence"); ax[0].set_ylabel("P(disease | positive)"); ax[0].set_title("(a) the same test at different prevalences",fontsize=10); ax[0].legend(frameon=False,fontsize=8)
T=J['thr']; B,H,se,sp,r=T['B'],T['H'],T['se'],T['sp'],T['r']; p=np.linspace(0,1,400)
ax[1].plot(p,p*B,color=NAVY,lw=2,label="withhold treatment")
ax[1].plot(p,(1-p)*H,color=RUST,lw=2,label="treat")
ax[1].plot(p,p*(1-se)*B+(1-p)*(1-sp)*H+r,color=TEAL,lw=2,label="test, treat if positive")
for v,l in [(T['test_low'],'test threshold'),(T['treat'],'treatment threshold'),(T['test_up'],'test treatment threshold')]:
    ax[1].axvline(v,color=GRAY,ls=':',lw=1)
ax[1].set_ylim(0,0.4); ax[1].set_xlabel("probability of disease $p$"); ax[1].set_ylabel("expected loss"); ax[1].set_title("(b) when to test, when to treat",fontsize=10); ax[1].legend(frameon=True,facecolor='white',edgecolor='none',fontsize=8,loc='center right',bbox_to_anchor=(1.0,0.6))
ax[1].text(T['test_low']/2,0.37,"no test\nno treat",ha='center',fontsize=7); ax[1].text((T['test_low']+T['test_up'])/2,0.37,"test",ha='center',fontsize=7); ax[1].text((1+T['test_up'])/2,0.37,"treat",ha='center',fontsize=7)
fig.tight_layout(); fig.savefig(FIGS / "med_diagnosis.pdf")
# Fig 2
fig,ax=plt.subplots(1,2,figsize=(9.4,3.4)); c=J['dca']['curve']
ax[0].plot(c['t'],c['model'],color=NAVY,lw=2,label="calibrated model")
ax[0].plot(c['t'],c['over'],color=RUST,lw=2,label="overconfident model, same AUC")
ax[0].plot(c['t'],c['all'],color=GRAY,lw=1.2,ls='--',label="treat all"); ax[0].axhline(0,color='k',lw=.8,label="treat none")
ax[0].set_ylim(-0.05,0.18); ax[0].set_xlabel("threshold probability $p_t$"); ax[0].set_ylabel("net benefit"); ax[0].set_title("(a) decision curves",fontsize=10); ax[0].legend(frameon=False,fontsize=7.5)
hr=np.linspace(0.5,1.3,400); res=J['trial']['res']; s=J['trial']['se']; sig=J['trial']['prior_sd']
b=math.log(.8)
for k,col in [('reference',GRAY),('skeptical',NAVY),('enthusiastic',RUST)]:
    m0={'reference':0,'skeptical':0,'enthusiastic':math.log(.8)}[k]; s0=1e6 if k=='reference' else sig
    v=1/(1/s**2+1/s0**2); m=v*(b/s**2+m0/s0**2)
    ax[1].plot(hr,stats.norm.pdf(np.log(hr),m,math.sqrt(v))/hr,color=col,lw=2,label=f"{k}: P(HR<0.9)={res[k]['P_lt09']:.2f}")
    if k=='skeptical': ax[1].plot(hr,stats.norm.pdf(np.log(hr),0,sig)/hr,color=col,lw=1,ls=':',label="skeptical prior")
ax[1].axvline(1,color='k',lw=.8); ax[1].axvline(.9,color=GRAY,lw=.8,ls=':')
ax[1].set_xlabel("hazard ratio"); ax[1].set_ylabel("posterior density"); ax[1].set_title("(b) one trial, three priors",fontsize=10); ax[1].set_ylim(0,8.2); ax[1].legend(frameon=False,fontsize=7,loc='upper right')
fig.tight_layout(); fig.savefig(FIGS / "med_curves_trial.pdf")
print('ok')
