# Generates the application figures for the "Twenty One Bayesian Vignettes" section (Section ch:vignettes).
# Writes app_*.pdf into the repo's figs/ whatever the working directory. Fixed seeds: all book numbers reproduce.
import numpy as np, matplotlib as mpl, pathlib
mpl.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
# ---- book-matching style ----
mpl.rcParams.update({
 'pdf.fonttype':42,'ps.fonttype':42,
 'font.family':'serif','font.serif':['CMU Serif','DejaVu Serif'],'mathtext.fontset':'cm',
 'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'legend.fontsize':9.5,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.8,
 'axes.edgecolor':'#333333','xtick.color':'#333333','ytick.color':'#333333',
 'axes.labelcolor':'#222222','text.color':'#222222','figure.dpi':200,
 'axes.grid':True,'grid.color':'#dddddd','grid.linewidth':0.6})
NAVY='#26456e'; RUST='#c0562f'; TEAL='#2a8f82'; GOLD='#d4a11e'; GRAY='#6b6b6b'
FIGS=pathlib.Path(__file__).resolve().parents[1]/'figs'
def finish(fig,name):
    fig.tight_layout(); fig.savefig(FIGS/(name+'.pdf'),bbox_inches='tight'); plt.close(fig); print('wrote',name)

# 1) A/B TEST -------------------------------------------------------------
rng=np.random.default_rng(1)
aA,bA=1+80,1+320   # 80/400
aB,bB=1+95,1+305   # 95/400
x=np.linspace(0.12,0.34,600)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.6,3.2))
ax1.fill_between(x,stats.beta.pdf(x,aA,bA),color=NAVY,alpha=.28)
ax1.plot(x,stats.beta.pdf(x,aA,bA),color=NAVY,lw=2,label=r'A: $\mathrm{Beta}(81,321)$')
ax1.fill_between(x,stats.beta.pdf(x,aB,bB),color=RUST,alpha=.28)
ax1.plot(x,stats.beta.pdf(x,aB,bB),color=RUST,lw=2,label=r'B: $\mathrm{Beta}(96,306)$')
ax1.set_xlabel('conversion rate  $\\theta$'); ax1.set_ylabel('posterior density')
ax1.set_title('(a)  Posterior conversion rates'); ax1.legend(frameon=False,loc='upper right')
ax1.set_yticks([])
A=rng.beta(aA,bA,400000); B=rng.beta(aB,bB,400000); d=B-A
pB=(d>0).mean(); lift=d.mean()*100
xs=np.linspace(d.mean()-4*d.std(),d.mean()+4*d.std(),600)
kde=stats.gaussian_kde(d[:60000])
ax2.plot(xs,kde(xs),color='#333',lw=1.8)
ax2.fill_between(xs[xs>0],kde(xs[xs>0]),color=TEAL,alpha=.35)
ax2.axvline(0,color=GRAY,lw=1,ls='--')
ax2.set_xlabel(r'uplift  $\delta=\theta_B-\theta_A$'); ax2.set_yticks([])
ax2.set_title('(b)  Posterior uplift')
ax2.annotate(r'$\Pr(\delta>0)=%.2f$'%pB,xy=(0.02,0.62),xycoords='axes fraction',
             color=TEAL,fontsize=11)
ax2.annotate('mean uplift\n$%.1f$ points'%lift,xy=(0.60,0.55),xycoords='axes fraction',fontsize=9.5,color='#333')
finish(fig,'app_abtest'); print('  P(B>A)=%.3f lift=%.2fpp'%(pB,lift))

# 2) HIERARCHICAL SHRINKAGE ----------------------------------------------
rng=np.random.default_rng(3)
K=12; n=np.array([45,53,28,60,39,72,33,50,41,58,26,47])
true=0.27+0.06*rng.standard_normal(K)
hits=rng.binomial(n,true); raw=hits/n
grand=raw.mean()
# empirical-Bayes shrinkage toward grand mean (Beta-Binomial)
tau2=max(raw.var()-np.mean(raw*(1-raw)/n),1e-4)
w=tau2/(tau2+raw*(1-raw)/n)
shrunk=grand+w*(raw-grand)
fig,ax=plt.subplots(figsize=(7.2,3.4))
for i in range(K):
    ax.plot([0,1],[raw[i],shrunk[i]],color=GRAY,lw=0.9,alpha=.7,zorder=1)
ax.scatter(np.zeros(K),raw,s=42,color=NAVY,zorder=3,label='raw rate  $y_i/n_i$')
ax.scatter(np.ones(K),shrunk,s=42,color=RUST,zorder=3,label='posterior mean (pooled)')
ax.axhline(grand,color=TEAL,lw=1.4,ls='--',label='grand mean')
ax.set_xlim(-0.18,1.18); ax.set_xticks([0,1]); ax.set_xticklabels(['no pooling','partial pooling'])
ax.set_ylabel('success rate'); ax.set_title('Hierarchical shrinkage: borrowing strength across %d groups'%K)
ax.legend(frameon=False,loc='upper right',ncol=1)
finish(fig,'app_shrinkage')

# 3) GAUSSIAN PROCESS REGRESSION -----------------------------------------
rng=np.random.default_rng(7)
def kern(a,b,l=1.1,s=1.0): return s**2*np.exp(-0.5*(a[:,None]-b[None,:])**2/l**2)
Xtr=np.array([-4.2,-3.1,-1.6,-0.4,0.9,2.2,3.6,4.4]); ytr=np.sin(Xtr)+0.15*rng.standard_normal(len(Xtr))
Xs=np.linspace(-5.2,5.4,300); sig=0.15
K=kern(Xtr,Xtr)+sig**2*np.eye(len(Xtr)); Ks=kern(Xs,Xtr); Kss=kern(Xs,Xs)
Ki=np.linalg.inv(K); mu=Ks@Ki@ytr; cov=Kss-Ks@Ki@Ks.T; sd=np.sqrt(np.clip(np.diag(cov),0,None))
fig,ax=plt.subplots(figsize=(7.2,3.4))
ax.fill_between(Xs,mu-1.96*sd,mu+1.96*sd,color=NAVY,alpha=.18,label='95% credible band')
L=np.linalg.cholesky(cov+1e-8*np.eye(len(Xs)))
for j in range(3): ax.plot(Xs,mu+L@rng.standard_normal(len(Xs)),color=NAVY,lw=0.7,alpha=.5)
ax.plot(Xs,mu,color=NAVY,lw=2,label='posterior mean')
ax.plot(Xs,np.sin(Xs),color=TEAL,lw=1.2,ls='--',label='truth  $\\sin x$')
ax.scatter(Xtr,ytr,s=38,color=RUST,zorder=5,label='observations')
ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)$'); ax.set_title('Gaussian-process regression with posterior uncertainty')
ax.legend(frameon=False,loc='lower left',ncol=2); finish(fig,'app_gp')

# 4) KALMAN FILTER --------------------------------------------------------
rng=np.random.default_rng(11); T=60; q,r=0.02,0.6
x=np.zeros(T); v=0.0
for t in range(1,T): v=0.92*v+np.sqrt(q)*rng.standard_normal(); x[t]=x[t-1]+v+0.05
obs=x+np.sqrt(r)*rng.standard_normal(T)
mu=np.zeros(T); P=np.zeros(T); mu[0]=obs[0]; P[0]=1.0
for t in range(1,T):
    mp=mu[t-1]; Pp=P[t-1]+q+0.05
    Kk=Pp/(Pp+r); mu[t]=mp+Kk*(obs[t]-mp); P[t]=(1-Kk)*Pp
sd=np.sqrt(P); tt=np.arange(T)
fig,ax=plt.subplots(figsize=(7.2,3.3))
ax.scatter(tt,obs,s=20,color=GRAY,alpha=.7,label='noisy observations')
ax.plot(tt,x,color=TEAL,lw=1.4,ls='--',label='true state')
ax.fill_between(tt,mu-1.96*sd,mu+1.96*sd,color=NAVY,alpha=.2,label='95% filter band')
ax.plot(tt,mu,color=NAVY,lw=2,label='Kalman posterior mean')
ax.set_xlabel('time  $t$'); ax.set_ylabel('position'); ax.set_title('Kalman filtering: recursive Bayesian state estimation')
ax.legend(frameon=False,loc='upper left',ncol=2); finish(fig,'app_kalman')

# 5) BANDIT REGRET --------------------------------------------------------
def run(strategy,T=1500,reps=200,seed=0):
    rng=np.random.default_rng(seed); mus=np.array([0.5,0.55,0.6]); best=mus.max()
    R=np.zeros((reps,T))
    for rp in range(reps):
        a=np.ones(3); b=np.ones(3); cnt=np.zeros(3); val=np.zeros(3); creg=0
        for t in range(T):
            if strategy=='ts': arm=np.argmax(rng.beta(a,b))
            elif strategy=='ucb':
                ucb=np.where(cnt==0,1e9,val+np.sqrt(2*np.log(t+1)/np.maximum(cnt,1))); arm=np.argmax(ucb)
            else:
                arm=rng.integers(3) if rng.random()<0.1 else np.argmax(val)
            rw=rng.random()<mus[arm]; creg+=best-mus[arm]; R[rp,t]=creg
            cnt[arm]+=1
            if rw: a[arm]+=1
            else: b[arm]+=1
            val[arm]=a[arm]/(a[arm]+b[arm]) if strategy!='eps' else val[arm]+(rw-val[arm])/cnt[arm]
    return R.mean(0),R.std(0)/np.sqrt(reps)
fig,ax=plt.subplots(figsize=(7.2,3.3)); T=1500; tt=np.arange(T)
for strat,col,lab in [('ts',RUST,'Thompson sampling'),('ucb',NAVY,'UCB'),('eps',GRAY,r'$\varepsilon$-greedy')]:
    m,se=run(strat,T=T); ax.plot(tt,m,color=col,lw=2,label=lab); ax.fill_between(tt,m-2*se,m+2*se,color=col,alpha=.18)
ax.set_xlabel('round  $t$'); ax.set_ylabel('cumulative regret'); ax.set_title('Multi-armed bandits: Thompson sampling vs.\\ alternatives')
ax.legend(frameon=False,loc='upper left'); finish(fig,'app_bandit')

# 6) CHANGE-POINT ---------------------------------------------------------
rng=np.random.default_rng(5); T=80; cp=48; y=np.concatenate([2.0+0.7*rng.standard_normal(cp),3.6+0.7*rng.standard_normal(T-cp)])
# posterior over the change point (flat prior), from the profile likelihood with each segment's own mean and variance
def logml(seg):
    n=len(seg); 
    if n<2: return -1e9
    return -0.5*n*np.log(seg.var()+1e-6)
lp=np.array([logml(y[:k])+logml(y[k:]) for k in range(2,T-1)])
lp-=lp.max(); post=np.exp(lp); post/=post.sum(); ks=np.arange(2,T-1)
fig,(a1,a2)=plt.subplots(2,1,figsize=(7.2,4.0),sharex=True,gridspec_kw={'height_ratios':[2,1]})
a1.plot(np.arange(T),y,color=GRAY,lw=1,marker='o',ms=3)
a1.axvline(cp,color=TEAL,ls='--',lw=1.4,label='true change-point')
khat=ks[np.argmax(post)]
a1.axvline(khat,color=RUST,lw=1.6,label='posterior mode $\\hat\\tau=%d$'%khat)
a1.set_ylabel('$y_t$'); a1.set_title('Bayesian change-point detection'); a1.legend(frameon=False,loc='upper left')
a2.fill_between(ks,post,color=NAVY,alpha=.4,step='mid'); a2.plot(ks,post,color=NAVY,lw=1.2,drawstyle='steps-mid')
a2.set_xlabel('time  $t$'); a2.set_ylabel(r'$\Pr(\tau=t\mid y)$'); a2.set_yticks([])
finish(fig,'app_changepoint'); print('  changepoint true=%d hat=%d'%(cp,khat))
print("ALL FIGURES DONE")
