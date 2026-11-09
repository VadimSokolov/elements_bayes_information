# Generates the application figures for the "Twenty One Bayesian Vignettes" section (Section ch:vignettes).
# Writes app_*.pdf into the repo's figs/ whatever the working directory. Fixed seeds: all book numbers reproduce.
import numpy as np, matplotlib as mpl, pathlib
mpl.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
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

# ---- 1) EIGHT SCHOOLS (the canonical hierarchical model) ----------------
y=np.array([28.,8.,-3.,7.,-1.,1.,18.,12.]); s=np.array([15.,10.,16.,11.,9.,11.,10.,18.])
names=['A','B','C','D','E','F','G','H']
# full Bayes: marginal posterior of tau (Gelman's half-Cauchy-ish flat prior on tau)
taus=np.linspace(0.01,30,4000)
def logp_tau(t):
    V=s**2+t**2; mu_hat=np.sum(y/V)/np.sum(1/V); Vmu=1/np.sum(1/V)
    return -0.5*np.sum(np.log(V))+0.5*np.log(Vmu)-0.5*np.sum((y-mu_hat)**2/V)
lp=np.array([logp_tau(t) for t in taus]); lp-=lp.max(); p=np.exp(lp); p/=np.trapezoid(p,taus)
rng=np.random.default_rng(0); S=40000
tsamp=rng.choice(taus,size=S,p=p/p.sum())
V=s**2+tsamp[:,None]**2
muhat=np.sum(y/V,axis=1)/np.sum(1/V,axis=1); Vmu=1/np.sum(1/V,axis=1)
mu=muhat+np.sqrt(Vmu)*rng.standard_normal(S)
Vj=1/(1/s**2+1/tsamp[:,None]**2); thmean=Vj*(y/s**2+mu[:,None]/tsamp[:,None]**2)
theta=thmean+np.sqrt(Vj)*rng.standard_normal((S,8))
post_mean=theta.mean(0); lo=np.percentile(theta,2.5,axis=0); hi=np.percentile(theta,97.5,axis=0)
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.8,3.5),gridspec_kw={'width_ratios':[1.35,1]})
yy=np.arange(8)[::-1]
a1.errorbar(y,yy+0.15,xerr=1.96*s,fmt='o',color=GRAY,ms=5,lw=1,capsize=2,label='observed effect $\\pm$ 2 s.e.')
a1.errorbar(post_mean,yy-0.15,xerr=[post_mean-lo,hi-post_mean],fmt='o',color=RUST,ms=5,lw=1.6,capsize=2,label='posterior (pooled)')
a1.axvline(mu.mean(),color=TEAL,ls='--',lw=1.3,label=r'population mean $\mu$')
a1.set_yticks(yy); a1.set_yticklabels(names); a1.set_xlabel('treatment effect (points)')
a1.set_title('(a)  Eight schools: shrinkage')
a1.margins(x=0.02); a1.legend(frameon=False,fontsize=8.5,loc='upper center',bbox_to_anchor=(0.5,-0.22),ncol=3,handletextpad=0.4,columnspacing=1.0)
a2.plot(taus,p,color=NAVY,lw=2); a2.fill_between(taus,p,color=NAVY,alpha=.25)
a2.set_xlabel(r'between-school s.d.  $\tau$'); a2.set_yticks([])
a2.set_title(r'(b)  Posterior for $\tau$')
a2.annotate('mass near $\\tau=0$:\ncomplete pooling\nis plausible',xy=(0.30,0.55),xycoords='axes fraction',fontsize=8.5,color='#333')
finish(fig,'app_eightschools')
print('  tau post median=%.2f  mu mean=%.2f  schoolA: raw %.1f -> %.1f'%(np.median(tsamp),mu.mean(),y[0],post_mean[0]))

# ---- 2) FUNNEL: centered vs noncentered geometry -----------------------
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.6,3.2),sharey=False)
a1.scatter(theta[:4000,0],np.log(tsamp[:4000]),s=3,alpha=.25,color=NAVY)
a1.set_xlabel(r'$\theta_1$'); a1.set_ylabel(r'$\log\tau$'); a1.set_title('(a)  Centered: the funnel')
a1.set_xlim(-25,45)
eta=(theta[:4000,0]-mu[:4000])/tsamp[:4000]
a2.scatter(eta,np.log(tsamp[:4000]),s=3,alpha=.25,color=TEAL)
a2.set_xlabel(r'$\eta_1=(\theta_1-\mu)/\tau$'); a2.set_ylabel(r'$\log\tau$')
a2.set_title('(b)  Noncentered: reparametrized'); a2.set_xlim(-4,4)
finish(fig,'app_funnel')

# ---- 3) POSTERIOR PREDICTIVE CHECK --------------------------------------
rng=np.random.default_rng(2)
ydat=rng.negative_binomial(6,0.45,180)          # overdispersed truth
# fit Poisson (wrong model): posterior on lambda with Gamma prior
a,b=1+ydat.sum(),1+len(ydat)
lam=rng.gamma(a,1/b,2000)
reps=np.array([rng.poisson(l,len(ydat)) for l in lam[:2000]])
Tobs=ydat.var()/ydat.mean(); Trep=reps.var(1)/reps.mean(1)
pval=(Trep>=Tobs).mean()
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.6,3.2))
bins=np.arange(0,ydat.max()+2)-0.5
for r in reps[:60]: a1.hist(r,bins=bins,histtype='step',color=NAVY,alpha=.12,density=True)
a1.hist(ydat,bins=bins,color=RUST,alpha=.55,density=True,label='observed')
a1.plot([],[],color=NAVY,alpha=.5,label='posterior predictive draws')
a1.set_xlabel('count $y$'); a1.set_ylabel('density'); a1.set_title('(a)  Replicated vs observed')
a1.legend(frameon=False,fontsize=8.5)
a2.hist(Trep,bins=30,color=NAVY,alpha=.6,density=True)
a2.axvline(Tobs,color=RUST,lw=2.2,label='observed $T$')
a2.set_xlabel(r'variance-to-mean ratio $T(y)$'); a2.set_yticks([])
a2.set_title('(b)  Predictive $p$-value $=%.3f$'%pval); a2.legend(frameon=False,fontsize=9)
finish(fig,'app_ppc'); print('  ppc p=%.4f Tobs=%.2f'%(pval,Tobs))

# ---- 4) MCMC DIAGNOSTICS: trace + running mean + R-hat ------------------
rng=np.random.default_rng(4)
def chain(n=4000,step=0.35,start=0.0):
    x=start; out=np.empty(n)
    for i in range(n):
        prop=x+step*rng.standard_normal()
        if np.log(rng.random())< (-0.5*prop**2)-(-0.5*x**2): x=prop
        out[i]=x
    return out
chains=[chain(start=st) for st in (-3.,0.,3.,1.5)]
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.8,3.1))
cols=[NAVY,RUST,TEAL,GOLD]
for c,col in zip(chains,cols): a1.plot(c,lw=0.6,alpha=.8,color=col)
a1.set_xlabel('iteration'); a1.set_ylabel(r'$\theta$'); a1.set_title('(a)  Four chains: mixing')
for c,col in zip(chains,cols): a2.plot(np.cumsum(c)/np.arange(1,len(c)+1),lw=1.4,color=col)
a2.axhline(0,color='#333',ls='--',lw=1)
a2.set_xlabel('iteration'); a2.set_ylabel('running mean'); a2.set_title('(b)  Convergence to the target mean')
a2.set_ylim(-1.2,1.2)
# R-hat
M=np.array([c[1000:] for c in chains]); n=M.shape[1]; m=M.shape[0]
B=n*M.mean(1).var(ddof=1); W=M.var(1,ddof=1).mean(); var=( (n-1)/n*W + B/n ); Rhat=np.sqrt(var/W)
a2.annotate(r'$\hat R=%.3f$'%Rhat,xy=(0.62,0.12),xycoords='axes fraction',fontsize=11,color='#333')
finish(fig,'app_mcmc'); print('  Rhat=%.4f'%Rhat)

# ---- 5) BAYESIAN LOGISTIC / CALIBRATION ---------------------------------
rng=np.random.default_rng(6)
n=600; X=rng.standard_normal((n,2)); beta=np.array([1.3,-0.9]); p=1/(1+np.exp(-(X@beta+0.3)))
yb=(rng.random(n)<p).astype(int)
# Laplace-approx posterior for logistic (with ridge prior)
def fit(X,y,lam=1.0,iters=60):
    Xd=np.c_[np.ones(len(X)),X]; b=np.zeros(Xd.shape[1])
    for _ in range(iters):
        eta=Xd@b; mu=1/(1+np.exp(-eta)); W=mu*(1-mu)+1e-9
        H=Xd.T@(Xd*W[:,None])+lam*np.eye(Xd.shape[1]); g=Xd.T@(y-mu)-lam*b
        b+=np.linalg.solve(H,g)
    return b,np.linalg.inv(H)
b,Cov=fit(X,yb)
draws=rng.multivariate_normal(b,Cov,4000)
Xd=np.c_[np.ones(n),X]; P=1/(1+np.exp(-(draws@Xd.T)))   # 4000 x n
pm=P.mean(0)
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.8,3.2))
# coefficient posteriors
for k,(lab,col,truth) in enumerate([(r'$\beta_1$',NAVY,1.3),(r'$\beta_2$',RUST,-0.9)]):
    d=draws[:,k+1]; xs=np.linspace(d.min(),d.max(),300)
    a1.plot(xs,stats.gaussian_kde(d)(xs),color=col,lw=2,label=lab)
    a1.axvline(truth,color=col,ls='--',lw=1.1)
a1.set_xlabel('coefficient value'); a1.set_yticks([]); a1.set_title('(a)  Posterior coefficients (dashed = truth)')
a1.legend(frameon=False)
# calibration curve with predictive bands
bins=np.linspace(0,1,11); idx=np.digitize(pm,bins)-1
xs=[];obs=[];lo_=[];hi_=[]
for k in range(10):
    m=idx==k
    if m.sum()<5: continue
    xs.append(pm[m].mean()); obs.append(yb[m].mean())
    bb=rng.binomial(m.sum(),pm[m].mean(),2000)/m.sum()
    lo_.append(np.percentile(bb,2.5)); hi_.append(np.percentile(bb,97.5))
a2.plot([0,1],[0,1],ls='--',color=GRAY,lw=1.2,label='perfect calibration')
a2.fill_between(xs,lo_,hi_,color=NAVY,alpha=.18,label='95% predictive band')
a2.plot(xs,obs,'o-',color=RUST,lw=1.8,ms=5,label='observed frequency')
a2.set_xlabel('predicted probability'); a2.set_ylabel('observed frequency')
a2.set_title('(b)  Calibration of the predictive'); a2.legend(frameon=False,fontsize=8.5,loc='upper left')
finish(fig,'app_logistic')

# ---- 6) DIRICHLET PROCESS MIXTURE / density estimation -------------------
rng=np.random.default_rng(9)
data=np.concatenate([rng.normal(-2.2,0.55,150),rng.normal(0.6,0.45,200),rng.normal(3.1,0.8,120)])
# collapsed Gibbs for DPM of Gaussians (fixed variance, conjugate normal base)
def dpm(data,alpha=1.0,sw=0.6,s0=3.0,iters=220):
    n=len(data); z=np.zeros(n,int); K=1
    for it in range(iters):
        for i in range(n):
            zi=z[i]; z[i]=-1
            counts=np.bincount(z[z>=0],minlength=max(K,1)).astype(float)
            logp=[]
            for k in range(len(counts)):
                if counts[k]==0: logp.append(-np.inf); continue
                xs=data[z==k]; v=1/(1/s0**2+len(xs)/sw**2); m=v*(xs.sum()/sw**2)
                logp.append(np.log(counts[k])+stats.norm.logpdf(data[i],m,np.sqrt(v+sw**2)))
            logp.append(np.log(alpha)+stats.norm.logpdf(data[i],0,np.sqrt(s0**2+sw**2)))
            logp=np.array(logp); logp-=logp.max(); pr=np.exp(logp); pr/=pr.sum()
            znew=rng.choice(len(pr),p=pr)
            if znew==len(counts): z[i]=len(counts); K=len(counts)+1
            else: z[i]=znew
        # relabel
        u,z=np.unique(z,return_inverse=True); K=len(u)
    return z,K
z,K=dpm(data)
xs=np.linspace(data.min()-1,data.max()+1,500)
dens=np.zeros_like(xs)
for k in range(K):
    xsk=data[z==k]
    if len(xsk)==0: continue
    v=1/(1/3.0**2+len(xsk)/0.6**2); m=v*(xsk.sum()/0.6**2)
    dens+=len(xsk)/len(data)*stats.norm.pdf(xs,m,np.sqrt(0.6**2))
fig,ax=plt.subplots(figsize=(7.2,3.3))
ax.hist(data,bins=40,density=True,color=GRAY,alpha=.35,label='data')
cols=[NAVY,RUST,TEAL,GOLD,'#7b5ea7','#8a8a2f']
for k in range(K):
    xsk=data[z==k]
    if len(xsk)<3: continue
    v=1/(1/3.0**2+len(xsk)/0.6**2); m=v*(xsk.sum()/0.6**2)
    ax.plot(xs,len(xsk)/len(data)*stats.norm.pdf(xs,m,0.6),lw=1.2,ls='--',color=cols[k%len(cols)])
ax.plot(xs,dens,color=NAVY,lw=2.4,label='DP mixture predictive (%d clusters)'%K)
ax.set_xlabel('$y$'); ax.set_ylabel('density')
ax.set_title('Dirichlet-process mixture: the model learns how many clusters')
ax.legend(frameon=False,loc='upper right'); finish(fig,'app_dpm'); print('  DPM clusters=%d'%K)
print("BATCH 2 DONE")
