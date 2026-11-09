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

# 1) META ANALYSIS FOREST PLOT -------------------------------------------
rng=np.random.default_rng(21)
studies=['Trial I','Trial II','Trial III','Trial IV','Trial V','Trial VI','Trial VII']
yi=np.array([0.62,0.28,0.45,0.10,0.55,0.33,0.71]); si=np.array([0.22,0.14,0.30,0.18,0.25,0.12,0.35])
taus=np.linspace(0.001,1.2,3000)
def lp(t):
    V=si**2+t**2; m=np.sum(yi/V)/np.sum(1/V); Vm=1/np.sum(1/V)
    return -0.5*np.sum(np.log(V))+0.5*np.log(Vm)-0.5*np.sum((yi-m)**2/V)
L=np.array([lp(t) for t in taus]); L-=L.max(); w=np.exp(L); w/=w.sum()
S=60000; ts=rng.choice(taus,S,p=w)
V=si**2+ts[:,None]**2; mh=np.sum(yi/V,1)/np.sum(1/V,1); Vm=1/np.sum(1/V,1)
mu=mh+np.sqrt(Vm)*rng.standard_normal(S)
Vj=1/(1/si**2+1/ts[:,None]**2); th=Vj*(yi/si**2+mu[:,None]/ts[:,None]**2)+np.sqrt(Vj)*rng.standard_normal((S,len(yi)))
pm=th.mean(0); plo=np.percentile(th,2.5,0); phi=np.percentile(th,97.5,0)
fig,ax=plt.subplots(figsize=(7.2,3.6)); yy=np.arange(len(yi))[::-1]
ax.errorbar(yi,yy+0.16,xerr=1.96*si,fmt='s',color=GRAY,ms=5,lw=1,capsize=2,label='study estimate')
ax.errorbar(pm,yy-0.16,xerr=[pm-plo,phi-pm],fmt='o',color=RUST,ms=5,lw=1.6,capsize=2,label='posterior (pooled)')
ax.axvline(mu.mean(),color=TEAL,lw=1.5,ls='--',label='overall effect')
ax.axvspan(np.percentile(mu,2.5),np.percentile(mu,97.5),color=TEAL,alpha=.12)
ax.axvline(0,color='#999',lw=1)
ax.set_yticks(yy); ax.set_yticklabels(studies); ax.set_xlabel('log odds ratio')
ax.set_title('Random effects meta analysis')
ax.legend(frameon=False,fontsize=8.5,loc='upper center',bbox_to_anchor=(0.5,-0.18),ncol=3)
finish(fig,'app_meta'); print('  overall=%.3f [%.3f, %.3f]'%(mu.mean(),np.percentile(mu,2.5),np.percentile(mu,97.5)))

# 2) SURVIVAL ANALYSIS ----------------------------------------------------
rng=np.random.default_rng(23)
n=160; shape_t,scale_t=1.6,9.0
T=scale_t*rng.weibull(shape_t,n); C=rng.exponential(14,n)
obs=np.minimum(T,C); ev=(T<=C).astype(int)
def loglik(p):
    k,lam=p
    if k<=0 or lam<=0: return -1e9
    return np.sum(ev*(np.log(k/lam)+(k-1)*np.log(obs/lam))) - np.sum((obs/lam)**k)
# grid posterior with flat priors
ks=np.linspace(1.0,2.4,140); ls=np.linspace(6.5,12.5,140)
LL=np.array([[loglik((k,l)) for l in ls] for k in ks]); LL-=LL.max()
P=np.exp(LL); P/=P.sum()
idx=rng.choice(P.size,4000,p=P.ravel()); ki,li=np.unravel_index(idx,P.shape)
kk,ll=ks[ki],ls[li]
t=np.linspace(0,22,300)
Sdraw=np.exp(-(t[None,:]/ll[:,None])**kk[:,None])
Sm=Sdraw.mean(0); Slo=np.percentile(Sdraw,2.5,0); Shi=np.percentile(Sdraw,97.5,0)
# Kaplan Meier
order=np.argsort(obs); ob=obs[order]; ee=ev[order]; surv=1.0; km_t=[0]; km_s=[1.0]; atrisk=n
for i,(ti,ei) in enumerate(zip(ob,ee)):
    if ei==1: surv*= (1-1/atrisk); km_t.append(ti); km_s.append(surv)
    atrisk-=1
fig,ax=plt.subplots(figsize=(7.2,3.3))
ax.step(km_t,km_s,where='post',color=GRAY,lw=1.4,label='empirical survival curve')
ax.fill_between(t,Slo,Shi,color=NAVY,alpha=.2,label='95% credible band')
ax.plot(t,Sm,color=NAVY,lw=2,label='posterior mean survival')
ax.plot(t,np.exp(-(t/scale_t)**shape_t),color=TEAL,ls='--',lw=1.3,label='truth')
ax.set_xlabel('time'); ax.set_ylabel('survival probability'); ax.set_ylim(0,1.02)
ax.set_title('Bayesian survival analysis with censored data')
ax.legend(frameon=False,fontsize=8.5); finish(fig,'app_survival')

# 3) SPATIAL KRIGING HEATMAP ---------------------------------------------
rng=np.random.default_rng(29)
ns=45; sx=rng.uniform(0,10,ns); sy=rng.uniform(0,10,ns)
def truef(a,b): return 2.2*np.sin(0.55*a)+1.5*np.cos(0.45*b)+0.35*a*0.3
z=truef(sx,sy)+0.35*rng.standard_normal(ns)
def K(x1,y1,x2,y2,l=2.4,s=2.0): 
    d2=(x1[:,None]-x2[None,:])**2+(y1[:,None]-y2[None,:])**2
    return s**2*np.exp(-0.5*d2/l**2)
g=np.linspace(0,10,70); GX,GY=np.meshgrid(g,g); gx=GX.ravel(); gy=GY.ravel()
Kss=K(sx,sy,sx,sy)+0.35**2*np.eye(ns); Ki=np.linalg.inv(Kss); Kgs=K(gx,gy,sx,sy)
mu=(Kgs@Ki@z).reshape(GX.shape)
var=(2.0**2-np.einsum('ij,jk,ik->i',Kgs,Ki,Kgs)).reshape(GX.shape); sd=np.sqrt(np.clip(var,0,None))
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.8,3.2))
im1=a1.imshow(mu,origin='lower',extent=[0,10,0,10],cmap='viridis',aspect='auto',interpolation='bilinear')
a1.scatter(sx,sy,s=14,facecolor='none',edgecolor='white',lw=0.8)
a1.set_title('(a)  Posterior mean surface'); a1.set_xlabel('easting'); a1.set_ylabel('northing')
plt.colorbar(im1,ax=a1,fraction=0.046)
im2=a2.imshow(sd,origin='lower',extent=[0,10,0,10],cmap='magma',aspect='auto',interpolation='bilinear')
a2.scatter(sx,sy,s=14,facecolor='none',edgecolor='white',lw=0.8)
a2.set_title('(b)  Posterior standard deviation'); a2.set_xlabel('easting')
plt.colorbar(im2,ax=a2,fraction=0.046)
for a in (a1,a2): a.grid(False)
finish(fig,'app_kriging')

# 4) EPIDEMIC NOWCASTING (time varying reproduction number) ---------------
rng=np.random.default_rng(31)
Tn=70; Rt_true=np.concatenate([np.full(24,1.8),np.linspace(1.8,0.75,16),np.full(30,0.85)])
w=stats.gamma(a=2.6,scale=1.8).pdf(np.arange(1,16)); w/=w.sum()
I=np.zeros(Tn); I[:5]=np.array([4,6,7,9,11])
for t in range(5,Tn):
    lam=Rt_true[t]*np.sum([w[k-1]*I[t-k] for k in range(1,min(15,t)+1)])
    I[t]=rng.poisson(max(lam,0.1))
# posterior for Rt on a sliding window, Gamma prior
a0,b0=1.5,0.4; win=7; Rm=np.full(Tn,np.nan); Rlo=np.full(Tn,np.nan); Rhi=np.full(Tn,np.nan)
for t in range(16,Tn):
    idx=range(t-win+1,t+1)
    Isum=sum(I[u] for u in idx)
    Lsum=sum(np.sum([w[k-1]*I[u-k] for k in range(1,min(15,u)+1)]) for u in idx)
    a=a0+Isum; b=b0+Lsum
    Rm[t]=a/b; Rlo[t]=stats.gamma.ppf(.025,a,scale=1/b); Rhi[t]=stats.gamma.ppf(.975,a,scale=1/b)
fig,(a1,a2)=plt.subplots(2,1,figsize=(7.2,4.0),sharex=True,gridspec_kw={'height_ratios':[1,1.15]})
a1.bar(np.arange(Tn),I,color=GRAY,alpha=.65,width=0.9)
a1.set_ylabel('new cases'); a1.set_title('Epidemic nowcasting: estimating transmission in real time')
tt=np.arange(Tn)
a2.fill_between(tt,Rlo,Rhi,color=NAVY,alpha=.22,label='95% credible band')
a2.plot(tt,Rm,color=NAVY,lw=2,label='posterior mean $R_t$')
a2.plot(tt,Rt_true,color=TEAL,ls='--',lw=1.3,label='true $R_t$')
a2.axhline(1,color=RUST,lw=1.2)
a2.set_xlabel('day'); a2.set_ylabel('$R_t$'); a2.set_ylim(0,3)
a2.legend(frameon=False,fontsize=8.5,ncol=3,loc='upper right')
finish(fig,'app_epidemic')

# 5) ITEM RESPONSE THEORY -------------------------------------------------
rng=np.random.default_rng(37)
nP,nI=400,6
ability=rng.standard_normal(nP); diff=np.linspace(-1.6,1.6,nI); disc=np.linspace(0.9,2.0,nI)
Pm=1/(1+np.exp(-disc*(ability[:,None]-diff[None,:])))
Y=(rng.random((nP,nI))<Pm).astype(int)
# estimate abilities by posterior mode with standard normal prior, items fixed at truth
th=np.linspace(-4,4,400)
post=np.zeros((nP,len(th)))
for i in range(nP):
    p=1/(1+np.exp(-disc[None,:]*(th[:,None]-diff[None,:])))
    ll=(Y[i]*np.log(p)+(1-Y[i])*np.log(1-p)).sum(1)+stats.norm.logpdf(th)
    ll-=ll.max(); e=np.exp(ll); post[i]=e/np.trapezoid(e,th)
pmean=np.array([np.trapezoid(th*post[i],th) for i in range(nP)])
psd=np.array([np.sqrt(np.trapezoid((th-pmean[i])**2*post[i],th)) for i in range(nP)])
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.8,3.2))
for j in range(nI):
    a1.plot(th,1/(1+np.exp(-disc[j]*(th-diff[j]))),lw=1.7,label='item %d'%(j+1))
a1.set_xlabel(r'ability $\theta$'); a1.set_ylabel('probability correct')
a1.set_title('(a)  Item characteristic curves'); a1.legend(frameon=False,fontsize=7.5,ncol=2)
o=np.argsort(pmean)
a2.plot(ability[o],pmean[o],'o',ms=3.5,color=NAVY,alpha=.6)
a2.errorbar(ability[o][::12],pmean[o][::12],yerr=1.96*psd[o][::12],fmt='none',ecolor=RUST,alpha=.7,lw=1)
lim=[-3,3]; a2.plot(lim,lim,ls='--',color=GRAY)
a2.set_xlabel('true ability'); a2.set_ylabel('posterior mean ability')
a2.set_title('(b)  Recovery, with uncertainty'); a2.set_xlim(lim); a2.set_ylim(-3.5,3.5)
finish(fig,'app_irt')

# 6) BAYESIAN OPTIMIZATION -----------------------------------------------
rng=np.random.default_rng(41)
def obj(x): return -(np.sin(3*x)+0.35*x**2-0.4*x)
def gp(Xtr,ytr,Xs,l=0.55,s=1.3,noise=1e-3):
    Kf=lambda a,b: s**2*np.exp(-0.5*(a[:,None]-b[None,:])**2/l**2)
    Km=Kf(Xtr,Xtr)+noise*np.eye(len(Xtr)); Ki=np.linalg.inv(Km); Ks=Kf(Xs,Xtr)
    mu=Ks@Ki@ytr; var=np.clip(s**2-np.einsum('ij,jk,ik->i',Ks,Ki,Ks),1e-9,None)
    return mu,np.sqrt(var)
Xs=np.linspace(-2,2.4,400)
X=np.array([-1.6,0.2,1.9]); y=obj(X)
fig,axes=plt.subplots(2,2,figsize=(7.8,4.6),sharex=True)
for step,ax in enumerate(axes.ravel()):
    mu,sd=gp(X,y,Xs)
    best=y.max(); z=(mu-best)/sd; ei=(mu-best)*stats.norm.cdf(z)+sd*stats.norm.pdf(z)
    ax.plot(Xs,obj(Xs),color=TEAL,ls='--',lw=1.1,label='objective' if step==0 else None)
    ax.fill_between(Xs,mu-1.96*sd,mu+1.96*sd,color=NAVY,alpha=.18)
    ax.plot(Xs,mu,color=NAVY,lw=1.8,label='posterior mean' if step==0 else None)
    ax.plot(X,y,'o',color=RUST,ms=5,label='evaluations' if step==0 else None)
    xn=Xs[np.argmax(ei)]
    ax.axvline(xn,color=GOLD,lw=1.4,label='next point' if step==0 else None)
    ax2=ax.twinx(); ax2.fill_between(Xs,0,ei/ei.max(),color=GOLD,alpha=.22); ax2.set_yticks([]); ax2.grid(False)
    ax.set_title('step %d'%(step+1),fontsize=10)
    X=np.append(X,xn); y=np.append(y,obj(xn))
axes[0,0].legend(frameon=False,fontsize=8,loc='lower center')
for ax in axes[1]: ax.set_xlabel('$x$')
fig.suptitle('Bayesian optimization: the posterior chooses the next experiment',y=1.00,fontsize=12)
finish(fig,'app_bayesopt'); print('  best found x=%.3f'%X[np.argmax(y)])
print("BATCH 3 DONE")
