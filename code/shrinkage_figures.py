# Shrinkage in pictures: posterior means, kappa densities, James Stein risk, baseball, global local heatmap, sparse recovery.
import numpy as np, matplotlib as mpl, pathlib
from scipy import stats, integrate
mpl.use('Agg'); import matplotlib.pyplot as plt
mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'serif',
 'font.serif':['CMU Serif','DejaVu Serif'],'mathtext.fontset':'cm','font.size':10.5,'axes.titlesize':11,
 'axes.labelsize':10.5,'legend.fontsize':8.6,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':0.8,'axes.edgecolor':'#333','figure.dpi':200,'axes.grid':True,'grid.color':'#ddd','grid.linewidth':0.6})
NAVY='#26456e'; RUST='#c0562f'; TEAL='#2a8f82'; GOLD='#d4a11e'; GRAY='#6b6b6b'; PLUM='#7b5ea7'
FIGS=pathlib.Path(__file__).resolve().parents[1]/'figs'
def fin(fig,n): fig.tight_layout(); fig.savefig(FIGS/(n+'.pdf'),bbox_inches='tight'); plt.close(fig); print('wrote',n)

# posterior means under four priors for y ~ N(beta,1)
lam=np.exp(np.linspace(-9,7,1600))                    # grid for local scale
def hs_mean(y,tau=1.0):
    v=tau**2*lam**2; w=stats.norm.pdf(y,0,np.sqrt(1+v))*2/(np.pi*(1+lam**2))*lam  # jacobian for log grid
    w/=w.sum(); return float(np.sum(w*v/(1+v))*y)
def num_mean(y,logprior):
    b=np.linspace(-14,14,5601); lp=stats.norm.logpdf(y,b,1)+logprior(b); p=np.exp(lp-lp.max()); p/=p.sum()
    return float(np.sum(p*b))
lasso=lambda b: -np.abs(b)*1.0
cauchy=lambda b: stats.cauchy.logpdf(b)
ridge=lambda b: stats.norm.logpdf(b,0,1.0)
ys=np.linspace(-6,6,241)
m_ridge=[num_mean(y,ridge) for y in ys]; m_lasso=[num_mean(y,lasso) for y in ys]
m_cauchy=[num_mean(y,cauchy) for y in ys]; m_hs=[hs_mean(y) for y in ys]
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.9,3.3))
a1.plot(ys,ys,color=GRAY,lw=1,ls=':',label='no shrinkage')
a1.plot(ys,m_ridge,color=NAVY,lw=2,label='Gaussian prior (ridge)')
a1.plot(ys,m_lasso,color=GOLD,lw=2,label='Laplace prior (lasso)')
a1.plot(ys,m_cauchy,color=PLUM,lw=2,label='Cauchy prior')
a1.plot(ys,m_hs,color=RUST,lw=2.3,label='horseshoe')
a1.set_xlabel('observation $y$'); a1.set_ylabel(r'posterior mean $E[\beta\mid y]$'); a1.set_title('(a)  Four shrinkage rules'); a1.legend(frameon=False,fontsize=7.8,loc='upper left')
a2.plot(ys,ys-np.array(m_ridge),color=NAVY,lw=2); a2.plot(ys,ys-np.array(m_lasso),color=GOLD,lw=2)
a2.plot(ys,ys-np.array(m_cauchy),color=PLUM,lw=2); a2.plot(ys,ys-np.array(m_hs),color=RUST,lw=2.3)
a2.axhline(0,color=GRAY,lw=0.8); a2.set_xlabel('observation $y$'); a2.set_ylabel(r'amount shrunk $y-E[\beta\mid y]$'); a2.set_title('(b)  Bounded against unbounded influence')
fin(fig,'shr_rules')
print('  shrinkage at y=6: ridge %.2f lasso %.2f cauchy %.2f horseshoe %.2f'%(6-m_ridge[-1],6-m_lasso[-1],6-m_cauchy[-1],6-m_hs[-1]))

# kappa densities: horseshoe Beta(1/2,1/2), Strawderman Berger, Laplace, Student t
k=np.linspace(0.005,0.995,500)
hs=stats.beta.pdf(k,0.5,0.5); sb=stats.beta.pdf(k,0.5,1.0)
# Laplace prior with scale 1: lambda^2 ~ Exp(1/2) => kappa density
def kappa_pdf_from_lam2(f_l2):   # lambda^2 = (1-k)/k
    l2=(1-k)/k; return f_l2(l2)/k**2
lap=kappa_pdf_from_lam2(lambda l2: 0.5*np.exp(-0.5*l2))
stt=kappa_pdf_from_lam2(lambda l2: stats.invgamma.pdf(l2,a=1.5,scale=1.5))   # t with 3 df
fig,ax=plt.subplots(figsize=(7.2,3.3))
for d,c,l in [(hs,RUST,r'horseshoe: $\mathrm{Beta}(\frac{1}{2},\frac{1}{2})$'),(sb,TEAL,r'Strawderman Berger: $\mathrm{Beta}(\frac{1}{2},1)$'),(lap,GOLD,'Laplace (lasso)'),(stt,PLUM,r'Student $t_3$')]:
    ax.plot(k,np.clip(d,0,6),color=c,lw=2.2,label=l); ax.fill_between(k,0,np.clip(d,0,6),color=c,alpha=.10)
ax.set_ylim(0,6); ax.set_xlabel(r'shrinkage weight $\kappa$    (0 = leave alone,  1 = shrink to zero)'); ax.set_ylabel('prior density')
ax.set_title(r'The prior on $\kappa$ is the shape of the rule'); ax.legend(frameon=False,fontsize=8.6)
fin(fig,'shr_kappa')

# James Stein risk
rng=np.random.default_rng(1)
def js_risk(p,theta,reps=4000):
    th=theta[None,:]; Y=th+rng.standard_normal((reps,p)); n2=(Y**2).sum(1,keepdims=True)
    js=(1-(p-2)/n2)*Y; jsp=np.maximum(1-(p-2)/n2,0)*Y
    return ((Y-th)**2).sum(1).mean(),((js-th)**2).sum(1).mean(),((jsp-th)**2).sum(1).mean()
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.9,3.3))
ps=np.arange(3,41,2); r=np.array([js_risk(p,np.zeros(p)) for p in ps])
a1.plot(ps,r[:,0],color=GRAY,lw=2,label='sample mean (risk $=p$)'); a1.plot(ps,r[:,1],color=NAVY,lw=2,label='James Stein')
a1.plot(ps,r[:,2],color=RUST,lw=2,ls='--',label='positive part'); a1.set_xlabel('dimension $p$'); a1.set_ylabel('total squared error risk')
a1.set_title(r'(a)  At $\theta=0$, risk collapses from $p$ to $2$'); a1.legend(frameon=False)
p=10; norms=np.linspace(0,12,25); r2=np.array([js_risk(p,np.r_[nm,np.zeros(p-1)]) for nm in norms])
a2.plot(norms,r2[:,0],color=GRAY,lw=2,label='sample mean'); a2.plot(norms,r2[:,1],color=NAVY,lw=2,label='James Stein')
a2.plot(norms,r2[:,2],color=RUST,lw=2,ls='--',label='positive part'); a2.set_xlabel(r'$\|\theta\|$'); a2.set_ylabel('risk, $p=10$')
a2.set_title('(b)  Dominance everywhere, gain near the origin'); a2.legend(frameon=False)
fin(fig,'shr_jamesstein')

# baseball (synthetic, Efron and Morris in spirit)
rng=np.random.default_rng(7); n=18; AB=45; truth=rng.normal(0.265,0.028,n); early=rng.binomial(AB,truth)/AB
rest=rng.binomial(400,truth)/400
mu=early.mean(); s2=mu*(1-mu)/AB; tau2=max(early.var()-s2,1e-5); B=s2/(s2+tau2); shr=mu+(1-B)*(early-mu)
o=np.argsort(-early)
fig,ax=plt.subplots(figsize=(7.2,3.6))
for i in o:
    ax.plot([0,1,2],[early[i],shr[i],rest[i]],color=GRAY,lw=0.8,alpha=.6)
ax.scatter(np.zeros(n),early,s=38,color=NAVY,zorder=3,label='first 45 at bats')
ax.scatter(np.ones(n),shr,s=38,color=RUST,zorder=3,label='shrunk estimate')
ax.scatter(np.full(n,2),rest,s=38,color=TEAL,zorder=3,label='rest of season')
ax.axhline(mu,color=GOLD,ls='--',lw=1.4,label='grand mean')
ax.set_xticks([0,1,2]); ax.set_xticklabels(['early average','shrunk','season outcome']); ax.set_ylabel('batting average')
ax.set_title('Shrinkage predicts the rest of the season'); ax.legend(frameon=False,fontsize=8.5,ncol=2)
fin(fig,'shr_baseball')
e_raw=np.mean((early-rest)**2); e_shr=np.mean((shr-rest)**2)
print('  baseball: prediction error raw %.5f shrunk %.5f  ratio %.2f  B=%.2f'%(e_raw,e_shr,e_raw/e_shr,B))

# global local heatmap: horseshoe posterior mean as a function of y and tau
taus=np.exp(np.linspace(np.log(0.02),np.log(3),120)); yy=np.linspace(0.15,6,118)
H=np.array([[hs_mean(y,t)/max(y,1e-9) for y in yy] for t in taus])
fig,ax=plt.subplots(figsize=(7.0,3.6))
im=ax.pcolormesh(yy,taus,H,cmap='viridis',vmin=0,vmax=1,shading='gouraud',rasterized=True); ax.set_yscale('log'); ax.grid(False)
cs=ax.contour(yy,taus,H,levels=[0.1,0.5,0.9],colors='white',linewidths=0.9); ax.clabel(cs,fmt='%.1f',fontsize=7.5,colors='white')
ax.set_xlabel('observation $|y|$'); ax.set_ylabel(r'global scale $\tau$'); ax.set_title(r'Fraction kept, $E[\beta\mid y]/y$, under the horseshoe')
cb=plt.colorbar(im,ax=ax); cb.set_label('fraction of $y$ retained'); ax.grid(False)
fin(fig,'shr_globallocal')

# sparse recovery
rng=np.random.default_rng(3); p=200; kk=12; beta=np.zeros(p); idx=rng.choice(p,kk,replace=False); beta[idx]=rng.choice([-1,1],kk)*rng.uniform(3,6,kk)
y=beta+rng.standard_normal(p)
tau_hat=0.25
def hs_post(yv,tau):
    v=tau**2*lam**2; w=stats.norm.pdf(yv,0,np.sqrt(1+v))*2/(np.pi*(1+lam**2))*lam; w/=w.sum()
    m=np.sum(w*v/(1+v))*yv; var=np.sum(w*(v/(1+v)+(v/(1+v)*yv)**2))-m**2
    return m,np.sqrt(max(var,1e-9))
post=np.array([hs_post(v,tau_hat) for v in y]); pm=post[:,0]; ps_=post[:,1]
noise=beta==0; cover=np.abs(pm-beta)<=1.96*ps_; kept=int((np.abs(pm[~noise])>np.abs(pm[noise]).max()).sum())
fig,(a1,a2)=plt.subplots(2,1,figsize=(7.6,4.4),sharex=True)
cols=np.where(beta!=0,RUST,NAVY)
a1.vlines(np.arange(p),0,y,color=cols,lw=0.9,alpha=.75); a1.scatter(idx,beta[idx],s=22,facecolor='none',edgecolor='#333',lw=0.9,zorder=4,label='true nonzero')
a1.set_ylabel('observation $y_i$'); a1.set_title('(a)  Twelve signals among two hundred noisy coordinates'); a1.legend(frameon=False,fontsize=8.5)
a2.errorbar(np.arange(p),pm,yerr=1.96*ps_,fmt='none',ecolor='#bbb',elinewidth=0.6)
a2.vlines(np.arange(p),0,pm,color=cols,lw=0.9,alpha=.85); a2.scatter(idx,beta[idx],s=22,facecolor='none',edgecolor='#333',lw=0.9,zorder=4)
a2.set_ylabel(r'horseshoe $E[\beta_i\mid y]$'); a2.set_xlabel('coordinate'); a2.set_title('(b)  Noise squashed, %d of %d signals kept'%(kept,kk))
fin(fig,'shr_sparse')
print('  sparse: mean |post| on noise %.3f (raw %.3f); mean |err| on signals %.3f (raw %.3f); mse raw %.3f hs %.3f'%(
  np.abs(pm[noise]).mean(),np.abs(y[noise]).mean(),np.abs(pm[~noise]-beta[~noise]).mean(),np.abs(y[~noise]-beta[~noise]).mean(),
  np.mean((y-beta)**2),np.mean((pm-beta)**2)))
print('  sparse: 95%% intervals cover %d of %d signals and %d of %d noise coordinates; %d signals above every noise posterior mean'%(
  cover[~noise].sum(),kk,cover[noise].sum(),noise.sum(),kept))
