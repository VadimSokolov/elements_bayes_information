# Hierarchical model figures: varying intercepts and slopes, growth curves, small area mapping.
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
def finish(fig,n): fig.tight_layout(); fig.savefig(FIGS/(n+'.pdf'),bbox_inches='tight'); plt.close(fig); print('wrote',n)

# ---- 1) VARYING INTERCEPTS WITH A GROUP LEVEL PREDICTOR ------------------
rng=np.random.default_rng(101)
J=45                                   # counties
u=rng.uniform(-1.2,1.2,J)              # group level predictor
a_true=1.55+0.72*u+0.22*rng.standard_normal(J)
b_true=-0.62
nj=rng.integers(1,18,J)                # very unequal sample sizes
X=[];Y=[];G=[]
for j in range(J):
    x=rng.binomial(1,0.4,nj[j]); y=a_true[j]+b_true*x+0.75*rng.standard_normal(nj[j])
    X.append(x);Y.append(y);G.append(np.full(nj[j],j))
X=np.concatenate(X);Y=np.concatenate(Y);G=np.concatenate(G)
# Gibbs sampler for the varying intercept model with group level regression
S=6000; a=np.zeros(J); g0,g1=0.0,0.0; b=0.0; sy=1.0; sa=1.0
keep=np.zeros((S,J)); kg=np.zeros((S,2)); kv=np.zeros((S,2))
for s in range(S):
    for j in range(J):
        idx=G==j; n=idx.sum(); mu_j=g0+g1*u[j]
        prec=n/sy**2+1/sa**2
        mean=((Y[idx]-b*X[idx]).sum()/sy**2+mu_j/sa**2)/prec
        a[j]=mean+rng.standard_normal()/np.sqrt(prec)
    # group level regression of a on u
    Z=np.c_[np.ones(J),u]; V=np.linalg.inv(Z.T@Z/sa**2+np.eye(2)*1e-4)
    m=V@(Z.T@a/sa**2); g0,g1=rng.multivariate_normal(m,V)
    r=a-(g0+g1*u); sa=np.sqrt(1/rng.gamma(1+J/2, 1/(0.5*np.sum(r**2)+0.5)))
    res=Y-a[G]; Vb=1/(np.sum(X**2)/sy**2+1e-4); mb=Vb*np.sum(X*(Y-a[G]))/sy**2
    b=mb+rng.standard_normal()*np.sqrt(Vb)
    e=Y-a[G]-b*X; sy=np.sqrt(1/rng.gamma(1+len(Y)/2, 1/(0.5*np.sum(e**2)+0.5)))
    keep[s]=a; kg[s]=(g0,g1); kv[s]=(sy,sa)
burn=2000; apost=keep[burn:].mean(0); asd=keep[burn:].std(0); gm=kg[burn:].mean(0)
nopool=np.array([Y[G==j].mean() for j in range(J)])
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.8,3.4))
pred=gm[0]+gm[1]*u                      # what the group level model predicts for each group
sy_h,sa_h=kv[burn:].mean(0)
den=nopool-pred; num=apost-pred
ok=np.abs(den)>0.25
wj=1-num[ok]/den[ok]
a1.scatter(nj[ok],np.clip(wj,0,1),s=34,color=RUST,zorder=3,alpha=.75,label='groups')
ns=np.linspace(0.8,nj.max()+0.5,200)
a1.plot(ns,(sy_h**2/ns)/(sy_h**2/ns+sa_h**2),color=NAVY,lw=2,
        label=r'$(\sigma^2/n_j)\,/\,(\sigma^2/n_j+\sigma_\alpha^2)$')
a1.set_xlabel('observations in group $n_j$'); a1.set_ylabel('fraction pooled')
a1.set_ylim(-0.03,1.03)
a1.set_title('(a)  Small groups are pooled hardest')
a1.legend(frameon=False,fontsize=8,loc='upper right')
uu=np.linspace(-1.35,1.35,50)
a2.errorbar(u,apost,yerr=1.96*asd,fmt='o',color=NAVY,ms=4,lw=1,capsize=2,label='group posterior')
a2.plot(uu,gm[0]+gm[1]*uu,color=RUST,lw=2,label='group level regression')
a2.set_xlabel('group level predictor $u_j$'); a2.set_ylabel('intercept $\\alpha_j$')
a2.set_title('(b)  Borrowing through a group predictor')
a2.legend(frameon=False,fontsize=8.5)
finish(fig,'app_multilevel'); print('  g0=%.2f g1=%.2f (truth 1.55, 0.72)'%(gm[0],gm[1]))

# ---- 2) LONGITUDINAL GROWTH CURVES, RANDOM SLOPES ------------------------
rng=np.random.default_rng(103)
N=20; t=np.arange(0,7.0,1.0)
A=rng.normal(5.0,1.1,N); B=rng.normal(1.25,0.45,N)
Yl=np.array([A[i]+B[i]*t+0.45*rng.standard_normal(len(t)) for i in range(N)])
# hierarchical fit: conjugate normal hierarchy on (a_i, b_i) via Gibbs
S=4000; mu=np.array([5.,1.]); Sig=np.eye(2); sy=0.5
Z=np.c_[np.ones(len(t)),t]; coef=np.zeros((N,2)); keepm=np.zeros((S,2)); keepc=np.zeros((S,N,2))
for s in range(S):
    Si=np.linalg.inv(Sig)
    for i in range(N):
        V=np.linalg.inv(Z.T@Z/sy**2+Si); m=V@(Z.T@Yl[i]/sy**2+Si@mu)
        coef[i]=rng.multivariate_normal(m,V)
    Vm=np.linalg.inv(N*Si+np.eye(2)*1e-3); mm=Vm@(Si@coef.sum(0))
    mu=rng.multivariate_normal(mm,Vm)
    Sc=np.cov((coef-mu).T)*N+np.eye(2)
    Sig=stats.invwishart.rvs(df=N+3,scale=Sc,random_state=int(rng.integers(1e9)))
    e=np.array([Yl[i]-Z@coef[i] for i in range(N)])
    sy=np.sqrt(1/rng.gamma(1+e.size/2,1/(0.5*np.sum(e**2)+0.5)))
    keepm[s]=mu; keepc[s]=coef
bn=1500; mpost=keepm[bn:].mean(0); cpost=keepc[bn:].mean(0)
tt=np.linspace(0,6,120)
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.8,3.3))
for i in range(N):
    a1.plot(t,Yl[i],color=GRAY,lw=0.7,alpha=.55,marker='o',ms=2.5)
    a1.plot(tt,cpost[i,0]+cpost[i,1]*tt,color=NAVY,lw=0.9,alpha=.55)
pop=keepm[bn:]
a1.plot(tt,mpost[0]+mpost[1]*tt,color=RUST,lw=2.6,label='population mean curve')
a1.set_xlabel('time'); a1.set_ylabel('response'); a1.set_title('(a)  Individual and population growth')
a1.legend(frameon=False,fontsize=8.5,loc='upper left')
sl=keepc[bn:][:,:,1].mean(0)
a2.hist(sl,bins=12,color=NAVY,alpha=.55,density=True,label='individual slopes')
xs=np.linspace(pop[:,1].min()-.3,pop[:,1].max()+.3,200)
a2.plot(xs,stats.gaussian_kde(pop[:,1])(xs),color=RUST,lw=2.2,label='population slope posterior')
a2.set_xlabel('slope'); a2.set_yticks([]); a2.set_title('(b)  Where the variation lives')
a2.legend(frameon=False,fontsize=8.5)
finish(fig,'app_growth')

# ---- 3) SMALL AREA DISEASE MAPPING --------------------------------------
rng=np.random.default_rng(107)
g=12; xs_,ys_=np.meshgrid(np.arange(g),np.arange(g))
latent=0.45*np.sin(xs_/2.1)+0.35*np.cos(ys_/2.6)
E=rng.integers(2,22,(g,g))                    # expected counts, small and very unequal
theta=np.exp(latent+0.15*rng.standard_normal((g,g)))
O=rng.poisson(E*theta)
raw=O/E                                        # standardized ratio
# hierarchical Poisson with a lognormal random effect, Metropolis within Gibbs
S=3000; lg=np.zeros((g,g)); mu=0.0; tau=0.5; keep=np.zeros((g,g))
cnt=0
for s in range(S):
    prop=lg+0.25*rng.standard_normal((g,g))
    lpc=O*lg-E*np.exp(lg)-0.5*((lg-mu)/tau)**2
    lpp=O*prop-E*np.exp(prop)-0.5*((prop-mu)/tau)**2
    acc=np.log(rng.random((g,g)))<(lpp-lpc); lg=np.where(acc,prop,lg)
    mu=lg.mean()+tau/np.sqrt(lg.size)*rng.standard_normal()
    r=lg-mu; tau=np.sqrt(1/rng.gamma(1+lg.size/2,1/(0.5*np.sum(r**2)+0.5)))
    if s>1000: keep+=np.exp(lg); cnt+=1
smooth=keep/cnt
fig,axes=plt.subplots(1,3,figsize=(8.2,2.9))
for ax,M,ttl,cm in [(axes[0],raw,'(a)  Raw ratio $O_i/E_i$','RdYlBu_r'),
                    (axes[1],smooth,'(b)  Posterior mean risk','RdYlBu_r'),
                    (axes[2],E,'(c)  Expected counts $E_i$','Greys')]:
    im=ax.imshow(M,cmap=cm,origin='lower',interpolation='nearest')
    ax.set_title(ttl,fontsize=10.5); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    plt.colorbar(im,ax=ax,fraction=0.046)
finish(fig,'app_smallarea')
print('  raw sd=%.3f  smoothed sd=%.3f'%(raw.std(),smooth.std()))
print("BATCH 4 DONE")
