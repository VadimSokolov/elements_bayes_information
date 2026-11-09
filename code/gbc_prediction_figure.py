# Generative Bayesian prediction: a quantile network trained by the check loss,
# compared against the exact posterior it never sees.
import numpy as np, matplotlib as mpl, pathlib
mpl.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,
 'font.family':'serif','font.serif':['CMU Serif','DejaVu Serif'],'mathtext.fontset':'cm',
 'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'legend.fontsize':9.5,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.8,
 'axes.edgecolor':'#333333','xtick.color':'#333333','ytick.color':'#333333',
 'axes.labelcolor':'#222222','text.color':'#222222','figure.dpi':200,
 'axes.grid':True,'grid.color':'#dddddd','grid.linewidth':0.6})
NAVY='#26456e'; RUST='#c0562f'; TEAL='#2a8f82'; GOLD='#d4a11e'; GRAY='#6b6b6b'
FIGS=pathlib.Path(__file__).resolve().parents[1]/'figs'
rng=np.random.default_rng(5)

# ---- model: theta ~ Gamma(a0,b0), y_i | theta ~ Poisson(theta), n obs.
a0,b0,n=2.0,1.0,12
# exact posterior given sufficient statistic s = sum y_i is Gamma(a0+s, b0+n)

# ---- simulate training pairs from the joint (prior predictive)
S=60000
th=rng.gamma(a0,1/b0,S)
s=rng.poisson(th*n)                       # sufficient statistic
tau=rng.random(S)

def feats(tau,s):
    # cosine embedding of tau (as in implicit quantile networks) + scaled statistic
    k=np.arange(1,5)[None,:]
    emb=np.cos(np.pi*k*tau[:,None])
    z=(np.log1p(s)-np.log1p(a0*n/b0))[:,None]
    return np.hstack([tau[:,None],emb,z,z**2])

X=feats(tau,s); Y=th
mu_x,sd_x=X.mean(0),X.std(0)+1e-9; Xn=(X-mu_x)/sd_x

# ---- tiny MLP trained with the pinball (check) loss, plain Adam
H=64; rs=np.random.default_rng(0)
W1=rs.standard_normal((Xn.shape[1],H))*np.sqrt(2/Xn.shape[1]); b1=np.zeros(H)
W2=rs.standard_normal((H,H))*np.sqrt(2/H); b2=np.zeros(H)
W3=rs.standard_normal((H,1))*np.sqrt(1/H); b3=np.zeros(1)
P=[W1,b1,W2,b2,W3,b3]; M=[np.zeros_like(p) for p in P]; V=[np.zeros_like(p) for p in P]
lr=2e-3; B=2048; t=0
for ep in range(9000):
    idx=rs.integers(0,S,B); xb=Xn[idx]; yb=Y[idx]; tb=tau[idx]
    h1=np.maximum(xb@P[0]+P[1],0); h2=np.maximum(h1@P[2]+P[3],0); q=(h2@P[4]+P[5]).ravel()
    u=yb-q
    dq=-(np.where(u>=0,tb,tb-1.0))/B                    # d(pinball)/dq
    g5=h2.T@dq[:,None]; g6=dq.sum(keepdims=True)
    d2=(dq[:,None]@P[4].T)*(h2>0)
    g3=h1.T@d2; g4=d2.sum(0)
    d1=(d2@P[2].T)*(h1>0)
    g1=xb.T@d1; g2=d1.sum(0)
    G=[g1,g2,g3,g4,g5,g6]; t+=1
    for i in range(6):
        M[i]=0.9*M[i]+0.1*G[i]; V[i]=0.999*V[i]+0.001*G[i]**2
        P[i]-=lr*(M[i]/(1-0.9**t))/(np.sqrt(V[i]/(1-0.999**t))+1e-8)
def _raw(tau_,s_):
    x=(feats(tau_,np.full_like(tau_,s_,dtype=float))-mu_x)/sd_x
    h1=np.maximum(x@P[0]+P[1],0); h2=np.maximum(h1@P[2]+P[3],0)
    return (h2@P[4]+P[5]).ravel()
GRID=np.linspace(0.001,0.999,600)
def gen(tau_,s_,w=41):
    # a quantile function is smooth and nondecreasing: lightly smooth the fitted
    # curve, then project to monotone, then interpolate
    q=_raw(GRID,s_)
    ker=np.ones(w)/w
    qs=np.convolve(np.pad(q,(w//2,w//2),mode='edge'),ker,mode='valid')[:len(q)]
    qs=np.maximum.accumulate(qs)
    return np.interp(tau_,GRID,qs)

fig,(a1,a2)=plt.subplots(1,2,figsize=(7.8,3.3))
tt=np.linspace(0.005,0.995,400)
for s_,c in [(6,NAVY),(18,TEAL),(40,GOLD)]:
    a1.plot(tt,gen(tt,s_),color=c,lw=2.1,label='generator, $s=%d$'%s_)
    a1.plot(tt,stats.gamma.ppf(tt,a0+s_,scale=1/(b0+n)),color=RUST,lw=1.1,ls='--')
a1.plot([],[],color=RUST,lw=1.1,ls='--',label='exact quantile function')
a1.set_xlabel(r'base uniform  $\tau$'); a1.set_ylabel(r'$\theta$')
a1.set_title('(a)  One network, every data set'); a1.legend(frameon=False,fontsize=8.3,loc='upper left')

lev=np.linspace(0.02,0.98,120)
for s_,c,mk in [(6,NAVY,'o'),(18,TEAL,'s'),(40,GOLD,'^')]:
    ex=stats.gamma.ppf(lev,a0+s_,scale=1/(b0+n))
    a2.plot(ex,gen(lev,s_),color=c,lw=2,label='$s=%d$'%s_)
lim=[0,4.6]
a2.plot(lim,lim,ls='--',color=RUST,lw=1.2,label='exact')
a2.set_xlim(lim); a2.set_ylim(lim)
a2.set_xlabel('exact posterior quantile'); a2.set_ylabel('generator quantile')
a2.set_title('(b)  Calibration of the generator')
a2.legend(frameon=False,fontsize=9,loc='upper left')
fig.tight_layout(); fig.savefig(FIGS/'app_gbc.pdf',bbox_inches='tight'); plt.close(fig)
for s_ in (6,18,40):
    ex=stats.gamma.ppf(lev,a0+s_,scale=1/(b0+n)); gq=gen(lev,s_)
    print('s=%2d  max abs quantile error %.4f  (posterior sd %.3f)'%(
        s_,np.abs(ex-gq).max(),np.sqrt((a0+s_))/(b0+n)))
