# Sufficiency as dimension reduction in generative Bayesian computation.
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

# Gamma-Poisson: theta ~ Gamma(a0,b0), y_i ~ Poisson(theta), n obs.
a0,b0,n=2.0,1.0,12
rng=np.random.default_rng(9)

def train(featfn, S=60000, iters=7000, seed=0):
    th=rng.gamma(a0,1/b0,S); Y=rng.poisson(th[:,None]*1.0,size=(S,n))
    tau=rng.random(S); X=featfn(tau,Y)
    mu,sd=X.mean(0),X.std(0)+1e-9; Xn=(X-mu)/sd
    rs=np.random.default_rng(seed); H=64
    P=[rs.standard_normal((Xn.shape[1],H))*np.sqrt(2/Xn.shape[1]),np.zeros(H),
       rs.standard_normal((H,H))*np.sqrt(2/H),np.zeros(H),
       rs.standard_normal((H,1))*np.sqrt(1/H),np.zeros(1)]
    M=[np.zeros_like(p) for p in P]; V=[np.zeros_like(p) for p in P]; t=0; B=2048
    for _ in range(iters):
        idx=rs.integers(0,S,B); xb=Xn[idx]; yb=th[idx]; tb=tau[idx]
        h1=np.maximum(xb@P[0]+P[1],0); h2=np.maximum(h1@P[2]+P[3],0); q=(h2@P[4]+P[5]).ravel()
        u=yb-q; dq=-(np.where(u>=0,tb,tb-1.0))/B
        g5=h2.T@dq[:,None]; g6=dq.sum(keepdims=True)
        d2=(dq[:,None]@P[4].T)*(h2>0); g3=h1.T@d2; g4=d2.sum(0)
        d1=(d2@P[2].T)*(h1>0); g1=xb.T@d1; g2=d1.sum(0)
        G=[g1,g2,g3,g4,g5,g6]; t+=1
        for i in range(6):
            M[i]=0.9*M[i]+0.1*G[i]; V[i]=0.999*V[i]+0.001*G[i]**2
            P[i]-=2e-3*(M[i]/(1-0.9**t))/(np.sqrt(V[i]/(1-0.999**t))+1e-8)
    def gen(tau_,Yrow):
        X=(featfn(tau_,np.repeat(Yrow[None,:],len(tau_),0))-mu)/sd
        h1=np.maximum(X@P[0]+P[1],0); h2=np.maximum(h1@P[2]+P[3],0)
        q=(h2@P[4]+P[5]).ravel()
        return q
    return gen

def emb(tau):
    k=np.arange(1,5)[None,:]; return np.cos(np.pi*k*tau[:,None])
def feat_suff(tau,Y):
    s=Y.sum(1); z=(np.log1p(s)-np.log1p(a0*n/b0))[:,None]
    return np.hstack([tau[:,None],emb(tau),z,z**2])
def feat_insuff(tau,Y):                 # the maximum alone: not sufficient
    m=Y.max(1); z=(np.log1p(m)-np.log1p(3.0))[:,None]
    return np.hstack([tau[:,None],emb(tau),z,z**2])

g_suff=train(feat_suff,seed=1); g_ins=train(feat_insuff,seed=2)
GRID=np.linspace(0.002,0.998,500)
def smooth_gen(g,Yrow,tau_):
    q=g(GRID,Yrow); w=41; ker=np.ones(w)/w
    qs=np.convolve(np.pad(q,(w//2,w//2),mode='edge'),ker,mode='valid')[:len(q)]
    return np.interp(tau_,GRID,np.maximum.accumulate(qs))

theta_true=1.5
Yobs=rng.poisson(theta_true,n); s_obs=Yobs.sum()
xs=np.linspace(0.4,3.2,400)
exact=stats.gamma.pdf(xs,a0+s_obs,scale=1/(b0+n))
lev=np.linspace(0.01,0.99,200)
q_suff=smooth_gen(g_suff,Yobs,lev); q_ins=smooth_gen(g_ins,Yobs,lev)
q_ex=stats.gamma.ppf(lev,a0+s_obs,scale=1/(b0+n))

fig,(a1,a2)=plt.subplots(1,2,figsize=(7.9,3.3))
a1.plot(q_ex,lev,color=RUST,lw=2.4,label='exact posterior')
a1.plot(q_suff,lev,color=NAVY,lw=2,ls='-',label=r'generator on $s=\sum y_i$ (sufficient)')
a1.plot(q_ins,lev,color=TEAL,lw=2,ls='--',label=r'generator on $\max y_i$ (insufficient)')
a1.set_xlabel(r'$\theta$'); a1.set_ylabel('posterior CDF')
a1.set_title('(a)  What an insufficient summary costs')
a1.legend(frameon=False,fontsize=8.2,loc='lower right')
# width comparison across many data sets
ws=[];wi=[];we=[]
for _ in range(120):
    th=rng.gamma(a0,1/b0); Y=rng.poisson(th,n); s=Y.sum()
    ex=stats.gamma.ppf([.05,.95],a0+s,scale=1/(b0+n))
    we.append(ex[1]-ex[0])
    qs=smooth_gen(g_suff,Y,np.array([.05,.95])); ws.append(qs[1]-qs[0])
    qi=smooth_gen(g_ins,Y,np.array([.05,.95])); wi.append(qi[1]-qi[0])
a2.boxplot([we,ws,wi],labels=['exact','sufficient','insufficient'],widths=.55,
           patch_artist=True,
           boxprops=dict(facecolor='#dfe6f0',color='#333'),medianprops=dict(color=RUST,lw=2))
a2.set_ylabel('width of 90 percent interval')
a2.set_title('(b)  Interval width over 120 data sets')
fig.tight_layout(); fig.savefig(FIGS/'app_sufficiency.pdf',bbox_inches='tight'); plt.close(fig)
print('median width  exact %.3f  sufficient %.3f  insufficient %.3f'%(
      np.median(we),np.median(ws),np.median(wi)))
