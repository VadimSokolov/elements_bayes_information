# Double descent in a random feature model, and ordered empirical Bayes shrinkage.
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

# ---------- 1) DOUBLE DESCENT ----------
rng=np.random.default_rng(3)
n, d, ntest, sig = 80, 25, 3000, 0.6
beta=rng.standard_normal(d)/np.sqrt(d)
def data(m):
    X=rng.standard_normal((m,d)); y=X@beta+sig*rng.standard_normal(m); return X,y
def rf(X,W): return np.maximum(X@W,0)/np.sqrt(W.shape[1])
ps=np.unique(np.round(np.concatenate([np.linspace(5,75,18),np.linspace(76,86,9),
                                      np.linspace(88,400,22)])).astype(int))
REP=26
risk_min=np.zeros((REP,len(ps))); risk_ridge=np.zeros((REP,len(ps)))
for r in range(REP):
    Xtr,ytr=data(n); Xte,yte=data(ntest)
    for j,p in enumerate(ps):
        W=rng.standard_normal((d,p))
        Ztr=rf(Xtr,W); Zte=rf(Xte,W)
        # minimum norm least squares (ridgeless)
        co=np.linalg.pinv(Ztr,rcond=1e-12)@ytr
        risk_min[r,j]=np.mean((Zte@co-yte)**2)
        # ridge with a fixed positive penalty
        lam=0.05
        if p<=n: co2=np.linalg.solve(Ztr.T@Ztr+lam*np.eye(p),Ztr.T@ytr)
        else:    co2=Ztr.T@np.linalg.solve(Ztr@Ztr.T+lam*np.eye(n),ytr)
        risk_ridge[r,j]=np.mean((Zte@co2-yte)**2)
m1=np.median(risk_min,0); m2=np.median(risk_ridge,0)
q1=np.percentile(risk_min,25,0); q3=np.percentile(risk_min,75,0)
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.9,3.3))
a1.fill_between(ps/n,q1,q3,color=NAVY,alpha=.18)
a1.plot(ps/n,m1,color=NAVY,lw=2,label='minimum norm (no penalty)')
a1.plot(ps/n,m2,color=RUST,lw=2,label='ridge penalty')
a1.axvline(1.0,color=GRAY,ls='--',lw=1.2)
a1.annotate('interpolation\nthreshold  $\\psi=1$',xy=(1.0,m1.max()*0.62),xytext=(1.5,m1.max()*0.72),
            fontsize=8.5,color='#333',arrowprops=dict(arrowstyle='->',color='#777',lw=0.9))
a1.axhline(sig**2,color=TEAL,ls=':',lw=1.2,label='noise floor $\\sigma^2$')
a1.set_yscale('log'); a1.set_xlabel(r'$\psi=p/n$'); a1.set_ylabel('test risk')
a1.set_title('(a)  Double descent'); a1.legend(frameon=False,fontsize=8.3)
# effective complexity: trace of the hat matrix
eff=[]
Xtr,ytr=data(n)
for p in ps:
    W=rng.standard_normal((d,p)); Z=rf(Xtr,W); lam=0.05
    S=Z@np.linalg.solve(Z.T@Z+lam*np.eye(p),Z.T)
    eff.append(np.trace(S))
a2.plot(ps/n,eff,color=NAVY,lw=2)
a2.axvline(1.0,color=GRAY,ls='--',lw=1.2)
a2.axhline(n,color=TEAL,ls=':',lw=1.2,label='$n$ observations')
a2.set_xlabel(r'$\psi=p/n$'); a2.set_ylabel('effective degrees of freedom')
a2.set_title('(b)  Complexity saturates'); a2.legend(frameon=False,fontsize=8.5)
fig.tight_layout(); fig.savefig(FIGS/'app_doubledescent.pdf',bbox_inches='tight'); plt.close(fig)
pk=ps[np.argmax(m1)]/n
print('peak at psi=%.2f  peak risk %.3f  risk at psi=5 %.3f  ridge peak %.3f'%(
      pk,m1.max(),m1[-1],m2.max()))

# ---------- 2) ORDERED EMPIRICAL BAYES ----------
rng=np.random.default_rng(11)
p=600; k=70
mu=np.zeros(p); mu[:k]=rng.normal(0,3.2,k)
# two independent measurements per coordinate: order by the first, estimate with the second
z=mu+rng.standard_normal(p); z2=mu+rng.standard_normal(p)
o=np.argsort(np.abs(z)); zs=z[o]; z2s=z2[o]; mus=mu[o]; a=np.abs(zs)

# unconstrained local empirical Bayes: bin by |z| and estimate a weight per bin.
# with few observations per bin the estimate is noisy and not monotone.
B=26; edges=np.quantile(a,np.linspace(0,1,B+1)); edges[-1]+=1e-9
bid=np.clip(np.digitize(a,edges)-1,0,B-1)
w_bin=np.zeros(B); cnt=np.zeros(B); ctr=np.zeros(B)
for b in range(B):
    m=bid==b; cnt[b]=m.sum(); ctr[b]=a[m].mean()
    w_bin[b]=np.clip(1-1/max((z2s[m]**2).mean(),1e-6),0,1)
def pava_inc(y,w):
    lvl=[]; 
    for v,wv in zip(y,w):
        lvl.append([v,wv])
        while len(lvl)>1 and lvl[-2][0]>lvl[-1][0]:
            v2,w2=lvl.pop(); v1,w1=lvl.pop(); lvl.append([(v1*w1+v2*w2)/(w1+w2),w1+w2])
    out=[]
    for v,wv in lvl: out+=[v]
    res=[]; i=0
    for v,wv in lvl:
        res+=[v]*int(round(wv))
    return np.array(res)
# isotonic (nondecreasing in |z|) fit to the binned weights
w_iso_bins=pava_inc(w_bin,cnt)[:0] if False else None
# run PAVA on the bin values weighted by counts, then expand back to bins
vals=[];wts=[]
for v,wv in zip(w_bin,cnt):
    vals.append([v,wv])
    while len(vals)>1 and vals[-2][0]>vals[-1][0]:
        v2,w2=vals.pop(); v1,w1=vals.pop(); vals.append([(v1*w1+v2*w2)/(w1+w2),w1+w2])
iso=[]; 
for v,wv in vals: iso.append(v)
# map pooled blocks back onto the B bins in order
blocks=[]; 
for v,wv in vals: blocks.append((v,wv))
w_ord_bin=np.empty(B); idx=0
for v,wv in blocks:
    m=0; tot=0
    while idx<B and tot<wv-1e-9:
        tot+=cnt[idx]; w_ord_bin[idx]=v; idx+=1
w_ord=w_ord_bin[bid]

tau2=max((z.var()-1),1e-6); wg=tau2/(1+tau2)
th_glob=wg*zs; th_ord=w_ord*zs
r_glob=np.mean((th_glob-mus)**2); r_ord=np.mean((th_ord-mus)**2); r_mle=np.mean((zs-mus)**2)
# the same bin weights without the monotone constraint
r_bin=np.mean((w_bin[bid]*zs-mus)**2)

fig,(b1,b2)=plt.subplots(1,2,figsize=(7.9,3.3))
b1.plot(ctr,w_bin,'o',ms=5,color=GRAY,alpha=.85,label='unconstrained bin estimate')
b1.step(ctr,w_ord_bin,where='mid',color=RUST,lw=2.2,label='ordered (isotonic) fit')
b1.axhline(wg,color=NAVY,lw=1.8,ls='--',label='single global weight')
b1.set_xlabel(r'$|z|$'); b1.set_ylabel(r'weight $1-\kappa$ on $z$'); b1.set_ylim(-0.05,1.05)
b1.set_title('(a)  Ordering removes the noise'); b1.legend(frameon=False,fontsize=8.3,loc='lower right')
b2.bar(['no shrinkage','global','ordered'],[r_mle,r_glob,r_ord],color=[GRAY,NAVY,RUST],width=.6)
for i,v in enumerate([r_mle,r_glob,r_ord]): b2.text(i,v*1.02,'%.2f'%v,ha='center',fontsize=9.5)
b2.set_ylabel('mean squared error'); b2.set_title('(b)  Risk')
fig.tight_layout(); fig.savefig(FIGS/'app_orderedeb.pdf',bbox_inches='tight'); plt.close(fig)
print('risk  mle %.3f  global %.3f  ordered %.3f'%(r_mle,r_glob,r_ord))
print('risk  unconstrained bins %.3f'%r_bin)
print('bin weights monotone already?',bool(np.all(np.diff(w_bin)>=0)))
