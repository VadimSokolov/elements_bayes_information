import pathlib
HERE = pathlib.Path(__file__).resolve().parent
FIGS = pathlib.Path(__file__).resolve().parents[1] / "figs"
import numpy as np, math, json
from scipy import stats
from scipy.special import betaln, gammaln
out={}
# 1 diagnosis
def ppv(prev,se,sp): return se*prev/(se*prev+(1-sp)*(1-prev))
out['mammo']=dict(ppv=ppv(.01,.9,.91), LRp=.9/.09, LRn=.1/.91, npv=(.91*.99)/(.91*.99+.1*.01))
# second independent positive test
p1=ppv(.01,.9,.91); out['mammo']['ppv2']=ppv(p1,.9,.91)
# natural frequencies of 1000
out['mammo']['freq']=dict(tp=.9*10, fp=.09*990)
# 2 thresholds: B=benefit treating diseased, H=harm treating healthy, test sens/spec, risk r (same units)
B,H,se,sp,r=1.0,0.25,0.85,0.9,0.01
pt=H/(H+B); plo=((1-sp)*H+r)/((1-sp)*H+se*B); pup=(sp*H-r)/(sp*H+(1-se)*B)
out['thr']=dict(B=B,H=H,se=se,sp=sp,r=r,treat=pt,test_low=plo,test_up=pup)
# 3 decision curves: simulate
rng=np.random.default_rng(3); n=200_000
x=rng.normal(size=n); lin=-2.0+1.2*x; p=1/(1+np.exp(-lin)); y=rng.random(n)<p
over=1/(1+np.exp(-(-2.0+2.4*x+0.8)))   # overconfident and shifted
auc=lambda s: (stats.rankdata(s)[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum())
out['dca']=dict(prev=float(y.mean()),auc=float(auc(p)),auc_over=float(auc(over)))
def nb(score,t): tp=((score>=t)&y).mean(); fp=((score>=t)&~y).mean(); return tp-fp*t/(1-t)
ts=np.round(np.arange(0.02,0.61,0.01),2)
out['dca']['curve']={'t':ts.tolist(),'model':[nb(p,t) for t in ts],'over':[nb(over,t) for t in ts],'all':[float(y.mean()-(1-y.mean())*t/(1-t)) for t in ts]}
for t in (0.1,0.2,0.3): out['dca'][f'at{t}']=dict(model=nb(p,t),over=nb(over,t),all=float(y.mean()-(1-y.mean())*t/(1-t)))
# 4 trial with priors: HR 0.80 (0.64,1.00)
b=math.log(.8); s=(math.log(1.0)-math.log(.64))/(2*1.96)
sig=-math.log(.8)/1.645
pri={'reference':(0,1e6),'skeptical':(0,sig),'enthusiastic':(math.log(.8),sig)}
res={}
for k,(m0,s0) in pri.items():
    v=1/(1/s**2+1/s0**2); m=v*(b/s**2+m0/s0**2); sd=math.sqrt(v)
    res[k]=dict(post_HR=math.exp(m),lo=math.exp(m-1.96*sd),hi=math.exp(m+1.96*sd),P_lt1=stats.norm.cdf(-m/sd),P_lt09=stats.norm.cdf((math.log(.9)-m)/sd))
out['trial']=dict(se=s,prior_sd=sig,res=res)
# 5 single arm predictive probability design
p0,Nmax,a0,b0,thT,thL=0.2,40,1,1,0.95,0.05
looks=[10,20,30,40]
def post_succ(x,n): return 1-stats.beta.cdf(p0,a0+x,b0+n-x)
def PP(x,n):
    m=Nmax-n; ys=np.arange(m+1)
    pm=np.exp(gammaln(m+1)-gammaln(ys+1)-gammaln(m-ys+1)+betaln(a0+x+ys,b0+n-x+m-ys)-betaln(a0+x,b0+n-x))
    return float((pm*(np.array([post_succ(x+yy,Nmax) for yy in ys])>thT)).sum())
# precompute final success count threshold
xs_final=min(x for x in range(Nmax+1) if post_succ(x,Nmax)>thT)
out['design']=dict(final_needed=xs_final)
def sim(p,R=100000,seed=1):
    r=np.random.default_rng(seed); X=(r.random((R,Nmax))<p).cumsum(1)
    succ=0; N=np.zeros(R); stopped=np.zeros(R,bool); win=np.zeros(R,bool)
    cache={}
    for L in looks:
        x=X[:,L-1]
        if L<Nmax:
            pp=np.array([cache.setdefault((xx,L),PP(xx,L)) for xx in range(L+1)])[x]
            fut=(~stopped)&(pp<thL); N[fut]=L; stopped|=fut
        else:
            act=~stopped; N[act]=Nmax; win[act]=x[act]>=xs_final
    return dict(reject=float(win.mean()),EN=float(N.mean()),early=float((N<Nmax).mean()))
out['design']['null']=sim(0.2); out['design']['alt']=sim(0.4,seed=2)
# fixed design comparison: exact binomial alpha at final threshold
out['design']['fixed_alpha']=float(stats.binom.sf(xs_final-1,Nmax,0.2)); out['design']['fixed_power']=float(stats.binom.sf(xs_final-1,Nmax,0.4))
# 6 safety surveillance: Poisson counts per week, baseline mean lam0=2, gamma mixture e-process on RR
lam0=2.0; a,bb=1.0,1.0  # prior on RR ~ Gamma(1,1)
def eproc(counts):
    S=np.cumsum(counts,1); T=np.arange(1,counts.shape[1]+1)*lam0
    # marginal under Poisson(RR*lam0) with RR~Gamma(a,b) over Poisson(lam0): ratio of
    logm=gammaln(a+S)-gammaln(a)+a*np.log(bb)-(a+S)*np.log(bb+T)+S*np.log(lam0)  # times prod lam0^c/c! cancels partly
    logn=-T+S*np.log(lam0)
    return logm-logn
r=np.random.default_rng(5); W=104
for rr,key in [(1.0,'null'),(1.5,'rr15'),(2.0,'rr2')]:
    C=r.poisson(rr*lam0,size=(20000,W)); E=eproc(C); hit=(E>=math.log(20)).any(1)
    first=np.where(hit,(E>=math.log(20)).argmax(1)+1,np.nan)
    out.setdefault('surv',{})[key]=dict(alarm=float(hit.mean()),median_week=float(np.nanmedian(first)) if hit.any() else None)
# 7 EVPI
m,sd,pop=0.05,0.10,100000
evpi=sd*stats.norm.pdf(m/sd)+m*stats.norm.cdf(m/sd)-max(m,0)
out['evpi']=dict(m=m,sd=sd,per_patient=evpi,population=evpi*pop,P_better=stats.norm.cdf(m/sd))
print(json.dumps({k:v for k,v in out.items() if k!='dca'},indent=1,default=float)); print({k:v for k,v in out['dca'].items() if k!='curve'})
json.dump(out,open(HERE/'med_numbers.json','w'),indent=1,default=float)
