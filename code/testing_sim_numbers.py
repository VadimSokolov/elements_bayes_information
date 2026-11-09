import pathlib
HERE = pathlib.Path(__file__).resolve().parent
FIGS = pathlib.Path(__file__).resolve().parents[1] / "figs"
import numpy as np, json, itertools, math
from scipy.special import comb
rng=np.random.default_rng(2602)
p0,p1=0.5,0.65
up,dn=math.log(p1/p0),math.log((1-p1)/(1-p0))
mu=p1*up+(1-p1)*dn; var=p1*(1-p1)*(up-dn)**2
out={'mu':mu,'sigma2':var}
# Table: stopping times under P1, 200k reps
R=200_000; T=4000
rows=[]
X=None
S=np.zeros(R); tau={b:np.full(R,-1) for b in [10,20,50,100,200]}
for t in range(1,T+1):
    x=rng.random(R)<p1
    S+=np.where(x,up,dn)
    for b in tau:
        m=(tau[b]<0)&(S>=math.log(b)); tau[b][m]=t
    if all((tau[b]>=0).all() for b in tau): break
for b in [10,20,50,100,200]:
    tb=tau[b].astype(float); lb=math.log(b)
    rows.append(dict(b=b,pred=lb/mu,mean=tb.mean(),median=float(np.median(tb)),sd=tb.std(),resid=(tb.mean()-lb/mu)/math.sqrt(lb),
        sd_pred=math.sqrt(var/mu**3*lb), unfinished=int((tau[b]<0).sum())))
out['table']=rows
# corollary tail bound check at b=20, t= 2*log b/mu
b=20; lb=math.log(b); tb=tau[20]
# rho is the length of the range of the increments, so Hoeffding's inequality applies
rho=up-dn; out['rho']=rho
for t in [100,150,200]:
    out.setdefault('tail',[]).append(dict(t=t,emp=float((tb>t).mean()),
        hoeffding=math.exp(-2*(mu*t-lb)**2/(t*rho**2)),gaussian=math.exp(-(mu*t-lb)**2/(2*var*t))))
# H1 500 paths T=200 crossing log 20
R2=500; x=rng.random((R2,200))<p1; St=np.cumsum(np.where(x,up,dn),1)
cross=(St>=lb).any(1); first=np.where(cross,(St>=lb).argmax(1)+1,np.nan)
out['h1_500']=dict(frac=float(cross.mean()),median=float(np.nanmedian(first)))
# Type I optional stopping under H0, T=500, 10k reps: LR and ML ratio
R3=10_000; T3=500
x=(rng.random((R3,T3))<p0).astype(float)
S0=np.cumsum(np.where(x==1,up,dn),1)
lr_rej=(S0>=lb).any(1).mean()
k=np.cumsum(x,1); t=np.arange(1,T3+1)
th=k/t
with np.errstate(divide='ignore',invalid='ignore'):
    ll=np.where(k>0,k*np.log(th),0)+np.where(t-k>0,(t-k)*np.log(1-th),0)
mlr=ll-t*math.log(0.5)
ml_rej=(mlr>=lb).any(1).mean()
# NML-normalized (fixed horizon 500) ratio, valid at fixed time but check optional stopping too
out['typeI']=dict(LR=float(lr_rej),LR_se=float(math.sqrt(lr_rej*(1-lr_rej)/R3)),ML=float(ml_rej),ML_se=float(math.sqrt(ml_rej*(1-ml_rej)/R3)))
# prequential KT mixture (Beta(1/2,1/2)) e-process under same stopping
from scipy.special import betaln
kt=betaln(k+0.5,t-k+0.5)-betaln(0.5,0.5)-t*math.log(0.5)
out['typeI']['KT']=float((kt>=lb).any(1).mean())
# misspecification
pt,pa=0.55,0.80
up2,dn2=math.log(pa/p0),math.log((1-pa)/(1-p0))
d=pt*up2+(1-pt)*dn2
x=rng.random((500,300))<pt; Sm=np.cumsum(np.where(x,up2,dn2),1)
out['misspec']=dict(drift=d,cross=int((Sm>=lb).any(1).sum()),KLt0=pt*math.log(pt/p0)+(1-pt)*math.log((1-pt)/(1-p0)),KLt1=pt*math.log(pt/pa)+(1-pt)*math.log((1-pt)/(1-pa)))
# NML horizon dependence q2^(N)(0|0)
def nml_marg(prefix,N):
    # sum over completions of PNML^(N)
    C=sum(comb(N,j)*((j/N)**j)*((1-j/N)**(N-j)) for j in range(N+1))
    s=0.0; m=len(prefix); k0=sum(prefix)
    for j in range(N-m+1):
        k=k0+j
        s+=comb(N-m,j)*((k/N)**k)*((1-k/N)**(N-k))
    return s/C
q=[]
for N in range(2,8):
    q.append(round(nml_marg([0,0],N)/nml_marg([0],N),4))
out['nml_q2']=q
out['C']=[sum(comb(N,j)*((j/N)**j)*((1-j/N)**(N-j)) for j in range(N+1)) for N in (1,2,3)]
# Brier
out['brier']=0.5*math.exp(-5/8)+0.5*math.exp(3/8); out['brier100']=out['brier']**100
# misspecification: Lundberg exponent, Ville bound b^(-theta*), and 100,000 paths over 1500 observations
from scipy.optimize import brentq
th=brentq(lambda s: pt*math.exp(s*up2)+(1-pt)*math.exp(s*dn2)-1,1e-6,20)
rng_ms=np.random.default_rng(2603); R5=100_000; T5=1500; bs=[20,100,1000]
Sm=np.zeros(R5); hit={b:np.full(R5,-1) for b in bs}
for t in range(1,T5+1):
    Sm+=np.where(rng_ms.random(R5)<pt,up2,dn2)
    for b in bs:
        m=(hit[b]<0)&(Sm>=math.log(b)); hit[b][m]=t
ms=out['misspec']; ms['theta_star']=th
ms['lundberg']={b:b**(-th) for b in bs}
ms['sim_cross']={b:float((hit[b]>=0).mean()) for b in bs}
ms['median_cross_time_b20']=float(np.median(hit[20][hit[20]>=0]))
print(json.dumps(out,indent=1,default=float))
json.dump(out,open(HERE/'testing_sim_numbers.json','w'),indent=1,default=float)
