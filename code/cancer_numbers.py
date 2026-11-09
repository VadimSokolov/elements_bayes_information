import pathlib
HERE = pathlib.Path(__file__).resolve().parent
FIGS = pathlib.Path(__file__).resolve().parents[1] / "figs"
import numpy as np, math, json
from scipy import stats, optimize
out={}
# QALY annuity
for conv in ['annual_end','continuous']:
    f=(1-1.02**-30)/0.02 if conv=='annual_end' else (1-math.exp(-0.6))/0.02
    out[conv]=dict(factor=f,q95=0.95*f,q90=0.90*f)
# value today of a QALY thirty years out, annual discounting as in the annuity above
out['disc30']={r:(1+r)**-30 for r in (0.02,0.035,0.05)}
# Beta checks
def cal(a,b,ph,eps=0.10): return stats.beta.cdf(ph+eps,a,b)-stats.beta.cdf(ph-eps,a,b)
out['OC']=dict(mean=26/33,cal=cal(26,7,.78),ci=stats.beta.ppf([.025,.975],26,7).tolist())
out['CP']=dict(mean=4/20,cal=cal(4,16,.20),ci=stats.beta.ppf([.025,.975],4,16).tolist())
out['B199']=dict(mean=1/100,ci=stats.beta.ppf([.025,.975],1,99).tolist(),q975=stats.beta.ppf(.975,1,99),q99=stats.beta.ppf(.99,1,99),P_gt05=stats.beta.sf(.05,1,99))
# rare state: P(|pi-0.01|<=0.10)=P(pi<=0.11) under Beta(0.01n,0.99n) as the effective sample size n varies
rare=lambda n: cal(0.01*n,0.99*n,0.01)
rmin=optimize.minimize_scalar(lambda ln: rare(math.exp(ln)),bounds=(math.log(0.5),math.log(50)),method='bounded')
out['rare_cal']=dict(n_to_0=rare(1e-6),n_min=math.exp(rmin.x),min=rmin.fun,n_100=rare(100.0))
# anchor: mean ph, quantile q equals 0.05 -> solve n
def anchor(ph,q,thr=0.05):
    f=lambda ln: stats.beta.ppf(q,ph*math.exp(ln),(1-ph)*math.exp(ln))-thr
    L=np.linspace(math.log(0.5),math.log(1e5),3000); v=np.array([f(x) for x in L])
    roots=[optimize.brentq(f,L[i],L[i+1]) for i in range(len(L)-1) if v[i]*v[i+1]<0]
    n=math.exp(roots[-1]); return ph*n,(1-ph)*n,[math.exp(r) for r in roots]
for ph in (0.01,0.02):
    for q in (0.95,0.975,0.99):
        a,b,rts=anchor(ph,q); out[f'anchor_{ph}_{q}']=dict(a=a,b=b,n=a+b,roots=rts,ci=stats.beta.ppf([.025,.975],a,b).tolist(),P_gt05=stats.beta.sf(.05,a,b))
# paper's stated Beta(0.494,48.90) etc
for a,b in [(0.494,48.90),(0.225,22.28),(0.827,81.92)]:
    out[f'paper_{a}']=dict(q95=stats.beta.ppf(.95,a,b),q975=stats.beta.ppf(.975,a,b),q99=stats.beta.ppf(.99,a,b))
# PSMA
def post(p,lr): o=p/(1-p)*lr; return o/(1+o)
se,sp=0.80,0.995; out['psma']=dict(LRp=se/(1-sp),LRn=(1-se)/sp,post_pos=post(.01,se/(1-sp)),post_neg=post(.01,(1-se)/sp),post_pos_200=post(.01,200))
# wedge for Patient A: pi=0.02, dU tail = 21.3-10 (missed LNI branch 8 to 12 QALY)
for dC in (0.05,0.10):
    for br in (8,10,12):
        out.setdefault('wedge',[]).append(dict(dC=dC,branch=br,tail_gain=0.02*(21.3-br),kappa=(21.3-br)/dC))
# value of PSMA test before choosing RP+ vs RP-: loss(RP+)=dC, loss(RP-)=pi*dU
def vot(pi,dC=0.075,dU=11.3,se=.8,sp=.995):
    no=min(dC,pi*dU); ppos=pi*se+(1-pi)*(1-sp)
    pp=pi*se/ppos; pn=pi*(1-se)/(1-ppos)
    w=ppos*min(dC,pp*dU)+(1-ppos)*min(dC,pn*dU)
    return no,w,no-w
out['vot']={str(pi):vot(pi) for pi in (0.005,0.01,0.02,0.04,0.08)}
grid=np.linspace(0.001,0.2,400); out['vot_grid']=[vot(p)[2] for p in grid]; out['grid']=grid.tolist()
out['vot_max']=dict(pi=float(grid[int(np.argmax(out['vot_grid']))]),v=float(max(out['vot_grid'])))
print(json.dumps({k:v for k,v in out.items() if k not in ('vot_grid','grid')},indent=1,default=float))
json.dump(out,open(HERE/'cancer_numbers.json','w'),default=float)
