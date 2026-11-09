import pathlib
HERE = pathlib.Path(__file__).resolve().parent
FIGS = pathlib.Path(__file__).resolve().parents[1] / "figs"
import numpy as np, math, json
from scipy import integrate, optimize, stats
out={}
# (1) exact Bayes boundary for k dims, w(theta)=||theta||^lam * exp(-||theta||^2/2) (prior times loss), pi0*L=1, thetahat ~ N(theta, I/n)
def boundary(n,k,lam):
    # reject iff int w(theta) f(thetahat|theta) dtheta > f(thetahat|0); radial symmetric -> function of r=||thetahat||
    def num(r):
        # Monte Carlo free: integrate over theta = r e1 + u/sqrt(n); use quadrature in polar for k=1, gauss-hermite in k dims
        z,wq=np.polynomial.hermite_e.hermegauss(40); wq=wq/np.sqrt(2*np.pi)
        if k==1:
            th=r+z/np.sqrt(n); vals=np.abs(th)**lam*np.exp(-th**2/2)
            return (wq*vals).sum()
        grids=np.meshgrid(*[z]*k,indexing='ij'); W=np.ones_like(grids[0])
        for i in range(k): W=W*wq[(slice(None),)+(None,)*0][...,] if False else W
        Wg=np.ones([40]*k)
        for i in range(k):
            shape=[1]*k; shape[i]=40; Wg=Wg*wq.reshape(shape)
        th0=grids[0]/np.sqrt(n)+r; rr=np.sqrt(th0**2+sum(g**2 for g in grids[1:])/n)
        return (Wg*rr**lam*np.exp(-rr**2/2)).sum()
    def g(r):  # log num - log null density
        return math.log(num(r)) - (k/2*math.log(n/(2*math.pi)) - n*r*r/2)
    rs=math.sqrt((k+lam)*math.log(n)/n)
    return optimize.brentq(g,1e-6,5*rs+1)
rows=[]
for k,lam in [(1,0),(1,2),(2,0),(3,0)]:
    for n in [10**2,10**3,10**4,10**5,10**6]:
        r=boundary(n,k,lam); a=math.sqrt((k+lam)*math.log(n)/n)
        rows.append(dict(k=k,lam=lam,n=n,exact=r,asym=a,ratio=r/a, tsq=n*r*r))
out['boundary']=rows
# (2) Bayes risk order for k=1, lam=0: risk(n) of optimal rule; compare with C(log n/n)^{1/2}
def risk(n,c):  # reject |sqrt n thetahat|>c ; prior on alt N(0,1) density times loss 1, pi0 L=1
    t1=2*stats.norm.sf(c)
    f=lambda th: stats.norm.pdf(th)*(stats.norm.cdf(c-math.sqrt(n)*th)-stats.norm.cdf(-c-math.sqrt(n)*th))
    t2=integrate.quad(f,-1,1,points=[0],limit=200)[0]*1
    t2+=integrate.quad(f,1,10)[0]*2*0  # negligible
    return t1,t2
rk=[]
for n in [10**2,10**3,10**4,10**5,10**6]:
    res=optimize.minimize_scalar(lambda c: sum(risk(n,c)),bounds=(0.5,8),method='bounded')
    c=res.x; t1,t2=risk(n,c)
    rk.append(dict(n=n,c=c,csq=c*c,logn=math.log(n),typeI=t1,typeII=t2,total=t1+t2,scaled=(t1+t2)/math.sqrt(math.log(n)/n)))
out['risk']=rk
# (3) Bayes risk efficiency: mean test vs sign test under N(theta,1)
# sign statistic: S=#positive ~ Bin(n,Phi(theta)); test rejects |S-n/2|>=m. Minimise over m.
def risk_sign(n):
    best=None
    ks=np.arange(n+1)
    for m in range(0,n//2+1):
        rej=np.abs(ks-n/2)>=m
        t1=stats.binom.pmf(ks,n,0.5)[rej].sum()
        f=lambda th: stats.norm.pdf(th)*stats.binom.pmf(ks,n,stats.norm.cdf(th))[~rej].sum()
        t2=integrate.quad(f,-1.5,1.5,limit=200)[0]
        r=t1+t2
        if best is None or r<best: best=r
        elif r>best*1.5: break
    return best
def risk_mean(n): return optimize.minimize_scalar(lambda c: sum(risk(n,c)),bounds=(0.5,8),method='bounded').fun
eff=[]
for ns in [50,100,200,400,800,1600]:
    rsg=risk_sign(ns)
    nm=optimize.brentq(lambda n: risk_mean(n)-rsg, 5, ns)
    eff.append(dict(n_sign=ns,risk=rsg,n_mean=nm,ratio=ns/nm))
out['efficiency']=eff; out['pi_over_2']=math.pi/2
print(json.dumps(out,indent=1))
json.dump(out,open(HERE/'testing_rs_numbers.json','w'),indent=1)
