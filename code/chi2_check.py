import numpy as np
from scipy.optimize import brentq
from scipy import stats
x=np.arange(1,7);
lam=brentq(lambda l:(x*np.exp(l*x)).sum()/np.exp(l*x).sum()-4.5,-5,5)
Ps=np.exp(lam*x); Ps/=Ps.sum()
H=-(Ps*np.log(Ps)).sum()
print('Brandeis dice tilt: lambda %.5f' % lam, 'p', np.round(Ps,5), 'mean %.4f' % (x@Ps), 'H %.5f' % H, 'log 6 %.5f' % np.log(6))
rng=np.random.default_rng(1)
for n in [200,1000]:
    C=rng.multinomial(n,Ps,size=2_000_000); keep=(C@x)==int(4.5*n); C=C[keep]
    Ph=C/n
    with np.errstate(divide='ignore',invalid='ignore'):
        D=np.where(Ph>0,Ph*np.log(Ph/Ps),0).sum(1)
    s=2*n*D; print(n,len(s),'mean',s.mean(),'var',s.var(),'KS vs chi2_4',stats.kstest(s,'chi2',args=(4,)).statistic)
