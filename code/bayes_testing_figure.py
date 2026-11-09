# Bayes risk balancing and the moderate deviation threshold.
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

# Gaussian testing: T ~ N(0,1) under H0, T ~ N(Delta,1) under H1, prior odds pi0/pi1 = b
def risk(c,Delta,b):
    pi0=b/(1+b); pi1=1/(1+b)
    alpha=stats.norm.sf(c)                 # type I
    beta=stats.norm.cdf(c-Delta)           # type II
    return pi0*alpha, pi1*beta, pi0*alpha+pi1*beta

fig,(a1,a2)=plt.subplots(1,2,figsize=(7.9,3.3))
b=1000.0; Delta=5.0
cs=np.linspace(0.0,Delta,900)
A,B,T=risk(cs,Delta,b)
a1.semilogy(cs,A,color=RUST,lw=2,label=r'$\pi_0\alpha(c)$  (false alarm)')
a1.semilogy(cs,B,color=TEAL,lw=2,label=r'$\pi_1\beta(c)$  (missed signal)')
a1.semilogy(cs,T,color=NAVY,lw=2.6,label='total Bayes risk')
cstar=cs[np.argmin(T)]
a1.axvline(cstar,color=GRAY,ls='--',lw=1.4)
a1.annotate('$c^*=%.2f$'%cstar,xy=(cstar,T.min()),xytext=(cstar+0.35,T.min()*22),
            fontsize=9.5,color='#333',arrowprops=dict(arrowstyle='->',color='#777',lw=0.9))
a1.set_xlabel('threshold $c$'); a1.set_ylabel('contribution to risk')
a1.set_title(r'(a)  The balance, $b=10^3$, $\Delta=5$')
a1.legend(frameon=False,fontsize=8.3,loc='lower left')

# the stationarity condition pi0 phi(c) = pi1 phi(c-Delta) gives c* = log(b)/Delta + Delta/2.
# At the detectability boundary Delta = sqrt(2 log b) this collapses to c* = sqrt(2 log b).
bs=np.logspace(1.5,10,40)
num=[];form=[]
for bb in bs:
    Dl=np.sqrt(2*np.log(bb))
    cc=np.linspace(0.05,3*Dl,20000)
    _,_,TT=risk(cc,Dl,bb); num.append(cc[np.argmin(TT)])
    form.append(np.log(bb)/Dl+Dl/2)
num=np.array(num); form=np.array(form)
a2.plot(np.log(bs),num,'o',ms=4.5,color=NAVY,label='numerical $c^*$')
a2.plot(np.log(bs),form,color=RUST,lw=2,label=r'$\log b/\Delta+\Delta/2$')
a2.plot(np.log(bs),np.sqrt(2*np.log(bs)),color=TEAL,lw=1.6,ls='--',label=r'$\sqrt{2\log b}$')
a2.set_xlabel(r'$\log b$  (log number of competing hypotheses)')
a2.set_ylabel('optimal threshold $c^*$')
a2.set_title(r'(b)  At $\Delta=\sqrt{2\log b}$')
a2.legend(frameon=False,fontsize=8.6)
fig.tight_layout(); fig.savefig(FIGS/'app_bayestesting.pdf',bbox_inches='tight'); plt.close(fig)
rel=np.abs(num-np.sqrt(2*np.log(bs)))/np.sqrt(2*np.log(bs))
print('c* at b=1e3, Delta=5: %.3f'%cstar)
print('max relative gap between numerical c* and sqrt(2 log b): %.4f'%rel.max())
print('max gap between numerical c* and the closed form: %.4f'%np.abs(num-form).max())
