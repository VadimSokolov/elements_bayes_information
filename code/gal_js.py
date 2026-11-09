# Gallery figure gal_js: James-Stein shrinkage of p = 12 normal means (Appendix T, "James--Stein shrinkage and empirical Bayes").
# Design: the means are the p quantiles theta_i = sqrt(3/2) Phi^{-1}((i-1/2)/p) of N(0, 3/2), a typical spread of the
# empirical-Bayes prior, and y_i ~ N(theta_i, 1) is one fixed draw (seed 1), so E||y||^2 = ||theta||^2 + p.
import numpy as np, matplotlib as mpl, pathlib
from scipy import stats
mpl.use('Agg'); import matplotlib.pyplot as plt
mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'serif',
 'font.serif':['CMU Serif','DejaVu Serif'],'mathtext.fontset':'cm','font.size':10.5,'axes.titlesize':11,
 'axes.labelsize':10.5,'legend.fontsize':8.6,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':0.8,'axes.edgecolor':'#333','figure.dpi':200,'axes.grid':True,'grid.color':'#ddd','grid.linewidth':0.6})
NAVY='#26456e'; RUST='#c0562f'; TEAL='#2a8f82'; GOLD='#d4a11e'; GRAY='#6b6b6b'; PLUM='#7b5ea7'
FIGS=pathlib.Path(__file__).resolve().parents[1]/'figs'
def fin(fig,n): fig.tight_layout(); fig.savefig(FIGS/(n+'.pdf'),bbox_inches='tight'); plt.close(fig); print('wrote',n)

p,tau2=12,1.5
rng=np.random.default_rng(1)
theta=np.sqrt(tau2)*stats.norm.ppf((np.arange(1,p+1)-0.5)/p)
y=theta+rng.normal(0.0,1.0,p)
S=float(y@y); B=1-(p-2)/S; js=B*y                      # James-Stein estimate (1-(p-2)/||y||^2) y
sse_mle=float(((y-theta)**2).sum()); sse_js=float(((js-theta)**2).sum())
# exact risk of James-Stein at these means: p-(p-2)^2 E[1/chi2_p(lam)], lam=||theta||^2,
# with E[1/chi2_p(lam)] = sum_k Poisson(k; lam/2)/(p+2k-2)
lam=float(theta@theta); k=np.arange(0,400)
risk_js=p-(p-2)**2*float(np.sum(stats.poisson.pmf(k,lam/2)/(p+2*k-2)))

fig,ax=plt.subplots(figsize=(5.2,3.2))
rows=np.arange(p)[::-1]
for r,a,b in zip(rows,y,js): ax.plot([a,b],[r,r],color=GRAY,lw=0.9,zorder=1)
ax.axvline(0,color='#222',lw=1.0,ls='--',zorder=0)
ax.scatter(y,rows,s=34,color=NAVY,zorder=3,label='MLE $y_i$')
ax.scatter(js,rows,s=34,color=RUST,marker='s',zorder=3,label='James\u2013Stein')
ax.set_yticks([]); ax.grid(axis='y',visible=False); ax.spines['left'].set_visible(False)
ax.set_xlabel('estimate'); ax.legend(frameon=True,loc='best')
fin(fig,'gal_js')
print('  p=%d tau2=%.2f ||theta||^2=%.2f E||y||^2=%.2f ||y||^2=%.2f factor=%.3f'%(p,tau2,lam,lam+p,S,B))
print('  this draw: summed squared error MLE %.2f James-Stein %.2f'%(sse_mle,sse_js))
print('  risk at these means: MLE %.2f James-Stein %.2f'%(p,risk_js))
