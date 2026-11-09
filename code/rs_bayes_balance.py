# Rubin--Sethuraman balance for the KS test (Section np:sec:rs-mc): the table of empirical Bayes-risk
# minimizers hat c_n^* and Figure np:fig:rs-bayes (figs/rs_bayes_balance.pdf). Null F_0 = standard normal,
# Pitman shift alternative F_0(x - mu_n) with mu_n = sqrt(4 pi log n / n), equal prior weights.
import numpy as np, matplotlib as mpl, pathlib
from scipy import stats
mpl.use('Agg'); import matplotlib.pyplot as plt
mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'serif',
 'font.serif':['CMU Serif','DejaVu Serif'],'mathtext.fontset':'cm','font.size':10.5,'axes.titlesize':11,
 'axes.labelsize':10.5,'legend.fontsize':8.6,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':0.8,'axes.edgecolor':'#333','figure.dpi':200,'axes.grid':True,'grid.color':'#ddd','grid.linewidth':0.6})
NAVY='#26456e'; RUST='#c0562f'; TEAL='#2a8f82'; GOLD='#d4a11e'; GRAY='#6b6b6b'; PLUM='#7b5ea7'
FIGS=pathlib.Path(__file__).resolve().parents[1]/'figs'
rng=np.random.default_rng(1965)

def ks_stats(n, reps, shift, chunk=200):
    """sqrt(n) sup_x |F_n(x) - Phi(x)| for reps samples of size n from N(shift, 1)."""
    out=[]; i=np.arange(1,n+1)
    for k in range(0,reps,chunk):
        m=min(chunk,reps-k)
        u=stats.norm.cdf(np.sort(rng.standard_normal((m,n))+shift,axis=1))
        out.append(np.sqrt(n)*np.maximum((i/n-u).max(axis=1),(u-(i-1)/n).max(axis=1)))
    return np.concatenate(out)

c_grid=np.round(np.arange(0.50,4.0+1e-9,0.01),2)
def curves(T0,T1):
    a=(T0[:,None]>c_grid[None,:]).mean(axis=0); b=(T1[:,None]<=c_grid[None,:]).mean(axis=0)
    return a,b,0.5*a+0.5*b

print(f"{'n':>6} {'Delta_n':>8} {'exact':>7} {'c_hat':>6} {'Delta/2':>8} {'B(c_hat)':>9} {'max T0':>7} {'min T1':>7} {'2e^-2c^2':>9}")
rows=[]
for n in [500,1000,2000,5000,10000]:
    mu=np.sqrt(4*np.pi*np.log(n)/n); D=np.sqrt(2*np.log(n)); Dx=np.sqrt(n)*(2*stats.norm.cdf(mu/2)-1)
    T0=ks_stats(n,1000,0.0); T1=ks_stats(n,1000,mu)
    a,b,B=curves(T0,T1); j=int(np.argmin(B)); c=c_grid[j]
    print(f"{n:>6d} {D:>8.3f} {Dx:>7.3f} {c:>6.2f} {D/2:>8.3f} {B[j]:>9.4f} {T0.max():>7.3f} {T1.min():>7.3f} {2*np.exp(-2*c*c):>9.2e}")
    rows.append(f"${n}$ & ${D:.2f}$ & ${c:.2f}$ & ${D/2:.2f}$\\\\")
print("\n".join(rows))

n=2000; mu=np.sqrt(4*np.pi*np.log(n)/n); D=np.sqrt(2*np.log(n))
T0=ks_stats(n,2000,0.0); T1=ks_stats(n,2000,mu)
a,b,B=curves(T0,T1); j=int(np.argmin(B)); c_hat=c_grid[j]
print(f"figure n={n}: Delta_n={D:.3f}, Delta_n/2={D/2:.3f}, minimizer={c_hat:.2f}, B min={B[j]:.4f}, "
      f"max null T={T0.max():.3f}, min alt T={T1.min():.3f}")
fig,ax=plt.subplots(figsize=(6.4,3.9))
ax.semilogy(c_grid,a,color=NAVY,lw=2,label=r'$\alpha_n(c)$, empirical')
ax.semilogy(c_grid,b,color=RUST,lw=2,label=r'$\beta_n(c)$, empirical')
ax.semilogy(c_grid,B,color=TEAL,lw=2.4,label=r'$B_n(c)=\frac{1}{2}\alpha_n+\frac{1}{2}\beta_n$')
ax.semilogy(c_grid,np.minimum(2*np.exp(-2*c_grid**2),1),color=NAVY,lw=1.2,ls='--',label=r'$2e^{-2c^2}$')
cc=c_grid[c_grid<D]
ax.semilogy(cc,np.minimum(2*np.exp(-2*(D-cc)**2),1),color=RUST,lw=1.2,ls='--',label=r'$2e^{-2(\Delta_n-c)^2}$')
ax.axvline(D/2,color=GRAY,lw=1,ls=':'); ax.axvline(c_hat,color=GOLD,lw=1.2)
ax.text(D/2,1.6,r'$\Delta_n/2$',ha='right',va='bottom',fontsize=9,color=GRAY)
ax.text(c_hat,1.6,' empirical minimizer',ha='left',va='bottom',fontsize=9,color=GOLD)
ax.set_ylim(1e-4,3); ax.set_xlim(c_grid[0],c_grid[-1])
ax.set_xlabel('threshold $c$'); ax.set_ylabel('error probability')
ax.set_title(r'KS test at $n=2000$, $\Delta_n=\sqrt{2\log n}$, $\pi_0=\pi_1=\frac{1}{2}$')
ax.legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(0.5,-0.17),ncol=3)
fig.tight_layout(); fig.savefig(FIGS/'rs_bayes_balance.pdf',bbox_inches='tight'); plt.close(fig); print('wrote rs_bayes_balance')
