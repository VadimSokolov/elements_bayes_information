# Stern and Cover lottery: maximum entropy over C(49,6) combinations given inclusion marginals,
# fitted by generalized iterative scaling without enumerating a combination.
import numpy as np, matplotlib as mpl, pathlib
from math import comb
mpl.use('Agg'); import matplotlib.pyplot as plt
mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'serif',
 'font.serif':['CMU Serif','DejaVu Serif'],'mathtext.fontset':'cm','font.size':10.5,'axes.titlesize':11,
 'axes.labelsize':10.5,'legend.fontsize':8.8,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':0.8,'axes.edgecolor':'#333','figure.dpi':200,'axes.grid':True,'grid.color':'#ddd','grid.linewidth':0.6})
NAVY='#26456e'; RUST='#c0562f'; TEAL='#2a8f82'; GOLD='#d4a11e'; GRAY='#6b6b6b'
FIGS=pathlib.Path(__file__).resolve().parents[1]/'figs'
M,m=49,6
def esym(x,k):
    """elementary symmetric polynomials e_0..e_k of vector x"""
    E=np.zeros(k+1); E[0]=1.0
    for xi in x:
        for j in range(k,0,-1): E[j]+=xi*E[j-1]
    return E
def marginals(x):
    E=esym(x,m); pi=np.zeros(M)
    for i in range(M):
        # e_{m-1}(x_{-i}) by removing x_i: e_k(x_-i) = e_k(x) - x_i e_{k-1}(x_-i)
        Ei=np.zeros(m); Ei[0]=1.0
        for k in range(1,m): Ei[k]=E[k]-x[i]*Ei[k-1]
        pi[i]=x[i]*Ei[m-1]/E[m]
    return pi,E
# "true" popularity weights: birthday numbers overplayed, a few lucky numbers, high numbers underplayed
x_true=np.ones(M)
x_true[:31]=1.0; x_true[31:]=np.linspace(0.42,0.33,M-31)
for n,w in [(7,1.7),(3,1.25),(11,1.2),(13,0.75),(17,1.15),(21,1.1)]: x_true[n-1]*=w
x_true/=x_true.mean()
pi_obs,_=marginals(x_true)
# GIS from the uniform start
x=np.ones(M); hist=[]
for it in range(400):
    pi_x,_=marginals(x)
    hist.append(np.max(np.abs(pi_x-pi_obs)))
    x=x*(pi_obs/pi_x)**(1.0/m)
pi_fit,E=marginals(x)
xn=x/x.mean(); err_w=np.max(np.abs(xn-x_true))
hist=np.array(hist)
print('marginal error after %d iterations: %.2e ; weight recovery error: %.2e'%(len(hist),hist[-1],err_w))
# popularity of a combination relative to uniform: p*(c)*C(49,6) with p*(c)=prod x /e_6
def pop(c): return np.prod(x[np.array(c)-1])/E[m]*comb(M,m)
C=comb(M,m); J=40e6; N=20e6
def ret(c):
    p=np.prod(x[np.array(c)-1])/E[m]
    return J/C*(1-(1-p)**(N+1))/((N+1)*p)
def breakeven(c):
    p=np.prod(x[np.array(c)-1])/E[m]; return C*((N+1)*p)/(1-(1-p)**(N+1))
tickets={'Six birthday numbers (3, 7, 11, 17, 21, 31)':[3,7,11,17,21,31],
         'Typical mixed ticket (5, 14, 23, 29, 36, 44)':[5,14,23,29,36,44],
         'Six high numbers (33, 36, 40, 42, 45, 47)':[33,36,40,42,45,47],
         'The six highest numbers (44 to 49)':[44,45,46,47,48,49]}
rows=[]
for k,c in tickets.items():
    rows.append((k,pop(c),ret(c),breakeven(c)/1e6)); print('%-48s pop %7.3fx  return $%.2f  breakeven $%.1fM'%rows[-1])
print('most/least popular single numbers:',np.argsort(-pi_obs)[:3]+1,np.argsort(pi_obs)[:3]+1)
print('ratio of most to least played marginal: %.2f'%(pi_obs.max()/pi_obs.min()))
print('popularity ratio birthday/highest: %.1f'%(rows[0][1]/rows[3][1]))

fig,ax=plt.subplots(1,3,figsize=(8.6,2.75))
ax[0].bar(np.arange(1,M+1),pi_obs,color=np.where(np.arange(M)<31,NAVY,GRAY),width=0.8)
ax[0].axhline(m/M,color=RUST,ls='--',lw=1.3,label='uniform play $6/49$')
ax[0].set_xlabel('number'); ax[0].set_ylabel(r'inclusion marginal $\pi_i$'); ax[0].set_title('(a)  Synthetic marginals'); ax[0].legend(frameon=False)
ax[1].semilogy(hist,color=NAVY,lw=2); ax[1].set_xlabel('GIS iteration'); ax[1].set_ylabel(r'$\max_i|\hat\pi_i-\pi_i|$')
ax[1].set_title('(b)  Geometric convergence')
ax[2].plot(x_true,xn,'o',ms=3.5,color=RUST); lim=[x_true.min()*0.95,x_true.max()*1.05]
ax[2].plot(lim,lim,ls='--',color=GRAY,lw=1); ax[2].set_xlabel('true weight $x_i$'); ax[2].set_ylabel('recovered weight')
ax[2].set_title('(c)  Weights recovered')
fig.tight_layout(); fig.savefig(FIGS/'fig_lottery_gis.pdf',bbox_inches='tight'); plt.close(fig)

# return figure
Ns=np.logspace(5.5,8,200)
fig,ax=plt.subplots(1,3,figsize=(8.6,2.75))
cols=[RUST,NAVY,TEAL,GOLD]
for (k,c),col in zip(tickets.items(),cols):
    p=np.prod(x[np.array(c)-1])/E[m]
    r=[J/C*(1-(1-p)**(n+1))/((n+1)*p) for n in Ns]
    ax[0].plot(Ns/1e6,r,color=col,lw=1.8,label=k.split(' (')[0])
ax[0].set_xscale('log'); ax[0].set_xlabel('tickets sold (millions)'); ax[0].set_ylabel('expected return per \\$1 ticket')
ax[0].set_title('(a)  Return at a \\$40M jackpot'); ax[0].legend(frameon=False,fontsize=7.2)
pops=np.logspace(-2,1.2,200)
be=[C*((N+1)*(q/C))/(1-(1-q/C)**(N+1))/1e6 for q in pops]
ax[1].plot(pops,be,color=NAVY,lw=2)
for (k,c),col in zip(tickets.items(),cols): ax[1].plot([pop(c)],[breakeven(c)/1e6],'o',color=col,ms=6)
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('popularity (relative to uniform)'); ax[1].set_ylabel('break even jackpot (\\$M)')
ax[1].set_title('(b)  Break even against popularity')
# Pythagorean identity: D(q||p0)=D(q||p*)+D(p*||p0) for q in the constraint set, on a 3 point toy
p0=np.array([0.6,0.3,0.1]); 
def KL(a,b): return np.sum(a*np.log(a/b))
# constraint: q1 = 0.5 ; I-projection of p0 : tilt second coordinate
lo,hi=-20.0,20.0
for _ in range(200):
    l=(lo+hi)/2; q=p0*np.exp(l*np.array([0,1,0])); q/=q.sum()
    if q[1]<0.5: lo=l
    else: hi=l
best=(0,q)
ps=best[1]; ts=np.linspace(0.02,0.48,80); L=[];R=[]
for t in ts:
    q=np.array([t,0.5,0.5-t]); L.append(KL(q,p0)); R.append(KL(q,ps)+KL(ps,p0))
ax[2].plot(L,R,color=TEAL,lw=2); ax[2].plot([min(L),max(L)],[min(L),max(L)],ls='--',color=GRAY,lw=1)
ax[2].set_xlabel(r'$D(q\,\|\,p_0)$'); ax[2].set_ylabel(r'$D(q\,\|\,p^\star)+D(p^\star\|\,p_0)$'); ax[2].set_title('(c)  Pythagorean identity')
fig.tight_layout(); fig.savefig(FIGS/'fig_lottery_return.pdf',bbox_inches='tight'); plt.close(fig)
print('pythagorean max |lhs-rhs|: %.2e'%np.max(np.abs(np.array(L)-np.array(R))))
