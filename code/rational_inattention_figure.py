# Matejka and McKay rational inattention, solved by Blahut Arimoto.
import numpy as np, itertools, matplotlib as mpl, pathlib
from scipy.optimize import brentq
mpl.use('Agg'); import matplotlib.pyplot as plt
mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'serif',
 'font.serif':['CMU Serif','DejaVu Serif'],'mathtext.fontset':'cm','font.size':10,'axes.titlesize':10.5,
 'axes.labelsize':10,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':0.8,'axes.edgecolor':'#333','figure.dpi':200,'axes.grid':True,'grid.color':'#ddd','grid.linewidth':0.6})
NAVY='#26456e'; RUST='#c0562f'; TEAL='#2a8f82'; GOLD='#d4a11e'; GRAY='#6b6b6b'
FIGS=pathlib.Path(__file__).resolve().parents[1]/'figs'
appeal=np.array([1.00,0.90,0.75,0.50,0.25,0.00]); J=6; sig=0.5
states=np.array(list(itertools.product([-sig,sig],repeat=J)))        # 64 equiprobable states
V=appeal[None,:]+states                                               # payoff of product j in state w
pw=np.full(len(states),1/len(states))
def BA(lam,iters=4000,start=None):
    P0=np.full(J,1/J) if start is None else start/start.sum()
    for _ in range(iters):
        W=P0[None,:]*np.exp((V-V.max(1,keepdims=True))/lam); Pjw=W/W.sum(1,keepdims=True)
        P0n=pw@Pjw
        if np.max(np.abs(P0n-P0))<1e-13: P0=P0n; break
        P0=P0n
    W=P0[None,:]*np.exp((V-V.max(1,keepdims=True))/lam); Pjw=W/W.sum(1,keepdims=True)
    I=np.sum(pw[:,None]*Pjw*np.log(np.clip(Pjw/np.clip(P0[None,:],1e-300,None),1e-300,None)))
    U=np.sum(pw[:,None]*Pjw*V)
    return P0,Pjw,I,U
lams=np.logspace(-1.6,1.0,400)
P0s=[];Is=[];Us=[]
for l in lams:
    P0,Pjw,I,U=BA(l,100000); P0s.append(P0); Is.append(I); Us.append(U)       # converged, so panel (a) shows the exits where they are
P0s=np.array(P0s); Is=np.array(Is); Us=np.array(Us)
def G(lam,P0):                                                        # G(a) of the consideration-set proposition
    K=np.exp((V-V.max(1,keepdims=True))/lam); return pw@(K/(K@P0)[:,None])
def exit_lam(j):                                                      # product j leaves where G_j=1 at the optimum over products 1..j
    f=lambda l: G(l,BA(l,100000,(np.arange(J)<j)*1.0)[0])[j]-1
    return brentq(f,0.02,10,xtol=1e-7) if f(0.02)>0 else np.nan
thr=[exit_lam(j) for j in range(1,J)]
print('exit thresholds lambda, products 2 to 6 (nan: never considered):',np.round(thr,3))
for l in (0.6,1.5):
    P0=BA(l)[0]; print('lambda=%.1f: P0'%l,np.round(P0,3),' G',np.round(G(l,P0),3))
for l in (0.05,1.5):
    P0,Pjw,I,U=BA(l); print('lambda=%.2f: I=%.3f nats (log 6 = %.3f), products alive=%d'%(l,I,np.log(J),(P0>1e-6).sum()))
fig=plt.figure(figsize=(8.6,4.6)); gs=fig.add_gridspec(2,3,height_ratios=[1.25,1])
a=fig.add_subplot(gs[0,0])
for j in range(J): a.plot(lams,P0s[:,j],lw=1.7,label='product %d'%(j+1))
a.set_xscale('log'); a.set_xlabel(r'price of attention $\lambda$'); a.set_ylabel('unconditional choice $P_0(j)$')
a.set_title('(a)  Products exit at sharp thresholds'); a.legend(frameon=False,fontsize=6.8,ncol=2)
b=fig.add_subplot(gs[0,1]); b.plot(Is,Us,color=NAVY,lw=2); b.set_xlabel('information $I$ (nats)'); b.set_ylabel('expected payoff')
b.set_title(r'(b)  The attention frontier, slope $\lambda$')
c=fig.add_subplot(gs[0,2]); c.plot(lams,Is,color=TEAL,lw=2); c.axhline(np.log(J),color=GRAY,ls='--',lw=1,label=r'$\log 6$')
c.set_xscale('log'); c.set_xlabel(r'$\lambda$'); c.set_ylabel('nats extracted'); c.set_title('(c)  Information bought'); c.legend(frameon=False)
for k,l in enumerate([0.05,0.5,1.5]):
    ax=fig.add_subplot(gs[1,k]); P0,Pjw,I,U=BA(l)
    order=np.argsort(-(V@np.ones(J)))[:24]
    im=ax.imshow(Pjw[order].T,aspect='auto',cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
    ax.set_yticks(range(J)); ax.set_yticklabels(['%d'%(j+1) for j in range(J)],fontsize=7); ax.set_xticks([])
    ax.set_xlabel('states of the world'); ax.set_title('(%s)  $\\lambda=%.2f$, $I=%.2f$'%('def'[k],l,I),fontsize=9.5); ax.grid(False)
    if k==0: ax.set_ylabel('product')
fig.tight_layout(); fig.savefig(FIGS/'fig_matejka_mckay.pdf',bbox_inches='tight'); plt.close(fig)
