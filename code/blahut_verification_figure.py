import numpy as np, matplotlib as mpl, pathlib
mpl.use('Agg'); import matplotlib.pyplot as plt
mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,
 'font.family':'serif','font.serif':['CMU Serif','DejaVu Serif'],'mathtext.fontset':'cm',
 'font.size':10.5,'axes.titlesize':11,'axes.labelsize':10.5,'legend.fontsize':9,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.8,
 'axes.edgecolor':'#333333','xtick.color':'#333333','ytick.color':'#333333',
 'axes.labelcolor':'#222222','text.color':'#222222','figure.dpi':200,
 'axes.grid':True,'grid.color':'#dddddd','grid.linewidth':0.6})
NAVY='#26456e'; RUST='#c0562f'; TEAL='#2a8f82'; GOLD='#d4a11e'; GRAY='#6b6b6b'
FIGS=pathlib.Path(__file__).resolve().parents[1]/'figs'

# ================= fig_blahut_verified =================
def blahut(p,D,lam,iters=400):
    """rate-distortion Blahut-Arimoto; returns rate, distortion, objective trace"""
    n,m=D.shape; q=np.ones(m)/m; obj=[]
    for _ in range(iters):
        E=np.exp(-lam*D)*q[None,:]
        W=E/E.sum(1,keepdims=True)
        q=(p[:,None]*W).sum(0)
        R=np.sum(p[:,None]*W*np.log(np.clip(W/q[None,:],1e-300,None)))
        Dd=np.sum(p[:,None]*W*D)
        obj.append(R+lam*Dd)
    return R,Dd,np.array(obj)
n=6; x=np.arange(n)
p=np.exp(-0.35*x); p/=p.sum()
D=(x[:,None]-x[None,:])**2/ (n-1)**2
lams=np.linspace(0.6,40,60)
Rs=[];Ds=[]
for l in lams:
    R,Dd,_=blahut(p,D,l); Rs.append(R/np.log(2)); Ds.append(Dd)
_,_,tr=blahut(p,D,6.0)
fig,ax=plt.subplots(1,3,figsize=(8.4,2.7))
ax[0].plot(tr[:40],color=NAVY,lw=2); ax[0].set_xlabel('half step'); ax[0].set_ylabel('objective')
ax[0].set_title('(a)  Monotone decrease')
ax[1].plot(Ds,Rs,color=RUST,lw=2.2); ax[1].set_xlabel('distortion $D$'); ax[1].set_ylabel('rate $R(D)$ (bits)')
ax[1].set_title(r'(b)  Sweeping $\lambda$ traces $R(D)$')
sl=np.gradient(np.array(Rs)*np.log(2),np.array(Ds))
ax[2].plot(lams,-sl,color=TEAL,lw=2,label='measured slope')
ax[2].plot(lams,lams,ls='--',color=GRAY,lw=1.4,label=r'$\lambda$')
ax[2].set_xlabel(r'$\lambda$'); ax[2].set_ylabel(r'$-dR/dD$'); ax[2].set_title('(c)  Slope equals $-\\lambda$')
ax[2].legend(frameon=False,fontsize=8.5)
fig.tight_layout(); fig.savefig(FIGS/'fig_blahut_verified.pdf',bbox_inches='tight'); plt.close(fig)
err=np.max(np.abs(-sl[5:-5]-lams[5:-5])/lams[5:-5])
print('blahut: max relative slope error %.3f'%err)
