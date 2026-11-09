import numpy as np, matplotlib as mpl, pathlib
from scipy.optimize import minimize
mpl.use('Agg'); import matplotlib.pyplot as plt
mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'serif',
 'font.serif':['CMU Serif','DejaVu Serif'],'mathtext.fontset':'cm','font.size':10.5,'axes.titlesize':11,
 'axes.labelsize':10.5,'legend.fontsize':8.8,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':0.8,'axes.edgecolor':'#333','figure.dpi':200,'axes.grid':True,'grid.color':'#ddd','grid.linewidth':0.6})
NAVY='#26456e'; RUST='#c0562f'; TEAL='#2a8f82'; GOLD='#d4a11e'; GRAY='#6b6b6b'
FIGS=pathlib.Path(__file__).resolve().parents[1]/'figs'
def KL(q,p): q=np.clip(q,1e-300,1); return float(np.sum(q*np.log(q/p)))

# ---------- cyclic I-projection onto two half spaces fails ----------
p0=np.array([0.70,0.20,0.10])
# constraints: q0 <= 0.35  and  q2 >= 0.40   (half spaces a.q <= c)
cons=[(np.array([1.,0,0]),0.35),(np.array([0,0,-1.]),-0.40)]
def proj_half(q,a,c):
    if a@q<=c+1e-13: return q.copy()
    lo,hi=0.0,300.0
    for _ in range(300):
        mid=(lo+hi)/2; w=q*np.exp(-mid*a); w/=w.sum()
        if a@w>c: lo=mid
        else: hi=mid
    w=q*np.exp(-hi*a); return w/w.sum()
feas=lambda q: all(a@q<=c+1e-9 for a,c in cons)
q=p0.copy(); path=[q.copy()]
for it in range(100):
    for a,c in cons:
        q=proj_half(q,a,c); path.append(q.copy())
        if feas(q): break
    if feas(q): break
q_naive=q.copy()
# true I-projection onto the intersection, by direct minimisation over the simplex
res=minimize(lambda z: KL(np.exp(z)/np.exp(z).sum(),p0), x0=np.log(p0),
             constraints=[{'type':'ineq','fun':(lambda z,a=a,c=c: c-a@(np.exp(z)/np.exp(z).sum()))} for a,c in cons],
             method='SLSQP',options={'ftol':1e-14,'maxiter':2000})
q_opt=np.exp(res.x)/np.exp(res.x).sum()
gap=(KL(q_naive,p0)-KL(q_opt,p0))/KL(q_opt,p0)*100
print('naive stops at',np.round(q_naive,4),'KL %.5f'%KL(q_naive,p0))
print('optimum       ',np.round(q_opt,4),'KL %.5f'%KL(q_opt,p0),'  naive excess %.1f%%'%gap)
print('active at optimum:',[bool(abs(a@q_opt-c)<1e-6) for a,c in cons])
def tern(p): return np.array([p[1]+0.5*p[2],(np.sqrt(3)/2)*p[2]])
fig,ax=plt.subplots(figsize=(5.2,4.5)); ax.axis('off'); ax.grid(False); ax.set_aspect('equal')
V=np.array([tern([1,0,0]),tern([0,1,0]),tern([0,0,1])]); ax.plot(*np.vstack([V,V[0]]).T,color='#333',lw=1.3)
g=np.linspace(0,1,300); pts=[]
for q1 in g:
    for q2 in g:
        q0=1-q1-q2
        if q0<-1e-9: continue
        qq=np.array([q0,q1,q2])
        if feas(qq): pts.append(tern(qq))
pts=np.array(pts); ax.plot(pts[:,0],pts[:,1],'.',ms=1.4,color='#d6e6e0',zorder=0)
# contour of KL around p0
res_=220; A=np.linspace(0,1,res_); Z=np.full((res_,res_),np.nan); XX=np.zeros((res_,res_)); YY=np.zeros((res_,res_))
for i,q1 in enumerate(A):
    for j,q2 in enumerate(A):
        q0=1-q1-q2
        if q0<0: continue
        xy=tern([q0,q1,q2]); XX[j,i],YY[j,i]=xy; Z[j,i]=KL(np.array([q0,q1,q2]),p0)
ax.contour(XX,YY,Z,levels=[0.03,0.08,0.15,0.25,0.4,0.6],colors=[NAVY],linewidths=0.7,alpha=.6)
P=np.array([tern(z) for z in path]); ax.plot(P[:,0],P[:,1],'--',color=RUST,lw=1.3)
for pt,lab,col,dx,dy in [(p0,'$p_0$',NAVY,-0.08,-0.01),(q_naive,'naive cyclic',RUST,0.025,0.02),(q_opt,'true projection',GOLD,0.025,-0.045)]:
    xy=tern(pt); ax.plot(*xy,'o',ms=8,color=col); ax.text(xy[0]+dx,xy[1]+dy,lab,fontsize=9.5,color=col)
ax.annotate('',xy=tern(q_opt),xytext=tern(p0),arrowprops=dict(arrowstyle='-|>',color=GOLD,lw=1.8))
ax.set_title('Cyclic projection halts %.1f%% above the optimum'%gap,fontsize=11,pad=10)
fig.tight_layout(); fig.savefig(FIGS/'fig_cyclic_fails.pdf',bbox_inches='tight'); plt.close(fig)

# ---------- Sims reverse water filling ----------
# eight sectoral variables driven by two common factors plus idiosyncratic noise
rng=np.random.default_rng(2)
L=np.array([[1.6,0.2],[1.4,-0.3],[1.2,0.5],[1.0,-0.6],[0.3,1.3],[0.2,1.1],[-0.4,0.9],[0.1,0.8]])
Sigma=L@L.T+np.diag(rng.uniform(0.15,0.35,8))
sig2=np.sort(np.linalg.eigvalsh(Sigma))[::-1]
print('eigenvalues:',np.round(sig2,3),' total prior variance %.2f'%sig2.sum())
def waterfill(sig2,kbits):
    C=kbits*np.log(2); lo,hi=1e-12,sig2.max()
    for _ in range(300):
        th=(lo+hi)/2; R=0.5*np.sum(np.log(np.maximum(sig2/th,1.0)))
        if R>C: lo=th
        else: hi=th
    th=(lo+hi)/2
    return th,np.minimum(sig2,th),0.5*np.log2(np.maximum(sig2/th,1.0))
th,post,bits=waterfill(sig2,2.0)
nz=int((bits<1e-9).sum())
print('kappa=2 bits: theta=%.3f, zero-bit directions %d of %d, bits=%s'%(th,nz,len(sig2),np.round(bits,2)))
print('loss D=%.2f of prior variance %.2f, so two bits buy back %.0f%%'%(post.sum(),sig2.sum(),100*(1-post.sum()/sig2.sum())))
fig,ax=plt.subplots(1,3,figsize=(8.6,2.75)); idx=np.arange(1,9)
ax[0].bar(idx,sig2,color='#cdd8e8',edgecolor=NAVY,label='prior variance')
ax[0].bar(idx,post,color=NAVY,label='posterior variance')
ax[0].axhline(th,color=RUST,ls='--',lw=1.5,label=r'water level $\theta$')
ax[0].set_xlabel('principal direction'); ax[0].set_ylabel('variance'); ax[0].set_title(r'(a)  $\kappa=2$ bits'); ax[0].legend(frameon=False,fontsize=8)
ks=np.linspace(0.05,9,240); B=np.array([waterfill(sig2,k)[2] for k in ks])
LS4=[('-',2.0,'#000000'),('--',1.5,NAVY),(':',2.0,'#444444'),('-.',1.5,'#777777')]
for j in range(8):
    st=LS4[j] if j<4 else None
    ax[1].plot(ks,B[:,j],lw=st[1] if st else 1.5,ls=st[0] if st else '-',color=st[2] if st else None,label='direction %d'%(j+1) if j<4 else None)
ax[1].axvline(2.0,color=GRAY,ls=':',lw=1.2)
ax[1].set_xlabel(r'capacity $\kappa$ (bits)'); ax[1].set_ylabel('bits allocated'); ax[1].set_title('(b)  Directions switch on one at a time'); ax[1].legend(frameon=False,fontsize=7.5)
D=np.array([waterfill(sig2,k)[1].sum() for k in ks])
ax[2].plot(D,ks,color=TEAL,lw=2.2); ax[2].set_xlabel('distortion $D(\\theta)$'); ax[2].set_ylabel(r'rate $\kappa$ (bits)'); ax[2].set_title('(c)  A rate distortion curve')
fig.tight_layout(); fig.savefig(FIGS/'fig_sims_waterfilling.pdf',bbox_inches='tight'); plt.close(fig)
# switch-on capacities for the text
on=[ks[np.argmax(B[:,j]>1e-9)] for j in range(8)]
print('capacity at which each direction switches on (bits):',np.round(on,2))
