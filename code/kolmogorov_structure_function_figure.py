# A computable surrogate for the Kolmogorov structure function, over a restricted model class.
import numpy as np, matplotlib as mpl, pathlib
mpl.use('Agg'); import matplotlib.pyplot as plt
mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,
 'font.family':'serif','font.serif':['CMU Serif','DejaVu Serif'],'mathtext.fontset':'cm',
 'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'legend.fontsize':9.5,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.8,
 'axes.edgecolor':'#333333','xtick.color':'#333333','ytick.color':'#333333',
 'axes.labelcolor':'#222222','text.color':'#222222','figure.dpi':200,
 'axes.grid':True,'grid.color':'#dddddd','grid.linewidth':0.6})
NAVY='#26456e'; RUST='#c0562f'; TEAL='#2a8f82'; GOLD='#d4a11e'; GRAY='#6b6b6b'
FIGS=pathlib.Path(__file__).resolve().parents[1]/'figs'

rng=np.random.default_rng(17)
# a string with genuine structure: three blocks of differing bias, then noise on top
n=600; truth_bp=[0,200,380,600]; rates=[0.12,0.78,0.35]
x=np.concatenate([rng.random(truth_bp[i+1]-truth_bp[i])<rates[i] for i in range(3)]).astype(int)

def H(p):
    p=np.clip(p,1e-12,1-1e-12); return -(p*np.log2(p)+(1-p)*np.log2(1-p))

def best_blocks(x,m):
    """least data codelength with m piecewise constant blocks, by dynamic programming"""
    n=len(x); cs=np.concatenate([[0],np.cumsum(x)])
    def cost(i,j):
        k=cs[j]-cs[i]; L=j-i
        return L*H(k/L) if L>0 else 0.0
    INF=1e18
    D=np.full((m+1,n+1),INF); D[0,0]=0.0; B=np.zeros((m+1,n+1),int)
    for b in range(1,m+1):
        for j in range(1,n+1):
            best=INF; arg=0
            for i in range(b-1,j):
                if D[b-1,i]>=INF: continue
                c=D[b-1,i]+cost(i,j)
                if c<best: best=c; arg=i
            D[b,j]=best; B[b,j]=arg
    return D[m,n]

ms=np.arange(1,13)
data_bits=np.array([best_blocks(x,m) for m in ms])
# model cost: boundaries (log2 C(n-1,m-1)) plus a rate per block at resolution 1/sqrt(n)
from math import lgamma, log2
def logC(n,k):
    if k<0 or k>n: return 0.0
    return (lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1))/np.log(2)
model_bits=np.array([logC(n-1,m-1)+m*0.5*log2(n) for m in ms])
total=data_bits+model_bits
mstar=ms[np.argmin(total)]

fig,(a1,a2)=plt.subplots(1,2,figsize=(7.9,3.3))
a1.plot(ms,model_bits,color=RUST,lw=2,marker='o',ms=4,label='model cost  $K(S)$')
a1.plot(ms,data_bits,color=NAVY,lw=2,marker='s',ms=4,label=r'data given model  $\log|S|$')
a1.plot(ms,total,color='#333',lw=2.4,marker='D',ms=4,label='two part total')
a1.axvline(mstar,color=TEAL,ls='--',lw=1.4)
a1.annotate('minimal sufficient\nstatistic  $m^*=%d$'%mstar,xy=(mstar,total.min()),
            xytext=(mstar+1.6,total.min()+120),fontsize=8.5,color='#333',
            arrowprops=dict(arrowstyle='->',color='#777',lw=0.9))
a1.set_xlabel('model complexity (number of blocks $m$)'); a1.set_ylabel('bits')
a1.set_title('(a)  The two part code'); a1.legend(frameon=False,fontsize=8.3)

# structure function: h(alpha) = min data bits subject to model cost <= alpha
al=np.linspace(model_bits.min(),model_bits.max(),200)
h=np.array([data_bits[model_bits<=a].min() if (model_bits<=a).any() else np.nan for a in al])
a2.plot(al,h,color=NAVY,lw=2.2,label=r'$h_x(\alpha)$')
a2.plot(al,al+h,color='#333',lw=1.6,ls='-',label=r'$\alpha+h_x(\alpha)$')
a2.plot(al,total.min()-al,color=GRAY,ls=':',lw=1.4,label='sufficiency line')
ia=np.nanargmin(al+h)
a2.plot([al[ia]],[h[ia]],'o',ms=7,color=TEAL)
a2.annotate('sufficiency\nkink',xy=(al[ia],h[ia]),xytext=(al[ia]+6,h[ia]+100),fontsize=8.5,
            color='#333',arrowprops=dict(arrowstyle='->',color='#777',lw=0.9))
a2.set_xlabel(r'bits allowed for the model  $\alpha$'); a2.set_ylabel('bits')
a2.set_title('(b)  Structure function'); a2.legend(frameon=False,fontsize=8.5,loc='upper right')
fig.tight_layout(); fig.savefig(FIGS/'app_kolmogorov.pdf',bbox_inches='tight'); plt.close(fig)
print('true blocks 3, selected m*=%d'%mstar)
print('total at m*=%.0f bits; raw string length %d bits; data bits at m*=%.0f, model bits=%.0f'%(
      total.min(),n,data_bits[mstar-1],model_bits[mstar-1]))
print('past the kink, m=%d to %d: data bits %.1f to %.1f, model bits %.1f to %.1f, slope %.2f'%(
      mstar,ms[-1],data_bits[mstar-1],data_bits[-1],model_bits[mstar-1],model_bits[-1],
      (data_bits[-1]-data_bits[mstar-1])/(model_bits[-1]-model_bits[mstar-1])))
