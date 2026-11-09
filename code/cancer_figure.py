import pathlib
HERE = pathlib.Path(__file__).resolve().parent
FIGS = pathlib.Path(__file__).resolve().parents[1] / "figs"
import numpy as np, matplotlib, json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
NAVY,RUST,TEAL,GOLD,GRAY,PLUM="#26456e","#c0562f","#2a8f82","#d4a11e","#6b6b6b","#7b5ea7"
plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"serif","font.serif":["CMU Serif","DejaVu Serif"],
 "mathtext.fontset":"cm","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
J=json.load(open(HERE/'cancer_numbers.json'))
fig,ax=plt.subplots(1,3,figsize=(10.4,3.3))
x=np.linspace(1e-4,0.12,600)
for q,c in [(0.95,RUST),(0.975,NAVY),(0.99,TEAL)]:
    d=J[f'anchor_0.01_{q}']; ax[0].plot(x,stats.beta.sf(x,d['a'],d['b']),color=c,lw=1.8,label=f"$q={q}$: Beta({d['a']:.2f}, {d['b']:.1f})")
ax[0].plot(x,stats.beta.sf(x,1,99),color=GRAY,lw=1.2,ls='--',label="Beta(1, 99)")
ax[0].axvline(0.05,color='k',lw=.8,ls=':'); ax[0].axvline(0.01,color=GRAY,lw=.8)
ax[0].set_yscale('log'); ax[0].set_ylim(1e-4,1); ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$P(\\pi_{\\mathrm{LNI}}>x)$"); ax[0].set_title("(a) priors over the nomogram",fontsize=10); ax[0].legend(frameon=False,fontsize=6.8)
p=np.linspace(0.005,0.3,300); ax[1].plot(p,1/p,color='k',lw=1.4,label="$\\kappa=1/\\pi_{\\mathrm{tail}}$")
ax[1].fill_between(p,1/p,300,color=RUST,alpha=.10,label="Bayes aggressive, MAP not")
ax[1].fill_between([0.0175,0.0225],[93,93],[266,266],color=NAVY,alpha=.6,label="Patient A, 2004")
ax[1].set_ylim(0,300); ax[1].set_xlim(0,0.3); ax[1].set_xlabel("tail mass $\\pi_{\\mathrm{tail}}$"); ax[1].set_ylabel("$\\kappa=\\Delta U_{\\mathrm{tail}}/\\Delta C$"); ax[1].set_title("(b) where MAP and Bayes part",fontsize=10); ax[1].legend(frameon=False,fontsize=7)
g=np.array(J['grid']); v=np.array(J['vot_grid'])
ax[2].plot(g,v*365.25,color=TEAL,lw=2)
for pi,c in [(0.01,NAVY),(0.02,RUST)]:
    ax[2].plot([pi],[J['vot'][str(pi)][2]*365.25],'o',color=c); ax[2].annotate(f"$\\pi={pi}$",(pi,J['vot'][str(pi)][2]*365.25),textcoords='offset points',xytext=(6,4),fontsize=8)
ax[2].set_xlim(0,0.06); ax[2].set_xlabel("prior $\\pi_{\\mathrm{LNI}}$"); ax[2].set_ylabel("value of scan, quality adjusted days"); ax[2].set_title("(c) what a scan is worth",fontsize=10)
fig.tight_layout(); fig.savefig(FIGS / "med_cancer_case.pdf"); print('ok')
