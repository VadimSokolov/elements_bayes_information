import numpy as np, matplotlib as mpl, pathlib
mpl.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,
 'font.family':'serif','font.serif':['CMU Serif','DejaVu Serif'],'mathtext.fontset':'cm',
 'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'legend.fontsize':9.5,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.8,
 'axes.edgecolor':'#333333','xtick.color':'#333333','ytick.color':'#333333',
 'axes.labelcolor':'#222222','text.color':'#222222','figure.dpi':200,
 'axes.grid':True,'grid.color':'#dddddd','grid.linewidth':0.6})
NAVY='#26456e'; RUST='#c0562f'; TEAL='#2a8f82'; GOLD='#d4a11e'; GRAY='#6b6b6b'
FIGS=pathlib.Path(__file__).resolve().parents[1]/'figs'

# ---------- 1) GBC pipeline diagram ----------
fig,ax=plt.subplots(figsize=(7.6,3.1)); ax.axis('off'); ax.grid(False)
ax.set_xlim(0,10); ax.set_ylim(0,4.1)
def box(x,y,w,h,txt,fc,ec,fs=9.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06,rounding_size=0.10",
                 facecolor=fc,edgecolor=ec,lw=1.3))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color='#1d1d1d')
def arrow(x1,y1,x2,y2,txt='',col='#555',rad=0.0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,
                 color=col,lw=1.3,connectionstyle="arc3,rad=%.2f"%rad))
    if txt: ax.text((x1+x2)/2,(y1+y2)/2+0.20,txt,ha='center',fontsize=8.2,color='#444')
# training row
ax.text(0.05,3.85,'Training, once',fontsize=9.5,color=RUST,style='italic')
box(0.2,2.65,2.0,0.95,'prior and model\n$\\pi(\\theta),\\ p(y\\mid\\theta)$','#eef2f8',NAVY)
box(2.85,2.65,2.0,0.95,'simulate pairs\n$(\\theta^{(s)},y^{(s)})$','#eef2f8',NAVY)
box(5.5,2.65,2.0,0.95,'summarize\n$S^{(s)}=h(y^{(s)})$','#eef2f8',NAVY)
box(8.15,2.65,1.7,0.95,'fit $f_\\varphi$\ncheck loss','#f6ece3',RUST)
arrow(2.2,3.12,2.85,3.12); arrow(4.85,3.12,5.5,3.12); arrow(7.5,3.12,8.15,3.12)
# use row
ax.text(0.05,1.85,'Use, for any data set',fontsize=9.5,color=TEAL,style='italic')
box(0.2,0.55,2.0,0.95,'new data $y$','#eef7f4',TEAL)
box(2.85,0.55,2.0,0.95,'summary $S=h(y)$','#eef7f4',TEAL)
box(5.5,0.55,2.0,0.95,'draw $\\tau\\sim U[0,1]^d$','#eef7f4',TEAL)
box(8.15,0.55,1.7,0.95,'$\\theta=f_\\varphi(\\tau,S)$','#f6ece3',RUST)
arrow(2.2,1.02,2.85,1.02); arrow(4.85,1.02,5.5,1.02); arrow(7.5,1.02,8.15,1.02)
arrow(9.0,2.65,9.0,1.5,'',RUST)
ax.text(9.15,2.05,'trained\nmap',fontsize=8.2,color=RUST,va='center')
ax.text(5.0,0.06,'no likelihood evaluation, no chain, and the cost of training is amortized over every later analysis',
        ha='center',fontsize=8.4,color='#555',style='italic')
fig.tight_layout(); fig.savefig(FIGS/'app_gbcpipeline.pdf',bbox_inches='tight'); plt.close(fig)
print('wrote app_gbcpipeline')

# ---------- 2) I-projection on the simplex ----------
def tern(p):            # barycentric to 2d
    return np.array([p[1]+0.5*p[2], (np.sqrt(3)/2)*p[2]])
fig,ax=plt.subplots(figsize=(5.4,4.6)); ax.axis('off'); ax.grid(False)
V=np.array([tern([1,0,0]),tern([0,1,0]),tern([0,0,1])])
ax.plot(*np.vstack([V,V[0]]).T,color='#333',lw=1.4)
P0=np.array([0.55,0.30,0.15])                     # generating law
# constraint: E[X]=c  with values (0,1,2)  ->  q1 + 2 q2 = c, a line in the simplex
c=1.05
grid=np.linspace(0,1,700)
pts=[]
for q1 in grid:
    q2=(c-q1)/2
    q0=1-q1-q2
    if q0>=-1e-9 and q2>=-1e-9: pts.append([q0,q1,q2])
pts=np.array(pts); XY=np.array([tern(p) for p in pts])
ax.plot(XY[:,0],XY[:,1],color=RUST,lw=2.4,label='constraint set $E$')
# divergence contours around P0
def KL(Q,P): 
    Q=np.clip(Q,1e-12,1); P=np.clip(P,1e-12,1); return np.sum(Q*np.log(Q/P),axis=-1)
res=260; A=np.linspace(0,1,res); B=np.linspace(0,1,res)
Z=np.full((res,res),np.nan); XX=np.zeros((res,res)); YY=np.zeros((res,res))
for i,q1 in enumerate(A):
    for j,q2 in enumerate(B):
        q0=1-q1-q2
        if q0<0: continue
        xy=tern([q0,q1,q2]); XX[j,i],YY[j,i]=xy; Z[j,i]=KL(np.array([q0,q1,q2]),P0)
cs=ax.contour(XX,YY,Z,levels=[0.02,0.06,0.12,0.22,0.36,0.55],colors=[NAVY],linewidths=0.8,alpha=.7)
# I-projection: minimise KL(Q||P0) over the line
k=KL(pts,P0); istar=np.nanargmin(k); Pst=pts[istar]
xy0=tern(P0); xys=tern(Pst)
ax.plot(*xy0,'o',ms=9,color=NAVY); ax.text(xy0[0]-0.07,xy0[1]-0.01,'$P$',fontsize=12,color=NAVY)
ax.plot(*xys,'o',ms=9,color=GOLD); ax.text(xys[0]+0.02,xys[1]-0.05,'$P^*$',fontsize=12,color='#a8801a')
ax.annotate('',xy=xys,xytext=xy0,arrowprops=dict(arrowstyle='-|>',color=TEAL,lw=2))
ax.text((xy0[0]+xys[0])/2-0.16,(xy0[1]+xys[1])/2+0.02,'$D(P^*\\Vert P)$',fontsize=10,color=TEAL)
ax.text(V[0][0]-0.03,V[0][1]-0.06,'$e_0$',ha='center',fontsize=10,color='#444')
ax.text(V[1][0]+0.03,V[1][1]-0.06,'$e_1$',ha='center',fontsize=10,color='#444')
ax.text(V[2][0]+0.07,V[2][1]-0.04,'$e_2$',ha='center',fontsize=10,color='#444')
ax.legend(frameon=False,fontsize=9,loc='upper left')
ax.set_title('The I-projection onto a moment constraint',fontsize=12,pad=16)
ax.set_aspect('equal')
fig.tight_layout(); fig.savefig(FIGS/'app_iprojection.pdf',bbox_inches='tight'); plt.close(fig)
print('wrote app_iprojection; P* =',np.round(Pst,3),' KL =',round(k[istar],4))
