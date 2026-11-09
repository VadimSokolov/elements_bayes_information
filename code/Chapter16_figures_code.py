"""
Figures and worked numbers for the monograph chapter
"Growth-Optimal Prediction: Kelly, Universal Portfolios, and Bayesian Asset Allocation".
Reproducible: fixed seeds, no external data.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist, t as t_dist, norm
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import json, pathlib

FIGS = pathlib.Path(__file__).resolve().parents[1] / "figs"
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
BLUE, RED, GREEN, GREY = "#1f3b73", "#b2182b", "#1b7837", "#666666"
out = {}

# ===================================================================
# FIGURE 1: Kelly growth and the information edge (even-money coin)
# ===================================================================
p = 0.55; q = 1 - p
f = np.linspace(0.0, 0.6, 500)
G = p*np.log(1+f) + q*np.log1p(-f)          # expected log-growth per bet
fstar = p - q                                # = 0.10
Gstar = np.log(2) + p*np.log(p) + q*np.log(q)
KL = p*np.log(p/0.5) + q*np.log(q/0.5)       # D(Ber(p)||Ber(1/2))
f2 = 2*fstar                                  # double Kelly
G2 = p*np.log(1+f2) + q*np.log(1-f2)
out.update(dict(p=p, fstar=fstar, Gstar_nats=Gstar, KL_nats=KL,
                Gstar_bits=Gstar/np.log(2), double_kelly_growth=G2))

# simulate wealth paths for half / full / double Kelly
def sim_paths(frac, N=250, paths=400, seed=1):
    rng = np.random.default_rng(seed)
    wins = rng.random((paths, N)) < p
    mult = np.where(wins, 1+frac, 1-frac)
    logW = np.concatenate([np.zeros((paths,1)), np.cumsum(np.log(mult), axis=1)], axis=1)
    return logW
N = 250
paths_half = sim_paths(fstar/2, N); paths_full = sim_paths(fstar, N); paths_dbl = sim_paths(2*fstar, N)
tgrid = np.arange(N+1)

fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
ax[0].plot(f, G, color=BLUE, lw=2)
ax[0].axvline(fstar, color=RED, ls="--", lw=1.2)
ax[0].plot([fstar], [Gstar], "o", color=RED)
ax[0].axhline(0, color=GREY, lw=0.8)
ax[0].annotate("$f^\\star=p-q=%.2f$\n$G(f^\\star)=D(p\\,\\|\\,1/2)=%.4f$ nats" % (fstar, Gstar),
               xy=(fstar, Gstar), xytext=(0.22, Gstar*0.55),
               arrowprops=dict(arrowstyle="->", color=RED), fontsize=8.5)
ax[0].set_xlabel("bet fraction $f$"); ax[0].set_ylabel("expected log-growth $G(f)$")
ax[0].set_title(f"(a) Growth is the information edge ($p={p}$)")

for lab, P, c in [("half Kelly", paths_half, GREEN), ("full Kelly", paths_full, BLUE),
                  ("double Kelly", paths_dbl, RED)]:
    ax[1].plot(tgrid, P.mean(0), color=c, lw=2, label=lab)
    ax[1].fill_between(tgrid, np.percentile(P,25,0), np.percentile(P,75,0), color=c, alpha=0.12)
ax[1].axhline(0, color=GREY, lw=0.8)
ax[1].set_xlabel("bets $n$"); ax[1].set_ylabel("log-wealth $\\log W_n$")
ax[1].set_title("(b) Fractional Kelly: mean and IQR")
ax[1].legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(FIGS / "fig_kelly.png"); plt.close(fig)

# ===================================================================
# FIGURE 2: Cover's universal portfolio -- volatility pumping
# ===================================================================
# asset 1 = cash (relative 1); asset 2 = stock alternating x2, x1/2
n = 40
s = np.array([2.0 if i % 2 == 0 else 0.5 for i in range(n)])   # stock price relatives
X = np.column_stack([np.ones(n), s])                            # (n,2) price relatives

def crp_logwealth(b2):
    """log wealth path of CRP with weight b2 on the stock."""
    fac = (1 - b2) * X[:,0] + b2 * X[:,1]
    return np.concatenate([[0.0], np.cumsum(np.log(fac))])

grid = np.linspace(1e-4, 1-1e-4, 4000)
w = beta_dist(0.5, 0.5).pdf(grid); w /= w.sum()               # Dirichlet(1/2,1/2) weights
# terminal-and-path wealth for every CRP on the grid
logW_crp = np.array([crp_logwealth(b) for b in grid])          # (grid, n+1)
W_crp = np.exp(logW_crp)
# universal portfolio wealth = nu-mixture of CRP wealths (Cover's identity)
W_up = (W_crp * w[:,None]).sum(0)
logW_up = np.log(W_up)
# best CRP in hindsight (per horizon and terminal)
best_b2 = grid[np.argmax(logW_crp[:,-1])]
logW_best_path = np.max(logW_crp, axis=0)                       # best-at-each-horizon (upper env)
cash = crp_logwealth(0.0); stock = crp_logwealth(1.0); bestCRP = crp_logwealth(best_b2)
out.update(dict(best_crp_weight=float(best_b2),
                logW_stock=float(stock[-1]), logW_cash=float(cash[-1]),
                logW_bestCRP=float(bestCRP[-1]), logW_UP=float(logW_up[-1]),
                regret_final=float(bestCRP[-1]-logW_up[-1])))

# regret vs log n (recompute UP and best over horizons)
regret = logW_best_path - logW_up

fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
t = np.arange(n+1)
ax[0].plot(t, cash, color=GREY, lw=1.6, label="cash (asset 1)")
ax[0].plot(t, stock, color=RED, lw=1.6, label="stock (asset 2)")
ax[0].plot(t, bestCRP, color=GREEN, lw=2, label=f"best CRP $b={best_b2:.2f}$")
ax[0].plot(t, logW_up, color=BLUE, lw=2, ls="--", label="universal portfolio")
ax[0].set_xlabel("day $n$"); ax[0].set_ylabel("log-wealth")
ax[0].set_title("(a) Volatility pumping: the best CRP grows")
ax[0].legend(fontsize=8, loc="upper left")

mask = t >= 1
ax[1].plot(np.log(t[mask]), regret[mask], color=BLUE, lw=2, label="regret $\\log S_n(b^\\star)-\\log S_n(\\hat b)$")
# reference slope 1/2 (d-1)=1/2 for d=2
c0 = regret[mask][-1] - 0.5*np.log(t[mask][-1])
ax[1].plot(np.log(t[mask]), 0.5*np.log(t[mask]) + c0, color=RED, ls=":", lw=1.6,
           label="slope $\\frac{d-1}{2}=\\frac{1}{2}$")
ax[1].set_xlabel("$\\log n$"); ax[1].set_ylabel("regret (nats)")
ax[1].set_title("(b) Regret grows like $\\frac{d-1}{2}\\log n$")
ax[1].legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(FIGS / "fig_universal.png"); plt.close(fig)

# ===================================================================
# FIGURE 3: Bayesian asset allocation -- fractional Kelly from uncertainty
# ===================================================================
# one risky asset, monthly return ~ N(mu, sig^2), rf = 0.
m_hat, sig = 0.008, 0.045          # sample mean / sd (monthly)
def neg_Elog_normal(ff, mean, sd):
    val, _ = quad(lambda r: norm.pdf(r, mean, sd)*np.log(max(1+ff*r, 1e-9)), mean-8*sd, mean+8*sd)
    return -val
def neg_Elog_t(ff, loc, scale, df):
    lo, hi = loc-12*scale, loc+12*scale
    val, _ = quad(lambda r: t_dist.pdf(r, df, loc, scale)*np.log(max(1+ff*r, 1e-9)), lo, hi, limit=200)
    return -val

f_plugin = minimize_scalar(neg_Elog_normal, args=(m_hat, sig), bounds=(0, 20), method="bounded").x
Ts = np.array([6, 9, 12, 18, 24, 36, 48, 60, 90, 120, 180, 240, 360, 600])
f_bayes = []
for T in Ts:
    scale = sig*np.sqrt(1 + 1.0/T)              # predictive scale, unknown mean & var
    df = T - 1
    fb = minimize_scalar(neg_Elog_t, args=(m_hat, scale, df), bounds=(0, 20), method="bounded").x
    f_bayes.append(fb)
f_bayes = np.array(f_bayes)
ratio = f_bayes / f_plugin
out.update(dict(f_plugin=float(f_plugin),
                f_bayes_T12=float(f_bayes[Ts.tolist().index(12)]),
                ratio_T12=float(ratio[Ts.tolist().index(12)]),
                ratio_T60=float(ratio[Ts.tolist().index(60)])))

fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
ax[0].axhline(f_plugin, color=RED, ls="--", lw=1.6, label=f"plug-in Kelly $={f_plugin:.2f}$")
ax[0].plot(Ts, f_bayes, "o-", color=BLUE, lw=2, label="Bayesian (predictive)")
ax[0].set_xscale("log"); ax[0].set_xlabel("sample size $T$ (months)")
ax[0].set_ylabel("optimal fraction $f$")
ax[0].set_title("(a) Estimation risk shrinks the Kelly bet")
ax[0].legend(fontsize=8, loc="lower right")

ax[1].plot(Ts, ratio, "o-", color=BLUE, lw=2)
ax[1].axhline(1.0, color=GREY, lw=0.9)
ax[1].axhline(0.5, color=GREEN, ls=":", lw=1.4, label="half Kelly")
ax[1].set_xscale("log"); ax[1].set_xlabel("sample size $T$ (months)")
ax[1].set_ylabel("$f_{\\mathrm{Bayes}}\\,/\\,f_{\\mathrm{plug}}$")
ax[1].set_title("(b) Fractional Kelly is Bayesian Kelly")
ax[1].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(FIGS / "fig_bayeskelly.png"); plt.close(fig)

print(json.dumps(out, indent=2))
