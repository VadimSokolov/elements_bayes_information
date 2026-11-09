"""
Figures for the monograph chapter "When Is Bayes Gaussian? Curvature of the Rate Function".
Reproducible: fixed seeds, no external data.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
import json, pathlib

FIGS = pathlib.Path(__file__).resolve().parents[1] / "figs"
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
BLUE, RED, GREEN, GREY, PURPLE = "#1f3b73", "#b2182b", "#1b7837", "#666666", "#762a83"
out = {}

def Dber(x, p):
    x = np.clip(x, 1e-9, 1-1e-9); p = np.clip(p, 1e-9, 1-1e-9)
    return x*np.log(x/p) + (1-x)*np.log((1-x)/(1-p))

# ===================================================================
# FIGURE 1: the three regimes as three shapes of the rate function J
# ===================================================================
fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.5))

# (A) regular: Beta-Bernoulli, J(theta)=D(p0||theta), smooth well = Fisher paraboloid
p0 = 0.4
th = np.linspace(0.02, 0.98, 500)
J = Dber(p0, th)
I0 = 1.0/(p0*(1-p0))                      # Fisher information
ax[0].plot(th, J, color=BLUE, lw=2, label="$J(\\theta)=D(p_0\\|p_\\theta)$")
ax[0].plot(th, 0.5*I0*(th-p0)**2, color=RED, ls="--", lw=1.4, label="Fisher paraboloid")
ax[0].axvline(p0, color=GREY, lw=0.8)
ax[0].set_ylim(0, 0.7); ax[0].set_title("(A) Regular: a smooth well")
ax[0].set_xlabel("$\\theta$"); ax[0].set_ylabel("rate $J(\\theta)$")
ax[0].legend(fontsize=8, loc="upper center")

# (B) non-regular: Uniform-Pareto, J=log(theta/theta0) for theta>=theta0, +inf else
th0 = 1.0
t = np.linspace(0.5, 3.0, 500)
Jnr = np.where(t >= th0, np.log(np.maximum(t, 1e-9)/th0), np.nan)
ax[1].plot(t[t >= th0], Jnr[t >= th0], color=BLUE, lw=2)
ax[1].plot([th0, th0], [0, 1.2], color=RED, lw=2)                  # infinite wall
ax[1].annotate("$J=+\\infty$", xy=(0.92, 0.6), color=RED, fontsize=9)
ax[1].annotate("slope $1/\\theta_0$", xy=(1.9, 0.55), color=BLUE, fontsize=9)
ax[1].set_ylim(0, 1.2); ax[1].set_title("(B) Non-regular: a corner")
ax[1].set_xlabel("$\\theta$")

# (C) random curvature: parabolas with random eta^2 -> mixed normal
th0c = 0.5
tc = np.linspace(0.05, 0.95, 400)
rng = np.random.default_rng(3)
for k in range(6):
    eta2 = rng.gamma(3.0, 1.0)             # random curvature
    ax[2].plot(tc, 0.5*eta2*(tc-th0c)**2, color=PURPLE, lw=1.1, alpha=0.55)
ax[2].axvline(th0c, color=GREY, lw=0.8)
ax[2].set_ylim(0, 0.7); ax[2].set_title("(C) Random curvature: mixed normal")
ax[2].set_xlabel("$\\theta$")
fig.tight_layout(); fig.savefig(FIGS / "fig_bvm_regimes.png"); plt.close(fig)

# ===================================================================
# FIGURE 2: the coexistence corner (two-component mixture)
# ===================================================================
a, b = 0.25, 0.75
x = np.linspace(0.02, 0.98, 600)
true_rate = np.minimum(Dber(x, a), Dber(x, b))     # non-convex double well
env = np.where((x >= a) & (x <= b), 0.0, true_rate) # convex envelope = 0 on [a,b]
out.update(dict(coexist_peak=float(min(Dber(0.5, a), Dber(0.5, b)))))

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(x, true_rate, color=BLUE, lw=2, label="true rate $\\min\\{D(x\\|a),D(x\\|b)\\}$")
ax.plot(x, env, color=RED, ls="--", lw=1.8, label="convex envelope (Laplace/VI sees this)")
ax.fill_between(x, env, true_rate, where=(true_rate > env), color=GREY, alpha=0.25)
for c, lab in [(a, "$a$"), (b, "$b$")]:
    ax.axvline(c, color=GREY, lw=0.7); ax.annotate(lab, xy=(c, 0.44), fontsize=10)
ax.set_ylim(0, 0.5); ax.set_xlabel("$x$"); ax.set_ylabel("rate")
ax.set_title("The coexistence corner: the information a unimodal posterior discards")
ax.legend(fontsize=8.5, loc="upper center")
fig.tight_layout(); fig.savefig(FIGS / "fig_bvm_coexist.png"); plt.close(fig)

# ===================================================================
# FIGURE 3: misspecification splits the curvature (N(theta,1) fit to t5)
# ===================================================================
# model N(theta,1): Fisher V=1 -> credible sd (x sqrt n) = 1
# t5 data: Var = 5/3 -> sandwich var V^-1 J V^-1 = 5/3 -> sampling sd = sqrt(5/3)
V = 1.0; sand = 5.0/3.0
z = np.linspace(-5, 5, 600)
post = norm.pdf(z, 0, np.sqrt(1.0/V))          # posterior / credible (width 1)
samp = norm.pdf(z, 0, np.sqrt(sand))           # sampling of theta-hat (sandwich)
cover = 2*norm.cdf(1.96/np.sqrt(sand)) - 1     # coverage of the 95% credible band
out.update(dict(sandwich_var=sand, credible_coverage=float(cover),
                too_narrow_factor=float(np.sqrt(sand))))

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(z, post, color=BLUE, lw=2, label="posterior / credible ($V^{-1}=1$)")
ax.plot(z, samp, color=RED, lw=2, label="sampling of $\\hat\\theta$ (sandwich $=5/3$)")
band = (z >= -1.96) & (z <= 1.96)
ax.fill_between(z[band], 0, samp[band], color=BLUE, alpha=0.15)
ax.set_xlabel("$\\sqrt{n}\\,(\\theta-\\theta_\\star)$"); ax.set_ylabel("density")
ax.set_title(f"Misspecification splits curvature: 95% band covers {cover*100:.0f}%")
ax.annotate(f"too narrow by\n$\\sqrt{{5/3}}\\approx{np.sqrt(sand):.2f}$",
            xy=(2.4, 0.12), fontsize=9, color=GREY)
ax.legend(fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig(FIGS / "fig_bvm_misspec.png"); plt.close(fig)

# ===================================================================
# FIGURE 4: Bayesian AR(1) -- three curvatures (estimator limit laws)
# ===================================================================
def ar1_ols(theta, n, paths, seed):
    rng = np.random.default_rng(seed)
    est = np.empty(paths)
    for i in range(paths):
        y = np.empty(n+1); y[0] = 0.0
        e = rng.standard_normal(n)
        for t in range(n):
            y[t+1] = theta*y[t] + e[t]
        y0, y1 = y[:-1], y[1:]
        est[i] = np.dot(y0, y1)/np.dot(y0, y0)
    return est

n, paths = 200, 4000
n_expl = 60   # at n=200, 1.2**n*(est-1.2) lies on a rounding lattice of spacing 1.2**200*2**-52 = 1.5
est_stat = ar1_ols(0.5, n, paths, 10)
est_unit = ar1_ols(1.0, n, paths, 11)
est_expl = ar1_ols(1.2, n_expl, paths, 12)

fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.4))
s_stat = np.sqrt(n)*(est_stat-0.5)
ax[0].hist(s_stat, bins=60, density=True, color=BLUE, alpha=0.65)
zz = np.linspace(-4, 4, 200); ax[0].plot(zz, norm.pdf(zz, 0, np.sqrt(1-0.25)), color=RED, lw=1.6)
ax[0].set_title("(A) stationary $\\theta=0.5$: Gaussian"); ax[0].set_xlabel("$\\sqrt{n}(\\hat\\theta-\\theta)$")

s_unit = n*(est_unit-1.0)
ax[1].hist(np.clip(s_unit, -30, 6), bins=60, density=True, color=BLUE, alpha=0.65)
ax[1].set_title("(B) unit root $\\theta=1$: Dickey--Fuller"); ax[1].set_xlabel("$n(\\hat\\theta-\\theta)$")

s_raw = (1.2**n_expl)*(est_expl-1.2)
s_expl = np.clip(s_raw, -6, 6)
ax[2].hist(s_expl, bins=80, density=True, color=BLUE, alpha=0.65)
ax[2].set_title("(C) explosive $\\theta=1.2$, $n=%d$: Cauchy-like" % n_expl); ax[2].set_xlabel("$\\theta^{n}(\\hat\\theta-\\theta)$ (clipped)")
out.update(dict(ar1_stat_mean=float(s_stat.mean()), ar1_stat_var=float(s_stat.var()),
                ar1_unit_skew=float(((s_unit-s_unit.mean())**3).mean()/s_unit.std()**3),
                ar1_expl_n=n_expl, ar1_expl_quartiles=[float(q) for q in np.quantile(s_raw, [0.25, 0.75])]))
fig.tight_layout(); fig.savefig(FIGS / "fig_bvm_ar1.png"); plt.close(fig)

print(json.dumps(out, indent=2))
