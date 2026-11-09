import pathlib
FIGS = pathlib.Path(__file__).resolve().parents[1] / 'figs'
"""
Full Monte Carlo verification of the misspecified Fourier example
(Section "Monte Carlo verification", Table np:tab:mc and figs/sanov_floor.pdf).

The KL risk is integrated over the whole period [0,1] (the integrand is smooth
and periodic, so the trapezoid rule on an equispaced grid is spectrally
accurate); an earlier version integrated over [0.001, 0.999] only, which drops
about 0.001 of the Sanov floor and pushed the residuals below zero.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

rng = np.random.default_rng(42)

theta = 0.3
K0 = 3
c = 0.5

def p0(x):
    return 1 + theta * np.sqrt(2) * np.cos(2 * np.pi * K0 * x)

def p_a(x, a):
    return 1 + a * np.sqrt(2) * np.cos(2 * np.pi * x)

def sample_p0(n):
    """Rejection sampling from p_0."""
    M = 1 + theta * np.sqrt(2)
    out = np.empty(0)
    while out.size < n:
        proposed = rng.uniform(0, 1, 2 * n)
        u = rng.uniform(0, 1, 2 * n)
        out = np.concatenate([out, proposed[u * M < p0(proposed)]])
    return out[:n]

grid_a = np.linspace(-c, c, 1000)            # 1000-point posterior grid, uniform prior on [-c, c]
x_eval = np.linspace(0.0, 1.0, 2001)         # full period
p0_grid = p0(x_eval)

def posterior_a(samples):
    cs = np.sqrt(2) * np.cos(2 * np.pi * samples)
    log_post = np.log1p(np.outer(grid_a, cs)).sum(axis=1)
    log_post -= log_post.max()
    post = np.exp(log_post)
    return post / np.trapezoid(post, grid_a)

def predictive_density(post):
    """Bayes predictive density: integral over a of p_a(x) times the posterior."""
    return np.trapezoid(p_a(x_eval[:, None], grid_a[None, :]) * post[None, :], grid_a, axis=1)

def kl_risk(p_hat):
    return np.trapezoid(p0_grid * np.log(p0_grid / p_hat), x_eval)

sanov_exact, _ = quad(lambda x: p0(x) * np.log(p0(x)), 0, 1, limit=200)
sanov_taylor = theta**2 / 2 + theta**4 / 8 + theta**6 / 12
print(f"Sanov floor (exact integral):     {sanov_exact:.6f}")
print(f"Sanov floor (3-term Taylor):      {sanov_taylor:.6f}")
print(f"KL quadrature at a = 0:           {kl_risk(np.ones_like(x_eval)):.6f}")
print()

ns = [25, 50, 100, 200, 500, 1000, 2000, 5000]
n_reps = 2000
results = []
for n in ns:
    risks = np.array([kl_risk(predictive_density(posterior_a(sample_p0(n)))) for _ in range(n_reps)])
    mean_risk = risks.mean()
    se = risks.std(ddof=1) / np.sqrt(n_reps)
    residual = mean_risk - sanov_exact
    results.append((n, mean_risk, se, residual))
    print(f"n={n:>5}: R_KL = {mean_risk:.6f} +/- {2*se:.6f}, residual = {residual:.6f}, "
          f"1/(2n) = {1/(2*n):.6f}, n*residual = {n*residual:.3f} +/- {2*n*se:.3f}")
print()
for n, m, se, r in results:
    print(f"${n}$ & ${m:.6f} \\pm {2*se:.6f}$ & ${r:.6f}$ & ${1/(2*n):.6f}$ & ${n*r:.3f} \\pm {2*n*se:.3f}$ \\\\")

ns_arr = np.array([r[0] for r in results])
risks_arr = np.array([r[1] for r in results])
ses_arr = np.array([r[2] for r in results])

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(ns_arr, risks_arr, yerr=2*ses_arr, fmt='o-', markersize=9, lw=2,
            label=f'Empirical KL risk ({n_reps} Monte Carlo)', color='C0', capsize=4)
ax.axhline(sanov_exact, color='C3', ls='--', lw=2, label=fr'Sanov floor $= {sanov_exact:.4f}$')
ax.plot(ns_arr, sanov_exact + 1/(2*ns_arr), ':', color='C2', lw=2,
        label=r'Sanov floor $+ 1/(2n)$')
ax.set_xscale('log')
ax.set_xlabel(r'sample size $n$', fontsize=12)
ax.set_ylabel(r'predictive KL risk $R_{\mathrm{KL}}(P_0;\hat p_\Pi)$', fontsize=12)
ax.set_title(r'Sanov floor in action: misspecified Fourier example ($\theta=0.3$, $K_0=3$)', fontsize=12)
ax.legend(loc='upper right', framealpha=0.95, fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / 'sanov_floor.pdf', dpi=150, bbox_inches='tight')
print(f"\nFigure saved to figs/sanov_floor.pdf")
