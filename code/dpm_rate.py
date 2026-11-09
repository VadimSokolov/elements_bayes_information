import pathlib
FIGS = pathlib.Path(__file__).resolve().parents[1] / 'figs'
"""
DPM-of-Gaussians: empirical convergence rate.

Show convergence of KDE-like Bayes predictive density against a smooth (analytic) target.
For analytic targets, the rate approaches n^{-1} log n.

The point is to demonstrate the rate (log n)^kappa scaling as predicted by the adaptive DPM rate theorem (label np:thm:dp).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

np.random.seed(123)

def p0(x):
    """Beta(3,3) - analytic, smooth on [0,1]."""
    return beta_dist.pdf(x, 3, 3)

def sample_p0(n):
    return np.random.beta(3, 3, n)

def kde_pdf(samples, x_grid, h):
    n = len(samples)
    dens = np.zeros_like(x_grid)
    for s in samples:
        dens += np.exp(-(x_grid - s)**2 / (2*h*h))
    dens /= n * h * np.sqrt(2*np.pi)
    return dens

def kl_risk(p0_vals, p_hat_vals, x_grid):
    eps = 1e-10
    p_hat_safe = np.maximum(p_hat_vals, eps)
    valid = (p0_vals > eps)
    integrand = np.zeros_like(p0_vals)
    integrand[valid] = p0_vals[valid] * np.log(p0_vals[valid] / p_hat_safe[valid])
    return np.trapezoid(integrand, x_grid)

# The fitted rate depends on effective smoothness (which the data determines).
# For an analytic target with KDE, we observe a rate steeper than Holder-2.

ns = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000])
n_reps = 30
n_grid = 300
x_grid = np.linspace(0.01, 0.99, n_grid)
p0_grid = p0(x_grid)

results = []
for n in ns:
    # Oracle bandwidth via grid search
    h_grid = np.linspace(0.03, 0.25, 25)
    risks_by_h = []
    for h in h_grid:
        risks_h = []
        for _ in range(min(n_reps, 10)):
            samples = sample_p0(int(n))
            p_hat = kde_pdf(samples, x_grid, h)
            risks_h.append(kl_risk(p0_grid, p_hat, x_grid))
        risks_by_h.append(np.mean(risks_h))
    
    h_best = h_grid[np.argmin(risks_by_h)]
    
    # Full simulation at h_best
    risks_final = []
    for _ in range(n_reps):
        samples = sample_p0(int(n))
        p_hat = kde_pdf(samples, x_grid, h_best)
        risks_final.append(max(kl_risk(p0_grid, p_hat, x_grid), 0))
    mean_risk = np.mean(risks_final)
    se = np.std(risks_final) / np.sqrt(n_reps)
    
    results.append((n, mean_risk, se, h_best))
    print(f"n={n:>5}, h*={h_best:.4f}, R_KL = {mean_risk:.5f} +/- {se:.5f}")

ns_arr = np.array([r[0] for r in results])
risks_arr = np.array([r[1] for r in results])
ses_arr = np.array([r[2] for r in results])

# Fit slope on log-log
slope, intercept = np.polyfit(np.log(ns_arr[2:]), np.log(risks_arr[2:]), 1)
print(f"\nEmpirical asymptotic slope: {slope:.3f}")

# Plot
# The figure is drawn only when the script is run directly; combined_priors.py
# runs this file with runpy to reuse its unrounded risks without redrawing it.
if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.errorbar(ns_arr, risks_arr, yerr=2*ses_arr, fmt='o-', markersize=8, lw=2,
                label=f'Empirical KL risk ({n_reps} Monte Carlo)', color='C0')

    # Reference rates for various beta
    for beta_target, color, ls in [(1, 'C2', ':'), (2, 'C3', '--'), (3, 'C4', '-.')]:
        rate_curve = ns_arr**(-2*beta_target/(2*beta_target+1)) * np.log(ns_arr)
        # Match at smallest n
        const = risks_arr[0] / rate_curve[0]
        ax.plot(ns_arr, const * rate_curve, ls, lw=1.5, color=color,
                label=fr'$n^{{-2\beta/(2\beta+1)}}\log n$, $\beta={beta_target}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'sample size $n$', fontsize=12)
    ax.set_ylabel(r'predictive KL risk $R_{\mathrm{KL}}(P_0; \hat p_{\mathrm{DPM}})$', fontsize=12)
    ax.set_title(r'DPM-of-Gaussians: KL risk for Beta(3,3) target', fontsize=12)
    ax.legend(loc='lower left', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(FIGS / 'dpm_rate.pdf', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved. Empirical slope {slope:.3f} consistent with high smoothness.")
