import pathlib
FIGS = pathlib.Path(__file__).resolve().parents[1] / 'figs'
"""
Pólya tree predictive: empirical convergence rate.

Implementation:
- Truncated Pólya tree at level L on the dyadic partition of [0,1] into 2^L cells.
- Symmetric Dirichlet(alpha, ..., alpha) prior on the 2^L cell masses, alpha = c * L^2:
  the canonical beta_eps = c * |eps|^2 at the finest level |eps| = L only. As a Pólya
  tree this prior has Beta parameters c * L^2 * 2^(L-m) at depth m, not the canonical
  node-wise c * m^2, so the predictive is a smoothed histogram, not the canonical tree.
- Predictive density is piecewise constant on dyadic cells of finest level.
- Bias O(2^{-L*beta}), variance O(2^L / n).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

np.random.seed(42)

def p0(x):
    """Beta(3,3) - smooth target."""
    return beta_dist.pdf(x, 3, 3)

def sample_p0(n):
    return np.random.beta(3, 3, n)

def pt_predictive(samples, L, c=0.5):
    """
    Predictive density of the Pólya tree truncated at depth L, with a symmetric
    Dirichlet(alpha, ..., alpha) prior on its 2^L dyadic cells, alpha = c * L^2.
    The posterior mean mass of cell j is (count[j] + alpha) / (n + 2^L * alpha).
    """
    n_cells = 2**L
    cell_width = 1.0 / n_cells
    
    # Count samples in each cell
    cell_indices = np.minimum(np.floor(samples * n_cells).astype(int), n_cells - 1)
    counts = np.bincount(cell_indices, minlength=n_cells)
    
    # Posterior mean of cell j under the Dirichlet(alpha, ..., alpha) prior:
    # (count[j] + alpha) / (n + 2^L * alpha). alpha = c*L^2 matches the canonical
    # beta_eps = c*|eps|^2 only at the finest level |eps| = L (see the module docstring).
    alpha = c * L**2 if L > 0 else c
    
    post_probs = (counts + alpha) / (n + n_cells * alpha)
    
    # Predictive density is piecewise constant
    densities = post_probs / cell_width  # density = probability / cell_width
    return densities, cell_indices

def kl_risk_pt(densities_grid, p0_vals, x_grid, n_cells):
    """KL(p_0 || hat p_PT) where hat p_PT is piecewise constant."""
    cell_indices_grid = np.minimum(np.floor(x_grid * n_cells).astype(int), n_cells - 1)
    hat_p = densities_grid[cell_indices_grid]
    eps = 1e-10
    hat_p_safe = np.maximum(hat_p, eps)
    valid = (p0_vals > eps)
    integrand = np.zeros_like(p0_vals)
    integrand[valid] = p0_vals[valid] * np.log(p0_vals[valid] / hat_p_safe[valid])
    return np.trapezoid(integrand, x_grid)

ns = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000])
n_reps = 30
n_grid = 500
x_grid = np.linspace(0.001, 0.999, n_grid)
p0_grid = p0(x_grid)

results = []
for n in ns:
    # Oracle level L: balance 2^{-L*beta} bias and 2^L/n variance
    # For Holder-1: L^* = log_2(n)/3.  For Holder-2: L^* = log_2(n)/5.
    # Try several levels and pick the best
    L_grid = list(range(1, min(int(np.log2(n)) + 1, 12)))
    risks_by_L = []
    for L in L_grid:
        risks_L = []
        n_cells = 2**L
        for _ in range(10):
            samples = sample_p0(int(n))
            densities, _ = pt_predictive(samples, L)
            risks_L.append(kl_risk_pt(densities, p0_grid, x_grid, n_cells))
        risks_by_L.append(np.mean(risks_L))
    
    L_best = L_grid[np.argmin(risks_by_L)]
    
    # Full simulation at L_best
    n_cells = 2**L_best
    risks_final = []
    for _ in range(n_reps):
        samples = sample_p0(int(n))
        densities, _ = pt_predictive(samples, L_best)
        risks_final.append(max(kl_risk_pt(densities, p0_grid, x_grid, n_cells), 0))
    mean_risk = np.mean(risks_final)
    se = np.std(risks_final) / np.sqrt(n_reps)
    
    results.append((n, mean_risk, se, L_best))
    print(f"n={n:>5}, L*={L_best:>2}, R_KL = {mean_risk:.5f} +/- {se:.5f}")

ns_arr = np.array([r[0] for r in results])
risks_arr = np.array([r[1] for r in results])
ses_arr = np.array([r[2] for r in results])

# Fit asymptotic slope
slope, intercept = np.polyfit(np.log(ns_arr[3:]), np.log(risks_arr[3:]), 1)
print(f"\nEmpirical asymptotic slope: {slope:.3f}")

# Plot
# The figure is drawn only when the script is run directly; combined_priors.py
# runs this file with runpy to reuse its unrounded risks without redrawing it.
if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.errorbar(ns_arr, risks_arr, yerr=2*ses_arr, fmt='s-', markersize=8, lw=2,
                label='Empirical KL risk (oracle level $L^*$)', color='C1')

    # Reference rates
    for beta_target, color, ls in [(1, 'C2', ':'), (2, 'C3', '--')]:
        rate_curve = ns_arr**(-2*beta_target/(2*beta_target+1)) * np.log(ns_arr)
        const = risks_arr[2] / rate_curve[2]
        ax.plot(ns_arr, const * rate_curve, ls, lw=1.5, color=color,
                label=fr'$n^{{-2\beta/(2\beta+1)}}\log n$, $\beta={beta_target}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'sample size $n$', fontsize=12)
    ax.set_ylabel(r'predictive KL risk $R_{\mathrm{KL}}(P_0; \hat p_{\mathrm{PT}})$', fontsize=12)
    ax.set_title('Pólya tree (Dirichlet cell prior): KL risk for Beta(3,3) target', fontsize=12)
    ax.legend(loc='lower left', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(FIGS / 'pt_rate.pdf', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved.")
