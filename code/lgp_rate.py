import pathlib
FIGS = pathlib.Path(__file__).resolve().parents[1] / 'figs'
"""
Logistic-Gaussian process predictive: empirical convergence rate.

Simpler implementation: penalized maximum likelihood with cosine basis.
log p(x) = sum_k c_k cos(pi*k*x), penalized by sum c_k^2 / sigma_k^2.

The Matern-nu kernel has spectral density decay k^{-2*nu-1}, so prior std 
sigma_k ~ k^{-nu - 1/2}. Length-scale rescaling: A multiplies scale.

The penalty used here, weights k^4 (the squared Sobolev H^2 norm), is that of
Matern-3/2 (nu=1.5): sigma_k ~ k^{-2}, since the Matern-nu RKHS on an interval
is H^{nu+1/2}. (Matern-5/2 would need sigma_k ~ k^{-3}, weights k^6.)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from scipy.optimize import minimize

np.random.seed(42)

def p0(x):
    return beta_dist.pdf(x, 3, 3)

def sample_p0(n):
    return np.random.beta(3, 3, n)

def lgp_fit(samples, x_grid, K_basis=30, lam=1.0):
    """
    Penalized cosine-basis fit:
    minimize -loglik(c) + lam * sum_k k^4 c_k^2
    
    This is the Sobolev-2 penalty (Matern-3/2 kernel decay).
    """
    n = len(samples)
    
    def basis(x):
        x_arr = np.atleast_1d(x)
        B = np.zeros((len(x_arr), K_basis))
        B[:, 0] = 1.0
        for k in range(1, K_basis):
            B[:, k] = np.sqrt(2) * np.cos(np.pi * k * x_arr)
        return B
    
    penalty_weights = np.array([0] + [k**4 for k in range(1, K_basis)])  # k^4 = Sobolev-2 = Matern-3/2
    
    eval_grid = np.linspace(0.001, 0.999, 200)
    B_eval = basis(eval_grid)
    B_samples = basis(samples)
    
    def neg_log_post(c):
        lp_samples = B_samples @ c
        lp_eval = B_eval @ c
        max_lp = lp_eval.max()
        log_Z = max_lp + np.log(np.trapezoid(np.exp(lp_eval - max_lp), eval_grid))
        nll = -lp_samples.sum() + n * log_Z
        penalty = lam * 0.5 * np.sum(penalty_weights * c**2)
        return nll + penalty
    
    c_init = np.zeros(K_basis)
    result = minimize(neg_log_post, c_init, method='L-BFGS-B', options={'maxiter': 200})
    c_hat = result.x
    
    B_pred = basis(x_grid)
    lp_pred = B_pred @ c_hat
    max_lp = lp_pred.max()
    Z = np.trapezoid(np.exp(lp_pred - max_lp), x_grid)
    p_hat = np.exp(lp_pred - max_lp) / Z
    return p_hat

def kl_risk(p0_vals, p_hat_vals, x_grid):
    eps = 1e-10
    p_hat_safe = np.maximum(p_hat_vals, eps)
    valid = (p0_vals > eps)
    integrand = np.zeros_like(p0_vals)
    integrand[valid] = p0_vals[valid] * np.log(p0_vals[valid] / p_hat_safe[valid])
    return np.trapezoid(integrand, x_grid)

ns = np.array([100, 200, 500, 1000, 2000, 5000])
n_reps = 20
n_grid = 200
x_grid = np.linspace(0.01, 0.99, n_grid)
p0_grid = p0(x_grid)

results = []
for n in ns:
    # Oracle penalty: search over lam, find best
    lam_grid = np.logspace(-4, 1, 20)
    risks_by_lam = []
    for lam in lam_grid:
        risks_l = []
        for _ in range(5):
            samples = sample_p0(int(n))
            p_hat = lgp_fit(samples, x_grid, lam=lam)
            risks_l.append(kl_risk(p0_grid, p_hat, x_grid))
        risks_by_lam.append(np.mean(risks_l))
    
    lam_best = lam_grid[np.argmin(risks_by_lam)]
    
    risks_final = []
    for _ in range(n_reps):
        samples = sample_p0(int(n))
        p_hat = lgp_fit(samples, x_grid, lam=lam_best)
        risks_final.append(max(kl_risk(p0_grid, p_hat, x_grid), 0))
    mean_risk = np.mean(risks_final)
    se = np.std(risks_final) / np.sqrt(n_reps)
    
    results.append((n, mean_risk, se, lam_best))
    print(f"n={n:>5}, lam*={lam_best:.4f}, R_KL = {mean_risk:.5f} +/- {se:.5f}")

ns_arr = np.array([r[0] for r in results])
risks_arr = np.array([r[1] for r in results])
ses_arr = np.array([r[2] for r in results])

slope, intercept = np.polyfit(np.log(ns_arr[1:]), np.log(risks_arr[1:]), 1)
print(f"\nEmpirical asymptotic slope: {slope:.3f}")

# The figure is drawn only when the script is run directly; combined_priors.py
# runs this file with runpy to reuse its unrounded risks without redrawing it.
if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.errorbar(ns_arr, risks_arr, yerr=2*ses_arr, fmt='^-', markersize=8, lw=2,
                label='Empirical KL risk (oracle penalty)', color='C4')

    for beta_target, color, ls in [(1, 'C2', ':'), (2, 'C3', '--')]:
        kappa_lgp = (4*beta_target + 1) / (2*beta_target + 1)  # LGP polylog exponent of eps_n^2 (van der Vaart and van Zanten 2009)
        rate_curve = ns_arr**(-2*beta_target/(2*beta_target+1)) * np.log(ns_arr)**kappa_lgp
        const = risks_arr[1] / rate_curve[1]
        ax.plot(ns_arr, const * rate_curve, ls, lw=1.5, color=color,
                label=fr'$n^{{-2\beta/(2\beta+1)}}(\log n)^{{({4*beta_target+1})/({2*beta_target+1})}}$, $\beta={beta_target}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'sample size $n$', fontsize=12)
    ax.set_ylabel(r'predictive KL risk $R_{\mathrm{KL}}(P_0; \hat p_{\mathrm{LGP}})$', fontsize=12)
    ax.set_title(r'Logistic-Gaussian process (cosine-Sobolev): KL risk for Beta(3,3) target', fontsize=12)
    ax.legend(loc='lower left', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(FIGS / 'lgp_rate.pdf', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved.")
