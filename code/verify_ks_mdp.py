import pathlib
FIGS = pathlib.Path(__file__).resolve().parents[1] / 'figs'
"""
KS-truncated predictive density: Monte Carlo of Section np:sec:ks-mc and Figure np:fig:ks-mdp.
The truncated predictive CDF F_prior(x) + clip(F_n(x) - F_prior(x), -a/sqrt(n), a/sqrt(n)),
with the uniform baseline F_prior(x) = x, is evaluated on a grid of truncation levels a. For each
n the KL risk is averaged over 15 replications, the same samples serving every a, and its grid
minimizer is compared with the KS distance Delta_n = sqrt(n) sup_x |F_{P0}(x) - x| between the
truth and the baseline, and with sqrt(2 log n). The grid runs past Delta_n at every n.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

np.random.seed(123)
THETA = 0.5

def p0(x):
    # smooth perturbation of the uniform
    return 1 + THETA * np.sqrt(2) * np.cos(2 * np.pi * x)

def F0_cdf(x):
    # CDF of p0
    return x + THETA * np.sqrt(2) * np.sin(2 * np.pi * x) / (2 * np.pi)

def sample_p0(n):
    samples = []
    while len(samples) < n:
        x = np.random.uniform(0, 1, n*2)
        u = np.random.uniform(0, 1, n*2)
        M = 1 + 0.5 * np.sqrt(2)
        accept = u * M < p0(x)
        samples.extend(x[accept].tolist()[:n - len(samples)])
    return np.array(samples[:n])

def truncated_risks(a_values, n, n_reps=15, n_grid=200):
    """Mean KL risk of the truncated predictive at every a in a_values (common samples)."""
    x_grid = np.linspace(0.005, 0.995, n_grid)
    dx = x_grid[1] - x_grid[0]
    p0_grid = p0(x_grid)
    risks = np.zeros((n_reps, len(a_values)))
    for r in range(n_reps):
        samples = sample_p0(n)
        F_n = np.array([np.mean(samples <= x) for x in x_grid])
        dev = F_n - x_grid                      # F_prior(x) = x
        for j, a in enumerate(a_values):
            a_eff = a / np.sqrt(n)
            F_hat = x_grid + np.clip(dev, -a_eff, a_eff)
            F_hat = np.clip(np.maximum.accumulate(F_hat), 0, 1)
            f_hat = gaussian_filter1d(np.gradient(F_hat, dx), sigma=1.5)   # small Gaussian smoothing
            f_hat = np.maximum(f_hat, 1e-4)
            kl = np.sum(p0_grid * np.log(p0_grid / f_hat)) * dx
            risks[r, j] = max(kl, 0)
    return risks.mean(axis=0)

a_grid = np.linspace(0.5, 12.5, 65)          # spacing 0.1875
ns = [100, 200, 500, 1000, 2000, 5000]
x_fine = np.linspace(0, 1, 200001)
sup_dev = np.max(np.abs(F0_cdf(x_fine) - x_fine))   # sqrt(2)/(4 pi) * 2 THETA
print(f"sup|F_P0 - x| = {sup_dev:.5f} (sqrt(2)*THETA/(2 pi) = {np.sqrt(2)*THETA/(2*np.pi):.5f})")
print(f"{'n':>6} {'a*':>7} {'Delta_n':>8} {'sqrt(2logn)':>11} {'risk*':>9} {'risk(a_max)':>11} {'a*/Delta_n':>10}")
optimal_a, deltas, rows = [], [], []
for n in ns:
    R = truncated_risks(a_grid, n)
    idx = int(np.argmin(R))
    a_star, D = a_grid[idx], np.sqrt(n) * sup_dev
    optimal_a.append(a_star); deltas.append(D)
    print(f"{n:>6d} {a_star:>7.3f} {D:>8.3f} {np.sqrt(2*np.log(n)):>11.3f} {R[idx]:>9.5f} {R[-1]:>11.5f} {a_star/D:>10.3f}")
    rows.append(f"${n}$ & ${a_star:.2f}$ & ${D:.2f}$ & ${np.sqrt(2*np.log(n)):.2f}$ \\\\")
print("\n".join(rows))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogx(ns, optimal_a, 'o-', label=r'grid minimizer $\hat a_n^*$ (Monte Carlo)', markersize=8, lw=2)
nn = np.geomspace(ns[0], ns[-1], 200)
ax.semilogx(nn, np.sqrt(nn) * sup_dev, '-', color='C2', lw=2,
            label=r'$\Delta_n=\sqrt{n}\,\sup_x|F_{P_0}(x)-x|$')
ax.semilogx(nn, np.sqrt(2*np.log(nn)), '--', color='C3', lw=2, label=r'$\sqrt{2\log n}$')
ax.set_xlabel(r'sample size $n$')
ax.set_ylabel(r'truncation level $a$')
ax.set_title(r'KS-truncated predictive: risk-minimizing level on the grid $a\in[0.5,12.5]$')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / 'ks_mdp_scaling.pdf', dpi=150, bbox_inches='tight')
print(f"\nFigure saved.")
