"""Numbers of Table dv:tab:thr, Corollary dv:cor:cauchy, the crossover, and the genome-wide worked example
(at equal prior odds, and with prior odds m against each association, which give {log(m^2 n)}^{1/2}).

Leading order (Theorem dv:thm:threshold, theta ~ C(0, sigma), equal prior odds):
BF01 = sqrt(pi n / 2) exp(-t^2 / 2) and t_crit = sqrt(log(pi n / 2)).
Exact: BF01 = N(xbar; 0, sigma^2/n) / int N(xbar; theta, sigma^2/n) C(theta; 0, sigma) dtheta by quadrature,
with xbar = t sigma / sqrt(n); the values do not depend on sigma, so sigma = 1.
No figure is written; the script only prints. Run time about one second.
"""
import numpy as np
from scipy import integrate, optimize, stats

T = 1.96
ROWS = [5, 50, 100, 1_000, 10_000, 100_000]


def bf01_lead(t, n):
    return np.sqrt(np.pi * n / 2) * np.exp(-t**2 / 2)


def bf01_exact(t, n):
    xbar, s = t / np.sqrt(n), 1 / np.sqrt(n)
    m0 = stats.norm.pdf(xbar, 0, s)
    f = lambda th: stats.norm.pdf(xbar, th, s) * stats.cauchy.pdf(th)
    lo, hi = xbar - 40 * s, xbar + 40 * s
    m1 = (integrate.quad(f, -np.inf, lo)[0] + integrate.quad(f, lo, hi, limit=500)[0]
          + integrate.quad(f, hi, np.inf)[0])
    return m0 / m1


def tcrit_exact(n):
    return optimize.brentq(lambda t: np.log(bf01_exact(t, n)), 0.01, 10)


print("Table dv:tab:thr (leading order of Theorem dv:thm:threshold)")
for n in ROWS:
    print(f"n={n:>7}  t_crit={np.sqrt(np.log(np.pi * n / 2)):.2f}  BF01(1.96)={bf01_lead(T, n):.2f}")
print(f"leading-order crossover (t_crit = 1.96, equivalently BF01(1.96) = 1): n = {2 * np.exp(T**2) / np.pi:.1f}")

print("Exact integration of the Cauchy marginal")
for n in ROWS:
    print(f"n={n:>7}  t_crit={tcrit_exact(n):.2f}  BF01(1.96)={bf01_exact(T, n):.2f}")
for n in [1, 2, 20, 21]:
    print(f"n={n:>7}  t_crit={tcrit_exact(n):.3f}  BF01(1.96)={bf01_exact(T, n):.3f}")
n_x = optimize.brentq(lambda n: np.log(bf01_exact(T, n)), 5, 60)
print(f"exact crossover (BF01(1.96) = 1, equivalently exact t_crit = 1.96): n = {n_x:.1f}")

m = 1e6
t_bayes = np.sqrt(np.log(m))
print(f"genome-wide: sqrt(log n) = {t_bayes:.2f}, two-sided p = {2 * stats.norm.sf(t_bayes):.1e}; "
      f"Bonferroni z for 5e-8 = {stats.norm.isf(2.5e-8):.2f}; sqrt(2 log m) = {np.sqrt(2 * np.log(m)):.2f}; "
      f"p ratio = {2 * stats.norm.sf(t_bayes) / 5e-8:.0f}")
n = m
t_odds = np.sqrt(np.log(m**2 * n))
print(f"genome-wide with prior odds m (n = m): sqrt(log(m^2 n)) = {t_odds:.2f}, two-sided p = {2 * stats.norm.sf(t_odds):.1e}")
