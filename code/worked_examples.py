# Prints the numbers of the worked examples of Section ch:examples (no figures, no randomness):
# Table tab:bern and the m = 2 sentence (Bernoulli components), the Beta(2,2) inequality example,
# and the Brandeis dice tilt. The null-constraint numbers (Table tab:trap, co-area and sigma-reweighted
# values) come from code/null_constraint_trap.py.
# Every quantity is an exact finite sum or a one-dimensional integral or root. Runs in a few seconds.
import numpy as np
from scipy import stats, integrate, optimize

TOL = 1e-9  # closed windows: include types on the boundary despite rounding


def block_law(N, p0, m, alpha=0.8):
    """Law of the number of ones among m fixed coordinates of Ber(p0)^N given |S/N - alpha| <= N^{-1/2}."""
    eps = N ** -0.5
    s = np.arange(N + 1)
    keep = np.abs(s - alpha * N) <= eps * N + TOL
    lw = stats.binom.logpmf(s[keep], N, p0)   # log scale: the window sits far in the tail of Bin(N, p0)
    w = np.exp(lw - lw.max())
    w = w / w.sum()
    j = np.arange(m + 1)
    law = np.array([np.sum(w * stats.hypergeom.pmf(jj, N, s[keep], m)) for jj in j])
    return law / law.sum()


def tv(a, b):
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()


def binom_law(m, q):
    return stats.binom.pmf(np.arange(m + 1), m, q)


print("== Bernoulli components (Table tab:bern): mu = Ber(0.2), horizon mean 0.8, m = 5, eps_N = N^{-1/2}")
print("   TV between exchangeable laws on {0,1}^m equals TV between the laws of the number of ones.")
m = 5
print("      N   eps_N  split-set   window    total   m(m-1)/(2N)  budget/split")
for N in [25, 100, 400, 800, 1600]:
    eps = N ** -0.5
    L = block_law(N, 0.2, m)
    bullet = binom_law(m, 0.8 - eps)   # window projection of Ber(0.2) onto |Q(1)-0.8| <= eps
    star = binom_law(m, 0.8)           # I-projection Ber(0.8)
    split, window, total = tv(L, bullet), tv(bullet, star), tv(L, star)
    budget = m * (m - 1) / (2 * N)
    print("   %5d   %.3f   %.4f    %.4f    %.4f    %.5f      %.2f" % (N, eps, split, window, total, budget, budget / split))

print("\n== Degenerate affine constraint, N = 800, m = 2: block laws for several baselines")
N, m = 800, 2
laws = {p0: block_law(N, p0, m) for p0 in [0.05, 0.2, 0.5, 0.7]}
for p0, L in laws.items():
    print("   baseline Ber(%.2f): TV to Ber(0.8)^2 = %.4f" % (p0, tv(L, binom_law(m, 0.8))))
pair = max(tv(laws[a], laws[b]) for a in laws for b in laws)
print("   largest pairwise TV between the block laws = %.4f" % pair)

print("\n== Inequality constraint Q(1) >= alpha, prior Beta(2,2), alpha = 0.6")
a = 0.6
prior = stats.beta(2, 2)
mass = prior.sf(a)
mean = integrate.quad(lambda x: x * prior.pdf(x), a, 1)[0] / mass
second = integrate.quad(lambda x: x * x * prior.pdf(x), a, 1)[0] / mass
print("   Pi(E) = %.4f;  E[mu | mu >= alpha] = %.4f;  sd = %.4f" % (mass, mean, np.sqrt(second - mean ** 2)))
print("   finite-horizon means E[mu | S/N >= alpha - eps_N], eps_N = N^{-1/2} (unrelaxed threshold alpha in brackets)")
for N in [25, 50, 100, 200, 400, 800, 1600, 3200]:
    s = np.arange(N + 1)
    pmf = stats.betabinom.pmf(s, N, 2, 2)
    post = (s + 2) / (N + 4)           # E[mu | S = s] under Beta(2,2)
    out = []
    for thr in [a - N ** -0.5, a]:
        keep = s >= thr * N - TOL
        out.append(np.sum(pmf[keep] * post[keep]) / np.sum(pmf[keep]))
    print("   N = %5d: %.4f   [%.4f]" % (N, out[0], out[1]))

print("\n== Brandeis dice: uniform baseline, mean constraint 4.5")
i = np.arange(1, 7)
lam = optimize.brentq(lambda l: np.sum(i * np.exp(l * i)) / np.sum(np.exp(l * i)) - 4.5, 0, 2, xtol=1e-14)
p = np.exp(lam * i) / np.sum(np.exp(lam * i))
H = -np.sum(p * np.log(p))
print("   lambda = %.5f;  p = (%s);  H = %.4f;  log 6 = %.4f;  log 6 - H = %.4f"
      % (lam, ", ".join("%.3f" % x for x in p), H, np.log(6), np.log(6) - H))
