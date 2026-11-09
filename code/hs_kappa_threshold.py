# Shrinkage weight of the horseshoe at the testing boundary, and the fixed-tau single-observation
# Kullback--Leibler risk at a null coordinate: the numbers of Remark hs:rem:rate, of
# Subsection hs:sec:channel2 and of Subsection hs:sec:kappa (Section ch:hsmdp).
# Model: y | theta ~ N(theta, 1), theta | lambda, tau ~ N(0, lambda^2 tau^2), lambda ~ C+(0, 1),
# kappa = 1/(1 + lambda^2 tau^2), so that the posterior of kappa is hs:eq:kpost with sigma = 1.
# The boundary is t_crit^2 = log n - 2 log log n, hs:eq:hsthresh with its O(1) term set to 0.
# Deterministic quadrature, no random numbers. Prints numbers only and draws no figure.
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp


def u_grid(tau, n1=6000, n2=400):
    """Nodes u = 1 - kappa and weights for integrals over kappa in (0, 1)."""
    s = np.linspace(np.log(tau**2 * 1e-20), np.log(0.5), n1)   # u = e^s on (0, 1/2]
    w1 = np.full(n1, s[1] - s[0]); w1[[0, -1]] /= 2
    x, w = np.polynomial.legendre.leggauss(n2)                  # kappa on [0, 1/2]
    return np.concatenate([np.exp(s), 1 - 0.25 * (x + 1)]), np.concatenate([w1 * np.exp(s), 0.25 * w])


def post_weights(y, tau, u, w):
    """Normalized posterior weights of hs:eq:kpost (sigma = 1) on the nodes u = 1 - kappa."""
    lp = -0.5 * np.log(u) - np.log(u + (1 - u) * tau**2) - 0.5 * y**2 * (1 - u)
    p = w * np.exp(lp - lp.max())
    return p / p.sum()


def kappa_moments(y, tau):
    u, w = u_grid(tau)
    p = post_weights(y, tau, u, w)
    m = p @ u
    return 1 - m, p @ (u - m)**2                                 # E[kappa | y], Var[kappa | y]


def tcrit(n):
    return np.sqrt(np.log(n) - 2 * np.log(np.log(n)))


def null_risk(tau, ny=1201, ymax=12.0, nz=100):
    """E_y KL(N(0,1) || predictive given y) at theta = 0, for one observation and fixed tau."""
    u, w = u_grid(tau, n1=2500, n2=200)
    z, wz = np.polynomial.hermite_e.hermegauss(nz)
    wz = wz / wz.sum()
    ys = np.linspace(0.0, ymax, ny)
    kl = np.empty(ny)
    for i, y in enumerate(ys):
        p = post_weights(y, tau, u, w)
        # log N(z; u y, 1 + u) - log phi(z), written without cancellation
        uc = u[:, None]
        d = -0.5 * np.log1p(uc) + uc * (z[None, :]**2 + 2 * z[None, :] * y - uc * y**2) / (2 * (1 + uc))
        x = p @ np.expm1(d)                                      # r(z) - 1, r = predictive / phi
        with np.errstate(divide='ignore'):
            lr = logsumexp(np.log(p)[:, None] + d, axis=0)       # log r(z), stable when r is far from 1
        small = np.abs(x) < 0.5
        kl[i] = wz @ np.where(small, x - np.log1p(np.where(small, x, 0.0)), np.expm1(lr) - lr)
    f = kl * np.exp(-ys**2 / 2) / np.sqrt(2 * np.pi)
    h = ys[1] - ys[0]                                            # Simpson's rule, doubled for y < 0
    return 2 * h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())


print('Posterior mean of kappa at |y| = t_crit (Subsection hs:sec:channel2)')
for n, tau in [(1e3, 1e3**-0.5), (1e4, 10 / 1e4), (1e6, 1e-5)]:
    print('  n = %.0e, tau = %.3g: t_crit = %.3f, E[kappa | y] = %.5f' % (n, tau, tcrit(n), kappa_moments(tcrit(n), tau)[0]))
print('Root of E[kappa | y] = 1/2 against sqrt(2 log(1/tau))')
for tau in [1e-2, 1e-3, 1e-4]:
    r = brentq(lambda y: kappa_moments(y, tau)[0] - 0.5, 1.0, 10.0, xtol=1e-10)
    print('  tau = %.0e: root %.3f, sqrt(2 log(1/tau)) = %.3f' % (tau, r, np.sqrt(2 * np.log(1 / tau))))
print('Posterior variance of kappa at |y| = t_crit (Subsection hs:sec:kappa)')
for n, tau in [(1e3, 1e3**-0.5), (1e4, 1e-2), (1e3, 1e-2), (1e4, 1e-3)]:
    print('  n = %.0e, tau = %.3g: Var[kappa | y] = %.6f' % (n, tau, kappa_moments(tcrit(n), tau)[1]))
print('Fixed-tau single-observation KL risk at theta = 0 (Remark hs:rem:rate)')
for k in range(2, 9):
    tau = 10.0**-k
    R = null_risk(tau)
    print('  tau = 1e-%d: risk %.4e, risk * sqrt(log(1/tau)) / tau = %.4f' % (k, R, R * np.sqrt(np.log(1 / tau)) / tau))
