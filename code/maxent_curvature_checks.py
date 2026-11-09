"""Numbers quoted in Section ch:maxent (Maximum Entropy Algorithms).

Checks the Bouchard bound (eq:me-bouchard), the Bohning bound (eq:me-bohning),
the Gibbs variational identity (eq:me-logZ), the curvature H_I^env of
(eq:me-Henv), the J=4, d=9 logit example after Proposition prop:me-notpsd with
its eta table, and the closed-form binary example in the proof of that
proposition. Prints numbers only and writes no figure. Fixed seed; a few
seconds.
"""
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp

rng = np.random.default_rng(2026)

# Design: a J-category logit on d standard normal covariates, n rows, with
# category J as the reference, so theta has K=(J-1)d free coordinates.
J, d, n = 4, 9, 200
K = (J - 1) * d
X = rng.standard_normal((n, d))


def probs(theta):
    eta = np.column_stack([X @ theta.reshape(J - 1, d).T, np.zeros(n)])
    eta -= eta.max(axis=1, keepdims=True)
    P = np.exp(eta)
    return P / P.sum(axis=1, keepdims=True)


def H_U(theta):
    """Minus the Hessian of the average logit log-likelihood (free of y)."""
    P = probs(theta)[:, :J - 1]
    W = P[:, :, None] * np.eye(J - 1)[None] - P[:, :, None] * P[:, None, :]
    return np.einsum('ijk,ia,ib->jakb', W, X, X).reshape(K, K) / n


B = 0.5 * np.kron(np.eye(J - 1) - np.ones((J - 1, J - 1)) / J, X.T @ X / n)


def jacobian(theta):
    """dq_{ij}/dtheta as an (nJ) x K matrix, q_{ij} the channel q(j | x_i)."""
    P = probs(theta)
    Jac = np.zeros((n, J, J - 1, d))
    for k in range(J - 1):
        dP = P * ((np.arange(J) == k)[None, :] - P[:, [k]])
        Jac[:, :, k, :] = dP[:, :, None] * X[:, None, :]
    return Jac.reshape(n * J, K)


def grad_I(theta):
    """Gradient of I_theta, the mutual information of x (uniform on rows) and y."""
    P = probs(theta)
    g = np.log(P / P.mean(axis=0)) / n
    return jacobian(theta).T @ g.reshape(-1)


def hess_I(theta, h):
    """Hessian of I_theta by central differences of the analytic gradient."""
    H = np.zeros((K, K))
    for m in range(K):
        e = np.zeros(K)
        e[m] = h
        H[:, m] = (grad_I(theta + e) - grad_I(theta - e)) / (2 * h)
    return 0.5 * (H + H.T)


def H_env(theta):
    """(dq/dtheta)^T [Hessian of I in q] (dq/dtheta), eq:me-Henv."""
    P = probs(theta)
    pbar = P.mean(axis=0)
    Hq = np.zeros((n, J, n, J))
    for j in range(J):
        Hq[:, j, :, j] = np.diag(1.0 / (n * P[:, j])) - 1.0 / (n * n * pbar[j])
    Jac = jacobian(theta)
    return Jac.T @ Hq.reshape(n * J, n * J) @ Jac


lmin = lambda A: np.linalg.eigvalsh(A)[0]

# 1. The example after Proposition prop:me-notpsd, at one random theta.
theta0 = rng.standard_normal(K)
HU0 = H_U(theta0)
HI_a, HI_b = hess_I(theta0, 1e-4), hess_I(theta0, 1e-5)
print('example J=%d d=%d n=%d: lambda_min(H_U) = %+.4f' % (J, d, n, lmin(HU0)))
print('  lambda_min(Hess I), h=1e-4: %+.6f   h=1e-5: %+.6f   max entry gap %.1e'
      % (lmin(HI_a), lmin(HI_b), np.abs(HI_a - HI_b).max()))
print('  eta table: lambda_min(H_U + eta Hess I)')
for eta in [0, 0.5, 1, 2, 5, 10, 25]:
    print('    eta = %4g: %+.4f' % (eta, lmin(HU0 + eta * HI_b)))
print('  lambda_min(H_I^env) at the same theta: %+.4f' % lmin(H_env(theta0)))

# 2. Bouchard bound (eq:me-bouchard) over random (u, alpha, xi), J terms.
N = 100_000
u = 2 * rng.standard_normal((N, J))
alpha = 2 * rng.standard_normal((N, 1))
xi = 2 * rng.standard_normal((N, J))
z = u - alpha
phi = np.tanh(xi / 2) / (4 * xi)
rhs = alpha[:, 0] + np.sum(0.5 * (z - xi) + phi * (z ** 2 - xi ** 2)
                           + np.logaddexp(0, xi), axis=1)
slack = rhs - logsumexp(u, axis=1)
xo = np.abs(z)                                   # the minimizing xi_j = |u_j - alpha|
phio = np.where(xo > 0, np.tanh(xo / 2) / (4 * xo), 1 / 8)
rhso = alpha[:, 0] + np.sum(0.5 * (z - xo) + phio * (z ** 2 - xo ** 2)
                            + np.logaddexp(0, xo), axis=1)
slacko = rhso - logsumexp(u, axis=1)
print('Bouchard: min slack over %d draws of (u, alpha, xi) = %.2e; at xi=|u-alpha| %.2e;'
      ' xi=|u-alpha| never worse: %s' % (N, slack.min(), slacko.min(),
                                           bool(np.all(slacko <= slack + 1e-12))))

# 3. Bohning bound (eq:me-bohning) at 400 random theta.
gaps = [lmin(B - H_U(rng.standard_normal(K))) for _ in range(400)]
print('Bohning: min over 400 random theta of lambda_min(B - H_U) = %.4f' % min(gaps))

# 4. H_I^env is positive semidefinite (eq:me-Henv) at 300 random theta.
envs = [lmin(H_env(rng.standard_normal(K))) for _ in range(300)]
print('H_I^env: min over 300 random theta of lambda_min = %.2e' % min(envs))

# 5. Gibbs variational identity (eq:me-logZ): the supremum found by a generic
#    optimizer, and the gradient of log Z by central differences.
err_val = err_grad = 0.0
for _ in range(200):
    m = 6
    qbar = rng.dirichlet(np.ones(m))
    uu = rng.standard_normal(m)
    logZ = logsumexp(uu, b=qbar)
    qstar = qbar * np.exp(uu - logZ)

    def negobj(w):
        q = np.exp(w - logsumexp(w))
        return -(q @ uu - np.sum(q * np.log(q / qbar)))

    res = minimize(negobj, np.zeros(m), method='BFGS', options={'gtol': 1e-12})
    err_val = max(err_val, abs(-res.fun - logZ))
    h = 1e-6
    fd = np.array([(logsumexp(uu + h * e, b=qbar) - logsumexp(uu - h * e, b=qbar))
                   / (2 * h) for e in np.eye(m)])
    err_grad = max(err_grad, np.abs(fd - qstar).max())
print('Gibbs identity: max |sup - log Z| = %.1e, max |FD grad - q*| = %.1e over 200 draws'
      % (err_val, err_grad))

# 6. Binary example of the proof: x = +-1 equiprobable, q(1|x) = s(theta x).
#    I(theta) = log 2 + s log s + (1-s) log(1-s), s = s(theta);
#    I''(theta) = s(1-s){1 - theta tanh(theta/2)}.
def I_bin(t):
    s = 1 / (1 + np.exp(-t))
    return np.log(2) + s * np.log(s) + (1 - s) * np.log(1 - s)


def I2_closed(t):
    s = 1 / (1 + np.exp(-t))
    return s * (1 - s) * (1 - t * np.tanh(t / 2))


tstar = brentq(lambda t: t * np.tanh(t / 2) - 1, 0.5, 3)
print('binary: I\'\' < 0 for |theta| > %.4f' % tstar)
for t in [1, 2, 3, 4]:
    h = 1e-4
    fd = (I_bin(t + h) - 2 * I_bin(t) + I_bin(t - h)) / h ** 2
    print('  theta = %d: I\'\' closed form %+.5f, finite difference %+.5f'
          % (t, I2_closed(t), fd))
