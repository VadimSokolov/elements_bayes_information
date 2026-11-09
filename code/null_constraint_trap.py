"""
The null-constraint trap: the Example in Section ch:projected and Table
tab:trap with its problem in Section ch:examples.

X = {1,2,3}, g(x) = x, E = {Q : E_Q[X] = 2.2}, prior Pi = Dir(1,1,1).
(i) On the segment mu(u) = (u-0.2, 1.2-2u, u), u in [0.2, 0.6], with
    sigma^2(mu(u)) = Var_mu(X) = 2u - 0.24, it prints E[mu_3] under the co-area
    disintegration (uniform in u) and under the sigma-reweighted law.
(ii) It prints the exact finite-N constraint posterior mean E[mu_3 | E_N],
    under exact conditioning {Xbar_N = 2.2} (N a multiple of 5) and under the
    window |Xbar_N - 2.2| <= N^{-1/2}. Under Dir(1,1,1) the type counts
    (n1, n2, n3) are uniform over the compositions of N and
    E[mu_3 | counts] = (1 + n3)/(N + 3), so E[mu_3 | E_N] is a finite average,
    computed here in exact integer arithmetic (no simulation).
Deterministic; runs in about a second.
"""
from fractions import Fraction

from scipy import integrate

# (i) the two candidate limits on the segment
sig = lambda u: (2 * u - 0.24) ** 0.5
coarea = integrate.quad(lambda u: u, 0.2, 0.6)[0] / 0.4
num = integrate.quad(lambda u: u * sig(u), 0.2, 0.6)[0]
den = integrate.quad(sig, 0.2, 0.6)[0]
v0, v1 = 0.16, 0.96                             # closed form with v = 2u - 0.24
closed = 0.5 * (0.4 * (v1 ** 2.5 - v0 ** 2.5) / (2 / 3 * (v1 ** 1.5 - v0 ** 1.5)) + 0.24)
print(f"co-area (uniform in u): E[mu_3 | E] = {coarea:.6f}")
print(f"sigma-reweighted: E[mu_3] = {num / den:.6f} (closed form {closed:.6f}), "
      f"difference {100 * (num / den - coarea):.3f} percentage points")


def post_mean(N, accept):
    """Exact E[mu_3 | E_N]: average of (1+n3)/(N+3) over accepted compositions."""
    total = count = 0
    for S in range(N, 3 * N + 1):               # S = n1 + 2 n2 + 3 n3
        if not accept(N, S):
            continue
        t = S - N                               # t = n2 + 2 n3
        lo, hi = max(0, t - N), t // 2          # n2 = t - 2 n3 >= 0, n1 = N - t + n3 >= 0
        if hi < lo:
            continue
        c = hi - lo + 1
        count += c
        total += c + (lo + hi) * c // 2         # sum of (1 + n3) over n3 = lo..hi
    return Fraction(total, count * (N + 3))


exact = lambda N, S: 5 * S == 11 * N                        # Xbar_N = 2.2
window = lambda N, S: (5 * S - 11 * N) ** 2 <= 25 * N       # |Xbar_N - 2.2| <= N^{-1/2}

print("exact conditioning {Xbar_N = 2.2}:")
for N in [5, 100, 400, 1000, 1600, 5000]:
    m = post_mean(N, exact)
    print(f"  N = {N:5d}: E[mu_3 | E_N] = {float(m):.5f}, N (0.4 - mean) = {float(N * (Fraction(2, 5) - m)):.4f}")
print("window |Xbar_N - 2.2| <= N^{-1/2}:")
for N in [100, 400, 1600, 6400]:
    m = post_mean(N, window)
    print(f"  N = {N:5d}: E[mu_3 | E_N] = {float(m):.5f}, N (0.4 - mean) = {float(N * (Fraction(2, 5) - m)):.4f}")
