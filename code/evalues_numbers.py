"""Numbers of Section ch:evalues (E-values, likelihood ratios, and the canonical evidence representation).

1. Bernoulli NML (Section ev:sec:code): the conditional q_2^(N)(0|0) = P^(N)(00) / P^(N)(0) at horizons
   N = 2..7 (0.800 and 0.785 in the text); the one-step sums s_t(x^{t-1}) = sum_{x_t} P^(t)(x^t) / P^(t-1)(x^{t-1})
   of the process that codes x^t by the NML law of horizon t, which average one under P^(t-1) but exceed
   one after x^2 = 01; and the mean of that process at the stopping time tau = 3 if x^2 = 01, else 2,
   which is the same for every full-support null and exceeds one.
2. Proposition ev:prop:logloss, Ber(1/2) null against Ber(3/4) alternative: the Brier score
   S(P, x) = -||P - e_x||^2 summed over both outcomes gives a one-step factor with null mean 0.995 and growth
   0.125 nats under the alternative; D(P_1 || P_0) = 0.131; twice the log score gives null mean 1.25.
3. Worked example on the sequential likelihood-ratio test (P_0 = N(0,1), P_1 = N(delta,1), delta = 0.5,
   b = 20): the leading term (log b) / (delta^2 / 2) = 24, with a Monte Carlo check (fixed seed) of E_1[tau_b],
   which includes the O(1) overshoot, and of P_0(tau_b <= 400) <= 1/b.
No figure is written; the script only prints. Run time a few seconds.
"""
import itertools
from math import comb, exp, log

import numpy as np


def ml(k, n):
    """Maximized Bernoulli likelihood of a binary sequence of length n with k ones (0^0 = 1)."""
    p = k / n
    return (p ** k) * ((1 - p) ** (n - k))


def shtarkov(n):
    return sum(comb(n, k) * ml(k, n) for k in range(n + 1))


def nml(prefix, n):
    """Probability of the prefix under the NML law of horizon n (marginal over the remaining symbols)."""
    t, k = len(prefix), sum(prefix)
    return sum(comb(n - t, j) * ml(k + j, n) for j in range(n - t + 1)) / shtarkov(n)


print("1. Bernoulli NML")
for N in range(2, 8):
    print(f"   N={N}  q_2^(N)(0|0) = {nml((0, 0), N) / nml((0,), N):.6f}")
for t in (2, 3, 4):
    sums = {p: sum(nml(p + (x,), t) for x in (0, 1)) / nml(p, t - 1)
            for p in itertools.product((0, 1), repeat=t - 1)}
    mean = sum(nml(p, t - 1) * s for p, s in sums.items())
    body = "  ".join(f"s({''.join(map(str, p))})={s:.4f}" for p, s in sums.items())
    print(f"   t={t}  {body}  mean under P^({t - 1}) = {mean:.4f}")
stopped = sum(nml(p, 2) for p in itertools.product((0, 1), repeat=2) if p != (0, 1))
stopped += sum(nml((0, 1, x), 3) for x in (0, 1))
print(f"   mean of the horizon-t NML process at tau (3 after x^2 = 01, else 2) = {stopped:.6f}")

print("2. Scoring rules, Ber(1/2) null, Ber(3/4) alternative")
P0, P1 = np.array([0.5, 0.5]), np.array([0.25, 0.75])   # (P(X=0), P(X=1))


def brier(P, x):
    return -float(np.sum((P - np.eye(2)[x]) ** 2))


f = np.array([exp(brier(P1, x) - brier(P0, x)) for x in (0, 1)])
D = float(np.sum(P1 * np.log(P1 / P0)))
print(f"   Brier factor: null mean = {np.sum(P0 * f):.4f}, growth under P_1 = {np.sum(P1 * np.log(f)):.4f} nats")
print(f"   D(P_1||P_0) = {D:.4f}")
print(f"   twice the log score, factor (P_1/P_0)^2: null mean = {np.sum(P0 * (P1 / P0) ** 2):.4f}")

print("3. Worked example: delta = 0.5, b = 20")
delta, b = 0.5, 20.0
print(f"   (log b) / (delta^2/2) = {log(b) / (delta ** 2 / 2):.2f}")
rng = np.random.default_rng(20261019)
H, R, CH = 400, 200_000, 20_000


def first_passage(mean):
    taus = []
    for _ in range(R // CH):
        x = rng.normal(mean, 1.0, size=(CH, H))
        s = np.cumsum(delta * x - delta ** 2 / 2, axis=1)
        hit = s >= log(b)
        tau = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, 0)
        taus.append(tau)
    return np.concatenate(taus)


tau1 = first_passage(delta)
print(f"   Monte Carlo under P_1 ({R} paths): E_1[tau_b] = {tau1.mean():.2f} (se {tau1.std() / np.sqrt(R):.2f}), "
      f"paths not stopped by {H}: {(tau1 == 0).sum()}")
tau0 = first_passage(0.0)
print(f"   Monte Carlo under P_0: P_0(tau_b <= {H}) = {(tau0 > 0).mean():.4f} against Ville's 1/b = {1 / b:.4f}")
