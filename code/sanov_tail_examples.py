"""
Numbers of the two worked examples on large-deviation tails: "a binomial tail
by types" (Section ch:types) and "the exponential tilt" (Section ch:sanov).

For each example it prints the Sanov exponent D, the rate-only estimate
exp(-n D), the exact tail, their ratio, and the Bahadur--Rao estimate, which
restores the Theta(n^{-1/2}) prefactor that the rate discards. Deterministic;
runs in about a second.
"""
from fractions import Fraction
from math import comb, exp, log, pi, sqrt

import numpy as np
from scipy import integrate, stats

# 1) Binomial tail by types: P = (1/2, 1/2), event {Phat_n(1) >= 0.75}, n = 100.
n, a = 100, 0.75
H = -a * log(a) - (1 - a) * log(1 - a)          # H(0.75) in nats
D = log(2) - H                                  # D(Q||P) at Q = (0.25, 0.75)
rate = exp(-n * D)
exact = float(Fraction(sum(comb(n, k) for k in range(75, n + 1)), 2 ** n))
lam = log(a / (1 - a))                          # natural parameter of the tilt
sig = sqrt(a * (1 - a))                         # sd of one flip under the tilt
br = rate / ((1 - exp(-lam)) * sig * sqrt(2 * pi * n))   # lattice of span 1
print(f"binomial: D = {D:.6f} nats, n D = {n * D:.4f}, exp(-n D) = {rate:.4e}")
print(f"binomial: exact P(Bin(100,1/2) >= 75) = {exact:.4e} "
      f"(scipy {stats.binom.sf(74, n, 0.5):.4e}), ratio rate/exact = {rate / exact:.2f}, "
      f"Bahadur--Rao estimate = {br:.4e}")

# 2) Exponential tilt: P = Exp(1), event {mean of n draws >= a}, a = 2, n = 50.
n, a = 50, 2.0
D = a - 1 - log(a)                              # D(Exp(1/a)||Exp(1)) = 1 - log 2
Dq = integrate.quad(lambda x: np.exp(-x / a) / a * (x * (1 - 1 / a) - np.log(a)),
                    0, np.inf)[0]               # the integral displayed in the text
rate = exp(-n * D)
exact = stats.gamma.sf(n * a, n)                # sum of n Exp(1) draws is Gamma(n, 1)
lam = 1 - 1 / a                                 # tilt Exp(1 - lam) = Exp(1/a)
sig = a                                         # sd of Exp(1/a)
br = rate / (lam * sig * sqrt(2 * pi * n))      # non-lattice Bahadur--Rao
print(f"exponential: D = a-1-log a = {D:.6f} (quadrature {Dq:.6f}), n D = {n * D:.4f}, "
      f"exp(-n D) = {rate:.4e}")
print(f"exponential: exact P(Gamma(50,1) >= 100) = {exact:.4e}, ratio rate/exact = "
      f"{rate / exact:.2f}, Bahadur--Rao estimate = {br:.4e}")
