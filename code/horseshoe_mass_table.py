# Prior mass near the null: the table of the Numerical Evidence subsection (hs:sec:sim).
# Horseshoe column: mass on |theta| <= a = n^{-1/2} of the lower bound (K/2) log(1+4/theta^2) of
# hs:thm:bounds (tau = 1), which integrates to K[a log(1+4/a^2) + 4 arctan(a/2)].
# Laplace column: exact mass 1 - e^{-a} of the prior exp(-|theta|)/2.
# Prints numbers only and draws no figure.
import numpy as np
K = (2*np.pi**3)**-0.5
def hs_mass(a): return K*(a*np.log1p(4/a**2) + 4*np.arctan(a/2))
def lap_mass(a): return -np.expm1(-a)
def gap(u):                                   # gap as a function of u = log log n
    a = np.exp(-0.5*np.exp(u))
    return -np.log(hs_mass(a)) + np.log(lap_mass(a))
ns = np.array([1e3, 1e4, 1e5, 1e6, 1e8])
print('%6s %10s %10s %8s %8s' % ('n', '-logPi_HS', '-logPi_Lap', 'gap', 'loglogn'))
for n in ns:
    a = n**-0.5; h = -np.log(hs_mass(a)); l = -np.log(lap_mass(a))
    print('%6.0e %10.3f %10.3f %8.3f %8.3f' % (n, h, l, h - l, np.log(np.log(n))))
u = np.log(np.log(ns))
print('slope of gap on log log n over the five rows: %.3f' % np.polyfit(u, gap(u), 1)[0])
for n in (1e8, 1e16, 1e64, 1e256):
    v = np.log(np.log(n)); d = 1e-4
    print('local slope d gap / d loglog n at n=%.0e: %.3f' % (n, (gap(v + d) - gap(v - d))/(2*d)))
