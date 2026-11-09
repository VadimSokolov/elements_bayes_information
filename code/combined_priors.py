import pathlib
FIGS = pathlib.Path(__file__).resolve().parents[1] / 'figs'
"""
Combined comparison plot: all three priors (DPM, PT, LGP) on the same target,
against the reference rate n^{-2/3} log n. The risks are read, unrounded, from
dpm_rate.py, pt_rate.py and lgp_rate.py, which this script reruns (without their figures).
"""
import numpy as np
import matplotlib.pyplot as plt

import contextlib
import io
import runpy

HERE = pathlib.Path(__file__).resolve().parent
NMAX = 5000  # common range of the three experiments


def load(script):
    """Run a rate script without drawing its figure; return its unrounded risks and s.e. for n <= NMAX."""
    with contextlib.redirect_stdout(io.StringIO()):
        g = runpy.run_path(str(HERE / script))
    keep = g['ns_arr'] <= NMAX
    return g['ns_arr'][keep], g['risks_arr'][keep], g['ses_arr'][keep]


ns, dpm_risks, dpm_ses = load('dpm_rate.py')   # oracle bandwidth
ns_pt, pt_risks, pt_ses = load('pt_rate.py')   # oracle level
ns_lgp, lgp_risks, lgp_ses = load('lgp_rate.py')  # oracle penalty
assert (ns == ns_pt).all() and (ns == ns_lgp).all()
for lab, r in [('DPM', dpm_risks), ('PT', pt_risks), ('LGP', lgp_risks)]:
    print(lab, ' '.join(f'{v:.6f}' for v in r))

fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(ns, dpm_risks, yerr=2*dpm_ses, fmt='o-', markersize=8, lw=2,
            label='DPM of Gaussians', color='C0')
ax.errorbar(ns, pt_risks, yerr=2*pt_ses, fmt='s-', markersize=8, lw=2,
            label='Pólya tree', color='C1')
ax.errorbar(ns, lgp_risks, yerr=2*lgp_ses, fmt='^-', markersize=8, lw=2,
            label='Logistic-GP', color='C4')

# Reference rate
ref = ns**(-2.0/3) * np.log(ns) / 5
ax.plot(ns, ref, '--', lw=1.5, color='gray', label=r'reference $n^{-2/3}\log n$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'sample size $n$', fontsize=12)
ax.set_ylabel(r'predictive KL risk $R_{\mathrm{KL}}(P_0;\hat p_\Pi)$', fontsize=12)
ax.set_title(r'Empirical predictive KL risk for three Bayesian nonparametric priors' + '\n' +
             r'(Beta(3,3) target, oracle adaptive parameter)', fontsize=12)
ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(FIGS / 'combined_priors.pdf', dpi=150, bbox_inches='tight')
print("Combined figure saved.")
print(f"\nFitted slopes:")
print(f"  DPM:  {np.polyfit(np.log(ns[1:]), np.log(dpm_risks[1:]), 1)[0]:.3f}")
print(f"  PT:   {np.polyfit(np.log(ns[1:]), np.log(pt_risks[1:]), 1)[0]:.3f}")
print(f"  LGP:  {np.polyfit(np.log(ns[1:]), np.log(lgp_risks[1:]), 1)[0]:.3f}")
print(f"  Predicted -2/3 = {-2/3:.3f}")
print(f"  Reference n^(-2/3) log n: {np.polyfit(np.log(ns[1:]), np.log(ref[1:]), 1)[0]:.3f}")
