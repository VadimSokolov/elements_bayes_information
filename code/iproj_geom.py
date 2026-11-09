import pathlib
FIGS = pathlib.Path(__file__).resolve().parents[1] / 'figs'
"""
Visualization of I-projection geometry for the misspecified Fourier example.

Plot:
- True density p_0(x) = 1 + theta*sqrt(2)*cos(2*pi*K_0*x) at K_0 = 3
- Prior family p_a(x) = 1 + a*sqrt(2)*cos(2*pi*x)
- I-projection: a* = 0 (uniform)
- Csiszar Pythagorean equality visible as orthogonality
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 200)
theta = 0.3
K0 = 3

# True density
p0 = 1 + theta * np.sqrt(2) * np.cos(2 * np.pi * K0 * x)

# Several priors in the family
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.plot(x, p0, 'k-', lw=2.5, label=r'$p_0$ (true, $K_0=3$)')
ax.plot(x, np.ones_like(x), 'r--', lw=2, label=r'$\pi^* = $ Uniform (I-projection)')

# Some sample p_a
for a, color in [(-0.3, 'C0'), (0.3, 'C2'), (0.5, 'C3')]:
    p_a = 1 + a * np.sqrt(2) * np.cos(2 * np.pi * x)
    ax.plot(x, p_a, color=color, lw=1.2, alpha=0.7, label=fr'$p_a$, $a={a}$')

ax.set_xlabel(r'$x$', fontsize=12)
ax.set_ylabel('density', fontsize=12)
ax.set_title(r'True density $p_0$ vs.\ prior family $\{p_a : a \in [-c,c]\}$', fontsize=12)
ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)

# Right panel: KL divergence as function of a
ax = axes[1]
a_grid = np.linspace(-0.6, 0.6, 300)
kl_a = []
for a in a_grid:
    p_a = 1 + a * np.sqrt(2) * np.cos(2 * np.pi * x)
    p_a_safe = np.maximum(p_a, 1e-6)
    kl = np.trapezoid(p0 * np.log(p0 / p_a_safe), x)
    kl_a.append(kl)

ax.plot(a_grid, kl_a, 'k-', lw=2)
ax.axvline(0, color='r', ls='--', lw=1.5, label=r'$a^* = 0$ (I-projection)')

# Theoretical curve: KL(p_0 || p_a) = KL(p_0 || pi^*) + a^2/2 + O(a^4) (Csiszar Pythagorean)
sanov = theta**2 / 2 + theta**4 / 8 + theta**6 / 12
ax.axhline(sanov, color='C2', ls=':', lw=1.5, label=fr'Sanov floor $= {sanov:.4f}$')
theory_curve = sanov + a_grid**2 / 2
ax.plot(a_grid, theory_curve, 'C0--', lw=1.5, alpha=0.7,
        label=r'Csisz\'ar: $\mathrm{KL}(p_0\|\pi^*) + a^2/2$')

ax.set_xlabel(r'$a$', fontsize=12)
ax.set_ylabel(r'$\mathrm{KL}(p_0 \,\|\, p_a)$', fontsize=12)
ax.set_title(r'KL divergence: I-projection minimizes at $a^* = 0$', fontsize=12)
ax.legend(loc='upper center', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS / 'iproj_geom.pdf', dpi=150, bbox_inches='tight')
print("Saved iproj_geom.pdf")
