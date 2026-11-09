#!/usr/bin/env python3
"""
Numerical experiments of the section "Linear Bayes Methods for Prediction-Powered Inference"
(label ch:ppi, subsection "Numerical Experiments", label sec:num of Elements_of_Bayes_INTERIOR.tex).

The section reports five experiments.  This script implements each of them with numpy and scipy
only (the two real data sets are embedded at the bottom of the file) and prints every value the
section reports, labelled by table and row.  It writes no files and runs from any directory:

    python3 ppi_simulations.py        # the replication counts of the text; a few seconds
    python3 ppi_simulations.py 10     # ten times as many replications (to see Monte Carlo error)

    Table tab:reg    sec:reg    regression under informative labelling      300 replications
    Table tab:surv   sec:surv   survival analysis, simulated                200 replications
    Table tab:mar    sec:mar    MAR sensitivity                             200 replications per gamma
    Table tab:ncctg  sec:ncctg  NCCTG lung cancer, one 80/20 split, B = 200 bootstrap draws
    Table tab:bwt    sec:bwt    birth weight, one informative chart-review draw (plus redraws)

The text leaves some details of the designs open.  The choices made here are the following.

 1. Regression (sec:reg, sec:mar).  X_1..X_4 are iid N(0,1) with an intercept first, so beta has five
    entries and beta_1 = 2 is the coefficient on X_1.  The surrogate Yhat = 0.85 X'beta + eta is an
    external prediction (nothing is fitted, so no cross-fitting is needed) and sd(eta) is set so
    that corr(Yhat, Y) = 0.75 in the population.
 2. Labelling.  A sample of fixed size n = 100 out of M = 700 is drawn by randomised systematic PPS,
    whose first-order inclusion probabilities are exactly pi_i = 100 p_i / sum_j p_j with
    p_i = logistic(1.5 X_i1) ("sized so n = 100 are labelled"); the known pi of the IPW rows are
    these pi_i.  In sec:mar, p_i = logistic(0.5 X_i1 + gamma (Y_i - Ybar)).
 3. Estimators.  Cross-PPI takes A and b1 from the unlabelled pool and the rectifier from the labelled
    sample, as in the Algorithm.  HT divides the weighted sum by M, Hajek by sum(1/pi).  The estimated
    propensity is a logistic regression of the label indicator on X.  Intervals are estimate +- 1.96 se.
    OLS uses the HC0 sandwich, Cross-PPI the variance A^-1 (Vs/n + Vu/N) A^-1 of the text, with Vs the
    covariance of X (Y - Yhat) over the labelled sample and Vu that of X (Yhat - X'beta_hat) over the pool
    (centring by X'beta_hat accounts for the estimation of A, since beta_hat - beta = A^-1 (b1 - A beta + b2);
    the covariance of X Yhat alone omits it), HT and Hajek the Poisson-type design variance of the rectifier
    plus the same Vu/N (the estimation of pi-hat is ignored).  Bayes linear is the GCV
    form of Theorem blgcv with the adjusted variance of eq. blvar; the accurate prior is
    N(beta_true, 0.5^2 I) and the vague prior N(0, 1000^2 I).
 4. sec:mar.  The IPW-Hajek row uses the known pi, which depends on Y_i and is computable for labelled
    units; the variant with an estimated X-only propensity is printed as an extra row.
 5. Survival (sec:surv).  T ~ Weibull(shape 1.5, scale 2), C ~ Unif(0,4); the first 80 of 200 units
    are labelled.  Kaplan-Meier and Berliner-Hill curves stay flat beyond the last observation; IPCW is
    Definition ipcw with the censoring Kaplan-Meier curve; the truth is the restricted mean of the
    Weibull on [0,6].  The surrogate is the exponential curve exp(-lam t) with lam the exponential
    MLE, computed on the unlabelled pool and on the labelled sample ('pool'), or one common rate from
    the labelled sample ('common', whose surrogate adjustment is identically zero).  Both are printed.
    P[S(5) > 0] is printed with the strict test and with a tolerance of 1e-9: without tied times the IPCW
    curve is the Kaplan-Meier curve up to rounding, so a strict test of a value that is zero up to
    rounding is not meaningful (the tolerance column is the one to read).
 6. NCCTG (sec:ncctg).  Row 14 (missing ph.ecog) is dropped, a seeded 80/20 split gives 181 and 46
    patients, time is in days (1 yr = 365 d), bootstrap percentile intervals resample both pools.
 7. Birth weight (sec:bwt).  Full-data OLS of bwt on smoke, age, lwt, factor(race), ht, ui, ftv with
    the classical variance and 1.96 (this reproduces the oracle row).  Labelled units are drawn by
    systematic PPS with pi_i = min(1, c exp(-bwt_i/500)), sum pi_i = 100.  The surrogate is a linear
    fit on the labelled sample, cross-fitted over 5 folds; the Bayes linear prior is the OLS fit on a
    pilot of 40 unlabelled births with covariance doubled (safety factor 2).
"""
import sys

import numpy as np
from scipy import integrate, optimize, special, stats

SEED = 20261018
SCALE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
Z95 = stats.norm.ppf(0.975)


def rng_for(k):
    return np.random.default_rng([SEED, k])


def head(title):
    print('\n' + '=' * 110 + '\n' + title + '\n' + '=' * 110)


# ------------------------------------------------------------------------------------ helpers
def pps_systematic(rng, pi):
    """Fixed-size randomised systematic PPS sample; first-order inclusion probabilities equal pi."""
    perm = rng.permutation(len(pi))
    cum = np.cumsum(pi[perm])
    pts = rng.uniform() + np.arange(int(round(cum[-1])))
    idx = np.minimum(np.searchsorted(cum, pts, side='right'), len(pi) - 1)
    lab = np.zeros(len(pi), bool)
    lab[perm[idx]] = True
    return lab


def ols(Z, y, robust, ridge=0.0):
    G = np.linalg.inv(Z.T @ Z + ridge * np.eye(Z.shape[1]))
    b = G @ (Z.T @ y)
    e = y - Z @ b
    V = G @ ((Z.T * e ** 2) @ Z) @ G if robust else G * (e @ e) / (len(y) - Z.shape[1])
    return b, V


def bayes_linear(b, V, mu0, V0):
    """Adjusted expectation mu0 + L (b - mu0), L = V0 (V0 + V)^-1, and adjusted variance L V."""
    L = V0 @ np.linalg.inv(V0 + V)
    return mu0 + L @ (b - mu0), L @ V


def logit_fit(Z, r):
    """Fitted probabilities of a logistic regression of the 0/1 vector r on Z (Newton's method)."""
    b = np.zeros(Z.shape[1])
    for _ in range(50):
        p = special.expit(Z @ b)
        step = np.linalg.solve((Z.T * (p * (1 - p))) @ Z, Z.T @ (r - p))
        b += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return special.expit(Z @ b)


def ppi_fit(Z, y, yhat, lab, kind, pi=None):
    """beta = A^-1 (b1 + b2) of the Cross-PPI Algorithm; returns the estimate and its variance.
    kind: 'cppi' plain rectifier, 'ht' Horvitz-Thompson / M, 'ht_n' HT / n (diagnostic), 'haj' Hajek."""
    M, unl = len(y), ~lab
    Zu, Zl = Z[unl], Z[lab]
    Nu, n = Zu.shape[0], Zl.shape[0]
    A = Zu.T @ Zu / Nu
    b1 = (Zu * yhat[unl][:, None]).mean(0)
    s = Zl * (y[lab] - yhat[lab])[:, None]
    if kind == 'cppi':
        b2, V2 = s.mean(0), np.cov(s.T) / n
    else:
        w = 1.0 / pi[lab]
        c = (1.0 - pi[lab]) * w ** 2
        if kind in ('ht', 'ht_n'):
            D = M if kind == 'ht' else n
            b2, V2 = (s * w[:, None]).sum(0) / D, (s.T * c) @ s / D ** 2
        else:
            b2 = (s * w[:, None]).sum(0) / w.sum()
            d = s - b2
            V2 = (d.T * c) @ d / w.sum() ** 2
    Ai = np.linalg.inv(A)
    beta = Ai @ (b1 + b2)
    du = Zu * (yhat[unl] - Zu @ beta)[:, None]      # pool term of the linearisation of A^-1 (b1 + b2)
    Vu = np.cov(du.T) / Nu
    return beta, Ai @ (V2 + Vu) @ Ai.T


def summarise(est, var, truth):
    est, se = np.asarray(est), np.sqrt(np.asarray(var))
    cov = np.mean(np.abs(est - truth) <= Z95 * se)
    return dict(cov=cov, width=np.mean(2 * Z95 * se), bias=np.mean(est - truth), sd=np.std(est, ddof=1),
                mean_se=np.mean(se), se_cov=np.sqrt(cov * (1 - cov) / len(est)),
                se_bias=np.std(est, ddof=1) / np.sqrt(len(est)))


# ------------------------------------------------------------- regression (sec:reg, sec:mar)
BETA = np.array([1.0, 2.0, -1.0, 0.5, 1.5])
SD_EPS, M_POP, N_LAB, RHO = 1.5, 700, 100, 0.75
VAR_SIG = np.sum(BETA[1:] ** 2)


def sd_eta(rho):
    """sd(eta) with corr(0.85 X'beta + eta, Y) = rho (attainable for rho <= 0.877)."""
    return np.sqrt((0.85 * VAR_SIG / rho) ** 2 / (VAR_SIG + SD_EPS ** 2) - 0.85 ** 2 * VAR_SIG)


SD_ETA = sd_eta(RHO)


def draw_regression(rng, mode, gamma, sd_eta_value=SD_ETA):
    X = rng.standard_normal((M_POP, 4))
    Z = np.column_stack([np.ones(M_POP), X])
    y = Z @ BETA + SD_EPS * rng.standard_normal(M_POP)
    yhat = 0.85 * (Z @ BETA) + sd_eta_value * rng.standard_normal(M_POP)
    p = special.expit(1.5 * X[:, 0]) if mode == 'reg' else special.expit(0.5 * X[:, 0] + gamma * (y - y.mean()))
    pi = N_LAB * p / p.sum()
    return Z, y, yhat, pi, pps_systematic(rng, pi)


def regression_rows(Z, y, yhat, pi, lab, mode):
    """(estimate, variance) of beta_1 for every row of Table reg or Table mar."""
    out = {}
    b, V = ols(Z[lab], y[lab], robust=True)
    out['OLS (labelled only)'] = (b[1], V[1, 1])
    if mode == 'reg':
        for name, mu0, V0 in (('BL vague prior', np.zeros(5), 1e6 * np.eye(5)),
                              ('BL accurate prior', BETA, 0.25 * np.eye(5))):
            bl, Vbl = bayes_linear(b, V, mu0, V0)
            out[name] = (bl[1], Vbl[1, 1])
    est, V = ppi_fit(Z, y, yhat, lab, 'cppi')
    out['Cross-PPI (no IPW)'] = (est[1], V[1, 1])
    pi_hat = logit_fit(Z, lab.astype(float))
    if mode == 'reg':
        for name, kind, p in (('IPW-PPI HT (known pi)', 'ht', pi), ('IPW-PPI Hajek (known pi)', 'haj', pi),
                              ('IPW-PPI Hajek (estimated pi)', 'haj', pi_hat),
                              ('diagnostic: HT with rectifier divided by n', 'ht_n', pi)):
            est, V = ppi_fit(Z, y, yhat, lab, kind, p)
            out[name] = (est[1], V[1, 1])
    else:
        for name, p in (('IPW-PPI Hajek (known pi incl. Y)', pi), ('variant: IPW-PPI Hajek (estimated X-only pi)', pi_hat)):
            est, V = ppi_fit(Z, y, yhat, lab, 'haj', p)
            out[name] = (est[1], V[1, 1])
    return out


def run_regression(key, reps, mode, gamma=0.0, sd_eta_value=SD_ETA):
    rng, acc = rng_for(key), {}
    for _ in range(reps):
        for name, (e, v) in regression_rows(*draw_regression(rng, mode, gamma, sd_eta_value), mode).items():
            acc.setdefault(name, []).append((e, v))
    return {name: summarise(*zip(*v), 2.0) for name, v in acc.items()}


def cross_ppi_population_bias():
    """Bias 0.15 (S_A^-1 S_l - I) beta of Cross-PPI in the design of Table reg, from population moments (sec:reg).
    S_l is the second moment of (1, X) among labelled units, which a unit with X_1 = x joins with probability
    pi(x) = (n/M) logistic(1.5 x) / E[logistic(1.5 X_1)], the limit of pi_i = 100 w_i / sum_j w_j; S_A is that of the
    units A and b1 average over: the unlabelled pool (the Algorithm) or all M units (Theorem ipwvalid(b)).
    Selection acts on X_1 alone, so only the (intercept, X_1) block of each moment matrix differs from I."""
    def moments(g):
        return [integrate.quad(lambda x: g(x) * x ** k * stats.norm.pdf(x), -np.inf, np.inf)[0] for k in range(3)]
    ew = moments(lambda x: special.expit(1.5 * x))[0]

    def p_lab(x):
        return N_LAB / M_POP * special.expit(1.5 * x) / ew

    S = {}
    for key, g in (('labelled', p_lab), ('pool', lambda x: 1.0 - p_lab(x)), ('all M', lambda x: 1.0)):
        m0, m1, m2 = moments(g)
        S[key] = np.eye(5)
        S[key][0, 1] = S[key][1, 0] = m1 / m0
        S[key][1, 1] = m2 / m0
    return {key: (1 - 0.85) * (np.linalg.solve(S[key], S['labelled']) - np.eye(5)) @ BETA for key in ('pool', 'all M')}


# ------------------------------------------------------------------------- survival machinery
def product_limit(t, e, add=0):
    """prod over event times <= t of (1 - d_j/(n_j + add)); add = 0 Kaplan-Meier, add = 1 Berliner-Hill.
    Returns the event times and the curve just after each; the curve is flat beyond the last datum."""
    order = np.argsort(t, kind='stable')
    t, e = t[order], e[order]
    ut, first = np.unique(t, return_index=True)
    d = np.add.reduceat(e, first)
    S = np.cumprod(1.0 - d / (len(t) - first + add))
    return ut[d > 0], S[d > 0]


def ipcw_curve(t, e):
    """S(t) = 1 - n^-1 sum_{i: delta_i = 1, T_i <= t} 1/G(T_i-), G the Kaplan-Meier curve of the censoring."""
    tc, Gc = product_limit(t, 1.0 - e)
    te = np.sort(t[e == 1])
    k = np.searchsorted(tc, te, side='left')
    G_left = np.where(k == 0, 1.0, Gc[np.maximum(k - 1, 0)]) if len(tc) else np.ones(len(te))
    return te, 1.0 - np.cumsum(1.0 / G_left) / len(t)


def step_at(te, S, t):
    k = np.searchsorted(te, t, side='right')
    return np.where(k == 0, 1.0, S[np.maximum(k - 1, 0)]) if len(te) else np.ones(len(t))


def step_rmst(te, S, tau):
    m = te < tau
    return float(np.sum(np.concatenate(([1.0], S[m])) * np.diff(np.concatenate(([0.0], te[m], [tau])))))


def exp_rmst(lam, tau):
    return (1.0 - np.exp(-lam * tau)) / lam


METHODS = ('Kaplan-Meier', 'Berliner-Hill', 'IPCW', 'PPI-Survival', 'IPW-PPI-Survival')


def survival_estimates(tl, el, tu, eu, surrogate, times, tau):
    """Survival at `times` and restricted mean on [0, tau] (None to skip) for the five methods."""
    km, bh, ip = product_limit(tl, el), product_limit(tl, el, 1), ipcw_curve(tl, el)
    lam_l = el.sum() / tl.sum()
    lam_u = eu.sum() / tu.sum() if surrogate == 'pool' else lam_l
    adj = np.exp(-lam_u * times) - np.exp(-lam_l * times)
    adj_r = exp_rmst(lam_u, tau) - exp_rmst(lam_l, tau) if tau else 0.0
    out = {}
    for name, curve, use in (('Kaplan-Meier', km, 0), ('Berliner-Hill', bh, 0), ('IPCW', ip, 0),
                             ('PPI-Survival', bh, 1), ('IPW-PPI-Survival', ip, 1)):
        out[name] = (step_at(*curve, times) + use * adj, step_rmst(*curve, tau) + use * adj_r if tau else None)
    return out


S_WEIBULL = lambda t: np.exp(-(t / 2.0) ** 1.5)
TRUE_RMST = integrate.quad(S_WEIBULL, 0, 6)[0]


def run_survival(key, reps, surrogate):
    rng, rm, tail = rng_for(key), {}, {}
    for _ in range(reps):
        T, C = 2.0 * rng.weibull(1.5, 200), rng.uniform(0.0, 4.0, 200)
        t, e = np.minimum(T, C), (T <= C).astype(float)
        for name, (s, r) in survival_estimates(t[:80], e[:80], t[80:], e[80:], surrogate, np.array([5.0]), 6.0).items():
            rm.setdefault(name, []).append(r)
            tail.setdefault(name, []).append(s[0])
    return {name: dict(bias=np.mean(np.array(rm[name]) - TRUE_RMST),
                       rmse=np.sqrt(np.mean((np.array(rm[name]) - TRUE_RMST) ** 2)),
                       pos=np.mean(np.array(tail[name]) > 0), pos_tol=np.mean(np.array(tail[name]) > 1e-9),
                       se_bias=np.std(rm[name], ddof=1) / np.sqrt(reps)) for name in METHODS}


def run_ncctg(key, surrogate, B=200):
    keep = np.arange(len(LUNG_TIME)) != LUNG_DROP
    t, e = np.array(LUNG_TIME, float)[keep], (np.array(LUNG_STATUS)[keep] == 2).astype(float)
    rng = rng_for(key)
    perm = rng.permutation(len(t))
    lab, unl = perm[:int(0.8 * len(t))], perm[int(0.8 * len(t)):]
    times = np.array([365.0, 730.0])
    est = survival_estimates(t[lab], e[lab], t[unl], e[unl], surrogate, times, None)
    boot = {m: [] for m in METHODS}
    for _ in range(B):
        bl, bu = rng.choice(lab, len(lab)), rng.choice(unl, len(unl))
        for m, (s, _) in survival_estimates(t[bl], e[bl], t[bu], e[bu], surrogate, times, None).items():
            boot[m].append(s)
    lam_l, lam_u = e[lab].sum() / t[lab].sum(), e[unl].sum() / t[unl].sum()
    ci = {m: np.percentile(np.array(boot[m]), [2.5, 97.5], axis=0) for m in METHODS}
    return len(t), e.mean(), len(lab), len(unl), est, ci, (np.exp(-lam_u * times) - np.exp(-lam_l * times))


def ncctg_split_spread(key, reps=200):
    """Kaplan-Meier and Berliner-Hill at 1 and 2 years on the 181 labelled patients of random 80/20 splits."""
    keep = np.arange(len(LUNG_TIME)) != LUNG_DROP
    t, e = np.array(LUNG_TIME, float)[keep], (np.array(LUNG_STATUS)[keep] == 2).astype(float)
    rng, times, out = rng_for(key), np.array([365.0, 730.0]), []
    for _ in range(reps):
        lab = rng.permutation(len(t))[:int(0.8 * len(t))]
        out.append(np.concatenate([step_at(*product_limit(t[lab], e[lab]), times),
                                   step_at(*product_limit(t[lab], e[lab], 1), times)]))
    return np.array(out)


# ----------------------------------------------------------------------- birth weight (sec:bwt)
def bwt_design():
    d = BW
    Z = np.column_stack([np.ones(len(d['bwt'])), d['smoke'], d['age'], d['lwt'], d['race'] == 2, d['race'] == 3,
                         d['ht'], d['ui'], d['ftv']]).astype(float)
    return Z, d['bwt'].astype(float)


def cross_fit_linear(Z, y, lab, K, rng):
    idx = np.flatnonzero(lab)
    yhat = np.empty(len(y))
    yhat[~lab] = Z[~lab] @ ols(Z[lab], y[lab], False)[0]
    for f in np.array_split(rng.permutation(idx), K):
        yhat[f] = Z[f] @ ols(Z[np.setdiff1d(idx, f)], y[np.setdiff1d(idx, f)], False)[0]
    return yhat


def bwt_rows(rng, Z, y, pi):
    """One chart-review draw; None when a design matrix is singular (a rare category is missing from a fit)."""
    try:
        return _bwt_rows(rng, Z, y, pi)
    except np.linalg.LinAlgError:
        return None


def _bwt_rows(rng, Z, y, pi):
    lab = pps_systematic(rng, pi)
    out = {}
    b, V = ols(Z[lab], y[lab], robust=False)
    out['OLS (labelled, informative)'] = (b[1], V[1, 1])
    pilot = rng.choice(np.flatnonzero(~lab), 40, replace=False)
    mu0, Vp = ols(Z[pilot], y[pilot], robust=False, ridge=1e-8)
    bl, Vbl = bayes_linear(b, V, mu0, 2.0 * Vp)
    out['BL (pilot prior)'] = (bl[1], Vbl[1, 1])
    yhat = cross_fit_linear(Z, y, lab, 5, rng)
    for name, kind, p in (('Cross-PPI (no IPW)', 'cppi', None), ('IPW-PPI HT', 'ht', pi), ('IPW-PPI Hajek', 'haj', pi)):
        est, V = ppi_fit(Z, y, yhat, lab, kind, p)
        out[name] = (est[1], V[1, 1])
    return out


def main():
    # ---------------------------------------------------------------- analytic values
    head('Analytic values quoted in the section')
    cens = integrate.quad(S_WEIBULL, 0, 4)[0] / 4
    print(f'E[T] = 2 Gamma(5/3) = {2 * special.gamma(5 / 3):.5f}  (text 1.805)')
    print(f'P(C<T) = E[min(T,4)]/4 = {cens:.4f}  (text 0.439)')
    print(f'true restricted mean int_0^6 S(t) dt = {TRUE_RMST:.4f}  (the truth used for bias and RMSE of Table surv)')
    print(f'chi2_5 0.95 quantile = {stats.chi2.ppf(0.95, 5):.2f}  (text 11.07)')
    print(f'sd(eta) for corr(Yhat, Y) = 0.75: {SD_ETA:.3f}   sd(Y) = {np.sqrt(VAR_SIG + SD_EPS ** 2):.3f}')
    bias = cross_ppi_population_bias()
    print(f"Cross-PPI bias of beta_1 in Table reg from population moments, 0.15 (S_A^-1 S_l - I) beta: "
          f"{bias['pool'][1]:.4f} with A and b1 over the pool (text 0.11), "
          f"{bias['all M'][1]:.4f} with A and b1 over all M units (text 0.08)")

    # ------------------------------------------------------------------------ Table reg
    R = 300 * SCALE
    head(f'Table reg: beta_1 = 2.0, n = {N_LAB}, N = {M_POP - N_LAB}, {R} replications; '
         f'Monte Carlo s.e. of a coverage near 0.95 = {np.sqrt(0.95 * 0.05 / R):.3f}')
    for name, r in run_regression(1, R, 'reg').items():
        print(f"Table reg | {name:<44s} | cov={r['cov']:.3f} width={r['width']:.3f} bias={r['bias']:+.3f} "
              f"(se cov {r['se_cov']:.3f}, se bias {r['se_bias']:.3f}; sd of estimates {r['sd']:.3f}, "
              f"mean standard error {r['mean_se']:.3f})")

    print('-- Cross-PPI (no IPW) when sd(eta) is calibrated to other correlations (the bias does not involve sd(eta)):')
    for rho in (0.60, 0.75, 0.85):
        r = run_regression(1, R, 'reg', sd_eta_value=sd_eta(rho))['Cross-PPI (no IPW)']
        print(f"Table reg sensitivity | rho={rho:.2f} sd(eta)={sd_eta(rho):.2f} | Cross-PPI cov={r['cov']:.3f} "
              f"width={r['width']:.3f} bias={r['bias']:+.3f}")

    # ----------------------------------------------------------------------- Table surv
    R = 200 * SCALE
    head(f'Table surv: restricted mean on [0,6] (truth {TRUE_RMST:.3f}), {R} replications, n_l = 80, N_u = 120')
    for surrogate in ('pool', 'common'):
        res = run_survival(2, R, surrogate)
        for m in METHODS:
            r = res[m]
            print(f"Table surv [surrogate={surrogate:<6s}] | {m:<17s} | bias={r['bias']:+.3f} rmse={r['rmse']:.3f} "
                  f"P[S(5)>0]={100 * r['pos']:.0f}% (tolerance 1e-9: {100 * r['pos_tol']:.0f}%) (se bias {r['se_bias']:.3f})")

    # ------------------------------------------------------------------------ Table mar
    head(f'Table mar: beta_1 = 2.0, {R} replications per gamma')
    for g in (0.0, 0.25, 0.5, 1.0, 2.0):
        res = run_regression(3, R, 'mar', g)
        o, c = res['OLS (labelled only)'], res['Cross-PPI (no IPW)']
        for name in ('IPW-PPI Hajek (known pi incl. Y)', 'variant: IPW-PPI Hajek (estimated X-only pi)'):
            h = res[name]
            print(f"Table mar | gamma={g:<4.2f} | OLS cov={o['cov']:.3f} bias={o['bias']:+.3f} | Cross-PPI cov={c['cov']:.3f} "
                  f"bias={c['bias']:+.3f} | {name}: cov={h['cov']:.3f} (bias={h['bias']:+.3f}) width={h['width']:.2f} "
                  f"sd of estimates={h['sd']:.3f} mean standard error={h['mean_se']:.3f}")

    # ---------------------------------------------------------------------- Table ncctg
    head('Table ncctg: NCCTG lung cancer, one seeded 80/20 split, bootstrap B = 200')
    for surrogate in ('pool', 'common'):
        n, ev, nl, nu, est, ci, adj = run_ncctg(4, surrogate)
        print(f'n = {n} (dropped row {LUNG_DROP + 1}), event fraction {ev:.3f}, n_l = {nl}, N_u = {nu}; '
              f'[surrogate={surrogate}] surrogate adjustment S_surr,u - S_surr,l at 1 yr, 2 yr = {adj[0]:+.4f}, {adj[1]:+.4f}')
        for m in METHODS:
            s, (lo, hi) = est[m][0], ci[m]
            print(f"Table ncctg [surrogate={surrogate:<6s}] | {m:<17s} | S(1yr)={s[0]:.3f} CI=({lo[0]:.3f}, {hi[0]:.3f}) "
                  f"| S(2yr)={s[1]:.3f} CI=({lo[1]:.3f}, {hi[1]:.3f})")
    t_all, e_all = np.array(LUNG_TIME, float), (np.array(LUNG_STATUS) == 2).astype(float)
    km = product_limit(t_all, e_all)
    print(f'reference: Kaplan-Meier on all 228 patients S(1yr), S(2yr) = {step_at(*km, np.array([365.0, 730.0]))}')
    sp = ncctg_split_spread(6)
    print(f'over {len(sp)} random 80/20 splits (labelled n = 181): Kaplan-Meier S(1yr) mean {sp[:, 0].mean():.3f} sd {sp[:, 0].std(ddof=1):.3f}, '
          f'S(2yr) mean {sp[:, 1].mean():.3f} sd {sp[:, 1].std(ddof=1):.3f}; Berliner-Hill minus Kaplan-Meier '
          f'1yr mean {np.mean(sp[:, 2] - sp[:, 0]):+.4f}, 2yr mean {np.mean(sp[:, 3] - sp[:, 1]):+.4f}, smallest difference '
          f'{min((sp[:, 2] - sp[:, 0]).min(), (sp[:, 3] - sp[:, 1]).min()):+.4f}')

    # ------------------------------------------------------------------------ Table bwt
    Z, y = bwt_design()
    b, V = ols(Z, y, robust=False)
    se = np.sqrt(V[1, 1])
    head('Table bwt: MASS::birthwt, 189 births, labelled n = 100, unlabelled N = 89')
    print(f"Table bwt | Oracle OLS (full data)                | beta_smoke={b[1]:+.0f} CI=({b[1] - Z95 * se:+.0f}, "
          f"{b[1] + Z95 * se:+.0f}) width={2 * Z95 * se:.0f}")
    w = np.exp(-y / 500.0)
    c = np.exp(optimize.brentq(lambda lc: np.minimum(1.0, np.exp(lc) * w).sum() - N_LAB, -50, 50))
    pi = np.minimum(1.0, c * w)
    print(f'selection: pi_i = min(1, {c:.1f} exp(-bwt/500)); {np.sum(pi == 1)} births have pi = 1, '
          f'smallest pi = {pi.min():.4f}, largest weight 1/pi = {1 / pi.min():.0f}')
    rng = rng_for(5)
    first = None
    while first is None:
        first = bwt_rows(rng, Z, y, pi)
    for name, (e, v) in first.items():
        s = np.sqrt(v)
        print(f"Table bwt | {name:<37s} | beta_smoke={e:+.0f} CI=({e - Z95 * s:+.0f}, {e + Z95 * s:+.0f}) width={2 * Z95 * s:.0f}")
    RB = 200 * SCALE
    acc, skipped = {}, 0
    for _ in range(RB):
        rows = bwt_rows(rng, Z, y, pi)
        skipped += rows is None
        for name, (e, v) in (rows or {}).items():
            acc.setdefault(name, []).append((e, 2 * Z95 * np.sqrt(v)))
    print(f'-- the same design redrawn {RB} times, {skipped} skipped for a singular design matrix '
          f'(the table is one draw, so only its distribution can be compared):')
    for name, v in acc.items():
        e, wd = np.array(v).T
        print(f"Table bwt redraws | {name:<37s} | mean={e.mean():+.0f} sd={e.std(ddof=1):.0f} 5%..95%=({np.percentile(e, 5):+.0f}, "
              f"{np.percentile(e, 95):+.0f}) P(est>0)={np.mean(e > 0):.3f} | width mean={wd.mean():.0f} median={np.median(wd):.0f} "
              f"5%..95%=({np.percentile(wd, 5):.0f}, {np.percentile(wd, 95):.0f})")


# ---------------------------------------------------------------------------- embedded data
# Public R data sets, copied from R: survival::lung (NCCTG, time in days, status 1 = censored, 2 = dead)
# and MASS::birthwt (189 births).  LUNG_DROP is the 0-based row with a missing covariate (R row 14).
LUNG_TIME = [
    306, 455, 1010, 210, 883, 1022, 310, 361, 218, 166, 170, 654, 728, 71, 567, 144, 613, 707,
    61, 88, 301, 81, 624, 371, 394, 520, 574, 118, 390, 12, 473, 26, 533, 107, 53, 122, 814,
    965, 93, 731, 460, 153, 433, 145, 583, 95, 303, 519, 643, 765, 735, 189, 53, 246, 689, 65,
    5, 132, 687, 345, 444, 223, 175, 60, 163, 65, 208, 821, 428, 230, 840, 305, 11, 132, 226,
    426, 705, 363, 11, 176, 791, 95, 196, 167, 806, 284, 641, 147, 740, 163, 655, 239, 88, 245,
    588, 30, 179, 310, 477, 166, 559, 450, 364, 107, 177, 156, 529, 11, 429, 351, 15, 181, 283,
    201, 524, 13, 212, 524, 288, 363, 442, 199, 550, 54, 558, 207, 92, 60, 551, 543, 293, 202,
    353, 511, 267, 511, 371, 387, 457, 337, 201, 404, 222, 62, 458, 356, 353, 163, 31, 340, 229,
    444, 315, 182, 156, 329, 364, 291, 179, 376, 384, 268, 292, 142, 413, 266, 194, 320, 181,
    285, 301, 348, 197, 382, 303, 296, 180, 186, 145, 269, 300, 284, 350, 272, 292, 332, 285,
    259, 110, 286, 270, 81, 131, 225, 269, 225, 243, 279, 276, 135, 79, 59, 240, 202, 235, 105,
    224, 239, 237, 173, 252, 221, 185, 92, 13, 222, 192, 183, 211, 175, 197, 203, 116, 188, 191,
    105, 174, 177
]
LUNG_STATUS = [
    2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2,
    2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2,
    2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1,
    2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1,
    2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1
]
LUNG_DROP = 13
BW = {
    'age': np.array([
        19, 33, 20, 21, 18, 21, 22, 17, 29, 26, 19, 19, 22, 30, 18, 18, 15, 25, 20, 28, 32, 31,
        36, 28, 25, 28, 17, 29, 26, 17, 17, 24, 35, 25, 25, 29, 19, 27, 31, 33, 21, 19, 23, 21,
        18, 18, 32, 19, 24, 22, 22, 23, 22, 30, 19, 16, 21, 30, 20, 17, 17, 23, 24, 28, 26, 20,
        24, 28, 20, 22, 22, 31, 23, 16, 16, 18, 25, 32, 20, 23, 22, 32, 30, 20, 23, 17, 19, 23,
        36, 22, 24, 21, 19, 25, 16, 29, 29, 19, 19, 30, 24, 19, 24, 23, 20, 25, 30, 22, 18, 16,
        32, 18, 29, 33, 20, 28, 14, 28, 25, 16, 20, 26, 21, 22, 25, 31, 35, 19, 24, 45, 28, 29,
        34, 25, 25, 27, 23, 24, 24, 21, 32, 19, 25, 16, 25, 20, 21, 24, 21, 20, 25, 19, 19, 26,
        24, 17, 20, 22, 27, 20, 17, 25, 20, 18, 18, 20, 21, 26, 31, 15, 23, 20, 24, 15, 23, 30,
        22, 17, 23, 17, 26, 20, 26, 14, 28, 14, 23, 17, 21
    ]),
    'lwt': np.array([
        182, 155, 105, 108, 107, 124, 118, 103, 123, 113, 95, 150, 95, 107, 100, 100, 98, 118,
        120, 120, 121, 100, 202, 120, 120, 167, 122, 150, 168, 113, 113, 90, 121, 155, 125, 140,
        138, 124, 215, 109, 185, 189, 130, 160, 90, 90, 132, 132, 115, 85, 120, 128, 130, 95,
        115, 110, 110, 153, 103, 119, 119, 119, 110, 140, 133, 169, 115, 250, 141, 158, 112,
        150, 115, 112, 135, 229, 140, 134, 121, 190, 131, 170, 110, 127, 123, 120, 105, 130,
        175, 125, 133, 134, 235, 95, 135, 135, 154, 147, 147, 137, 110, 184, 110, 110, 120, 241,
        112, 169, 120, 170, 186, 120, 130, 117, 170, 134, 135, 130, 120, 95, 158, 160, 115, 129,
        130, 120, 170, 120, 116, 123, 120, 130, 187, 105, 85, 150, 97, 128, 132, 165, 105, 91,
        115, 130, 92, 150, 200, 155, 103, 125, 89, 102, 112, 117, 138, 130, 120, 130, 130, 80,
        110, 105, 109, 148, 110, 121, 100, 96, 102, 110, 187, 122, 105, 115, 120, 142, 130, 120,
        110, 120, 154, 105, 190, 101, 95, 100, 94, 142, 130
    ]),
    'race': np.array([
        2, 3, 1, 1, 1, 3, 1, 3, 1, 1, 3, 3, 3, 3, 1, 1, 2, 1, 3, 1, 3, 1, 1, 3, 3, 1, 1, 1, 2,
        2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 3, 1, 3, 1, 3, 1, 1, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 2, 1, 3, 3, 2, 1, 2, 1, 1, 2, 1, 1, 1, 3, 3, 3, 3, 3,
        1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 2, 1, 1, 1, 2, 1, 3, 1, 1, 1, 3,
        1, 3, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 3, 3, 3, 3, 2, 3, 1, 1, 1, 3, 3, 1,
        1, 2, 1, 3, 3, 3, 1, 1, 1, 1, 3, 2, 1, 2, 3, 1, 3, 3, 3, 2, 1, 3, 3, 1, 1, 2, 2, 2, 3,
        3, 1, 1, 1, 1, 2, 3, 3, 1, 3, 1, 3, 3, 2, 1
    ]),
    'smoke': np.array([
        0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1,
        0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1,
        1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1
    ]),
    'ptl': np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0,
        0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0
    ]),
    'ht': np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1
    ]),
    'ui': np.array([
        1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]),
    'ftv': np.array([
        0, 3, 1, 2, 0, 0, 1, 1, 1, 0, 0, 1, 0, 2, 0, 0, 0, 3, 0, 1, 2, 3, 1, 0, 2, 0, 0, 2, 0,
        1, 1, 1, 1, 1, 0, 2, 2, 0, 2, 1, 2, 2, 1, 0, 0, 0, 4, 0, 2, 0, 1, 0, 0, 2, 0, 0, 0, 0,
        0, 0, 0, 2, 0, 0, 0, 1, 2, 6, 1, 2, 0, 2, 1, 0, 0, 0, 1, 4, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 2, 4, 2, 1, 2, 1, 0, 1,
        0, 0, 2, 1, 1, 0, 1, 0, 2, 2, 1, 0, 1, 1, 0, 2, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0,
        2, 2, 0, 0, 0, 1, 2, 0, 0, 0, 0, 3, 1, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 1, 0, 1, 0, 0, 0,
        0, 0, 1, 3, 0, 2, 1, 3, 0, 0, 2, 2, 0, 0, 3
    ]),
    'bwt': np.array([
        2523, 2551, 2557, 2594, 2600, 2622, 2637, 2637, 2663, 2665, 2722, 2733, 2751, 2750,
        2769, 2769, 2778, 2782, 2807, 2821, 2835, 2835, 2836, 2863, 2877, 2877, 2906, 2920,
        2920, 2920, 2920, 2948, 2948, 2977, 2977, 2977, 2977, 2922, 3005, 3033, 3042, 3062,
        3062, 3062, 3062, 3062, 3080, 3090, 3090, 3090, 3100, 3104, 3132, 3147, 3175, 3175,
        3203, 3203, 3203, 3225, 3225, 3232, 3232, 3234, 3260, 3274, 3274, 3303, 3317, 3317,
        3317, 3321, 3331, 3374, 3374, 3402, 3416, 3430, 3444, 3459, 3460, 3473, 3544, 3487,
        3544, 3572, 3572, 3586, 3600, 3614, 3614, 3629, 3629, 3637, 3643, 3651, 3651, 3651,
        3651, 3699, 3728, 3756, 3770, 3770, 3770, 3790, 3799, 3827, 3856, 3860, 3860, 3884,
        3884, 3912, 3940, 3941, 3941, 3969, 3983, 3997, 3997, 4054, 4054, 4111, 4153, 4167,
        4174, 4238, 4593, 4990, 709, 1021, 1135, 1330, 1474, 1588, 1588, 1701, 1729, 1790, 1818,
        1885, 1893, 1899, 1928, 1928, 1928, 1936, 1970, 2055, 2055, 2082, 2084, 2084, 2100,
        2125, 2126, 2187, 2187, 2211, 2225, 2240, 2240, 2282, 2296, 2296, 2301, 2325, 2353,
        2353, 2367, 2381, 2381, 2381, 2410, 2410, 2410, 2414, 2424, 2438, 2442, 2450, 2466,
        2466, 2466, 2495, 2495, 2495, 2495
    ]),
}

if __name__ == '__main__':
    main()
