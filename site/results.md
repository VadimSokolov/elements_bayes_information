# Results by chapter <small>Elements of Bayes and Information Theory</small>

The results of [Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations](https://vsokolov.org/books/elements_bayes_information/) that are new in the book, that come from the authors' own papers, or that correct one of those papers, as the book states them: 60 entries (41 proved, 12 conditional, 6 open, 1 counterexample). Results of other authors that the book presents are credited in the [chapter summaries](https://github.com/VadimSokolov/elements_bayes_information/tree/main/summaries).

Status: proved; conditional (proved assuming the named statements, which the book does not prove); open (stated as an open question); counterexample (a proved counterexample); withdrawn (found wrong after release; the entry says what was wrong and which version corrects it).

## Chapter 3. The Tilted de Finetti Synthesis

<a id="constrained_horizon_master_inequality"></a>

### Componentwise constrained horizon master inequality

Theorem 3.2, page 30 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2025a), De Finetti + Sanov = Bayes: Forward and Inverse Sanov in Bayesian Prediction, Statistical Science (revised short communication).

Status: proved.

Statement. Let the alphabet have k elements, let P be a law with min_x P(x)>0, and let E be a nonempty closed convex polyhedral set of laws defined by finitely many linear moment equalities and inequalities, with window E_eps obtained by relaxing each bound by eps. Let P^star be the I projection of P onto E, assume min_x P^star(x)>0, and assume strong regularity of the windowed projections, ||P^star_eps - P^star_eps'||_TV <= C_0 |eps - eps'|. Let eps_N decrease to 0 with N eps_N > c_rnd eventually, where c_rnd/N bounds the shift of each moment under rounding to a type with denominator N. Let L_{N,I,P} be the law of the coordinates X_I of an i.i.d. P sample of size N conditioned on the empirical measure lying in E_{eps_N}. Then there is C(P,E)<infinity such that for all large N, every m element index set I and every delta>0, the total variation distance between L_{N,I,P} and the m fold product of P^star is at most m delta + m(m-1)/(2N) + C(P,E)(N+1)^{2k} exp(-2N delta^2) + C_0 m eps_N.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `thm:master`.

<a id="master_inequality_tuning"></a>

### Tuned rate for the constrained block law

Corollary 3.3, page 30 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2025a), De Finetti + Sanov = Bayes: Forward and Inverse Sanov in Bayesian Prediction, Statistical Science (revised short communication).

Status: proved.

Statement. Under the hypotheses of the componentwise constrained horizon master inequality, taking delta_N = sqrt((k+1) log(N+1)/N) gives, for fixed m, that the total variation distance between the constrained block law L_{N,I,P} and the m fold product of the I projection P^star is O(m eps_N + m sqrt(log N / N) + m^2/N), where k is the alphabet size and eps_N is the window width.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `cor:tuning`.

<a id="projected_de_finetti_limit"></a>

### Projected de Finetti limit

Theorem 3.4, page 32 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2025a), De Finetti + Sanov = Bayes: Forward and Inverse Sanov in Bayesian Prediction, Statistical Science (revised short communication).

Status: proved.

Statement. Let Pi be a prior on laws mu over a finite alphabet, let E be a closed convex set of laws given by finitely many linear moment equalities and inequalities, with window E_eps obtained by relaxing each bound by eps, and let eps_N be window widths. Let T_E(mu) be the I projection of mu onto E, the minimizer of D(Q||mu) over Q in E; let Q_{N,mu} be the law of an i.i.d. mu sequence conditioned on the empirical measure Phat_N of its first N terms lying in E_{eps_N}; and let Pi_{N,E} be the constraint posterior, proportional to P_mu(Phat_N in E_{eps_N}) Pi(d mu). Suppose (i) Pi_{N,E} converges weakly to a law Pi_E; (ii) T_E is continuous on a neighborhood of the support of Pi_E; and (iii) for every fixed m, the Pi_{N,E}-average of the total variation distance between Q_{N,mu}(X_{1:m} in .) and T_E(mu)^{otimes m} tends to 0. Then for every fixed m and every sequence of m element index sets I_N, the law of X_{I_N} under the mixture of the Q_{N,mu} over Pi_{N,E} converges in total variation to the integral of T_E(mu)^{otimes m} against Pi_E(d mu). With Pitilde_E the pushforward of Pi_E under T_E, the laws given by the integrals of q^{otimes m} against Pitilde_E(dq) are projectively consistent and define an infinite exchangeable law whose random directing measure is Q^star = T_E(mu) with mu distributed as Pi_E.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `thm:projected`.

<a id="bayesian_prediction_under_constrained_limit"></a>

### Bayesian prediction under the constrained limit

Corollary 3.5, page 33 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2025a), De Finetti + Sanov = Bayes: Forward and Inverse Sanov in Bayesian Prediction, Statistical Science (revised short communication).

Status: proved.

Statement. Let Pi_E be a law on laws mu over a finite alphabet (in the projected de Finetti limit, the weak limit of the constraint posterior), let P^star_mu be the I projection of mu onto a closed convex constraint set E, and let Ptilde_E be the infinite exchangeable law whose random directing measure is Q^star = P^star_mu with mu distributed as Pi_E. For fixed t and a history x_1, ..., x_t of positive probability under Ptilde_E, Ptilde_E(X_{t+1} = x | X_{1:t} = x_{1:t}) equals the integral of P^star_mu(x) prod_{j<=t} P^star_mu(x_j) against Pi_E(d mu) divided by the integral of prod_{j<=t} P^star_mu(x_j) against Pi_E(d mu). This is the standard de Finetti posterior predictive under the induced prior, the pushforward of Pi_E under mu -> P^star_mu, and Ptilde_E(X_{t+1} in . | X_{1:t}) converges almost surely to Q^star.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `cor:predict`.

<a id="reachability_and_degeneracy"></a>

### Reachability and degeneracy of moment constraints

Proposition 3.7, page 34 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2025a), De Finetti + Sanov = Bayes: Forward and Inverse Sanov in Bayesian Prediction, Statistical Science (revised short communication).

Status: conditional on proofs of its four clauses, which the book states without proof, in particular the concentration rate max(eps_N, N^{-1/2}) and the disintegration limit in clause (ii) and the transverse dimension k-1-r_0 in clause (iv).

Statement. Let Pi be a prior on laws mu over a finite alphabet with k elements, let E be the polyhedral set of laws Q with <g_j,Q> = alpha_j for j <= r_0 and <g_j,Q> >= alpha_j for r_0 < j <= r, let E_eps be its window obtained by relaxing each bound by eps, let E_N be the event that the empirical measure Phat_N of N observations lies in E_{eps_N}, let Pi_{N,E}(d mu) be proportional to P_mu(E_N) Pi(d mu) (the constraint posterior), let Pi_E be its weak limit, let T_E(mu) be the I projection of mu onto E, and let I_E(mu) = inf over Q in E of D(Q||mu), which vanishes exactly on E. Pointwise in mu, (1/N) log P_mu(E_N) tends to -I_E(mu), so Pi_{N,E}(d mu) is proportional to exp(-N I_E(mu) + o_mu(N)) Pi(d mu). (i) If Pi(E) > 0 and Pi assigns zero mass to the boundary of E in the simplex, then Pi_E is Pi conditioned on E and T_E is the identity on E. (ii) If Pi(E) = 0 but E meets the support of Pi, then Pi_{N,E} concentrates on E at rate max(eps_N, N^{-1/2}), Pi_E is the disintegration of Pi along the moment map at alpha, and T_E is the identity on E. (iii) If I^star = inf over the support of Pi of I_E is positive, then Pi_{N,E} concentrates on the set where I_E = I^star and the surviving components are genuine tilts. (iv) For affine E with r_0 equality constraints, T_E(mu) = T_E(nu) if and only if mu and nu lie in a common orbit of the exponential family generated by g_1, ..., g_{r_0}, and the surviving randomness is the component of Pi_E transverse to the tilt family, of dimension k-1-r_0.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `prop:reach`.

## Chapter 4. Risk, Coding, and Shrinkage

<a id="predictive_bayes_risk_decomposition"></a>

### Decomposition of nonparametric predictive Bayes risk into Sanov and contraction terms

Theorem 4.1, page 59 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026i), Nonparametric Predictive Bayes Risk, working paper.

Status: proved.

Statement. Let P_0 be a law on [0,1] with Lebesgue density p_0 satisfying 0 < c_0 <= p_0 <= C_0 < infinity and log p_0 in the Holder class H^beta(L) for some beta in (0, B] and L < infinity. Let the prior Pi have convex, Kullback Leibler closed support on which P -> D(P_0 || P) attains its infimum at a unique element Pi^* P_0 (the reverse I projection), whose density pi^* satisfies c_0 <= pi^* <= C_0. Assume Hellinger contraction: for some epsilon_n -> 0 with n epsilon_n^2 -> infinity and constants M, c > 0, the posterior mass of { P : d_H(P, Pi^* P_0) > M epsilon_n } is at most exp(-c n epsilon_n^2) in P_0^{\otimes n} probability, where d_H^2(P, Q) = (1/2) integral (sqrt p - sqrt q)^2 dx. Assume the Wong and Shen condition: there are eta > 0, M_WS >= 1 and C_WS < infinity with D(pi^* || p) <= C_WS d_H(P, Pi^* P_0)^2 whenever d_H(P, Pi^* P_0) <= eta and pi^* <= M_WS p. Let hat p_Pi be the posterior predictive density, R_KL(P_0; hat p_Pi) = E_{P_0^{\otimes n}} D(p_0 || hat p_Pi) and J_n(Pi, P_0) = E_{P_0^{\otimes n}} D(Pi^* P_0 || hat p_Pi). Then R_KL(P_0; hat p_Pi) = D(P_0 || Pi^* P_0) + J_n(Pi, P_0) + r_n, where |r_n| <= C ( sqrt(D(P_0 || Pi^* P_0) J_n(Pi, P_0)) + J_n(Pi, P_0) ) with C = e C_0 / c_0. If J_n -> 0 and D(P_0 || Pi^* P_0) > 0 then R_KL = D(P_0 || Pi^* P_0)(1 + o(1)); if D(P_0 || Pi^* P_0) = 0 then R_KL = J_n(Pi, P_0)(1 + o(1)).

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `np:thm:abc-csiszar`.

<a id="conditional_sanov_tilted_de_finetti"></a>

### Conditional Sanov and tilted de Finetti

Theorem 4.5, page 61 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2025a), De Finetti + Sanov = Bayes: Forward and Inverse Sanov in Bayesian Prediction, Statistical Science (revised short communication).

Status: proved.

Statement. Let (X_i) be infinitely exchangeable with de Finetti mixing measure Pi, and let E be a closed convex subset of the probability measures with inf over P in E of D(P || Pi^dagger) finite, for a de Finetti baseline Pi^dagger. Conditional on the event that the empirical measure of X_1, ..., X_n lies in E, the predictive law of any fixed block of size m converges to P^circ = argmin over P in E of D(P || Pi^dagger), at rate O(m / n^{1/3}).

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `np:thm:tilted`.

<a id="rubin_sethuraman_risk_balancing_ks"></a>

### Bayes risk optimal Kolmogorov Smirnov threshold at the moderate deviation scale

Theorem 4.9, page 65 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson, Sokolov and Zantedeschi (2026b), Bayes Risk for Goodness of Fit Tests.

Status: proved.

Statement. For the Kolmogorov Smirnov test of the book with Type I error given by Kolmogorov's tail and Type II error equal to the stated bound on the Pitman family with detection scale Delta_n, and prior odds b_n = pi_1 / pi_0, the unique Bayes risk minimizing threshold is c_n^* = Delta_n / 2 - (log b_n) / (4 Delta_n) (1 + o(1)). In the interior regime log b_n = o(Delta_n^2), c_n^* = (1/2) Delta_n (1 + o(1)) and the minimum Bayes risk is B_n(c_n^*) = 4 pi_0 sqrt(b_n) exp( -Delta_n^2 / 2 - (log b_n)^2 / (8 Delta_n^2) ) (1 + o(1)) = O( b_n^{-1/2} e^{-Delta_n^2/2} ). At the boundary Delta_n = sqrt(2 log n), b_n = n, one has c_n^* = (3/8) Delta_n (1 + o(1)) and B_n(c_n^*) = Theta(n^{-25/16}).

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `np:thm:rs`.

<a id="rubin_sethuraman_risk_balancing_general"></a>

### General moderate deviation risk balancing threshold for tests

Theorem 4.11, page 68 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Datta, Polson, Sokolov and Zantedeschi (2026), A New Look at Bayesian Testing.

Status: proved.

Statement. Let T_n / sigma_n converge in law to a nondegenerate limit Z, and let the alternatives sit at detection scale Delta_n -> infinity with P_0(T_n > a sigma_n) = exp(-a^2/2 + o(Delta_n^2)) and P_1(T_n <= a sigma_n) = exp(-(Delta_n - a)^2/2 + o(Delta_n^2)) uniformly in 0 < a < Delta_n. For the Bayes risk R_n(a) = pi_0 P_0(T_n > a sigma_n) + pi_1 P_1(T_n <= a sigma_n) with log(pi_0 / pi_1) / Delta_n^2 -> lambda in (-1/2, 1/2), every Bayes risk minimizing threshold satisfies a_n^* = ( Delta_n / 2 + log(pi_0/pi_1) / Delta_n )(1 + o(1)) and R_n(a_n^*) = pi_0 exp( -(a_n^*)^2/2 + o(Delta_n^2) ). When pi_0 / pi_1 = b_n >= 1, a_n^* >= sqrt(2 log b_n) (1 + o(1)) and R_n(a_n^*) <= pi_1 < 1 / b_n.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `np:thm:rs-general`.

<a id="optimal_truncation_moderate_deviation_scale"></a>

### Optimal truncation at the moderate deviation scale

Conjecture 4.15, page 72 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. New in this book.

Status: open.

Statement. Under the density, Holder, convex support, Hellinger contraction and Wong and Shen assumptions of the predictive Bayes risk decomposition (a density p_0 bounded away from 0 and infinity with log p_0 Holder, a prior with convex Kullback Leibler closed support and reverse I projection Pi^* P_0 with density bounded above and below, Hellinger contraction at rate epsilon_n, and the Wong and Shen condition), let P_0 (possibly depending on n) have CDF F_{P_0} and let Delta_n = sqrt(n) sup_x |F_{P_0}(x) - F_Pi^*(x)|, where F_Pi^* is the CDF of the reverse I projection Pi^* P_0. If Delta_n -> infinity and h_n Delta_n^2 -> infinity, then the Bayes risk optimal truncation level a_n^* of the truncated predictive density of the book satisfies a_n^* = Delta_n (1 + o(1)). In particular at Delta_n = sqrt(2 log n) with h_n log n -> infinity, a_n^* = sqrt(2 log n) (1 + o(1)).

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `np:conj:ks-mdp`.

<a id="master_decomposition_contraction_rate"></a>

### Master decomposition and the contraction term rate

Theorem 4.16, page 74 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026i), Nonparametric Predictive Bayes Risk, working paper.

Status: proved.

Statement. Under the assumptions of the predictive Bayes risk decomposition (a density p_0 bounded away from 0 and infinity with log p_0 Holder, a prior with convex Kullback Leibler closed support and reverse I projection Pi^* P_0 with density pi^* bounded above and below, and Hellinger contraction at rate epsilon_n), let U_n = { P : d_H(P, Pi^* P_0) <= M epsilon_n and pi^* <= M_WS p }, with epsilon_n the Hellinger contraction rate and M_WS the Wong and Shen constant. If E_{P_0^{\otimes n}} of the posterior integral over the complement of U_n of D(pi^* || p) is O(epsilon_n^2), then J_n(Pi, P_0) = O(epsilon_n^2). For a prior adapting over a grid of size b_n, epsilon_n is the contraction rate of the adaptive posterior.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `np:thm:master`.

<a id="adaptive_dpm_rate"></a>

### Adaptive Dirichlet process mixture predictive rate

Theorem 4.18, page 75 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026i), Nonparametric Predictive Bayes Risk, working paper.

Status: conditional on the data driven bandwidth stays within constant factors of the optimal bandwidth, which the book assumes and does not prove.

Statement. For a density p_0 in the Holder class H^beta(L) with beta in (0, B] unknown, if the data driven bandwidth of the Dirichlet process mixture of Gaussians stays within constant factors of h_n^* = n^{-1/(2 beta + 1)}, then J_n(DPM, P_0) is bounded above, up to a constant, by n^{-2 beta/(2 beta + 1)} log n.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `np:thm:dp`.

<a id="adaptive_polya_tree_rate"></a>

### Adaptive Polya tree predictive rate

Theorem 4.19, page 76 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026i), Nonparametric Predictive Bayes Risk, working paper.

Status: conditional on the marginal likelihood truncation level stays within O(1) of the optimal level, which the book assumes and does not prove.

Statement. For p_0 in the Holder class H^beta(L) with beta in (0, 1) unknown, if the marginal likelihood truncation level of the canonical Polya tree stays within O(1) of L_n^* = (log_2 n) / (2 beta + 1), then J_n(PT, P_0) is bounded above, up to a constant, by n^{-2 beta/(2 beta + 1)} log n.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `np:thm:pt`.

<a id="adaptive_logistic_gaussian_process_rate"></a>

### Adaptive logistic Gaussian process predictive rate

Theorem 4.20, page 77 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026i), Nonparametric Predictive Bayes Risk, working paper.

Status: conditional on the posterior contracts in Hellinger at the rate that van der Vaart and van Zanten (2009) prove for the squared exponential kernel; the remaining hypotheses of the master decomposition theorem (its density and Wong and Shen assumptions and its tail condition, that the posterior expected divergence D(pi^* || p) over laws P with d_H(P, Pi^* P_0) > M epsilon_n or pi^* > M_WS p is O(epsilon_n^2)), which the book assumes and does not check for this prior.

Statement. For p_0 in the Holder class H^beta(L) with beta in (0, B] unknown, and a logistic Gaussian process prior with rescaled Matern nu kernel K_A(x, y) = K((x - y)/A), nu >= B, and a length scale hyperprior on [1, n^{1/(2 nu + 1)}], J_n(LGP, P_0) is bounded above, up to a constant, by n^{-2 beta/(2 beta + 1)} (log n)^{kappa(beta)} with kappa(beta) = (4 beta + 1)/(2 beta + 1), twice the polylog exponent of van der Vaart and van Zanten (2009), provided the posterior contracts in Hellinger at the rate n^{-beta/(2 beta + 1)} (log n)^{(4 beta + 1)/(4 beta + 2)} that they prove for the squared exponential kernel, and the remaining hypotheses of the master decomposition theorem (its density and Wong and Shen assumptions and its tail condition) hold.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `np:thm:gp`.

<a id="horseshoe_super_efficiency_sharp_constant"></a>

### Sharp log log n term in horseshoe super efficiency

Theorem 4.27, page 122 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. New in this book.

Status: conditional on the sharp lower bound, which the book gives only in outline, namely that n R_n = (1/2) log(n/2 pi) - 1/2 - E_0 log m_n(ybar) with the marginal density m_n of the sample mean kept below a constant multiple of log n, so that the sharp coefficient of log log n is proved only as an upper bound.

Statement. Let y_1, y_2, ... be independent N(theta_0, sigma^2) and let R_n = n^{-1} sum_{j=1}^{n} E D(p_{theta_0} || hat p_j) be the Cesaro average Kullback Leibler risk of the Bayes predictive density hat p_j after j - 1 observations. Under the horseshoe prior with theta_0 = 0, R_n = n^{-1}( (1/2) log n - log log n + O(1) ).

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `hs:thm:super`.

<a id="scale_mixture_prior_pole_levy_measure"></a>

### Pole of a scale mixture prior at the origin from its Levy measure

Proposition 4.31, page 123 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson, Sokolov and Zantedeschi (2026c), Horseshoe Priors and MDP.

Status: proved.

Statement. For a scale mixture of normals prior pi(theta) with mixing (Levy) measure nu(ds) = s^{-gamma} ds on (0, 1), with the part on s >= 1 contributing a bounded term, pi(theta) = 2^{gamma - 1/2} / sqrt(2 pi) |theta|^{1 - 2 gamma} times the integral from theta^2/2 to infinity of u^{gamma - 3/2} e^{-u} du, plus O(1). As theta -> 0 the integral stays bounded if and only if gamma > 1/2, and then pi(theta) ~ C_gamma |theta|^{1 - 2 gamma} with C_gamma = 2^{gamma - 1/2} Gamma(gamma - 1/2) / sqrt(2 pi). Hence pi(0) is finite for gamma < 1/2, pi(theta) is of order -log|theta| for gamma = 1/2, and pi(theta) is of order |theta|^{-alpha_0} with alpha_0 = 2 gamma - 1 for gamma > 1/2.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `hs:prop:levy`.

<a id="moderate_deviation_threshold_gaussian_testing"></a>

### Moderate deviation risk optimal testing boundary for the normal mean

Theorem 4.32, page 125 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Datta, Polson, Sokolov and Zantedeschi (2026), A New Look at Bayesian Testing.

Status: proved.

Statement. Let the sample mean be N(theta, sigma^2/n) with sigma known, and test H_0: theta = 0 against H_a: theta != 0, with a prior under H_a that is C^3 with at most polynomial growth and density pi(0) = c_pi in (0, infinity) at the origin. Under 0-1 loss with prior weights pi_0 and pi_a, both positive, the risk optimal boundary t_crit on the standardized mean sqrt(n) |xbar| / sigma satisfies t_crit^2 = log n + log(c_pi^{-2}) - log(2 pi sigma^2) + 2 log(pi_0 / pi_a) + O( sqrt((log n)/n) ). The leading term log n is universal across regular models and priors, and the additive constants depend on c_pi, the Fisher information and the prior model odds.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `hs:thm:mdp`.

<a id="mdp_principle_for_ih_models"></a>

### Moderate deviation principle for non regular Ibragimov and Has'minskii models

Conjecture 4.41, page 138 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. New in this book.

Status: open.

Statement. Conjecture. Let a non regular location model of the Ibragimov and Has'minskii (IH) classification have posterior concentration rate n^p with p = 1/2 and limiting likelihood ratio process Z(u), and suppose the limiting Bayes factor, the integral of Z(u) du, has null tail P(integral Z(u) du > t) = t^{-gamma_p + o(1)} for some gamma_p >= 1. Under symmetric 0-1 loss and a prior pi(theta) positive and smooth at theta_0, the Bayes risk optimal threshold in n^p scaled coordinates is t_crit^{(p)} = sqrt(gamma_p log n) + o(sqrt(log n)), and both Type I and Type II Bayes risks decay as n^{-gamma_p/2 + o(1)}. On the original scale, |theta|_crit = n^{-p} sqrt(gamma_p log n).

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `nr:thm:mdp`.

<a id="modified_abos_for_mild_cusp"></a>

### Modified asymptotic Bayes optimality under sparsity for the mild cusp

Conjecture 4.43, page 139 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. New in this book.

Status: open.

Statement. Conjecture. For the exponential power location family EP(beta), with density beta / (2 Gamma(1/beta)) exp(-|x - theta|^beta), beta in (1/2, 1), sparsity pi_n -> 0 and signal strength mu_n = n^{-1/2} sqrt((2/beta) log n), the horseshoe with global scale tau of order pi_n satisfies 1 <= limsup_{n} BR_n^horseshoe / BR_n^oracle <= 2 / beta, where BR_n is the Bayes risk of the multiple testing rule.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `nr:prop:mildcusp`.

<a id="weak_abos_at_the_boundary"></a>

### Weak ABOS for the horseshoe at the boundary beta one half

Conjecture 4.44, page 139 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. New in this book.

Status: open.

Statement. Conjecture. For the exponential power location family EP(beta) with beta = 1/2, the horseshoe attains BR_n^horseshoe = O(BR_n^oracle), but exact asymptotic Bayes optimality under sparsity fails: there is a constant C_* in (1, infinity) with limsup_{n} BR_n^horseshoe / BR_n^oracle = C_*.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `nr:prop:boundary`.

<a id="abos_failure_for_strong_cusp"></a>

### Failure of asymptotic Bayes optimality for the strong cusp

Conjecture 4.45, page 140 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. New in this book.

Status: open.

Statement. Conjecture. For the exponential power location family EP(beta) with beta < 1/2, liminf_{n} BR_n^horseshoe / BR_n^oracle = infinity.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `nr:prop:strongcusp`.

<a id="abos_of_the_beta_horseshoe"></a>

### ABOS of the beta horseshoe for exponential power noise

Conjecture 4.48, page 142 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. New in this book.

Status: open.

Statement. Conjecture. (i) For EP(beta') with beta' in (1/2, 2), sparsity pi_n -> 0 and signal strength mu_n = n^{-p(beta')} sqrt((2/beta') log n), the beta' horseshoe with tau of order pi_n satisfies BR_n^{beta'-HS} / BR_n^oracle -> 1. (ii) For EP(beta') with beta' <= 1/2, the beta' horseshoe with tau of order pi_n^{1/(2 beta')} and polynomial scale threshold t_crit^{(p)} = n^{p(1/(2 beta') - 1)} achieves BR_n^{beta'-HS} / BR_n^oracle -> 1.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `nr:thm:betaHS`.

## Chapter 5. Predictive Geometry and Maximum Entropy

<a id="gaussian_localization_constraint_chart"></a>

### Gaussian localization on the constraint chart

Theorem 5.8, page 155 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2025b), Bayesian Prediction under Moment Conditioning.

Status: conditional on the bad type suppression and good type Gaussian control lemmas and the fixed chart Taylor, Stirling and coarea remainder constants, which the book states with outline proofs only and defers to the Supplementary Material of the source paper.

Statement. Fix a moment adequate finite partition, assume the regularity assumption of the section and lattice regularity, and work in the exact feasibility case along the cofinal set of sample sizes $n$ where the constraint is exactly attainable. For $1\le m\le n$ the constrained predictive law of a block of $m$ draws satisfies $\mu_{n,m}^{(r)}(A)=\int_{\|v\|_2\le\rho_n}\{p_r^*(v)\}^{\otimes m}(A)\,\varphi_{H_r^*}(v)\,dv+R_{n,m,r}(A)$, where $p_r^*(v)=p_r^*+V_rv/\sqrt n$, $V_r$ is a basis of the tangent space of the constraint chart at the I projection $p_r^*$, $\varphi_{H_r^*}$ is the Gaussian density on $\mathbb R^{s_r}$, $s_r$ the dimension of the tangent space, with precision the reduced Hessian $H_r^*$, and $\rho_n$ is the localization radius of the section. For fixed $r$, $\sup_A|R_{n,m,r}(A)|\le C_r\{(1+m)\,n^{-1/2}(\log n)^{3/2}+m^2/n\}$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `thm:gdf`.

<a id="exchangeable_block_collapse"></a>

### Exchangeable block collapse under moment conditioning

Theorem 5.10, page 156 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2025b), Bayesian Prediction under Moment Conditioning.

Status: conditional on the Gaussian localization theorem on the constraint chart, whose proof the book only outlines, deferring the bad type suppression and good type Gaussian control lemmas and the fixed chart remainder constants to the Supplementary Material of the source paper; the step from localization to the collapse bound is proved in the book.

Statement. Fix a moment adequate finite partition of the sample space with $k_r$ cells, cell level reference law $q_r$, and let $p_r^*$ be the I projection of $q_r$ onto the moment constraint, with reduced Hessian $H_r^*$ on the constraint chart. Under the log moment generating function of the moment map under the baseline law is essentially smooth and steep, the target lies in the gradient image of the interior of its domain, and its Hessian at the solution is positive definite and lattice regularity, in the exact feasibility case along the cofinal set of sample sizes $n$ for which the constraint is exactly attainable, and for $1\le m\le n$, the law $\mu_{n,m}^{(r)}$ of a block of $m$ draws from the sample conditioned on its empirical moments satisfies $\|\mu_{n,m}^{(r)}-(p_r^*)^{\otimes m}\|_{TV}\le C_r\bigl[m\,(k_r\,\mathrm{tr}\{(H_r^*)^{-1}\}/n)^{1/2}+m(m-1)/n+(1+m)(\log n)^{3/2}/\sqrt n\bigr]$, with a constant $C_r$ depending only on the chart. In particular, for every fixed $r$ and fixed block size $m$ the right side tends to $0$ as $n\to\infty$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `thm:collapse`.

<a id="window_robust_collapse"></a>

### Window robust collapse without lattice regularity

Theorem 5.11, page 156 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2025b), Bayesian Prediction under Moment Conditioning.

Status: conditional on the conditional concentration of the empirical cell frequencies around the slab minimizer at rate the square root of log n over n, and the distance of order the tolerance between the slab minimizer and the I projection, both cited in the book's outline proof from the Supplementary Material of the source paper.

Statement. Fix a moment adequate finite partition and let the regularity assumption of the section hold. Take a moment tolerance $\eta_n$ with $\eta_n\to0$ and $n\eta_n\to\infty$; lattice regularity is not assumed. Then for $1\le m\le n$ the law $\mu_{n,m}^{(r)}$ of a block of $m$ draws from the sample conditioned to satisfy its moment constraint within tolerance $\eta_n$ obeys $\|\mu_{n,m}^{(r)}-(p_r^*)^{\otimes m}\|_{TV}\le C_r\bigl(m\eta_n+m\sqrt{(\log n)/n}+m^2/n\bigr)$, where $p_r^*$ is the I projection onto the exact constraint and $C_r$ depends only on the chart.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `thm:window`.

<a id="lifted_diagonal_collapse"></a>

### Lifted collapse to the ambient predictive law under refinement

Corollary 5.15, page 159 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2025b), Bayesian Prediction under Moment Conditioning.

Status: conditional on the projection approximation lemma (Lemma S3 of the Supplementary Material of the source paper), which bounds the total variation between the lifted chart projection and the ambient I projection by the cell oscillation, and the window robust collapse bound.

Statement. Fix the block size $m$, take every partition $\Pi_r$ of a refining sequence moment adequate, and lift the chart predictive law to $\mathcal X^m$ by filling in each observed cell label independently from the baseline law $Q$ restricted to that cell. Then there is a deterministic sequence $r_n\to\infty$ such that the total variation distance between the lifted law and $(P^*)^{\otimes m}$ tends to $0$, where $P^*$ is the information projection of $Q$ onto the moment constraint.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `cor:joint`.

<a id="projection_induced_moment_criterion"></a>

### Projection induced moment criterion is a generalized method of moments

Proposition 5.20, page 163 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2025b), Bayesian Prediction under Moment Conditioning.

Status: conditional on the dual expansion proof in the Supplementary Material of the source paper, which the book does not reproduce.

Statement. Fix a cell moment matrix $A_r$ and a twice continuously differentiable moment target $\alpha(\theta)$. Suppose the cell labels are independent with law $p_{\theta_0,r}^*$, the parameter $\theta_0$ is locally identified, and the cell moment covariance $\Omega_{\theta,r}=A_r\Sigma_{\theta,r}A_r^\top$ is nonsingular near $\theta_0$, where $\Sigma_{\theta,r}=\mathrm{diag}(p_{\theta,r}^*)-p_{\theta,r}^*p_{\theta,r}^{*\top}$. Suppose also that with probability tending to one the empirical cell moment $A_r\hat P_n^{(r)}$ lies in the relative interior of the finite mean domain, so its dual solution exists. Then, uniformly over $\|\theta-\theta_0\|_2\le Cn^{-1/2}$ and up to an additive term not depending on $\theta$, the negative log induced conditional likelihood equals $\tfrac n2\,\bar g_{n,r}(\theta)^\top W_{\theta,r}\,\bar g_{n,r}(\theta)+O_P(n^{-1/2})$, where $\bar g_{n,r}(\theta)=A_r\hat P_n^{(r)}-\alpha(\theta)$ with $\hat P_n^{(r)}$ the empirical cell frequencies and $p_{\theta,r}^*$ the I projection of the cell reference law onto the constraint with target $\alpha(\theta)$ and the weight is $W_{\theta,r}=\Omega_{\theta,r}^{-1}$. The parameter information is $\dot\alpha(\theta)^\top\Omega_{\theta,r}^{-1}\dot\alpha(\theta)$, the optimal covariance weighting of generalized method of moments.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `thm:gmm`.

<a id="mutual_information_hessian_not_psd"></a>

### Mutual information Hessian in the channel parameter need not be positive semidefinite

Proposition 5.33, page 188 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026d), Maximum Entropy Algorithms for Predictive Rules: Blahut-Arimoto, Conditional Limits, and Iterative Scaling, working paper.

Status: counterexample, proved.

Statement. The Hessian $\nabla^2_\theta I_\theta$ of the mutual information $I_\theta=I(X;Y)$ with respect to the parameter $\theta$ of a parametric channel $q_\theta(y\mid x)$ need not be positive semidefinite. Example: let $Y$ take $J=2$ values, let $x=\pm1$ with probability $1/2$ each and $q_\theta(1\mid x)=s(\theta x)$ with $s(z)=1/(1+e^{-z})$. Then the marginal of $Y$ is uniform for every $\theta$, $I_\theta=\log2+s\log s+(1-s)\log(1-s)$ with $s=s(\theta)$, and $I_\theta''=s(1-s)\{1-\theta\tanh(\theta/2)\}$, which is negative for $|\theta|>1.55$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `prop:me-notpsd`.

## Chapter 6. Bayes, E-values, and Testing

<a id="canonical_sequential_evidence"></a>

### Likelihood ratio is the canonical sequential evidence under log loss

Theorem 6.1, page 195 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson, Sokolov and Zantedeschi (2026a), Bayes, E-values, and Testing.

Status: proved.

Statement. Test $H_0$ against $H_1$, either possibly composite, with priors $\Pi_0,\Pi_1$ over the model classes and prior predictive mixtures $M_j^{(n)}=\int P_F^{\otimes n}\,\Pi_j(dF)$, and assume the two mixtures are mutually absolutely continuous. Let $H_j$ have prior probability $\pi_j$ and let $L_{ij}>0$ be the loss of deciding $H_i$ when $H_j$ holds ($i\ne j$). The Bayes test rejects $H_0$ when the likelihood ratio process $\Lambda_n=dM_1^{(n)}/dM_0^{(n)}$ exceeds $b=\pi_0L_{10}/(\pi_1L_{01})$, and $\mathbb E_{M_0}[\Lambda_n]=1$, so $\Lambda_n$ is an e variable under the mixture null. If $D(M_1^{(n)}\,\|\,M_0^{(n)})<\infty$, then $\Lambda_n$ is the unique maximizer of the expected log growth $\mathbb E_{M_1}[\log E]$ among e variables $E$ under $M_0^{(n)}$, with maximum $D(M_1^{(n)}\,\|\,M_0^{(n)})$. In this log loss sense the canonical multiplicative evidence is $E_n=\Lambda_n$, with additive evidence $W_n=\log\Lambda_n$. General e processes remain valid but need not recover this rejection region.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `ev:thm:canonical`.

<a id="sequential_liftability_codelength"></a>

### Sequential liftability of a codelength to an e process

Theorem 6.3, page 197 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson, Sokolov and Zantedeschi (2026a), Bayes, E-values, and Testing.

Status: proved.

Statement. For a finite codelength $\ell$ and a null $P_0$ of full support, the sequence $(E_t)$ built from $\ell$ is an e process under $P_0$ if and only if the induced one step factors $q_t(x_t\mid x^{t-1})=\exp\{-\ell(x^{t-1},x_t)+\ell(x^{t-1})\}$ form an $\mathcal F_{t-1}$ measurable sub probability kernel, that is, $\sum_{x_t}q_t(x_t\mid x^{t-1})\le1$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `ev:thm:lift`.

<a id="complexity_validity_tradeoff"></a>

### Complexity validity tradeoff between minimax regret codes and e processes

Corollary 6.4, page 198 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson, Sokolov and Zantedeschi (2026a), Bayes, E-values, and Testing.

Status: proved.

Statement. Unless the normalized maximum likelihood (NML) laws of successive horizons are consistent, each the marginal of the next (in the Bernoulli model they are not), no code that attains the minimax individual sequence regret at every horizon gives an e process. At each horizon only NML attains that regret (Shtarkov 1987), and its one step factor sums from horizon $t-1$ to $t$ average one under the NML law of horizon $t-1$, which forces the conditional factors to exceed unit mass on some history.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `ev:cor:tradeoff`.

<a id="log_loss_uniqueness_evidence"></a>

### Log loss is the unique scoring rule giving a martingale evidence process

Proposition 6.5, page 198 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson, Sokolov and Zantedeschi (2026a), Bayes, E-values, and Testing.

Status: proved.

Statement. Let $P_0$ have full support on a finite alphabet. Among strictly proper scoring rules $S$, the log score is, up to adding a function of the outcome, the unique rule whose induced evidence process $E_t^S=\prod_{s\le t}e^{S(P_1,X_s)}/e^{S(P_0,X_s)}$ is a $P_0$ martingale for every alternative $P_1$. Another strictly proper rule need not give a valid process; when its one step factor $f=e^{S(P_1,\cdot)-S(P_0,\cdot)}$ has $\mathbb E_{P_0}[f]\le1$, its growth rate $\mathbb E_{P_1}[\log f]$ under $P_1$ is at most $D(P_1\,\|\,P_0)$, with equality only if $f=P_1/P_0$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `ev:prop:logloss`.

<a id="risk_optimal_bayes_factor_threshold"></a>

### Risk optimal Bayes factor threshold on the moderate deviation band

Theorem 6.9, page 210 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Datta, Polson, Sokolov and Zantedeschi (2026), A New Look at Bayesian Testing.

Status: proved.

Statement. Let $\bar X_n\sim N(\theta,\sigma^2/n)$ with $\sigma$ known, null $\theta_0=0$, and an alternative prior $\pi$ that is $C^3$ near the null with $\pi(0)=c_\pi>0$ and at most polynomial growth. Then, uniformly for $\bar x$ on the moderate deviation band $|\bar x|=O(\sqrt{(\log n)/n})$, with $t=\sqrt n\,\bar x/\sigma$, $BF_{01}(\bar x)=\dfrac{\sqrt n}{\sigma\sqrt{2\pi}\,c_\pi}\,e^{-t^2/2}\bigl(1+O(\sqrt{(\log n)/n})\bigr)$, and setting $BF_{01}=\pi_a/\pi_0$ (the ratio of prior alternative to null probabilities) gives $t_{\mathrm{crit}}^2=\log n+\log(c_\pi^{-2})-\log(2\pi\sigma^2)+2\log(\pi_0/\pi_a)+O(\sqrt{(\log n)/n})$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `dv:thm:threshold`.

## Chapter 7. Posterior Asymptotics and the Gaussian Regime

<a id="moment_gap_martingale_posterior"></a>

### Moment gap of martingale posteriors

Proposition 7.8, page 238 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026a), Martingale Posterior Predictive Coherence: Hausdorff Moment Hierarchy.

Status: proved.

Statement. Let $(\theta_n)_{n\ge1}$ be a first moment martingale path with values in $(0,1)$, that is $\theta_n=\mathbb{E}[\theta_{n+1}\mid Y_{1:n}]$ for the binary observations $Y_i$ (the Fong, Holmes and Walker condition), and let $\theta_\infty$ be its limit. For every $k\ge2$ the current value $\theta_n$ alone does not fix the $k$ step run probability: the map $\theta_n\mapsto\mathbb{E}[(1-\theta_\infty)^k\mid Y_{1:n}]$ is set valued, since infinitely many priors $\Pi$ on $[0,1]$ have posterior mean $\theta_n$ given $Y_{1:n}$ and they do not all assign the same $k$ step run probability. The whole path, given as a function of every history with $\theta_{n+1}$ taking different values after $Y_{n+1}=0$ and $Y_{n+1}=1$, does fix it, because the martingale condition then determines every predictive probability $P(Y_{n+1}=1\mid Y_{1:n})$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `crit:prop:gap`.

<a id="predictive_variance_gap"></a>

### Predictive variance gap for martingale posteriors

Corollary 7.9, page 238 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026a), Martingale Posterior Predictive Coherence: Hausdorff Moment Hierarchy.

Status: proved.

Statement. Let $X_1,X_2,\dots$ be exchangeable binary with directing parameter $\theta_\infty$ and suppose the martingale condition $\mathbb{E}[\theta_{n+1}\mid X_{1:n}]=\theta_n$ of Fong, Holmes and Walker holds. Given only the current value $0<\theta_n<1$ and $k\ge2$, the predictive variance $\mathrm{Var}(X_{n+1}+\cdots+X_{n+k}\mid X_{1:n})=k\,\mathbb{E}[\theta_\infty(1-\theta_\infty)\mid X_{1:n}]+k^2\,\mathrm{Var}(\theta_\infty\mid X_{1:n})$ is not determined, because $\mathrm{Var}(\theta_\infty\mid X_{1:n})$ requires the second conditional moment $\mathbb{E}[\theta_\infty^2\mid X_{1:n}]$, which the martingale condition does not constrain and which enters with coefficient $k^2-k$ (for $k=1$ it cancels, leaving $\theta_n(1-\theta_n)$). The coverage of a prediction set $C$ for $X_{n+1}+\cdots+X_{n+k}$ derived from the martingale posterior, when $C$ contains some but not all of $\{0,\dots,k\}$, is likewise indeterminate: it is the conditional mean of $\mathbb{P}(\mathrm{Bin}(k,\theta_\infty)\in C)$, which for $k\ge2$ is not affine in $\theta_\infty$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `crit:cor:sequential`.

<a id="admissibility_classes_non_nested"></a>

### Blackwell admissible, anytime valid and conformal rules are non nested

Proposition 7.11, page 242 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026b), Bayes with No Shame: Admissibility Geometries of Predictive Inference.

Status: proved.

Statement. Let $\mathcal{A}_B$, $\mathcal{A}_{AV}$ and $\mathcal{A}_C$ be the classes of Blackwell admissible, anytime valid admissible and conformal valid rules. Then $\mathcal{A}_B$, $\mathcal{A}_{AV}$ and $\mathcal{A}_C$ are pairwise non nested. The Fong, Holmes and Walker martingale condition $\mathbb{E}[\theta_N\mid Y_{1:N-1}]=\theta_{N-1}$ is necessary for $\mathcal{A}_B$ but is satisfied by rules in all three classes as well as by rules in none of them.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `crit:prop:separation`.

<a id="bayesian_sanov"></a>

### Bayesian Sanov principle under misspecification

Proposition 7.20, page 255 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026j), Bayesian Asymptotics: Merging, Martingales, and Moderate Deviations, SSRN 7013738.

Status: proved.

Statement. Let the alphabet $\mathcal{X}$ be finite, let $\Pi$ be a prior on distributions on $\mathcal{X}$ with support $\operatorname{supp}\Pi$, let $P_0$ be the true law and let $Q^\star=\arg\min_{Q\in\operatorname{supp}\Pi}\mathrm{KL}(P_0\,\|\,Q)$ be the reverse information projection of $P_0$ onto the model, assumed unique with $\mathrm{KL}(P_0\,\|\,Q^\star)<\infty$. Then under $P_0$ the posterior $\Pi(\cdot\mid X^n)$ obeys a large deviation principle at speed $n$ with rate function $I_\Pi(Q)=\mathrm{KL}(P_0\,\|\,Q)-\mathrm{KL}(P_0\,\|\,Q^\star)$ on $\operatorname{supp}\Pi$ and $I_\Pi=+\infty$ off it: for every set $E$ with $\inf_{E^\circ}I_\Pi=\inf_{\bar E}I_\Pi$, $\frac1n\log\Pi(Q\in E\mid X^n)\to-\inf_{Q\in E}I_\Pi(Q)$ $P_0$ almost surely. In particular the posterior concentrates on $Q^\star$. In a parametric submodel $\{P_\theta\}$, $Q^\star=P_{\theta^\star}$ with $\theta^\star$ the Kullback Leibler projection, and when the model is correct $Q^\star=P_0$ and the rate is $\mathrm{KL}(P_0\,\|\,Q)$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `ba:prop:bayes-sanov`.

<a id="non_coverage_exponents_of_credible_sets"></a>

### Non coverage exponents of credible sets on moderate scales

Corollary 7.25, page 260 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026j), Bayesian Asymptotics: Merging, Martingales, and Moderate Deviations, SSRN 7013738.

Status: proved.

Statement. Assume the model is locally asymptotically normal at $\theta_0$ with nonsingular Fisher information $\mathcal{I}$, the prior density is continuous and positive at $\theta_0$, the normalized score $\Delta_n$ satisfies a moderate deviation principle on scale $a_n$ with rate $\frac12x^\top\mathcal{I}^{-1}x$, and the local likelihood remainder and tails are exponentially negligible on moderate balls (the moderate deviation assumption of the book), with $a_n=t_n\to\infty$ and $t_n=o(\sqrt n)$. Let $C_n(t)=\{\theta:\sqrt n\,|\mathcal{I}^{1/2}(\theta-\hat\theta)|\le t\}$ be the credible ball of radius $t/\sqrt n$ in the information metric, centered at the maximum likelihood estimator $\hat\theta$. Then $t_n^{-2}\log\pi_n(\theta\notin C_n(t_n)\mid X^n)\to-\frac12$ in $P_0$ probability. If in addition $\sqrt n(\hat\theta-\theta_0)-\mathcal{I}^{-1}\Delta_n$ is exponentially negligible at speed $t_n^2$, that is $t_n^{-2}\log P_0(\|\sqrt n(\hat\theta-\theta_0)-\mathcal{I}^{-1}\Delta_n\|>\varepsilon t_n)\to-\infty$ for every $\varepsilon>0$, then the frequentist non coverage probability $P_0(\theta_0\notin C_n(t_n))$ decays at the same Gaussian exponent.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `ba:cor:coverage`.

<a id="three_regimes_one_rate_function"></a>

### Three regimes of the posterior governed by one quadratic form

Proposition 7.26, page 261 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026j), Bayesian Asymptotics: Merging, Martingales, and Moderate Deviations, SSRN 7013738.

Status: proved.

Statement. In a locally asymptotically normal model satisfying the hypotheses of the Bernstein and von Mises theorem and the moderate deviation assumption of the book, and, for the last regime, on a finite alphabet with $P_{\theta_0}>0$, the deviation behavior of the posterior is governed by the single quadratic form $\frac12v^\top\mathcal{I}v$ ($\mathcal{I}$ the Fisher information at $\theta_0$) across three scales of $\theta-\theta_0$. For $\theta-\theta_0=O(n^{-1/2})$ (with an efficient estimator $\hat\theta_n$) the law of $\sqrt n(\theta-\hat\theta_n)$ tends to $\mathcal{N}(0,\mathcal{I}^{-1})$. For $\theta-\theta_0=a_nn^{-1/2}$ with $1\ll a_n\ll\sqrt n$ the rate is $\frac12v^\top\mathcal{I}v$ at speed $a_n^2$. For $\theta-\theta_0=O(1)$ the rate is $\mathrm{KL}(P_{\theta_0}\,\|\,P_\theta)$ at speed $n$. The three are one rate function at three magnifications, since $\frac12v^\top\mathcal{I}v$ is the inverse covariance of the Bernstein and von Mises Gaussian, the moderate deviation rate, and the Hessian at $\theta_0$ of the large deviation rate $\theta\mapsto\mathrm{KL}(P_{\theta_0}\,\|\,P_\theta)$, because $\nabla^2_\theta\mathrm{KL}(P_{\theta_0}\,\|\,P_\theta)|_{\theta=\theta_0}=\mathcal{I}$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `ba:prop:three`.

<a id="functional_bvm_proportional_asymptotics"></a>

### Functional Bernstein and von Mises and bulk non contraction at proportional rate

Theorem 7.30, page 271 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026j), Bayesian Asymptotics: Merging, Martingales, and Moderate Deviations, SSRN 7013738.

Status: proved.

Statement. Let $y=X\theta_0+\varepsilon$ with $\varepsilon\sim\mathcal{N}(0,\sigma^2I_n)$, let $X\in\mathbb{R}^{n\times d_n}$ have i.i.d. $\mathcal{N}(0,1)$ entries, take a flat prior on $\theta$ and let $d_n/n\to\gamma\in(0,1)$. For deterministic unit vectors $c_n\in\mathbb{R}^{d_n}$, with $\psi=c_n^\top\theta$ and $\hat\psi=c_n^\top\hat\theta$ the least squares value, $\sqrt n(\psi-\hat\psi)\mid y$ converges in law to $\mathcal{N}(0,\sigma^2/(1-\gamma))$ in $P_0$ probability, while the full vector posterior does not contract: $\mathbb{E}[\|\theta-\hat\theta\|^2\mid y]\to\sigma^2\gamma/(1-\gamma)>0$ in probability. Consequently for $a_n\to\infty$ with $a_n=o(\sqrt n)$, $a_n^{-2}\log\mathbb{P}(\sqrt n|\psi-\hat\psi|\ge a_nt\mid y)\to-(1-\gamma)t^2/(2\sigma^2)$ in probability.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `ba:thm:prop`.

<a id="moderate_scale_calibration_misspecification"></a>

### Moderate scale calibration of tempered posteriors under misspecification

Proposition 7.31, page 272 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026j), Bayesian Asymptotics: Merging, Martingales, and Moderate Deviations, SSRN 7013738.

Status: proved.

Statement. Under the moderate deviation assumption of the book with $V$ in place of the Fisher information, consider a misspecified model with Kullback Leibler projection $\theta^\star$, expected curvature $V$ (the Hessian of $\theta\mapsto\mathrm{KL}(P_0\,\|\,P_\theta)$ at $\theta^\star$) and score covariance $\Sigma$. On the moderate scale, the tempered (Gibbs) posterior with learning rate $w$, proportional to the prior times $\exp\{-w\sum_iL(\theta,X_i)\}$, obeys a moderate deviation principle with rate $\frac w2v^\top Vv$, while the frequentist non coverage exponent of its credible balls is $\frac12v^\top(V\Sigma^{-1}V)v$. The two coincide in every direction $v$ if and only if $wV=V\Sigma^{-1}V$, that is $w\Sigma=V$. Hence a scalar learning rate restores moderate scale calibration exactly when $\Sigma^{-1}V$ is a scalar multiple of the identity, with $w^\star$ that multiple; otherwise no scalar suffices, and the spread of the eigenvalues of $\Sigma^{-1}V$ is an irreducible miscalibration that only a matrix temperature can remove.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `ba:prop:calib`.

<a id="best_scalar_temperature"></a>

### Best scalar temperature for a misspecified posterior

Corollary 7.32, page 273 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026j), Bayesian Asymptotics: Merging, Martingales, and Moderate Deviations, SSRN 7013738.

Status: proved.

Statement. With $V$ the expected curvature, $\Sigma$ the score covariance and $d$ the parameter dimension, when $\Sigma$ is not proportional to $V$ no scalar learning rate fully calibrates, but the rate matching the expected credible radius to nominal, equivalently matching the first moment of the coverage statistic, is $w^\star=d/\operatorname{tr}(V^{-1}\Sigma)$, which reduces to $w^\star=1$ under correct specification. The residual miscalibration is governed by the dispersion of the eigenvalues of $V^{-1}\Sigma$ about their mean.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `ba:cor:wstar`.

<a id="mean_field_variational_anti_conservatism"></a>

### Moderate deviation form of mean field variational undercoverage

Proposition 7.33, page 274 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026j), Bayesian Asymptotics: Merging, Martingales, and Moderate Deviations, SSRN 7013738.

Status: proved.

Statement. In the Gaussian Bernstein and von Mises limit $\mathcal{N}(\hat\theta,\mathcal{I}^{-1})$, the Kullback Leibler optimal product approximation is $\prod_i\mathcal{N}(\hat\theta_i,\mathcal{I}_{ii}^{-1})$, and the per coordinate moderate deviation rate of mean field variational Bayes satisfies $J^{\mathrm{VB}}_i(v)=\frac12\mathcal{I}_{ii}v^2\ge J_i(v)=\frac12v^2/(\mathcal{I}^{-1})_{ii}$, with $\kappa_i:=\mathcal{I}_{ii}(\mathcal{I}^{-1})_{ii}\ge1$ and equality if and only if $\mathcal{I}_{i,-i}=0$. Mean field credible intervals therefore undercover at the moderate scale, their half width too small by the factor $\kappa_i^{-1/2}$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `ba:prop:vb`.

<a id="dependent_data_calibration"></a>

### Calibration under dependent data and the long run variance

Proposition 7.34, page 275 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026j), Bayesian Asymptotics: Merging, Martingales, and Moderate Deviations, SSRN 7013738.

Status: conditional on the hypothesis that the posterior precision is asymptotically n V under stationary alpha mixing data and a possibly misspecified model, which the book assumes and does not derive.

Statement. Let the data be stationary and $\alpha$ mixing at a rate and with moments that give the score a central limit theorem, the model possibly misspecified, with $\theta^\star$ the Kullback Leibler projection, $V$ the expected per observation curvature and $\Sigma_{\mathrm{LRV}}=\sum_{k\in\mathbb{Z}}\mathrm{Cov}_0(\dot\ell_0(\theta^\star),\dot\ell_k(\theta^\star))$ the long run variance of the score, and suppose that the posterior precision is asymptotically $nV$. Then $\sqrt n(\hat\theta_n-\theta^\star)$ converges in law to $\mathcal{N}(0,V^{-1}\Sigma_{\mathrm{LRV}}V^{-1})$; hence the Bernstein and von Mises limit is the sandwich form with $\Sigma=\Sigma_{\mathrm{LRV}}$, and the moderate scale calibrating temperature is, in the scalar case, $w^\star=V/\Sigma_{\mathrm{LRV}}$. When the one observation score satisfies the Bartlett identity, positive dependence inflates $\Sigma_{\mathrm{LRV}}$ above $V$ and forces $w^\star<1$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `ba:prop:lrv`.

## Chapter 10. Growth, Filtering, and Monte Carlo

<a id="jeffreys_prior_equalizes_universal_portfolio_regret"></a>

### The Jeffreys prior equalizes the regret of the universal portfolio

Theorem 10.10, page 349 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson (2026), A Bayesian Interpretation of Cover's Universal Portfolio, submitted to Entropy.

Status: conditional on the Laplace expansion of the mixture wealth, which the book states without a formal proof.

Statement. Assume a stationary ergodic market with relatives bounded away from zero and infinity and a unique interior maximizer $b^\star$ of the growth rate $W(b)=\mathbb{E}\log b^\top x$, and write $\mathcal{I}(b)=\mathbb{E}[xx^\top/(b^\top x)^2]$ for the market information matrix on the tangent space of the simplex. Then, almost surely, the regret $R_T$ of the universal portfolio satisfies $R_T=\frac{d-1}2\log\frac{T}{2\pi}+\frac12\log\det\mathcal{I}(b^\star)-\log\pi(b^\star)+o(1)$, where $\pi$ is the prior density. Suppose further that the market is one of a family indexed by the growth optimal portfolio, and read $\mathcal{I}(b)$ as the information of the member whose growth optimal portfolio is $b$. If $\pi$ is the Jeffreys prior $\pi_J(b)\propto\sqrt{\det\mathcal{I}(b)}$ with finite volume $C=\int_\Delta\sqrt{\det\mathcal{I}(b)}\,db$, then $R_T=\frac{d-1}2\log\frac T{2\pi}+\log C+o(1)$, which does not depend on $b^\star$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `up:thm:regret`.

## Chapter 12. Causal Machine Learning

<a id="generalized_control_variate_unification"></a>

### Generalized control variate estimator, optimal shrinkage and variance

Theorem 12.6, page 394 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson, Sokolov and Zantedeschi (2026f), Linear Bayes Methods for Prediction-Powered Inference, SSRN 6744239.

Status: proved.

Statement. Let $\hat\theta_{\mathrm{base}}$ be a baseline estimator of a scalar $\theta$, let $\hat\theta_{\mathrm{truth},\ell}$ be a low bias estimator on the labeled subsample, and let $r_\ell=\hat\theta_{\mathrm{truth},\ell}-\hat\theta_{\mathrm{base},\ell}$ be the rectifier, where $\hat\theta_{\mathrm{base},\ell}$ is the baseline estimator computed on the labeled subsample. The generalized control variate (GCV) estimator is $\hat\theta_{GCV}(\lambda)=\hat\theta_{\mathrm{base}}+\lambda r_\ell$ for $\lambda\in\mathbb R$ (for $\theta\in\mathbb R^p$, $\lambda$ is replaced by a matrix $\Lambda\in\mathbb R^{p\times p}$). Suppose $\hat\theta_{\mathrm{base}}$ and $\hat\theta_{\mathrm{truth},\ell}$ have finite second moments and $\mathrm{Var}[r_\ell]>0$ (respectively $\mathrm{Var}[r_\ell]\succ0$ in the matrix case). Let $V(\lambda)=\mathrm{Var}[\hat\theta_{GCV}(\lambda)-\theta]$, the variance taken in the inner product space natural to the method. Then $V$ is quadratic in $\lambda$ and is minimized at $\lambda^*=\mathrm{Cov}[\theta-\hat\theta_{\mathrm{base}},r_\ell]/\mathrm{Var}[r_\ell]=-\mathrm{Cov}[\hat\theta_{\mathrm{base}}-\theta,r_\ell]/\mathrm{Var}[r_\ell]$, with minimized value $V(\lambda^*)=\mathrm{Var}[\hat\theta_{\mathrm{base}}-\theta](1-\rho_b^2)=\mathrm{Var}[\hat\theta_{\mathrm{truth},\ell}-\theta](1-\rho_t^2)$, where $\rho_b=\mathrm{Corr}[\hat\theta_{\mathrm{base}}-\theta,r_\ell]$ and, when $\hat\theta_{\mathrm{base}}=\hat\theta_{\mathrm{base},\ell}$, $\rho_t=\mathrm{Corr}[\hat\theta_{\mathrm{truth},\ell}-\theta,r_\ell]$. The matrix form $\Lambda^*=\mathrm{Cov}[\theta-\hat\theta_{\mathrm{base}},r_\ell]\,\mathrm{Var}[r_\ell]^{-1}$ minimizes $V$ in the Loewner order.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `thm:gcv`.

<a id="bl_and_cross_ppi_gain_on_residual"></a>

### Bayes linear and cross prediction powered inference as gain on residual updates

Theorem 12.10, page 397 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson, Sokolov and Zantedeschi (2026f), Linear Bayes Methods for Prediction-Powered Inference, SSRN 6744239.

Status: proved.

Statement. Both the Bayes linear (BL) estimator and the cross prediction powered inference (Cross PPI) estimator take the form $\hat\beta=\mu_{\mathrm{eff}}+G\,X_\ell^\top(Y_\ell-\text{baseline fit})$, where $X_\ell,Y_\ell$ are the labeled design and responses ($n$ labeled units). For BL, $\mu_{\mathrm{eff}}=\mu_0$ (the prior mean), the baseline fit is $X_\ell\mu_0$, and the gain is $G_{BL}=\mathrm{Var}_D[\beta]/\sigma^2=(V_0^{-1}+X_\ell^\top X_\ell/\sigma^2)^{-1}/\sigma^2$. For Cross PPI, $\mu_{\mathrm{eff}}=\hat\beta_{\mathrm{surr},u}$ is the least squares fit of the surrogate predictions on the $N$ unlabeled units with design $X_u$, the baseline fit is the cross fitted prediction of $Y_\ell$, and the gain is $G_{CPPI}=(X_u^\top X_u/N)^{-1}/n$. The gains differ in that $G_{BL}$ uses the posterior covariance, which includes the labeled data precision $X_\ell^\top X_\ell/\sigma^2$, whereas $G_{CPPI}$ uses the unlabeled pool covariance alone, carries no shrinkage denominator and does not adapt to $n$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `thm:duality`.

<a id="uniform_a_n_rule_is_cid_martingale_posterior_cdf"></a>

### Uniform A_(n) predictive rule is conditionally identically distributed with martingale posterior CDF

Theorem 12.17, page 401 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson, Sokolov and Zantedeschi (2026f), Linear Bayes Methods for Prediction-Powered Inference, SSRN 6744239.

Status: proved.

Statement. The uniform $A_{(n)}$ predictive rule (the next observation falls in each of the $n+1$ intervals cut by the ordered sample with probability $1/(n+1)$, the mass of each interval sitting at its upper end point and that of the last interval at $+\infty$) is conditionally identically distributed in the sense of Berti, Pratelli and Rigo (2004), and the martingale posterior for the distribution function $F$ satisfies $\mathrm{E}[F(x)\mid X_{(1:n)}]=\#\{i:X_i\le x\}/(n+1)$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `thm:cid`.

<a id="berliner_hill_limit_of_bayes_linear"></a>

### Berliner and Hill survival estimator as a limit of Bayes linear adjusted expectations

Theorem 12.18, page 402 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson, Sokolov and Zantedeschi (2026f), Linear Bayes Methods for Prediction-Powered Inference, SSRN 6744239.

Status: proved.

Statement. The Berliner--Hill survival estimator is $\hat S_{BH}(t)=\prod_{t_{(j)}\le t}\bigl(1-d_j/(n_j+1)\bigr)$, with $d_j$ events among $n_j$ at risk at event time $t_{(j)}$. Each factor $1-d_j/(n_j+1)$ is the limit, as the prior mean $p_0\uparrow1$, of the Bayes linear adjusted expectation of the conditional survival probability $p_j$ under prior mean $p_0$ with prior equivalent sample size $m=1$ per risk set. For uncensored data the estimator is the martingale posterior mean of $S(t)$ under the uniform $A_{(n)}$ prior.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `thm:bhbl`.

<a id="qnn_posterior_wasserstein_consistency"></a>

### Wasserstein consistency of the quantile neural network posterior

Theorem 12.30, page 413 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson, Sokolov and Zantedeschi (2026), Generative Bayesian Computation: A Measure-Theoretic Framework for Utility-Based Causal Inference, working paper.

Status: proved.

Statement. Let $\theta$ be a scalar parameter, $D_N$ the data, $h(D_N)$ a summary statistic, and let $f_{\varphi_S}:[0,1]\times H\to\mathbb R$ be a quantile neural network trained on $S=S_N$ draws $(\theta^{(s)},D^{(s)})$ from the joint prior predictive by minimizing the pinball loss $\frac1S\sum_{s=1}^S\mathrm{E}_\tau[\rho_\tau(\theta^{(s)}-f_\varphi(\tau,h(D^{(s)})))]$, where $\rho_\tau(u)=u(\tau-\mathbf 1_{u<0})$ and $\tau\sim\mathrm{Unif}[0,1]$. Let $W_2$ be the quadratic Wasserstein distance and $Q(\cdot\mid h)$ the quantile function of $p(\theta\mid h)$. Assume (Q1) $f_{\varphi_S}$ minimizes the pinball loss over a ReLU network class whose size grows with $S$; (Q2) the prior, hence every posterior, is supported in a fixed bounded interval; (Q3) $W_2(p(\theta\mid h(D_N)),p(\theta\mid D_N))\to0$ in probability; (Q4) $S_N$ grows fast enough that $\int_0^1|f_{\varphi_S}(\tau,h(D_N))-Q(\tau\mid h(D_N))|^2\,d\tau\to0$ in probability. Write $f_{\varphi_S,\#}\mathrm{Unif}$ for the law of $f_{\varphi_S}(U,h(D_N))$ with $U\sim\mathrm{Unif}[0,1]$. Then $W_2(f_{\varphi_S,\#}\mathrm{Unif},p(\theta\mid D_N))\to0$ in probability as $S,N\to\infty$, and for finite $S$, with $h=h(D_N)$, $W_2(f_{\varphi_S,\#}\mathrm{Unif},p(\theta\mid D_N))\le(\int_0^1|f_{\varphi_S}(\tau,h)-Q(\tau\mid h)|^2\,d\tau)^{1/2}+W_2(p(\theta\mid h),p(\theta\mid D_N))$. The limit uses only this bound, (Q3) and (Q4).

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `gbc:thm:qnn`.

<a id="gbc_ate_parametric_efficiency"></a>

### Efficiency of the GBC average treatment effect estimator in a parametric structural causal model

Theorem 12.35, page 415 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson, Sokolov and Zantedeschi (2026), Generative Bayesian Computation: A Measure-Theoretic Framework for Utility-Based Causal Inference, working paper.

Status: proved.

Statement. Let $W\in\{0,1\}$ be a treatment, $Y$ an outcome and $X$ covariates, observed in an i.i.d. sample $D_N$ of size $N$, and assume unconfoundedness $W\perp\!\!\!\perp(Y(0),Y(1))\mid X$, overlap $0<e(x)<1$ for the propensity score $e(x)=P(W=1\mid X=x)$, and the stable unit treatment value assumption. Let a parametric structural causal model with parameter $\theta\in\Theta\subset\mathbb R^d$ be correctly specified at an interior point $\theta_0$ and differentiable in quadratic mean there, with score $\dot\ell_{\theta_0}$ and nonsingular Fisher information $\mathcal I(\theta_0)$; let the average treatment effect $\tau(\theta)$ be Lipschitz and continuously differentiable, with $\tau_0=\tau(\theta_0)$; and let the prior meet the conditions of the parametric Bernstein and von Mises theorem for the posterior mean. Let $\hat\tau_{\mathrm{GBC}}=R^{-1}\sum_{r=1}^R\tau^{(r)}$, where the $\tau^{(r)}=\tau(\theta^{(r)})$ come from draws $\theta^{(r)}$ of the GBC posterior whose law is within $o_P(N^{-1/2})$ of $p(\theta\mid D_N)$ in the quadratic Wasserstein distance. Then $\sqrt N(\hat\tau_{\mathrm{GBC}}-\tau_0)$ converges in law to $N(0,V_{\mathrm{par}})$ as $N,S,R\to\infty$, where $S$ is the number of training simulations and $V_{\mathrm{par}}=\nabla\tau(\theta_0)^\top\mathcal I(\theta_0)^{-1}\nabla\tau(\theta_0)$. Moreover $V_{\mathrm{par}}\le V_{\mathrm{eff}}=\mathrm{E}[\psi(Y,W,X;\tau_0)^2]$, the efficiency bound of the nonparametric model, where $\psi(Y,W,X;\tau)=\frac{W-e(X)}{e(X)(1-e(X))}(Y-\mu_W(X))+\mu_1(X)-\mu_0(X)-\tau$ and $\mu_w(x)=\mathrm{E}[Y\mid W=w,X=x]$, with equality if and only if $\psi(Y,W,X;\tau_0)$ lies in the linear span of the components of $\dot\ell_{\theta_0}$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `gbc:thm:efficiency`.

<a id="gbc_meu_policy_posterior_contraction"></a>

### Posterior contraction of the GBC maximum expected utility policy

Theorem 12.41, page 421 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson, Sokolov and Zantedeschi (2026), Generative Bayesian Computation: A Measure-Theoretic Framework for Utility-Based Causal Inference, working paper.

Status: proved.

Statement. Let $\mathcal A$ be a class of policies $\pi$, $U(\pi,\theta,Y)$ a utility, $u(\pi,\theta)=\int U(\pi,\theta,y)\,p(y\mid\theta,\mathrm{do}(\pi))\,dy$, $\pi_N$ the posterior given the sample $D_N$, $EU(\pi\mid D_N)=\int u(\pi,\theta)\,\pi_N(d\theta)$, and $\|\pi\|^2=\mathrm{E}_X[\pi(X)^2]$, the squared $L_2$ norm under the covariate law. Let $\pi^*_0=\arg\max_{\pi\in\mathcal A}\mathrm{E}_{P_0}[U(\pi,\theta_0,Y)]$ be the oracle policy and $\pi^*_N=\arg\max_{\pi\in\mathcal A}EU(\pi\mid D_N)$. Assume (B1) the model is correctly specified at $\theta_0$, so that $\mathrm{E}_{P_0}[U(\pi,\theta_0,Y)]=u(\pi,\theta_0)$, and both maximizers exist; (B2) $|u(\pi,\theta)-u(\pi,\theta')|\le L_U\,d(\theta,\theta')$ for all $\pi\in\mathcal A$ and a metric $d$ on $\Theta$; (B3) $\mathrm{E}_{P_0}[\int d(\theta,\theta_0)\,\pi_N(d\theta)]\le C\varepsilon_N$ for a constant $C$; (B4) $\mathrm{E}_{P_0}[U(\pi^*_0,\theta_0,Y)-U(\pi,\theta_0,Y)]\ge\kappa\|\pi-\pi^*_0\|^2$ for all $\pi\in\mathcal A$, with $\kappa>0$. Then $\mathrm{E}_{P_0}[\|\pi^*_N-\pi^*_0\|^2]=O(L_U\varepsilon_N/\kappa)$. In parametric models where (B3) holds with $\varepsilon_N=N^{-1/2}$, the policy converges at rate $(L_U/\kappa)^{1/2}N^{-1/4}$ in root mean square.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `gbc:thm:contraction`.

<a id="gbc_meu_bayes_regret"></a>

### Bayes regret of the GBC maximum expected utility policy

Corollary 12.42, page 422 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson, Sokolov and Zantedeschi (2026), Generative Bayesian Computation: A Measure-Theoretic Framework for Utility-Based Causal Inference, working paper.

Status: proved.

Statement. Let $\mathcal A$ be a class of policies $\pi$, $U(\pi,\theta,Y)$ a utility, $u(\pi,\theta)=\int U(\pi,\theta,y)\,p(y\mid\theta,\mathrm{do}(\pi))\,dy$, $\pi_N$ the posterior given the sample $D_N$, $EU(\pi\mid D_N)=\int u(\pi,\theta)\,\pi_N(d\theta)$, $V(\pi)=\mathrm{E}_{P_0}[U(\pi,\theta_0,Y)]$, $\pi^*_0=\arg\max_{\pi\in\mathcal A}V(\pi)$ and $\pi^*_N=\arg\max_{\pi\in\mathcal A}EU(\pi\mid D_N)$. Assume (B1) the model is correctly specified at $\theta_0$, so that $V(\pi)=u(\pi,\theta_0)$, and both maximizers exist; (B2) $|u(\pi,\theta)-u(\pi,\theta')|\le L_U\,d(\theta,\theta')$ for all $\pi\in\mathcal A$ and a metric $d$ on $\Theta$; (B3) $\mathrm{E}_{P_0}[\int d(\theta,\theta_0)\,\pi_N(d\theta)]\le C\varepsilon_N$ with a constant $C$ free of $\theta_0$. Then the Bayes regret satisfies $R_N=\mathrm{E}_{D_N,\theta_0}[V(\pi^*_0)-V(\pi^*_N)]=O(L_U\varepsilon_N)$, and $R_N=O(N^{-1/2})$ in parametric models in which (B3) holds with $\varepsilon_N=N^{-1/2}$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `gbc:cor:regret`.

<a id="method_of_types_e_value"></a>

### Method of types e value with Kullback Leibler growth rate

Theorem 12.53, page 430 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson, Sokolov and Zantedeschi (2026), E-Values and Sanov's Theorem, working paper.

Status: proved.

Statement. Let $X_1,X_2,\dots$ be i.i.d. with law $P$ on a finite alphabet $\mathcal X$, let $\widehat P_n$ be the empirical measure (type) of $X_1,\dots,X_n$, let $P_0$ be a law on $\mathcal X$, and let $D(Q\,\|\,R)=\sum_xQ(x)\log(Q(x)/R(x))$ be the Kullback Leibler divergence. Define $E_n=(n+1)^{-|\mathcal X|}\exp(nD(\widehat P_n\,\|\,P_0))$. Then $\mathrm{E}_{P_0}[E_n]\le1$, so $E_n$ is an e value for the simple null $H_0:P=P_0$ (a nonnegative statistic with expectation at most one under the null). Moreover, under any $P$, $\frac1n\log E_n\to D(P\,\|\,P_0)$ almost surely, the limit taken in $[0,\infty]$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `ev:thm:motev`.

## Chapter 13. Applications, Worked Examples, and Federated Learning

<a id="federated_product_of_experts"></a>

### Federated product of experts from exchangeability

Theorem 13.2, page 473 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026f), Federated Learning, de Finetti, and Sanov, working paper.

Status: proved.

Statement. Let $K$ clients hold data $D_k=(X^{(k)}_1,\dots,X^{(k)}_{n_k})$ with $N=\sum_k n_k$. Suppose $\theta\mapsto p(\cdot\mid\theta)$ is injective, the full array $\{X^{(k)}_i\}_{i,k}$ is jointly exchangeable (invariant in law under all permutations of its $N$ entries) and extends to an infinite exchangeable sequence whose directing measure lies almost surely in $\{p(\cdot\mid\theta):\theta\in\Theta\}$. Then there exists a prior law $\Pi$ on $\Theta$, unique as the mixing law of the extended sequence, such that given $\theta\sim\Pi$ the extended sequence is i.i.d. from $p(\cdot\mid\theta)$ and, when $\Pi$ has a density $\pi$, $\pi(\theta\mid D_1,\dots,D_K)\propto\pi(\theta)\prod_{k=1}^K p(D_k\mid\theta)$. With size weighted local pseudo posteriors $\pi_k(\theta)\propto\pi(\theta)^{n_k/N}p(D_k\mid\theta)$, the global posterior is the product of local experts, $\pi(\theta\mid D_1,\dots,D_K)\propto\prod_{k=1}^K\pi_k(\theta)$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `thm:fed-poe`.

<a id="federated_sanov_concentration"></a>

### Federated Sanov posterior concentration at a Kullback Leibler rate

Theorem 13.8, page 475 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026f), Federated Learning, de Finetti, and Sanov, working paper.

Status: proved.

Statement. Let $\theta_0$ lie in the Kullback Leibler support of the prior $\Pi$, fix $\varepsilon>0$, and let $\bar\ell_N(\theta)$ be the average over all $N$ observations of $\log\{p(X\mid\theta_0)/p(X\mid\theta)\}$. Let $c_\varepsilon=\inf_{\|\theta-\theta_0\|>\varepsilon}\mathrm{KL}\bigl(p(\cdot\mid\theta_0)\,\|\,p(\cdot\mid\theta)\bigr)$ and suppose $c_\varepsilon>0$ and that, $P_{\theta_0}$-almost surely, for every $\delta>0$, $\inf_{\|\theta-\theta_0\|>\varepsilon}\bar\ell_N(\theta)\ge c_\varepsilon-\delta$ for all large $N$ (a one sided uniform law of large numbers). Then, $P_{\theta_0}$-almost surely, $\limsup_{N\to\infty}\frac1N\log\Pi(\|\theta-\theta_0\|>\varepsilon\mid D_1,\dots,D_K)\le-c_\varepsilon$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `thm:fed-conc`.

<a id="federated_bernstein_von_mises"></a>

### Federated Bernstein and von Mises theorem

Corollary 13.9, page 476 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026f), Federated Learning, de Finetti, and Sanov, working paper.

Status: proved.

Statement. Assume the conditions of the federated Sanov concentration theorem for every $\varepsilon>0$, and suppose also that the model is differentiable in quadratic mean at $\theta_0$ (local asymptotic normality), that the Fisher information $\mathcal I(\theta_0)$ is nonsingular, that the federated maximum likelihood estimator $\hat\theta_N$ is asymptotically efficient, and that the prior $\Pi$ has a density $\pi$ positive and continuous at $\theta_0$. Then the posterior law of $\sqrt N(\theta-\hat\theta_N)$ converges to $N(0,\mathcal I(\theta_0)^{-1})$ in total variation, in $P_{\theta_0}$-probability; in particular $\sqrt N(\theta-\hat\theta_N)\mid D_1,\dots,D_K\to N(0,\mathcal I(\theta_0)^{-1})$ in law.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `cor:fed-bvm`.

<a id="federated_horseshoe_sufficient_statistics"></a>

### Sufficient statistics for the federated horseshoe

Theorem 13.16, page 480 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026f), Federated Learning, de Finetti, and Sanov, working paper.

Status: proved.

Statement. In the linear model client $k$ observes $Y_k\mid\theta\sim N(X_k\theta,\sigma^2I_{n_k})$ with known $\sigma^2$ and a design $X_k\in\mathbb R^{n_k\times d}$ of full column rank, with prior $\theta\sim N(0,\tau^2\Lambda)$, $\Lambda=\mathrm{diag}(\lambda_1^2,\dots,\lambda_d^2)$ (the horseshoe prior conditionally on the local and global scales). Let $\mathrm{RSS}_k$ be the residual sum of squares of the least squares fit of $Y_k$ on $X_k$ and $T_k=(X_k^\top Y_k,\mathrm{RSS}_k,X_k^\top X_k)$. The marginal likelihood $p(Y_k\mid\Lambda,\tau)=\int p(Y_k\mid\theta)\pi(\theta\mid\Lambda,\tau)\,d\theta$ depends on $Y_k$ only through $T_k$. If $X_k^\top X_k$ is known to the server, client $k$ need only transmit $(X_k^\top Y_k,\mathrm{RSS}_k)\in\mathbb R^d\times\mathbb R$, which is $O(d)$ scalars regardless of $n_k$; if $X_k^\top X_k$ must be sent as well, the cost is $O(d^2)$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `thm:hs-suff`.

<a id="federated_horseshoe_gaussian_bias_bounds"></a>

### Gaussian approximation bias of the federated horseshoe

Theorem 13.25, page 484 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026f), Federated Learning, de Finetti, and Sanov, working paper.

Status: proved.

Statement. In the normal means model where client $k$ observes $y^{(k)}_j\sim N(\theta_j,\sigma^2)$ for coordinate $j$, under the horseshoe prior with local scale $\lambda_j\sim C^+(0,1)$ and shrinkage weights $\kappa^{(k)}_j=\sigma^2/(\sigma^2+\lambda_j^2\tau^2)$, define for client $k$ and coordinate $j$ the variance of shrinkage term $\Delta^{(k)}_j=(y^{(k)}_j)^2\,\mathrm{Var}[\kappa^{(k)}_j\mid y^{(k)}_j,\tau]$. Then (i) for a signal, $\Delta^{(k)}_j\to0$ as $|y^{(k)}_j|\to\infty$; (ii) for noise, $\theta_{0j}=0$, $\Delta^{(k)}_j\le\frac14(y^{(k)}_j)^2$, so $\mathrm{E}[\Delta^{(k)}_j]\le\sigma^2/4$; (iii) universally, $\Delta^{(k)}_j\le\frac14(y^{(k)}_j)^2$ and $\sum_j\Delta^{(k)}_j\le\frac14\|Y_k\|^2$.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `thm:hs-bias`.

<a id="federated_horseshoe_bias_vanishes_many_clients"></a>

### Federated horseshoe variance bias vanishes with many clients

Theorem 13.26, page 484 of Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1. First stated in Polson and Zantedeschi (2026f), Federated Learning, de Finetti, and Sanov, working paper.

Status: proved.

Statement. Take homogeneous clients ($n_k\equiv n$, each with noise variance $\sigma^2$) and a fixed global scale $\tau$. For coordinate $j$ let $\hat\pi_j$ be the product of the $K$ Gaussian approximations $\hat\pi^j_k$, with variances $V^j_k=\sigma^2(1-\bar\kappa^{(k)}_j)$ and $\bar\kappa^{(k)}_j$ the Gaussian approximation's shrinkage weight, and let $\pi^*_j$ be the product of the $K$ Gaussians with the means and variances of the local horseshoe posteriors. Let $w_k=(V^j_k)^{-1}/\sum_i(V^j_i)^{-1}$ and let $\bar\kappa_0<1$ be the value of $\bar\kappa^{(k)}_j$ at $y^{(k)}_j=0$, which depends only on $\tau/\sigma$. The variance of $\pi^*_j$ exceeds that of $\hat\pi_j$ by the aggregate bias $\Delta^{\mathrm{agg}}_j$, and $\Delta^{\mathrm{agg}}_j\le\sum_{k=1}^K w_k^2\Delta^{(k)}_j\le\sum_{k=1}^K(y^{(k)}_j)^2/\{4K^2(1-\bar\kappa_0)^2\}=O_p(1/K)$. As $K\to\infty$ with $\sigma^2$ and $\tau$ fixed, $\Delta^{\mathrm{agg}}_j\to0$ in probability, while the variance of $\hat\pi_j$, between $\sigma^2(1-\bar\kappa_0)/K$ and $\sigma^2/K$, shrinks at the same rate. The means $\hat\mu_j$ and $\mu^*_j$ coincide when all $V^j_k$ are equal but not in general, and the transport distance between $\hat\pi=\prod_j\hat\pi_j$ and $\pi^*=\prod_j\pi^*_j$ obeys $W_2^2(\hat\pi,\pi^*)-\sum_j(\hat\mu_j-\mu^*_j)^2\le\sum_j\Delta^{\mathrm{agg}}_j=O_p(d/K)$. The location error $\hat\mu_j-\mu^*_j$ is $O_p(K^{-1/2})$ for a noise coordinate, $\theta_{0j}=0$, but need not vanish for a signal.

Source: [`Elements_of_Bayes_INTERIOR.tex`](https://github.com/VadimSokolov/elements_bayes_information/blob/v1.0.0/Elements_of_Bayes_INTERIOR.tex), label `thm:hs-bias-K`.

***

<small>License: Text, figures and PDF: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code: [MIT](https://opensource.org/license/mit).</small>

<small>This page is generated from the book's repository ([main](https://github.com/VadimSokolov/elements_bayes_information/tree/main)). The numbers and pages are those of the PDF, [version 1.0.0](https://github.com/VadimSokolov/elements_bayes_information/releases/tag/v1.0.0).</small>

<!-- Generated by bookrepo.py from book.yaml and results.yaml. Edit those files, not this one. -->
