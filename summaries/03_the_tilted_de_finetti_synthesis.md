# Chapter 3. The Tilted de Finetti Synthesis

<!-- cite:begin -->
From *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations* by Nick Polson and Vadim Sokolov (2026), chapter 3. Cite as Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1, https://doi.org/10.5281/zenodo.24067804. Book page: https://vsokolov.org/books/elements_bayes_information/. Source: https://github.com/VadimSokolov/elements_bayes_information.
<!-- cite:end -->

This chapter proves the book's central result, the tilted de Finetti theorem: conditioning an exchangeable sequence on an empirical moment constraint yields predictive laws that are exponential family I projections of the directing measure. For a finite alphabet, a polyhedral moment window, an interior I projection and strong regularity, a split set master inequality bounds the total variation distance between a constrained block of m coordinates and the product of the I projection, with tuned rate O(m eps_N + m sqrt(log N / N) + m^2/N). Mixing over the constraint posterior gives the projected de Finetti limit, its predictive corollary, and a classification of reachable, null and unreachable constraints, with the null constraint trap. The posterior is a Gibbs measure (inverse Sanov), with the Ganesh and O'Connell large deviation principle and the curvature identity linking Fisher information to the rate function. It closes with Diaconis's finite de Finetti theorem, conditional limits and conjugate priors.

## From the literature

- The inverse Sanov theorem for posteriors (Ganesh and O'Connell 1999, 2000), with the exponential family case of Macci and Piccioni (2023).
- Gibbs conditioning (van Campenhout and Cover 1981; Csiszar 1984).
- The finite de Finetti theorem (Diaconis and Freedman 1980a) and the conditional limit theorem for exponential families (Diaconis and Freedman 1988).
- The characterization of conjugate priors for exponential families (Diaconis and Ylvisaker 1979).
- Links between maximum entropy and Bayes (Jaynes 1957; Dinwoodie and Zabell 1992; Fortini, Ladelli and Regazzini 2000) and PAC Bayes bounds (Catoni 2007).

## New results

These results come from the authors' paper De Finetti + Sanov = Bayes and are stated for a finite alphabet.

<!-- results:begin -->
- [Componentwise constrained horizon master inequality](../RESULTS.md#constrained_horizon_master_inequality) (Theorem 3.2, page 30; proved): A fixed block of coordinates drawn from a sample conditioned on a moment window is close in total variation to independent draws from the I projection, with an explicit finite sample bound with three separately controlled terms. First stated in Polson and Zantedeschi (2025a), De Finetti + Sanov = Bayes: Forward and Inverse Sanov in Bayesian Prediction, Statistical Science (revised short communication).
- [Tuned rate for the constrained block law](../RESULTS.md#master_inequality_tuning) (Corollary 3.3, page 30; proved): Choosing the split radius of order the square root of log N over N gives a block law rate of order m eps_N plus m sqrt(log N / N) plus m squared over N. First stated in Polson and Zantedeschi (2025a), De Finetti + Sanov = Bayes: Forward and Inverse Sanov in Bayesian Prediction, Statistical Science (revised short communication).
- [Projected de Finetti limit](../RESULTS.md#projected_de_finetti_limit) (Theorem 3.4, page 32; proved): After mixing over the constraint posterior, the block laws of an exchangeable sequence conditioned on a moment event converge to an exchangeable law whose random directing measure is the I projection of the original directing measure. First stated in Polson and Zantedeschi (2025a), De Finetti + Sanov = Bayes: Forward and Inverse Sanov in Bayesian Prediction, Statistical Science (revised short communication).
- [Bayesian prediction under the constrained limit](../RESULTS.md#bayesian_prediction_under_constrained_limit) (Corollary 3.5, page 33; proved): Under the projected de Finetti limit, the predictive probability of the next observation is the usual de Finetti posterior predictive under the prior induced by the I projection map. First stated in Polson and Zantedeschi (2025a), De Finetti + Sanov = Bayes: Forward and Inverse Sanov in Bayesian Prediction, Statistical Science (revised short communication).
- [Reachability and degeneracy of moment constraints](../RESULTS.md#reachability_and_degeneracy) (Proposition 3.7, page 34; conditional on proofs of its four clauses, which the book states without proof, in particular the concentration rate max(eps_N, N^{-1/2}) and the disintegration limit in clause (ii) and the transverse dimension k-1-r_0 in clause (iv)): Whether a moment constraint is reachable by the prior decides whether the I projection acts, with reachable constraints truncating or disintegrating the prior, unreachable ones leaving genuine tilts, and affine constraints able to collapse the randomness; the book states this without proof. First stated in Polson and Zantedeschi (2025a), De Finetti + Sanov = Bayes: Forward and Inverse Sanov in Bayesian Prediction, Statistical Science (revised short communication).
<!-- results:end -->

## Key terms

tilted de Finetti theorem, moment constraint, I projection, master inequality, constraint posterior, directing measure, reachability, null constraint, Borel Kolmogorov paradox, inverse Sanov, Gibbs measure posterior, Fisher information, curvature, Bernstein and von Mises, finite de Finetti, conjugate prior, maximum entropy
