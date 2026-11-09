# Appendix Q. Stein's Method and Stein Discrepancies

<!-- cite:begin -->
From *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations* by Nick Polson and Vadim Sokolov (2026), appendix Q. Cite as Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1, https://doi.org/10.5281/zenodo.24067804. Book page: https://vsokolov.org/books/elements_bayes_information/. Source: https://github.com/VadimSokolov/elements_bayes_information.
<!-- cite:end -->

Appendix Q is a short background toolkit on Stein's method and Stein discrepancies, which characterize a distribution by an operator that annihilates it. For a target density p the Langevin Stein operator applied to a smooth vector field f is the score of p times f plus the divergence of f; its expectation under p is zero for f vanishing at infinity, and its expectation under a law q vanishes for all test functions only when q equals p, so the supremum of that expectation measures how far q is from p using only the score, never the normalizing constant. Taking the test functions in the unit ball of an RKHS gives the kernel Stein discrepancy (KSD), a closed form discrepancy computable from samples of q and the score of p alone, which powers score based goodness of fit tests, sample quality diagnostics for MCMC and Stein variational gradient descent, a deterministic particle sampler; Tweedie's formula is the empirical Bayes shadow of the same score identity. Solving the Stein equation and bounding its solution converts a discrepancy into a bound on the difference of expectations of a test function, giving Berry Esseen rates for the central limit theorem and the Gaussian approximation error of the Laplace and Bernstein and von Mises expansions. The appendix presents standard results and proves no new theorem.

## From the literature

Stein's method, the Stein operator and Stein identity, kernel Stein discrepancy, Stein variational gradient descent, Berry Esseen rates by Stein's method, Tweedie's formula.

## New results

<!-- results:begin -->
No new results of this appendix are catalogued.
<!-- results:end -->

## Key terms

Stein's method, Stein operator, Langevin Stein operator, Stein identity, Stein equation, Stein discrepancy, kernel Stein discrepancy, KSD, Stein variational gradient descent, SVGD, score function, unnormalized density, goodness of fit, sampling quality, Berry Esseen, approximation bounds, background
