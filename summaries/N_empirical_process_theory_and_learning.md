# Appendix N. Empirical Process Theory and Learning

<!-- cite:begin -->
From *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations* by Nick Polson and Vadim Sokolov (2026), appendix N. Cite as Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1, https://doi.org/10.5281/zenodo.24067804. Book page: https://vsokolov.org/books/elements_bayes_information/. Source: https://github.com/VadimSokolov/elements_bayes_information.
<!-- cite:end -->

Appendix N is a short background toolkit on empirical process theory and statistical learning: the uniform control, over a class of functions, of the deviation between the empirical measure and the true law, which governs generalization and the cross fitting guarantees of double machine learning. It defines Glivenko Cantelli classes (a uniform law of large numbers) and Donsker classes (a uniform central limit theorem with a Gaussian process limit), with the empirical distribution function as the canonical example and the Kolmogorov Smirnov statistic as the supremum of the limiting Brownian bridge. It then lists the complexity measures that decide when these uniform laws hold: the Vapnik Chervonenkis dimension, which bounds the growth function of a binary class and gives a uniform deviation of order the square root of VC log n over n; the empirical and population Rademacher complexity, with the symmetrization bound of the expected supremum by twice the Rademacher complexity; and covering and bracketing numbers, which give metric entropy integrals through Dudley's bound. Finally it explains why cross fitting, by evaluating each nuisance estimate on a held out fold, removes the empirical process term, so that double machine learning attains root n inference for the target parameter with nuisance rates of only o of n to the minus one quarter, bypassing rather than assuming a Donsker condition, which matters when nuisances are fit by flexible machine learners. It collects standard results and proves no new theorem.

## From the literature

Glivenko Cantelli theorem, Donsker's theorem, Vapnik and Chervonenkis uniform convergence (Vapnik and Chervonenkis 1971), Rademacher complexity and symmetrization, covering numbers and Dudley's entropy bound (Dudley 1967).

## New results

<!-- results:begin -->
No new results of this appendix are catalogued.
<!-- results:end -->

## Key terms

empirical process, Glivenko Cantelli, Donsker class, uniform law of large numbers, uniform central limit theorem, Brownian bridge, Kolmogorov Smirnov, VC dimension, growth function, Rademacher complexity, symmetrization, covering number, bracketing number, Dudley entropy integral, statistical learning theory, generalization bound, double machine learning, cross fitting, background
