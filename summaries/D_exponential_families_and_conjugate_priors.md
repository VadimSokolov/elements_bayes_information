# Appendix D. Exponential Families and Conjugate Priors

<!-- cite:begin -->
From *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations* by Nick Polson and Vadim Sokolov (2026), appendix D. Cite as Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1, https://doi.org/10.5281/zenodo.24067804. Book page: https://vsokolov.org/books/elements_bayes_information/. Source: https://github.com/VadimSokolov/elements_bayes_information.
<!-- cite:end -->

This appendix is background. It reviews exponential families and conjugate priors. An exponential family has natural parameter, sufficient statistic and log partition function; by the Pitman Koopman Darmois theorem, under regularity, these are the only families with a fixed dimensional sufficient statistic for all sample sizes. For a minimal regular family the map from natural to mean parameters is a bijection with inverse given by the gradient of the conjugate, the Fisher information is the Hessian of the log partition function, and maximum likelihood is moment matching, an instance of the reverse I projection of the empirical distribution onto the family. The conjugate prior updates by adding the sufficient statistic to one hyperparameter and the sample size to the other, so the posterior mean of the mean parameter is a convex combination of prior and data means, the shrinkage of the hierarchical chapters. Standard pairs are listed: Beta Bernoulli, Gamma Poisson, Normal Normal, Dirichlet Multinomial and Normal inverse Wishart Gaussian.

## From the literature

The Pitman Koopman Darmois theorem (Pitman 1936; Koopman 1936; Darmois 1935). Conjugate priors for exponential families and their linear posterior mean (Diaconis and Ylvisaker 1979). Maximum likelihood as moment matching.

## New results

<!-- results:begin -->
No new results of this appendix are catalogued.
<!-- results:end -->

## Key terms

exponential family, natural parameter, mean parameter, sufficient statistic, log partition function, Pitman Koopman Darmois theorem, minimal regular family, maximum likelihood, moment matching, conjugate prior, prior sample size, posterior update, shrinkage, Beta Bernoulli, Gamma Poisson, Dirichlet Multinomial
