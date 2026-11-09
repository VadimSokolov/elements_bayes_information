# Chapter 8. Scalable Computation and Deep Learning

<!-- cite:begin -->
From *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations* by Nick Polson and Vadim Sokolov (2026), chapter 8. Cite as Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1, https://doi.org/10.5281/zenodo.24067804. Book page: https://vsokolov.org/books/elements_bayes_information/. Source: https://github.com/VadimSokolov/elements_bayes_information.
<!-- cite:end -->

How can the predictive theory be turned into algorithms? The chapter derives computation from the information view of prediction. It starts from an axiomatic characterization of the logarithmic score and a rate distortion frontier for entropy regularized prediction, then treats data augmentation, scale mixtures and latent variable Bayes (Polya Gamma Gibbs sampling for logistic and multinomial models), deep learning as hierarchical nonlinear regression with a coding and evidence view of regularization, and generative Bayesian prediction in which the posterior is a trained sampler built from quantile maps and noise outsourcing. It also discusses double descent, ordered empirical Bayes and the selection of complexity by evidence, and proximal algorithms with optimization as inference (penalized estimation and Langevin sampling as two readings of one iteration). It proves only a few statements of its own and mostly presents known methods in the book's predictive and maximum entropy language.

## From the literature

Characterization of the logarithmic score (Bernardo 1979); rate distortion theory (Cover and Thomas 2006); the Polya Gamma augmentation for logistic regression (Polson, Scott and Windle 2013) and its uniform ergodicity (Choi and Hobert 2013); noise outsourcing (Kallenberg); minimum description length and evidence (Rissanen 1978; Watanabe 2009); proximal algorithms in statistics (Polson, Scott and Willard 2015); the horseshoe prior (Carvalho, Polson and Scott 2010).

## New results

No new results of this chapter are catalogued; the log score characterization and the rate distortion frontier are presented after Bernardo (1979) and Cover and Thomas (2006).

<!-- results:begin -->
No new results of this chapter are catalogued.
<!-- results:end -->

## Key terms

scalable Bayesian computation, data augmentation, Polya Gamma, scale mixtures, latent variable models, Gibbs sampling, deep learning, neural networks as Bayesian models, entropy regularized inference, rational inattention, rate distortion, log score, generative Bayesian computation, quantile regression, amortized inference, noise outsourcing, double descent, interpolation threshold, ordered empirical Bayes, isotonic regression, proximal algorithms, optimization as inference
