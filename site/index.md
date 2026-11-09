# Elements of Bayes and Information Theory <small>A Predictive Theory of Inference: Information, Algorithms, and Large Deviations</small>

By Nick Polson and [Vadim Sokolov](https://orcid.org/0000-0002-6618-2965).

[![PDF](https://vsokolov.org/books/icon-pdf.svg)](https://github.com/VadimSokolov/elements_bayes_information/releases/download/v1.0.0/elements_bayes_information.pdf "PDF") [![Source on GitHub](https://vsokolov.org/books/icon-github.svg)](https://github.com/VadimSokolov/elements_bayes_information/tree/v1.0.0 "Source on GitHub") [![DOI 10.5281/zenodo.24067804](https://zenodo.org/badge/DOI/10.5281/zenodo.24067804.svg)](https://doi.org/10.5281/zenodo.24067804)

## Description

Bayesian inference is a predictive act. This book builds it from three ideas: de Finetti supplies the structure, writing exchangeable beliefs as mixtures of independent laws; Sanov supplies the asymptotics, the exponential concentration of empirical measures at a rate given by the Kullback Leibler divergence; and Bayes supplies the updating, an exponential tilt of the predictive law. Together they give a tilted de Finetti theorem: conditioning an exchangeable sequence on empirical moment constraints yields predictive laws that are I projections of a baseline measure, so parameters emerge as limits of empirical functionals and maximum entropy appears as a predictive limit rather than a postulate. Testing, shrinkage, asymptotics and decision are then read on one moderate deviation scale.

Results the book proves, most first stated in the authors' papers with Daniel Zantedeschi and Jyotishka Datta: a finite sample total variation bound between a block drawn under a moment window and independent draws from the I projection, and the projected de Finetti limit behind it; a split of the predictive Kullback Leibler Bayes risk into a Sanov term and a posterior contraction term,; a condition for a scalar learning rate to calibrate credible sets at moderate scale, namely that model curvature and score covariance be proportional; one control variate form for Bayes linear and prediction powered inference estimators, with the optimal gain in closed form; the global posterior of federated learning as a product of local experts under exchangeability; and a convergence rate for the policy that maximizes posterior expected utility in generative Bayesian computation. Several results are stated conditionally or as open problems, and the book says which.

Later chapters treat computation, universal portfolios, causal machine learning and medicine. Scripts reproducing every computed figure and number are in the repository.

## How to cite

Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1, https://doi.org/10.5281/zenodo.24067804.

```bibtex
@book{polsonsokolov2026elements_bayes_information,
  author = {Polson, Nick and Sokolov, Vadim},
  title = {{Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations}},
  year = {2026},
  publisher = {Independently published},
  doi = {10.5281/zenodo.24067804},
  url = {https://vsokolov.org/books/elements_bayes_information/}
}
```

## Main results

- [Componentwise constrained horizon master inequality](https://vsokolov.org/books/elements_bayes_information/results.html#constrained_horizon_master_inequality) (Theorem 3.2, page 30): A fixed block of coordinates drawn from a sample conditioned on a moment window is close in total variation to independent draws from the I projection, with an explicit finite sample bound with three separately controlled terms.
- [Projected de Finetti limit](https://vsokolov.org/books/elements_bayes_information/results.html#projected_de_finetti_limit) (Theorem 3.4, page 32): After mixing over the constraint posterior, the block laws of an exchangeable sequence conditioned on a moment event converge to an exchangeable law whose random directing measure is the I projection of the original directing measure.
- [Decomposition of nonparametric predictive Bayes risk into Sanov and contraction terms](https://vsokolov.org/books/elements_bayes_information/results.html#predictive_bayes_risk_decomposition) (Theorem 4.1, page 59): The predictive KL Bayes risk splits into a population Sanov term, a posterior contraction term and a small cross term, so misspecified models have risk equal to the Sanov term and well specified ones have risk equal to the contraction term.
- [Moderate scale calibration of tempered posteriors under misspecification](https://vsokolov.org/books/elements_bayes_information/results.html#moderate_scale_calibration_misspecification) (Proposition 7.31, page 272): A scalar learning rate calibrates credible balls at moderate scale only when the model curvature and score covariance are proportional, and otherwise a matrix temperature is needed.
- [Generalized control variate estimator, optimal shrinkage and variance](https://vsokolov.org/books/elements_bayes_information/results.html#generalized_control_variate_unification) (Theorem 12.6, page 394): The estimators of Bayes linear, cross prediction powered inference and inverse probability weighted prediction powered inference are all a baseline plus a shrunken labeled data rectifier, and the best shrinkage is a covariance over variance ratio.
- [Posterior contraction of the GBC maximum expected utility policy](https://vsokolov.org/books/elements_bayes_information/results.html#gbc_meu_policy_posterior_contraction) (Theorem 12.41, page 421): Under a Lipschitz utility, posterior contraction at a rate and a quadratic margin condition, the policy that maximizes posterior expected utility converges in mean square to the oracle policy at a rate proportional to the contraction rate, so at rate N to the minus one quarter in parametric models.
- [Federated product of experts from exchangeability](https://vsokolov.org/books/elements_bayes_information/results.html#federated_product_of_experts) (Theorem 13.2, page 473): If the pooled client data are exchangeable, the global posterior is the prior times the product of the client likelihoods, equivalently a product of size weighted local pseudo posteriors.

All 60 results of the book (41 proved, 12 conditional, 6 open, 1 counterexample) are on the page [Results by chapter](https://vsokolov.org/books/elements_bayes_information/results.html), each with its statement, status and number.

***

<small>License: Text, figures and PDF: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code: [MIT](https://opensource.org/license/mit).</small>

<small>This page is generated from the book's repository ([main](https://github.com/VadimSokolov/elements_bayes_information/tree/main)). The numbers and pages are those of the PDF, [version 1.0.0](https://github.com/VadimSokolov/elements_bayes_information/releases/tag/v1.0.0).</small>

<!-- Generated by bookrepo.py from book.yaml and results.yaml. Edit those files, not this one. -->
