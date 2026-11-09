# Chapter 13. Applications, Worked Examples, and Federated Learning

<!-- cite:begin -->
From *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations* by Nick Polson and Vadim Sokolov (2026), chapter 13. Cite as Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1, https://doi.org/10.5281/zenodo.24067804. Book page: https://vsokolov.org/books/elements_bayes_information/. Source: https://github.com/VadimSokolov/elements_bayes_information.
<!-- cite:end -->

What does the book's apparatus deliver when applied end to end, and how do de Finetti and Sanov license federated Bayesian learning? The chapter has five parts. Genome wide screening tests a million hypotheses with approximate Bayes factors, whose rejection boundary grows like the square root of the log of the number of tests, with the horseshoe prior and a follow up decision problem. Bayesian decision making in medicine covers diagnostic tests as likelihood ratios, treatment thresholds, decision curves, trials under skeptical and enthusiastic priors and the value of information. Twenty one vignettes apply posterior reasoning to A/B tests, hierarchical models and small area maps. Worked examples treat the thermodynamic reading of constrained conditioning (microcanonical and canonical ensembles) with Bernoulli components, a null constraint set, the Brandeis dice and Gaussian scale mixtures. The federated section shows that exchangeability of pooled client data gives a global posterior equal to a product of local experts, adds client effects, a federated evidence lower bound, Sanov style concentration and a Bernstein and von Mises limit at the centralized rate, then treats the federated horseshoe: sufficient statistics and Gaussian approximation bias.

## From the literature

- De Finetti's theorem and the Aldous and Hoover representation for partially exchangeable arrays (Aldous 1981; Hoover 1979).
- Sanov's theorem and large deviation rates for posterior concentration (Sanov 1957).
- Federated averaging (McMahan et al. 2017).
- The horseshoe prior (Carvalho, Polson and Scott 2010) and Wakefield's approximate Bayes factor (Wakefield 2009).
- Diagnostic threshold and decision curve methods (Pauker and Kassirer 1980; Vickers and Elkin 2006), and Bayesian clinical trial design (Spiegelhalter, Freedman and Parmar 1994; Berry 2006).

## New results

The catalogued results are the federated ones, from the federated learning paper.

<!-- results:begin -->
- [Federated product of experts from exchangeability](../RESULTS.md#federated_product_of_experts) (Theorem 13.2, page 473; proved): If the pooled client data are exchangeable, the global posterior is the prior times the product of the client likelihoods, equivalently a product of size weighted local pseudo posteriors. First stated in Polson and Zantedeschi (2026f), Federated Learning, de Finetti, and Sanov, working paper.
- [Federated Sanov posterior concentration at a Kullback Leibler rate](../RESULTS.md#federated_sanov_concentration) (Theorem 13.8, page 475; proved): The federated posterior mass outside any fixed neighborhood of the truth decays exponentially at a rate given by the smallest Kullback Leibler divergence from the truth to the outside of the neighborhood. First stated in Polson and Zantedeschi (2026f), Federated Learning, de Finetti, and Sanov, working paper.
- [Federated Bernstein and von Mises theorem](../RESULTS.md#federated_bernstein_von_mises) (Corollary 13.9, page 476; proved): The federated posterior is asymptotically normal around the federated maximum likelihood estimate with the centralized rate and variance, so nothing is lost by not pooling raw data. First stated in Polson and Zantedeschi (2026f), Federated Learning, de Finetti, and Sanov, working paper.
- [Sufficient statistics for the federated horseshoe](../RESULTS.md#federated_horseshoe_sufficient_statistics) (Theorem 13.16, page 480; proved): In the linear model each client needs to send only a vector of size d and a residual sum of squares for the server to evaluate its marginal likelihood under the horseshoe prior. First stated in Polson and Zantedeschi (2026f), Federated Learning, de Finetti, and Sanov, working paper.
- [Gaussian approximation bias of the federated horseshoe](../RESULTS.md#federated_horseshoe_gaussian_bias_bounds) (Theorem 13.25, page 484; proved): The variance lost when each client's horseshoe posterior is replaced by a Gaussian vanishes for strong signals and is bounded by a quarter of the squared observation for every coordinate. First stated in Polson and Zantedeschi (2026f), Federated Learning, de Finetti, and Sanov, working paper.
- [Federated horseshoe variance bias vanishes with many clients](../RESULTS.md#federated_horseshoe_bias_vanishes_many_clients) (Theorem 13.26, page 484; proved): With many homogeneous clients the variance bias of the product of Gaussian approximations is of order one over K, the same order as the posterior variance itself, so the bound alone does not make the relative error small. First stated in Polson and Zantedeschi (2026f), Federated Learning, de Finetti, and Sanov, working paper.
<!-- results:end -->

## Key terms

applications of Bayesian statistics, genome wide association, GWAS, screening, approximate Bayes factor, Wakefield, horseshoe prior, multiple testing, medical decision making, diagnostic testing, decision curve analysis, clinical trials, A/B testing, small area estimation, thermodynamic reading, Brandeis dice, federated learning, federated Bayes, product of experts, de Finetti, Sanov, Aldous Hoover, federated Bernstein von Mises, federated horseshoe, FedAvg
