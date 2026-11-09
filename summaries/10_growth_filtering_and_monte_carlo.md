# Chapter 10. Growth, Filtering, and Monte Carlo

<!-- cite:begin -->
From *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations* by Nick Polson and Vadim Sokolov (2026), chapter 10. Cite as Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1, https://doi.org/10.5281/zenodo.24067804. Book page: https://vsokolov.org/books/elements_bayes_information/. Source: https://github.com/VadimSokolov/elements_bayes_information.
<!-- cite:end -->

How does the predictive distribution drive action and dynamic estimation over time? The chapter takes growth optimal prediction, the Kelly criterion and its portfolio form, as the case where the predictive translates directly into betting, with growth rate equal to relative entropy and side information worth mutual information. Cover's universal portfolio is then shown to be a Bayes rule: wealth is the marginal likelihood, regret is a log Bayes factor and Bayesian redundancy, and its asymptotic regret expansion matches the Rissanen and Shtarkov stochastic complexity, with worked examples on Parrondo, roulette and market data. The chapter then treats nonlinear state space models and Bayesian filtering, Hamiltonian and Langevin Monte Carlo, and particle MCMC with SMC squared and pseudo marginal methods. The recurring point is that filtering is prediction run forward and Monte Carlo makes the recursion executable.

## From the literature

The Kelly criterion (Kelly 1956; Thorp 2006); growth optimal portfolios (Breiman 1961; Algoet and Cover 1988); Cover's universal portfolio (Cover 1991) and its regret bound with the Dirichlet one half prior (Cover and Ordentlich 1996); minimax redundancy and the Jeffreys prior (Clarke and Barron 1990, 1994); the Krichevsky and Trofimov estimator (1981); stochastic complexity (Shtarkov 1987; Rissanen 1996); Hamiltonian Monte Carlo (Duane et al. 1987; Neal 2011); the no U turn sampler (Hoffman and Gelman 2014); the pseudo marginal principle (Andrieu and Roberts 2009); particle MCMC (Andrieu, Doucet and Holenstein 2010).

## New results

The catalogued result is the Jeffreys prior equalizer for the regret of the universal portfolio; the wealth identity and the regret bound it reads as evidence and Bayesian redundancy are Cover's (1991) and Cover and Ordentlich's (1996).

<!-- results:begin -->
- [The Jeffreys prior equalizes the regret of the universal portfolio](../RESULTS.md#jeffreys_prior_equalizes_universal_portfolio_regret) (Theorem 10.10, page 349; conditional on the Laplace expansion of the mixture wealth, which the book states without a formal proof): In a stationary ergodic market the regret of the Bayes universal portfolio has a Laplace expansion of the Rissanen and Shtarkov type, and under the Jeffreys prior of a market family indexed by its growth optimal portfolio the constant does not depend on that portfolio, the equalizer property of Clarke and Barron carried to portfolios; the book states this without a formal proof, and for the uniform prior the first expansion recovers Cover (1991). First stated in Polson (2026), A Bayesian Interpretation of Cover's Universal Portfolio, submitted to Entropy.
<!-- results:end -->

## Key terms

Kelly criterion, growth optimal portfolio, log optimal betting, universal portfolio, Cover, constant rebalanced portfolio, regret, Bayesian redundancy, stochastic complexity, Jeffreys prior, Dirichlet prior, Parrondo paradox, state space model, Bayesian filtering, particle filter, Hamiltonian Monte Carlo, HMC, Langevin, particle MCMC, SMC squared, pseudo marginal
