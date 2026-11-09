# Chapter 11. Sequential Decisions and Reinforcement Learning

<!-- cite:begin -->
From *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations* by Nick Polson and Vadim Sokolov (2026), chapter 11. Cite as Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1, https://doi.org/10.5281/zenodo.24067804. Book page: https://vsokolov.org/books/elements_bayes_information/. Source: https://github.com/VadimSokolov/elements_bayes_information.
<!-- cite:end -->

How should a Bayesian choose actions when the actions change the data? The chapter treats sequential design and reinforcement learning with the predictive distribution as both guide and consequence of exploration. It covers Bayesian optimization with Gaussian process surrogates and expected improvement, Thompson sampling and Bayesian bandits with regret bounds, reinforcement learning and control as inference (KL regularized control, soft Q learning, Gibbs optimal policies), and approximate Bayesian computation (ABC, likelihood free inference) for simulators whose likelihood cannot be evaluated. The moderate deviation scale returns as the threshold behind the regret bounds, tying the decision problems to the testing and shrinkage boundaries of earlier chapters. The chapter is a survey in the book's predictive language and catalogues no new theorem.

## From the literature

Expected improvement and efficient global optimization (Jones, Schonlau and Welch 1998); Thompson sampling (Thompson 1933) and its regret analysis (Agrawal and Goyal 2012; Russo and Van Roy 2014); control as inference and KL regularized control (Todorov 2009; Kappen 2005; Levine 2018); approximate Bayesian computation (Beaumont, Zhang and Balding 2002; Marin et al. 2012).

## New results

No new results of this chapter are catalogued; the chapter is a survey of known methods.

<!-- results:begin -->
No new results of this chapter are catalogued.
<!-- results:end -->

## Key terms

Bayesian optimization, expected improvement, sequential design, Thompson sampling, multi armed bandit, regret bound, exploration and exploitation, reinforcement learning, control as inference, KL regularized control, soft Q learning, Gibbs policy, approximate Bayesian computation, ABC, likelihood free inference, simulation based inference
