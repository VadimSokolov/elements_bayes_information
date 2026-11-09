# Appendix M. Concentration Inequalities

<!-- cite:begin -->
From *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations* by Nick Polson and Vadim Sokolov (2026), appendix M. Cite as Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1, https://doi.org/10.5281/zenodo.24067804. Book page: https://vsokolov.org/books/elements_bayes_information/. Source: https://github.com/VadimSokolov/elements_bayes_information.
<!-- cite:end -->

Appendix M is a short background toolkit on concentration inequalities, the bounds that control the fluctuation of sums and functions of independent variables and that underlie the PAC Bayes, bandit and generalization results of the text. It defines sub Gaussian variables through a variance proxy on the moment generating function and derives the Chernoff tail bound, shows by Hoeffding's lemma that a variable in an interval of length b minus a is sub Gaussian with variance proxy at most (b minus a) squared over four, and states Hoeffding's inequality for the mean of n independent bounded terms, the concentration behind the central limit scale of the deviation regimes. Bernstein's inequality for centered summands bounded by M interpolates between a sub Gaussian regime for small deviations and a sub exponential regime for large ones, and Bennett's inequality is its sharper Poisson tail version; these variance aware bounds give the fast rates in bandit regret and empirical Bayes risk. McDiarmid's bounded differences inequality gives a function of independent variables that changes by at most c_i in coordinate i a variance proxy of one quarter of the sum of the c_i squared. The entropy (Herbst) method extends concentration to smooth functions through log Sobolev inequalities, and Talagrand's inequality gives dimension free concentration of suprema of empirical processes about their mean, which turns the expectation bounds of Appendix N into high probability bounds. The appendix states standard results and proves no new theorem.

## From the literature

Hoeffding's inequality (Hoeffding 1963), Hoeffding's lemma, Chernoff bound, Bernstein's inequality (Bernstein 1917), Bennett's inequality (Bennett 1962), McDiarmid's bounded differences inequality (McDiarmid 1989), the entropy method of Herbst and log Sobolev inequalities, Talagrand's concentration inequality (Talagrand 1995).

## New results

<!-- results:begin -->
No new results of this appendix are catalogued.
<!-- results:end -->

## Key terms

concentration inequalities, sub Gaussian, sub exponential, variance proxy, Chernoff bound, Hoeffding lemma, Hoeffding inequality, Bernstein inequality, Bennett inequality, bounded differences, McDiarmid inequality, entropy method, Herbst argument, log Sobolev inequality, Talagrand inequality, PAC Bayes, tail bounds, background
