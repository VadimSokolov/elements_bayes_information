# Appendix I. Proper Scoring Rules and Calibration

<!-- cite:begin -->
From *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations* by Nick Polson and Vadim Sokolov (2026), appendix I. Cite as Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1, https://doi.org/10.5281/zenodo.24067804. Book page: https://vsokolov.org/books/elements_bayes_information/. Source: https://github.com/VadimSokolov/elements_bayes_information.
<!-- cite:end -->

This appendix is a short reference on proper scoring rules and calibration; the substance (the Savage representation, Bregman divergences of scores, the singling out of the log score, and the calibration and sharpness decomposition) sits in the main text. A scoring rule is proper when expected reward is maximized by reporting the true law, strictly proper when the maximizer is unique. The logarithmic score is the unique local strictly proper rule given at least three outcomes, and its running total is a codelength; the Brier score and the continuous ranked probability score are proper alternatives. The Savage representation writes any proper score through a convex function whose regret is a Bregman divergence: log score gives Kullback Leibler divergence, Brier gives squared error. Two bench facts follow: the mean, quantiles and expectiles are elicitable but the variance alone is not, and calibration is checked by a reliability diagram and repaired by Platt or isotonic recalibration, with the stated effects on the reliability and resolution terms of Murphy's decomposition.

## From the literature

Proper scoring rules and the Savage representation (Gneiting and Raftery 2007). The Brier score. Murphy's calibration and resolution decomposition. Platt scaling (Platt 2000) and isotonic recalibration.

## New results

<!-- results:begin -->
No new results of this appendix are catalogued.
<!-- results:end -->

## Key terms

proper scoring rule, strictly proper, logarithmic score, log loss, Brier score, continuous ranked probability score, CRPS, Savage representation, Bregman divergence, elicitability, expectile, quantile, calibration, reliability diagram, sharpness, resolution, Murphy decomposition, Platt scaling, isotonic regression
