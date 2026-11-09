# Appendix G. Optimization and Stochastic Approximation

<!-- cite:begin -->
From *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations* by Nick Polson and Vadim Sokolov (2026), appendix G. Cite as Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1, https://doi.org/10.5281/zenodo.24067804. Book page: https://vsokolov.org/books/elements_bayes_information/. Source: https://github.com/VadimSokolov/elements_bayes_information.
<!-- cite:end -->

This appendix is background. It collects optimization facts behind the proximal, variational and deep learning sections, and the stochastic approximation results that make sampling and optimization two readings of one iteration. For convex L smooth functions gradient descent with step 1 over L has an O(1 over k) rate, Nesterov acceleration gives O(1 over k squared), and strong convexity gives a linear rate governed by the condition number; proximal gradient keeps these rates for a smooth plus proximable nonsmooth objective. Mirror descent uses Bregman projections, and with negative entropy gives exponentiated gradient (multiplicative weights) on the simplex with regret scaling as root log d. The Robbins Monro iteration converges under the stated step size, drift and growth conditions; stochastic gradient descent is the special case with g equal to the gradient, constant step size tracks a stationary distribution (the Langevin connection), and Polyak Ruppert averaging attains the optimal statistical rate.

## From the literature

Gradient descent rates for smooth convex and strongly convex functions. Nesterov acceleration. Mirror descent (Nemirovski and Yudin 1983). Exponentiated gradient and multiplicative weights. The Robbins Monro stochastic approximation. Polyak Ruppert averaging.

## New results

<!-- results:begin -->
No new results of this appendix are catalogued.
<!-- results:end -->

## Key terms

gradient descent, Nesterov acceleration, strongly convex, condition number, proximal gradient, proximal operator, mirror descent, Bregman projection, exponentiated gradient, multiplicative weights, regret, Robbins Monro, stochastic approximation, stochastic gradient descent, Polyak Ruppert averaging, Langevin
