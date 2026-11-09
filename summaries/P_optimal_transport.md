# Appendix P. Optimal Transport

<!-- cite:begin -->
From *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations* by Nick Polson and Vadim Sokolov (2026), appendix P. Cite as Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1, https://doi.org/10.5281/zenodo.24067804. Book page: https://vsokolov.org/books/elements_bayes_information/. Source: https://github.com/VadimSokolov/elements_bayes_information.
<!-- cite:end -->

Appendix P is a short background toolkit on optimal transport, the geometry behind the transport maps of generative Bayesian computation (GBC) and the Wasserstein distances used to compare predictive distributions. It presents the Monge problem of a map pushing one law to another at minimal cost and its Kantorovich relaxation to couplings, which always has a solution; for quadratic cost the square root of the minimum is the 2 Wasserstein distance, a metric on laws with finite second moment that metrizes weak convergence plus moment convergence. Brenier's theorem gives, for quadratic cost and an absolutely continuous source, a unique optimal map that is the gradient of a convex potential, which makes the GBC posterior a deterministic transport of a base distribution: the quantile map in one dimension and the Brenier map in higher dimensions, although the coordinatewise pinball loss of GBC estimates the triangular Knothe Rosenblatt map, which has the same pushforward. Kantorovich duality underlies the Wasserstein GAN objective and dual solvers. The Benamou and Brenier formula recasts the 2 Wasserstein distance as a least action geodesic between densities, and entropic optimal transport adds a Kullback Leibler penalty and is solved by Sinkhorn's algorithm, alternating I projections onto the two marginal constraints, the same maximum entropy iteration as in the main text. The appendix states standard results and proves no new theorem.

## From the literature

Monge and Kantorovich problems (Kantorovich 1942), Brenier's theorem (Brenier 1991), Kantorovich duality, the Benamou and Brenier dynamic formulation (Benamou and Brenier 2000), the Knothe Rosenblatt map, entropic transport and Sinkhorn distances (Cuturi 2013); general references Villani 2009 and Peyre and Cuturi 2019.

## New results

<!-- results:begin -->
No new results of this appendix are catalogued.
<!-- results:end -->

## Key terms

optimal transport, Monge problem, Kantorovich relaxation, coupling, Wasserstein distance, 2 Wasserstein, Brenier theorem, Brenier map, Knothe Rosenblatt map, quantile map, Kantorovich duality, Wasserstein GAN, Benamou Brenier, displacement interpolation, entropic transport, Sinkhorn algorithm, iterative scaling, transport map, background
