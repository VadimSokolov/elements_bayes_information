# Appendix O. Reproducing Kernel Hilbert Spaces

<!-- cite:begin -->
From *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations* by Nick Polson and Vadim Sokolov (2026), appendix O. Cite as Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1, https://doi.org/10.5281/zenodo.24067804. Book page: https://vsokolov.org/books/elements_bayes_information/. Source: https://github.com/VadimSokolov/elements_bayes_information.
<!-- cite:end -->

Appendix O is a short background toolkit on reproducing kernel Hilbert spaces (RKHS), the function spaces behind Gaussian processes and kernel methods, in which the kernel is at once a covariance and an inner product. A symmetric positive definite kernel defines a Hilbert space of functions in which evaluation is a bounded linear functional represented by the kernel (the reproducing property); the Moore Aronszajn theorem gives a unique such space for every positive definite kernel, and Mercer's theorem expands a continuous kernel on a compact domain in eigenfunctions, so the space consists of the functions with finite weighted norm. The representer theorem reduces regularized empirical risk minimization over the space to n coefficients on the kernel sections at the data, and the Gaussian process posterior mean is this kernel ridge solution for squared error loss with the penalty equal to the noise variance; the Gaussian process prior is the Gaussian measure whose Cameron Martin space is the RKHS. Kernel mean embeddings of distributions give the maximum mean discrepancy (MMD), a metric when the kernel is characteristic, which underlies kernel two sample tests and kernelized model criticism diagnostics and connects to the Stein discrepancies of Appendix Q. The appendix presents standard results and proves no new theorem.

## From the literature

Reproducing kernels and the Moore Aronszajn theorem (Aronszajn 1950), Mercer's theorem (Mercer 1909), the representer theorem, kernel mean embeddings and the maximum mean discrepancy.

## New results

<!-- results:begin -->
No new results of this appendix are catalogued.
<!-- results:end -->

## Key terms

RKHS, reproducing kernel Hilbert space, positive definite kernel, reproducing property, Moore Aronszajn theorem, Mercer theorem, representer theorem, kernel ridge regression, Cameron Martin space, kernel mean embedding, maximum mean discrepancy, MMD, characteristic kernel, two sample test, kernel methods, Gaussian process, background
