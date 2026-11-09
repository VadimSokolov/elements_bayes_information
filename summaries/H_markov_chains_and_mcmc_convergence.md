# Appendix H. Markov Chains and MCMC Convergence

<!-- cite:begin -->
From *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations* by Nick Polson and Vadim Sokolov (2026), appendix H. Cite as Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1, https://doi.org/10.5281/zenodo.24067804. Book page: https://vsokolov.org/books/elements_bayes_information/. Source: https://github.com/VadimSokolov/elements_bayes_information.
<!-- cite:end -->

This appendix is background. It states the convergence theory that justifies the MCMC samplers of the computation chapters. A kernel has stationary law pi if pi K equals pi; detailed balance is sufficient, makes the kernel self adjoint on L2 of pi, and is what the Metropolis Hastings acceptance ratio enforces, whereas the Gibbs sampler is stationary because each complete conditional update leaves pi invariant, although a systematic scan need not be reversible. An irreducible, aperiodic, positive recurrent chain obeys the ergodic theorem; geometric ergodicity gives finite Monte Carlo standard errors and a Markov chain central limit theorem under a 2 plus delta moment condition, with asymptotic variance inflated by autocorrelation. For reversible chains the second largest eigenvalue modulus and the spectral gap control mixing. The integrated autocorrelation time gives the effective sample size, and remedies for poor mixing include reparameterization, Hamiltonian gradient information, tempering, and tuning the likelihood estimate variance in pseudo marginal chains.

## From the literature

The ergodic theorem for Markov chains. Detailed balance and the Metropolis Hastings algorithm. Geometric ergodicity and the Markov chain central limit theorem. Spectral gap and mixing for reversible chains. Pseudo marginal MCMC (Andrieu and Roberts 2009).

## New results

<!-- results:begin -->
No new results of this appendix are catalogued.
<!-- results:end -->

## Key terms

Markov chain, stationary distribution, detailed balance, reversibility, Metropolis Hastings, Gibbs sampler, ergodic theorem, geometric ergodicity, Markov chain central limit theorem, asymptotic variance, spectral gap, integrated autocorrelation time, effective sample size, mixing, tempering, pseudo marginal MCMC
