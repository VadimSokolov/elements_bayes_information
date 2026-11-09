# Appendix R. Causal Foundations

<!-- cite:begin -->
From *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations* by Nick Polson and Vadim Sokolov (2026), appendix R. Cite as Polson and Sokolov (2026), *Elements of Bayes and Information Theory: A Predictive Theory of Inference: Information, Algorithms, and Large Deviations*, edition 1, https://doi.org/10.5281/zenodo.24067804. Book page: https://vsokolov.org/books/elements_bayes_information/. Source: https://github.com/VadimSokolov/elements_bayes_information.
<!-- cite:end -->

Appendix R is a short background toolkit on causal foundations for the causal machine learning chapter, giving the formal language of interventions in both graphical and potential outcomes form. A structural causal model assigns each variable a function of its parents and an independent noise, so the joint law factorizes along a directed acyclic graph (DAG); d separation reads conditional independences off the graph, and the intervention do(X = x) deletes the edges into X and fixes its value, defining an interventional distribution that the passive conditional need not equal. A causal effect is identified when it can be written in terms of the observational distribution: the back door criterion licenses adjustment for a set blocking all confounding paths, the front door criterion handles unmeasured confounding with an observed mediator, and Pearl's do calculus, three rewrite rules, is complete, so an effect is identified if and only if the rules with ordinary probability calculus rewrite it free of do. In the Neyman Rubin potential outcomes framework the average treatment effect is identified under unconfoundedness and overlap; its efficient influence function is the augmented inverse propensity score, whose Neyman orthogonality, insensitivity to first order nuisance error, is what double machine learning exploits. The appendix states standard results and proves no new theorem.

## From the literature

Structural causal models, causal diagrams and the do calculus (Pearl 1995), completeness of the do calculus (Shpitser and Pearl 2006; Huang and Valtorta 2006), the back door and front door criteria, Neyman Rubin potential outcomes, the augmented inverse propensity score, Neyman orthogonality (Neyman 1959), double machine learning (Chernozhukov et al. 2018).

## New results

<!-- results:begin -->
No new results of this appendix are catalogued.
<!-- results:end -->

## Key terms

causal inference, structural causal model, SCM, DAG, directed acyclic graph, d separation, intervention, do calculus, back door criterion, front door criterion, identification, potential outcomes, unconfoundedness, overlap, average treatment effect, augmented inverse propensity weighting, AIPW, efficient influence function, Neyman orthogonality, double machine learning, DML, confounding, background
