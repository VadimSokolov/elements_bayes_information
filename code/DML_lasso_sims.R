# DML-LASSO rows of Table tab:sims (Section "Simulation Studies" of the causal chapter):
# n = 500, B = 50 replications, theta0 = 1, p = 10 and p = 100, for the three DGPs.
# Uses the data generating processes and the cross-fitted DML-LASSO estimator of
# DML_meets_GBC_code.R itself (evaluated from that file by name), without running its
# GBC part, which is under repair.
# Run from the repository root:  Rscript code/DML_lasso_sims.R   (about 1 minute)
suppressMessages(library(glmnet))

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
exprs <- parse(file = file.path(here, "DML_meets_GBC_code.R"))
keep <- c("dgp_sparse", "dgp_nonlinear", "dgp_highdim", "dml_lasso")
for (e in exprs) {
  if (is.call(e) && identical(e[[1]], as.name("<-")) && is.name(e[[2]]) &&
      as.character(e[[2]]) %in% keep) eval(e, globalenv())
}
stopifnot(all(sapply(keep, exists)))

set.seed(2024)
n <- 500; B <- 50; theta0 <- 1
for (dgp in c("sparse", "nonlinear", "highdim")) for (p in c(10, 100)) {
  th <- cov <- numeric(B)
  for (b in 1:B) {
    dat <- switch(dgp, sparse = dgp_sparse(n, p), nonlinear = dgp_nonlinear(n, p),
                  highdim = dgp_highdim(n, p))
    f <- dml_lasso(as.numeric(dat$Y), as.numeric(dat$D), dat$X)
    th[b] <- f$theta
    cov[b] <- (f$ci_lo <= theta0 && theta0 <= f$ci_hi)
  }
  cat(sprintf("%-9s p=%3d bias=%6.3f sd=%.3f rmse=%.3f cov=%.2f\n", dgp, p,
              mean(th) - theta0, sd(th), sqrt(mean((th - theta0)^2)), mean(cov)))
}
