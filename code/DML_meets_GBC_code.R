# NOTE (audit): the GBC simulator pools scalar per-draw statistics (near-singular design)
# and the Lopes-Polson Gibbs step does not yet update l11,l22; until those are rebuilt the
# GBC rows of tab:sims and the tab:card numbers should be read as ILLUSTRATIVE. rho now uses
# the correct l21/sqrt(l21^2+l22^2). KNOWN LIMITATIONS documented here.
# gbc_posterior reports the SD of the fitted posterior quantile function and its quantiles
# at levels 0.025 and 0.975 as the 95% interval (section 3b).

# =============================================================================
# Double Machine Learning Meets Generative Bayesian Computation
# Nareklishvili, Polson, Sokolov (2026)
#
# Complete R code for all simulations and empirical illustrations
#
# Sections:
#   0. Packages
#   1. Data Generating Processes (DGPs)
#   2. DML Estimators (LASSO and Random Forest)
#   3. GBC Estimator (Quantile Transport Map)
#   4. Monte Carlo Simulation Study
#   5. Visualisation
#   6. Empirical Application: Card (1995) Returns to Education
#   7. Bayesian IV: Lopes-Polson (2014) Cholesky Prior
#   8. Hybrid DML-GBC Estimator
#   9. Full Comparison on a Single Dataset
# =============================================================================


# =============================================================================
# 0.  PACKAGES
# =============================================================================

# install.packages(c(
#   "MASS", "glmnet", "randomForest", "quantreg",
#   "hdm", "ggplot2", "dplyr", "coda"
# ))

library(MASS)
library(glmnet)
library(randomForest)
library(quantreg)
library(hdm)          # for rlasso (DML)
library(ivmodel)      # for card.data (Card 1995)
library(ggplot2)
library(dplyr)
library(coda)

set.seed(2024)


# =============================================================================
# 1.  DATA GENERATING PROCESSES
# =============================================================================

# ------------------------------------------------------------------------------
# DGP 1: Sparse linear confounding
#   g0(X) = X[,1:s0] %*% 1,  m0(X) = X[,1:s0] %*% 0.5
#   True theta_0 = 1
# ------------------------------------------------------------------------------
dgp_sparse <- function(n, p, theta0 = 1, s0 = 5) {
  beta_g <- c(rep(1,   s0), rep(0, p - s0))
  beta_m <- c(rep(0.5, s0), rep(0, p - s0))
  X <- matrix(rnorm(n * p), n, p)
  D <- X %*% beta_m + rnorm(n)
  Y <- D * theta0 + X %*% beta_g + rnorm(n)
  list(Y = Y, D = D, X = X)
}

# ------------------------------------------------------------------------------
# DGP 2: Dense nonlinear confounding
#   g0 = sin(X1) + cos(X2) + 0.5*X3^2 + rowMeans(X[,4:10])
#   m0 = tanh(X1) + 0.5*X2*X3 + 0.3*rowSums(X[,1:5]^2)
# ------------------------------------------------------------------------------
dgp_nonlinear <- function(n, p, theta0 = 1) {
  X  <- matrix(rnorm(n * p), n, p)
  g0 <- sin(X[, 1]) + cos(X[, 2]) + 0.5 * X[, 3]^2 +
        rowMeans(X[, 4:min(p, 10), drop = FALSE])
  m0 <- tanh(X[, 1]) + 0.5 * X[, 2] * X[, 3] +
        0.3 * rowSums(X[, 1:min(p, 5), drop = FALSE]^2)
  D  <- m0 + rnorm(n)
  Y  <- D * theta0 + g0 + rnorm(n)
  list(Y = Y, D = D, X = X, g0 = g0, m0 = m0)
}

# ------------------------------------------------------------------------------
# DGP 3: Very high-dimensional dense confounding
#   Signal spread across all p dimensions via exponentially decaying weights
# ------------------------------------------------------------------------------
dgp_highdim <- function(n, p, theta0 = 1) {
  X         <- matrix(rnorm(n * p), n, p)
  weights_g <- exp(-0.1  * (1:p))
  weights_m <- exp(-0.05 * (1:p))
  g0 <- X %*% weights_g + 0.3 * (X[, 1] * X[, 2])
  m0 <- X %*% weights_m
  D  <- m0 + rnorm(n)
  Y  <- D * theta0 + g0 + rnorm(n)
  list(Y = Y, D = D, X = X, g0 = g0, m0 = m0)
}


# =============================================================================
# 2.  DML ESTIMATORS
# =============================================================================

# ------------------------------------------------------------------------------
# 2a.  DML with LASSO nuisance estimators
#      Cross-fitting with K folds; heteroskedasticity-robust variance
# ------------------------------------------------------------------------------
dml_lasso <- function(Y, D, X, K = 5) {
  n     <- length(Y)
  folds <- sample(rep(1:K, length.out = n))
  Y_res <- D_res <- numeric(n)

  for (k in 1:K) {
    train <- folds != k
    test  <- folds == k

    cv_Y <- cv.glmnet(X[train, ], Y[train], alpha = 1, nfolds = 5)
    cv_D <- cv.glmnet(X[train, ], D[train], alpha = 1, nfolds = 5)

    Y_res[test] <- Y[test] - predict(cv_Y, X[test, ], s = "lambda.min")
    D_res[test] <- D[test] - predict(cv_D, X[test, ], s = "lambda.min")
  }

  theta_hat <- sum(D_res * Y_res) / sum(D_res^2)
  psi <- D_res * (Y_res - D_res * theta_hat)
  V   <- mean(D_res^2)^(-2) * mean(psi^2) / n
  se  <- sqrt(V)

  list(theta  = theta_hat,
       se     = se,
       ci_lo  = theta_hat - 1.96 * se,
       ci_hi  = theta_hat + 1.96 * se)
}

# ------------------------------------------------------------------------------
# 2b.  DML with Random Forest nuisance estimators
# ------------------------------------------------------------------------------
dml_rf <- function(Y, D, X, K = 5, ntree = 200) {
  n     <- length(Y)
  folds <- sample(rep(1:K, length.out = n))
  df    <- as.data.frame(X)
  Y_res <- D_res <- numeric(n)

  for (k in 1:K) {
    train <- folds != k
    test  <- folds == k

    rf_Y <- randomForest(y = Y[train], x = df[train, ],
                         ntree = ntree, importance = FALSE)
    rf_D <- randomForest(y = D[train], x = df[train, ],
                         ntree = ntree, importance = FALSE)

    Y_res[test] <- Y[test] - predict(rf_Y, df[test, ])
    D_res[test] <- D[test] - predict(rf_D, df[test, ])
  }

  theta_hat <- sum(D_res * Y_res) / sum(D_res^2)
  psi <- D_res * (Y_res - D_res * theta_hat)
  V   <- mean(D_res^2)^(-2) * mean(psi^2) / n
  se  <- sqrt(V)

  list(theta  = theta_hat,
       se     = se,
       ci_lo  = theta_hat - 1.96 * se,
       ci_hi  = theta_hat + 1.96 * se)
}


# =============================================================================
# 3.  GBC ESTIMATOR  (Quantile Transport Map)
# =============================================================================

# ------------------------------------------------------------------------------
# 3a.  Simulate training data from the prior
#      Returns N draws of (theta, Y, D, X) under the GBC generative model
# ------------------------------------------------------------------------------
gbc_simulate_prior <- function(N, p, sigma_theta = 2) {
  theta_sim  <- rnorm(N, 0, sigma_theta)
  beta_g_sim <- matrix(rnorm(N * p, 0, 1 / sqrt(p)), N, p)
  beta_m_sim <- matrix(rnorm(N * p, 0, 1 / sqrt(p)), N, p)
  X_sim      <- matrix(rnorm(N * p), N, p)
  Y_sim      <- D_sim <- numeric(N)

  for (i in 1:N) {
    m_i      <- sum(X_sim[i, ] * beta_m_sim[i, ])
    g_i      <- sum(X_sim[i, ] * beta_g_sim[i, ])
    D_sim[i] <- m_i + rnorm(1)
    Y_sim[i] <- D_sim[i] * theta_sim[i] + g_i + rnorm(1)
  }

  list(theta = theta_sim, Y = Y_sim, D = D_sim, X = X_sim)
}

# ------------------------------------------------------------------------------
# 3b.  GBC posterior via quantile regression on simulated data
#
#      The transport map T(U, summary(W)) -> theta is approximated by a
#      quantile regression of theta_sim on (tau, sufficient statistics).
#      Sufficient statistics used: partialled-out OLS estimate, var(D_res),
#      cor(Y_res, D_res) — these are low-dimensional summaries of (Y, D, X).
#
#      A full production implementation would substitute a deep QReLU network
#      (e.g. via torch or keras) trained jointly over all quantile levels.
#
#      Posterior summaries: the fitted quantile function is evaluated on the
#      midpoint grid of M levels (j - 0.5)/M, an equally weighted stratified
#      sample from the approximate posterior, whose mean and standard deviation
#      estimate the posterior mean and SD; the equal-tailed credible interval is
#      the pair of quantiles fitted at levels (1 - level)/2 and (1 + level)/2.
#      (The earlier version took the SD of 19 quantiles at levels 0.05, ..., 0.95,
#      which understates a normal SD by 11%, and the 2.5% and 97.5% sample
#      quantiles of those 19 values, about the 7% and 93% posterior quantiles,
#      so its "95%" interval had about 86% posterior probability.)
# ------------------------------------------------------------------------------
gbc_posterior <- function(Y_obs, D_obs, X_obs,
                           N_train = 5000,
                           M       = 200,
                           level   = 0.95) {
  p <- ncol(X_obs)

  # --- Observed sufficient statistics ---
  lm_Y_obs <- lm(Y_obs ~ X_obs)
  lm_D_obs <- lm(D_obs ~ X_obs)
  Y_res_obs <- residuals(lm_Y_obs)
  D_res_obs <- residuals(lm_D_obs)
  theta_ols_obs <- sum(D_res_obs * Y_res_obs) / sum(D_res_obs^2)
  var_D_obs     <- var(D_res_obs)
  cor_YD_obs    <- cor(Y_res_obs, D_res_obs)

  # --- Simulate training data ---
  sim       <- gbc_simulate_prior(N_train, p)
  theta_sim <- sim$theta

  lm_Y_s <- lm(sim$Y ~ sim$X)
  lm_D_s <- lm(sim$D ~ sim$X)
  Y_res_s  <- residuals(lm_Y_s)
  D_res_s  <- residuals(lm_D_s)
  ols_s    <- sum(D_res_s * Y_res_s) / sum(D_res_s^2)
  var_D_s  <- var(D_res_s)
  cor_YD_s <- cor(Y_res_s, D_res_s)

  # --- Quantile regression for each tau ---
  # Features: (U, ols_s, var_D_s, cor_YD_s)
  U_sim    <- runif(N_train)
  feat_sim <- cbind(U_sim, ols_s, var_D_s, cor_YD_s)

  post_quantile <- function(tau) {
    feat_obs <- c(tau, theta_ols_obs, var_D_obs, cor_YD_obs)
    qr_fit   <- rq(theta_sim ~ feat_sim, tau = tau)
    as.numeric(coef(qr_fit) %*% c(1, feat_obs))
  }

  taus       <- (seq_len(M) - 0.5) / M
  theta_post <- vapply(taus, post_quantile, numeric(1))
  ci         <- vapply(c((1 - level) / 2, (1 + level) / 2), post_quantile,
                       numeric(1))

  list(theta               = mean(theta_post),
       se                  = sd(theta_post),
       ci_lo               = ci[1],
       ci_hi               = ci[2],
       posterior_quantiles = theta_post,
       taus                = taus)
}


# =============================================================================
# 4.  MONTE CARLO SIMULATION STUDY
# =============================================================================

# ------------------------------------------------------------------------------
# run_simulation():
#   Runs B replications of a given (n, p, dgp) scenario and returns a
#   data frame of bias, RMSE, coverage, and average CI width for each method.
# ------------------------------------------------------------------------------
run_simulation <- function(n = 500, p = 20, dgp = "sparse",
                            B = 100, theta0 = 1) {

  theta_dml_lasso <- theta_dml_rf <- theta_gbc <- numeric(B)
  cover_lasso <- cover_rf <- cover_gbc         <- numeric(B)
  width_lasso <- width_rf <- width_gbc         <- numeric(B)

  for (b in 1:B) {
    dat <- switch(dgp,
                  "sparse"    = dgp_sparse(n, p),
                  "nonlinear" = dgp_nonlinear(n, p),
                  "highdim"   = dgp_highdim(n, p))
    Y <- dat$Y; D <- dat$D; X <- dat$X

    # DML-LASSO
    fit_lasso <- tryCatch(dml_lasso(Y, D, X),
                          error = function(e)
                            list(theta = NA, ci_lo = NA, ci_hi = NA))
    theta_dml_lasso[b] <- fit_lasso$theta
    cover_lasso[b] <- !is.na(fit_lasso$ci_lo) &&
                      fit_lasso$ci_lo <= theta0 && theta0 <= fit_lasso$ci_hi
    width_lasso[b] <- fit_lasso$ci_hi - fit_lasso$ci_lo

    # DML-RF  (skipped for large p to keep runtime manageable)
    if (p <= 50) {
      fit_rf <- tryCatch(dml_rf(Y, D, X),
                         error = function(e)
                           list(theta = NA, ci_lo = NA, ci_hi = NA))
      theta_dml_rf[b] <- fit_rf$theta
      cover_rf[b] <- !is.na(fit_rf$ci_lo) &&
                     fit_rf$ci_lo <= theta0 && theta0 <= fit_rf$ci_hi
      width_rf[b] <- fit_rf$ci_hi - fit_rf$ci_lo
    } else {
      theta_dml_rf[b] <- NA
    }

    # GBC
    fit_gbc <- tryCatch(gbc_posterior(Y, D, X, N_train = 3000),
                        error = function(e)
                          list(theta = NA, se = NA, ci_lo = NA, ci_hi = NA))
    theta_gbc[b] <- fit_gbc$theta
    cover_gbc[b] <- !is.na(fit_gbc$ci_lo) &&
                    fit_gbc$ci_lo <= theta0 && theta0 <= fit_gbc$ci_hi
    width_gbc[b] <- fit_gbc$ci_hi - fit_gbc$ci_lo
  }

  # Compile summary table
  results <- data.frame()
  for (mth in c("DML-LASSO", "DML-RF", "GBC")) {
    thetas <- switch(mth,
                     "DML-LASSO" = theta_dml_lasso,
                     "DML-RF"    = theta_dml_rf,
                     "GBC"       = theta_gbc)
    covers <- switch(mth,
                     "DML-LASSO" = cover_lasso,
                     "DML-RF"    = cover_rf,
                     "GBC"       = cover_gbc)
    widths <- switch(mth,
                     "DML-LASSO" = width_lasso,
                     "DML-RF"    = width_rf,
                     "GBC"       = width_gbc)

    thetas_clean <- thetas[!is.na(thetas)]
    if (length(thetas_clean) == 0) next

    results <- rbind(results, data.frame(
      method   = mth,
      bias     = round(mean(thetas_clean) - theta0, 4),
      rmse     = round(sqrt(mean((thetas_clean - theta0)^2)), 4),
      coverage = round(mean(covers, na.rm = TRUE), 3),
      ci_width = round(mean(widths,  na.rm = TRUE), 4)
    ))
  }
  results$n   <- n
  results$p   <- p
  results$dgp <- dgp
  results
}

# --- Run across scenarios (increase B for publication) ---
cat("Running Monte Carlo simulations...\n")

scenarios <- expand.grid(
  n   = c(500, 1000),
  p   = c(10, 50, 100),
  dgp = c("sparse", "nonlinear", "highdim"),
  stringsAsFactors = FALSE
)

all_results <- do.call(rbind, lapply(1:nrow(scenarios), function(i) {
  cat(sprintf("  n = %4d | p = %3d | dgp = %s\n",
              scenarios$n[i], scenarios$p[i], scenarios$dgp[i]))
  run_simulation(n   = scenarios$n[i],
                 p   = scenarios$p[i],
                 dgp = scenarios$dgp[i],
                 B   = 50)   # <-- increase to 500 for the paper
}))

print(all_results)


# =============================================================================
# 5.  VISUALISATION
# =============================================================================

# --- RMSE vs dimension ---
rmse_plot <- all_results %>%
  filter(!(method == "DML-RF" & p > 50)) %>%
  ggplot(aes(x = p, y = rmse, color = method, linetype = method)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  facet_grid(n ~ dgp, labeller = label_both) +
  scale_color_manual(values = c("DML-LASSO" = "#E74C3C",
                                 "DML-RF"    = "#F39C12",
                                 "GBC"       = "#2980B9")) +
  labs(title    = "RMSE vs Dimension: DML vs GBC",
       subtitle = "True theta_0 = 1; 50 Monte Carlo replications",
       x = "Dimension p", y = "RMSE",
       color = "Method", linetype = "Method") +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom")

print(rmse_plot)

# --- Coverage vs dimension ---
cov_plot <- all_results %>%
  filter(!(method == "DML-RF" & p > 50)) %>%
  ggplot(aes(x = p, y = coverage, color = method)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_hline(yintercept = 0.95, linetype = "dashed", color = "black") +
  facet_grid(n ~ dgp, labeller = label_both) +
  scale_color_manual(values = c("DML-LASSO" = "#E74C3C",
                                 "DML-RF"    = "#F39C12",
                                 "GBC"       = "#2980B9")) +
  ylim(0, 1) +
  labs(title    = "95% CI Coverage vs Dimension",
       subtitle = "Dashed line = nominal 95%",
       x = "Dimension p", y = "Empirical Coverage") +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom")

print(cov_plot)


# =============================================================================
# 6.  EMPIRICAL APPLICATION: CARD (1995) RETURNS TO EDUCATION
# =============================================================================

data(card.data, package = "ivmodel")

# Outcome, treatment, controls
Y_raw <- log(card.data$wage)
D_raw <- card.data$educ
Z_raw <- card.data$nearc4    # instrument: near 4-year college

X_raw <- model.matrix(
  ~ exper + expersq + black + south + smsa +
    reg661 + reg662 + reg663 + reg664 + reg665 +
    reg666 + reg667 + reg668 + smsa66 +
    momdad14 + sinmom14 + step14 + fatheduc + motheduc,
  data = card.data
)[, -1]

# Drop rows with missing values
complete <- complete.cases(Y_raw, D_raw, Z_raw, X_raw)
Y <- Y_raw[complete]
D <- D_raw[complete]
Z <- Z_raw[complete]
X <- X_raw[complete, ]
cat(sprintf("\nCard data: n = %d, p = %d\n", length(Y), ncol(X)))

# --- DML-LASSO ---
fit_card_dml <- dml_lasso(Y, D, X, K = 5)
cat("\n--- DML-LASSO (Card) ---\n")
cat(sprintf("Theta: %.4f  SE: %.4f  95%% CI: [%.4f, %.4f]\n",
            fit_card_dml$theta, fit_card_dml$se,
            fit_card_dml$ci_lo, fit_card_dml$ci_hi))

# --- GBC ---
fit_card_gbc <- gbc_posterior(Y, D, X, N_train = 5000)
cat("\n--- GBC (Card) ---\n")
cat(sprintf("Theta: %.4f  SE: %.4f  95%% CI: [%.4f, %.4f]\n",
            fit_card_gbc$theta, fit_card_gbc$se,
            fit_card_gbc$ci_lo, fit_card_gbc$ci_hi))

# --- GBC posterior quantile plot ---
post_df <- data.frame(theta = fit_card_gbc$posterior_quantiles,
                      tau   = fit_card_gbc$taus)

ggplot(post_df, aes(x = tau, y = theta)) +
  geom_line(color = "#2980B9", linewidth = 1.2) +
  geom_hline(yintercept = fit_card_dml$theta,
             linetype = "dashed", color = "#E74C3C") +
  geom_ribbon(aes(ymin = fit_card_dml$ci_lo,
                  ymax = fit_card_dml$ci_hi),
              alpha = 0.15, fill = "#E74C3C") +
  labs(title    = "GBC Posterior Quantiles vs DML Point Estimate",
       subtitle = "Card (1995) Returns to Education",
       x = "Quantile level tau",
       y = "Returns to schooling (log wage per year)",
       caption  = "Blue: GBC  |  Red dashed: DML-LASSO ± 95% CI") +
  theme_bw()


# =============================================================================
# 7.  BAYESIAN IV: LOPES-POLSON (2014) CHOLESKY PRIOR
# =============================================================================
#
#  Model:
#    Y = D*theta + Xd*beta + eps1
#    D = Xs*pi   + eps2
#    (eps1, eps2) ~ N(0, Sigma),   Sigma = L L',  L lower-triangular
#
#  Cholesky prior:
#    l11, l22 ~ LogNormal(0, 1)
#    l21      ~ Normal(0, 1)      (governs endogeneity correlation rho)
#
#  Posterior sampled via Gibbs + Metropolis-Hastings on l21.
# =============================================================================

# --- Log bivariate-normal density (vectorised over rows of X) ---
dmvnorm_log <- function(X_mat, sigma) {
  p   <- ncol(X_mat)
  sig_inv <- solve(sigma)
  -(p / 2) * log(2 * pi) - 0.5 * log(det(sigma)) -
    0.5 * rowSums((X_mat %*% sig_inv) * X_mat)
}

bayes_iv_lopes_polson <- function(Y, D, Z, X,
                                   n_iter = 8000, n_burn = 2000) {
  n  <- length(Y)
  Xd <- cbind(1, X)      # outcome design  (intercept + controls)
  Xs <- cbind(1, X, Z)   # first-stage design (intercept + controls + IV)

  # --- Initialise ---
  theta <- 0.1
  beta  <- rep(0, ncol(Xd))
  pi_s  <- rep(0, ncol(Xs))
  l11   <- 1; l22 <- 1; l21 <- 0
  Sigma <- matrix(c(l11^2, l11 * l21,
                    l11 * l21, l21^2 + l22^2), 2, 2)

  # --- Storage ---
  n_keep <- n_iter - n_burn
  store  <- matrix(NA_real_, n_keep, 4)
  colnames(store) <- c("theta", "l11", "l22", "l21")
  idx <- 0

  for (iter in 1:n_iter) {

    # Step 1: Sample theta  (conjugate normal)
    rho      <- Sigma[1, 2] / Sigma[2, 2]
    sig1_2   <- Sigma[1, 1] - Sigma[1, 2]^2 / Sigma[2, 2]
    eps2     <- D - Xs %*% pi_s
    Y_adj    <- Y - Xd %*% beta - rho * eps2
    V_theta  <- 1 / (sum(D^2) / sig1_2 + 1 / 100)
    M_theta  <- V_theta * sum(D * Y_adj) / sig1_2
    theta    <- rnorm(1, M_theta, sqrt(V_theta))

    # Step 2: Sample first-stage pi  (conjugate normal)
    V_pi <- solve(t(Xs) %*% Xs / Sigma[2, 2] +
                    diag(ncol(Xs)) * 0.01)
    M_pi <- V_pi %*% (t(Xs) %*% D / Sigma[2, 2])
    pi_s <- as.vector(mvrnorm(1, M_pi, V_pi))

    # Step 3: Sample outcome nuisance beta  (conjugate normal)
    Y_tilde <- Y - D * theta
    V_beta  <- solve(t(Xd) %*% Xd / sig1_2 + diag(ncol(Xd)) * 0.01)
    M_beta  <- V_beta %*% (t(Xd) %*% Y_tilde / sig1_2)
    beta    <- as.vector(mvrnorm(1, M_beta, V_beta))

    # Step 4: Metropolis step for l21  (endogeneity parameter)
    eps1      <- Y - D * theta - Xd %*% beta
    eps2      <- D - Xs %*% pi_s
    eps_mat   <- cbind(eps1, eps2)

    l21_prop  <- l21 + rnorm(1, 0, 0.05)
    Sig_prop  <- matrix(c(l11^2,
                           l11 * l21_prop,
                           l11 * l21_prop,
                           l21_prop^2 + l22^2), 2, 2)

    log_ar <- (sum(dmvnorm_log(eps_mat, Sig_prop)) -
               sum(dmvnorm_log(eps_mat, Sigma)) +
               dnorm(l21_prop, 0, 1, log = TRUE) -
               dnorm(l21,      0, 1, log = TRUE))

    if (log(runif(1)) < log_ar) {
      l21   <- l21_prop
      Sigma <- Sig_prop
    }

    # Store post-burn draws
    if (iter > n_burn) {
      idx            <- idx + 1
      store[idx, ]   <- c(theta, l11, l22, l21)
    }
  }

  # --- Posterior summaries ---
  rho_post <- store[, "l21"] /
    sqrt(store[, "l21"]^2 + store[, "l22"]^2)

  list(theta_mean  = mean(store[, "theta"]),
       theta_sd    = sd(store[, "theta"]),
       ci_lo       = quantile(store[, "theta"], 0.025),
       ci_hi       = quantile(store[, "theta"], 0.975),
       rho_mean    = mean(rho_post),
       posterior   = store[, "theta"])
}

# --- Run Bayesian IV ---
set.seed(2024)
bayes_iv_fit <- bayes_iv_lopes_polson(Y, D, Z, X,
                                       n_iter = 8000, n_burn = 2000)

cat("\n--- Bayesian IV (Lopes-Polson Cholesky Prior) ---\n")
cat(sprintf("Posterior mean : %.4f  (SD: %.4f)\n",
            bayes_iv_fit$theta_mean, bayes_iv_fit$theta_sd))
cat(sprintf("95%% Credible I : [%.4f, %.4f]\n",
            bayes_iv_fit$ci_lo, bayes_iv_fit$ci_hi))
cat(sprintf("Endogeneity rho: %.4f\n", bayes_iv_fit$rho_mean))

# --- Four-method comparison table ---
comparison <- data.frame(
  Method   = c("OLS", "DML-LASSO", "GBC", "Bayesian IV (LP)"),
  Estimate = c(coef(lm(Y ~ D + X))[2],
               fit_card_dml$theta,
               fit_card_gbc$theta,
               bayes_iv_fit$theta_mean),
  CI_Lo    = c(NA,
               fit_card_dml$ci_lo,
               fit_card_gbc$ci_lo,
               bayes_iv_fit$ci_lo),
  CI_Hi    = c(NA,
               fit_card_dml$ci_hi,
               fit_card_gbc$ci_hi,
               bayes_iv_fit$ci_hi)
)
print(comparison)

# --- Posterior density plot ---
post_df2 <- data.frame(
  theta  = c(fit_card_gbc$posterior_quantiles, bayes_iv_fit$posterior),
  method = c(rep("GBC",              length(fit_card_gbc$posterior_quantiles)),
             rep("Bayesian IV (LP)", length(bayes_iv_fit$posterior)))
)

ggplot(post_df2, aes(x = theta, fill = method, color = method)) +
  geom_density(alpha = 0.35, linewidth = 1) +
  geom_vline(xintercept = fit_card_dml$theta,
             linetype = "dashed", color = "#E74C3C", linewidth = 1) +
  scale_fill_manual(values  = c("GBC"              = "#2980B9",
                                 "Bayesian IV (LP)" = "#27AE60")) +
  scale_color_manual(values = c("GBC"              = "#2980B9",
                                 "Bayesian IV (LP)" = "#27AE60")) +
  labs(title    = "Posterior Distributions: GBC vs Bayesian IV (Lopes-Polson)",
       subtitle = "Card (1995)  |  Red dashed = DML-LASSO point estimate",
       x = "Returns to Schooling", y = "Density") +
  theme_bw()


# =============================================================================
# 8.  HYBRID DML-GBC ESTIMATOR
# =============================================================================
#
#  Uses bootstrap-averaged random forests as a proxy for the GBC posterior mean
#  in the nuisance estimation step of DML.  A full implementation would use
#  a trained QReLU network (e.g. via torch) as the nuisance learner.
# =============================================================================

dml_gbc_hybrid <- function(Y, D, X, K = 5, n_boot = 10, ntree = 100) {
  n     <- length(Y)
  folds <- sample(rep(1:K, length.out = n))
  df    <- as.data.frame(X)
  Y_res <- D_res <- numeric(n)

  for (k in 1:K) {
    train_idx <- which(folds != k)
    test_idx  <- which(folds == k)

    gbc_Y <- matrix(0, length(test_idx), n_boot)
    gbc_D <- matrix(0, length(test_idx), n_boot)

    for (j in 1:n_boot) {
      boot_idx <- sample(train_idx, replace = TRUE)
      rf_Y_j   <- randomForest(y = Y[boot_idx], x = df[boot_idx, ],
                                ntree = ntree, importance = FALSE)
      rf_D_j   <- randomForest(y = D[boot_idx], x = df[boot_idx, ],
                                ntree = ntree, importance = FALSE)
      gbc_Y[, j] <- predict(rf_Y_j, df[test_idx, ])
      gbc_D[, j] <- predict(rf_D_j, df[test_idx, ])
    }

    # GBC nuisance = posterior mean (average over bootstrap draws)
    Y_res[test_idx] <- Y[test_idx] - rowMeans(gbc_Y)
    D_res[test_idx] <- D[test_idx] - rowMeans(gbc_D)
  }

  theta_hat <- sum(D_res * Y_res) / sum(D_res^2)
  psi <- D_res * (Y_res - D_res * theta_hat)
  V   <- mean(D_res^2)^(-2) * mean(psi^2) / n
  se  <- sqrt(V)

  list(theta  = theta_hat,
       se     = se,
       ci_lo  = theta_hat - 1.96 * se,
       ci_hi  = theta_hat + 1.96 * se)
}


# =============================================================================
# 9.  FULL FOUR-METHOD COMPARISON ON A SINGLE DATASET
# =============================================================================

set.seed(42)
dat_nl <- dgp_nonlinear(n = 500, p = 50)
theta0 <- 1

fit1 <- dml_lasso(dat_nl$Y, dat_nl$D, dat_nl$X)
fit2 <- dml_rf(dat_nl$Y, dat_nl$D, dat_nl$X)
fit3 <- gbc_posterior(dat_nl$Y, dat_nl$D, dat_nl$X)
fit4 <- dml_gbc_hybrid(dat_nl$Y, dat_nl$D, dat_nl$X)

summary_table <- data.frame(
  Method   = c("DML-LASSO", "DML-RF", "GBC", "DML-GBC Hybrid"),
  Estimate = c(fit1$theta, fit2$theta, fit3$theta, fit4$theta),
  SE       = c(fit1$se,    fit2$se,    fit3$se,    fit4$se),
  CI_Lo    = c(fit1$ci_lo, fit2$ci_lo, fit3$ci_lo, fit4$ci_lo),
  CI_Hi    = c(fit1$ci_hi, fit2$ci_hi, fit3$ci_hi, fit4$ci_hi),
  Covers   = c(fit1$ci_lo <= theta0 & theta0 <= fit1$ci_hi,
               fit2$ci_lo <= theta0 & theta0 <= fit2$ci_hi,
               fit3$ci_lo <= theta0 & theta0 <= fit3$ci_hi,
               fit4$ci_lo <= theta0 & theta0 <= fit4$ci_hi)
)

cat("\n=== Single-dataset comparison (nonlinear DGP, n=500, p=50) ===\n")
print(summary_table)

# --- Forest plot ---
summary_table$Method <- factor(summary_table$Method,
                                levels = rev(summary_table$Method))

ggplot(summary_table, aes(x = Estimate, y = Method, color = Covers)) +
  geom_point(size = 3) +
  geom_errorbarh(aes(xmin = CI_Lo, xmax = CI_Hi), height = 0.2) +
  geom_vline(xintercept = theta0, linetype = "dashed") +
  scale_color_manual(values = c("TRUE" = "#27AE60", "FALSE" = "#E74C3C"),
                     labels = c("TRUE" = "Covers truth",
                                "FALSE" = "Misses truth")) +
  labs(title    = "Four-Method Comparison: Point Estimates and 95% Intervals",
       subtitle = "Nonlinear DGP | n = 500 | p = 50 | True theta = 1",
       x = "Estimated treatment effect", y = NULL,
       color = NULL) +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom")
