// File: ancova_conj_pp.stan
data {
  // Current data
  int<lower=1> N;
  int<lower=1> K;
  vector[N] y;
  matrix[N, K] X;

  // Historical data (same columns/order as X)
  int<lower=0> N0;
  vector[N0] y0;
  matrix[N0, K] X0;

  // Baseline (vague) NIG prior
  vector[K] m00;
  matrix[K, K] L_V00;   // lower Cholesky of V00
  real<lower=0> alpha00;
  real<lower=0> beta00;

  // Power prior control
  real<lower=0, upper=1> a0_fixed; // used if estimate_a0==0
  int<lower=0, upper=1> estimate_a0;

  // Historical max log-likelihood (MLE): ll0_hat = max_{beta,sigma} log p(y0|X0,beta,sigma)
  real ll0_hat;
}
parameters {
  vector[K] beta;
  real<lower=0> sigma2;
  real<lower=0, upper=1> a0;      // only used if estimate_a0==1
}
transformed parameters {
  real<lower=0, upper=1> a0_eff = estimate_a0 == 1 ? a0 : a0_fixed;
  real sigma = sqrt(sigma2);
}
model {
  // Baseline NIG prior
  beta  ~ multi_normal_cholesky(m00, sqrt(sigma2) * L_V00);
  sigma2 ~ inv_gamma(alpha00, beta00);

  // Optional prior on a0
  if (estimate_a0 == 1) a0 ~ beta(1, 1);

  // Normalized (conditional) power prior contribution
  if (N0 > 0) {
    target += a0_eff * (normal_lpdf(y0 | X0 * beta, sigma) - ll0_hat);
  }

  // Current likelihood
  y ~ normal(X * beta, sigma);
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | dot_product(row(X, n), beta), sqrt(sigma2));
  }
}
