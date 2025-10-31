// File: ancova_conj_pp.stan
data {
  // Current data
  int<lower=1> n;
  int<lower=1> p;
  vector[n] Y;
  matrix[n, p] X;

  // Historical data (same columns/order as X)
  int<lower=0> n_hst;
  int<lower=1> p_hst;
  vector[n_hst] Y_hst;
  matrix[n_hst, p_hst] X_hst;

  // Baseline (vague) NIG prior
  vector[p] beta_vag;
  matrix[p, p] L_Sigma_vag;   // lower Cholesky of vague covariance matrix
  real<lower=0> a_vag;
  real<lower=0> b_vag;

  // Power prior control
  real<lower=0, upper=1> w_fixed; // used if estimate_w==0
  int<lower=0, upper=1> estimate_w;

  // Historical max log-likelihood (MLE): ll0_hat = max_{beta,sigma} log p(y0|X0,beta,sigma)
  //real ll0_hat;
}
parameters {
  vector[p] beta;
  real<lower=0> sigma2;
  real<lower=0, upper=1> w;      // only used if estimate_w==1
}
transformed parameters {
  real<lower=0, upper=1> w_eff = estimate_w == 1 ? w : w_fixed;
  real sigma = sqrt(sigma2);
}
model {
  // Baseline NIG prior
  beta  ~ multi_normal_cholesky(beta_vag, sigma * L_Sigma_vag);
  sigma2 ~ inv_gamma(a_vag, b_vag);

  // Optional prior on w
  if (estimate_w == 1) w ~ beta(1, 1);

  // Normalized (conditional) power prior contribution
  if (n_hst - p > 0) {
    target += w_eff * normal_lpdf(Y_hst | X_hst * beta, sigma);
  }

  // Current likelihood
  Y ~ normal(X * beta, sigma);
}