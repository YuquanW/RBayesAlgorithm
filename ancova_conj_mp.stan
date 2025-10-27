// File: ancova_conj_mp.stan
data {
  int<lower=1> n;
  int<lower=1> p;
  vector[n] Y;
  matrix[n, p] X;

  // NIG prior (from historical data)
  vector[p] beta_hst;
  matrix[p, p] Sigma_hst;   // lower Cholesky of V_inf

  // Vague NIG
  vector[p] beta_vag;
  matrix[p, p] Sigma_vag;   // lower Cholesky of V_vag

  // Mixture weight control
  real<lower=0, upper=1> w_fixed;  // used if estimate_w==0
  int<lower=0, upper=1> estimate_w;
}
parameters {
  vector[p] beta;
  real<lower=0> sigma2;
  real<lower=0, upper=1> w;  // only used if estimate_w==1
}
transformed parameters {
  real<lower=0, upper=1> w_eff = estimate_w == 1 ? w : w_fixed;
  real sigma = sqrt(sigma2);
}
model {
  // optional prior on w
  if (estimate_w == 1) w ~ beta(1, 1);

  // Mixture NIG prior on (beta, sigma2): mixture of joint densities
  {
    real lp_inf = multi_normal_cholesky_lpdf(beta | m_inf, sqrt(sigma2) * L_V_inf)
                  + inv_gamma_lpdf(sigma2 | alpha_inf, beta_inf);
    real lp_vag = multi_normal_cholesky_lpdf(beta | m_vag, sqrt(sigma2) * L_V_vag)
                  + inv_gamma_lpdf(sigma2 | alpha_vag, beta_vag);
    target += log_mix(w_eff, lp_inf, lp_vag);
  }

  // Likelihood
  y ~ normal(X * beta, sigma);
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | dot_product(row(X, n), beta), sqrt(sigma2));
  }
}
