// File: ancova_conj_mp.stan
data {
  int<lower=1> n;
  int<lower=1> p;
  vector[n] Y;
  matrix[n, p] X;

  // NIG prior (from historical data)
  vector[p] beta_hst;
  matrix[p, p] L_Sigma_hst;   // lower Cholesky of Sigma_hst
  real<lower=0> a_hst;
  real<lower=0> b_hst;

  // Vague NIG
  vector[p] beta_vag;
  matrix[p, p] L_Sigma_vag;   // lower Cholesky of Sigma_vag
  real<lower=0> a_vag;
  real<lower=0> b_vag;

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
    real lp_ancova_conj_hst = multi_normal_cholesky_lpdf(beta | beta_hst, sqrt(sigma2) * L_Sigma_hst)
                  + inv_gamma_lpdf(sigma2 | a_hst, b_hst);
    real lp_ancova_conj_vag = multi_normal_cholesky_lpdf(beta | beta_vag, sqrt(sigma2) * L_Sigma_vag)
                  + inv_gamma_lpdf(sigma2 | a_vag, b_vag);
    target += log_mix(w_eff, lp_inf, lp_vag);
  }

  // Likelihood
  y ~ normal(X * beta, sigma);
}
