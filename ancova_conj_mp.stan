// File: ancova_conj_mp.stan
data {
  int<lower=1> n;
  int<lower=1> p;
  vector[n] Y;
  matrix[n, p] X;

  // Historical prior
  vector[p] beta_hst;
  matrix[p, p] L_Sigma_hst;

  // Vague prior
  vector[p] beta_vag;
  matrix[p, p] L_Sigma_vag;

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
  real<lower=0> sigma = sqrt(sigma2);
  real<lower=0, upper=1> w_eff = estimate_w == 1 ? w : w_fixed;
}
model {
  // optional prior on w
  if (estimate_w == 1) w ~ beta(1, 1);

  // Mixture prior on beta and improper prior on sigma2
  {
    target += log_mix(w_eff, 
                      multi_normal_cholesky_lpdf(beta | beta_hst, L_Sigma_hst), 
                      multi_normal_cholesky_lpdf(beta | beta_vag, L_Sigma_vag))
               + normal_id_glm_lpdf(Y | X, 0, beta, sigma) - log(sigma2);
  }
}

generated quantities {
  real ll = normal_id_glm_lpdf(Y | X, 0, beta, sigma);
}