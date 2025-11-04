// File: ancova_conj_path.stan
data {
  real<lower = 0, upper = 1> wknot;

  // Historical data
  int<lower=0> n_hst;
  int<lower=1> p_hst;
  vector[n_hst] Y_hst;
  matrix[n_hst, p_hst] X_hst;

  // Baseline (vague) NIG prior
  vector[p_hst] beta_vag;
  matrix[p_hst, p_hst] L_Sigma_vag;   // lower Cholesky of vague covariance matrix
  real<lower=0> a_vag;
  real<lower=0> b_vag;
}
parameters {
  vector[p_hst] beta;
  real<lower=0> sigma2;
}
model {
  sigma2 ~ inv_gamma(a_vag, b_vag);
  beta  ~ multi_normal_cholesky(beta_vag, sqrt(sigma2) * L_Sigma_vag);

  // Power prior
  target += wknot * normal_id_glm_lpdf(Y_hst | X_hst, 0, beta, sqrt(sigma2));
}
generated quantities {
  real ll = normal_id_glm_lpdf(Y_hst | X_hst, 0, beta, sqrt(sigma2));
}