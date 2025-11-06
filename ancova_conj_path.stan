// File: ancova_conj_path.stan
data {
  real<lower = 0, upper = 1> wknot;

  // Historical data
  int<lower=1> p;
  vector[p] beta_hst;
  matrix[p, p] L_Sigma_hst;

  // Baseline (vague) NIG prior
  vector[p] beta_vag;
  matrix[p, p] L_Sigma_vag;
}
parameters {
  vector[p] beta;
}
model {
  // Power prior
  target += wknot * multi_normal_cholesky_lpdf(beta | beta_hst, L_Sigma_hst)
             + multi_normal_cholesky_lpdf(beta | beta_vag, L_Sigma_vag);
}
generated quantities {
  real ll = multi_normal_cholesky_lpdf(beta | beta_hst, L_Sigma_hst);
}