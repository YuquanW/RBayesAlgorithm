// File: ancova_conj_path.stan
data {
  real<lower = 0, upper = 1> wknot;

  // Historical data
  int<lower=1> p;
  real tau_hst;
  real<lower=0> se_hst;

  // Baseline (vague) NIG prior
  real tau_vag;
  real<lower=0> se_vag;
}
parameters {
  real tau;
}
model {
  // Power prior
  target += wknot * normal_lpdf(tau | tau_hst, se_hst)
             + normal_lpdf(tau | tau_vag, se_vag);
}
generated quantities {
  real ll = normal_lpdf(tau | tau_hst, se_hst);
}