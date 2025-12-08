parameters {
  vector[S] eta0;        // p0 on logit scale
  real z;                // unconstrained seed for delta
  real<lower=0,upper=1> w;  // mixture weight if estimated
}
transformed parameters {
  vector[S] p0 = inv_logit(eta0);

  // Compute global feasible bounds L < 0 < U
  real L = -p0[1];
  real U = 1 - p0[1];
  for (s in 2:S) {
    L = fmax(L, -p0[s]);           // max over strata
    U = fmin(U, 1 - p0[s]);        // min over strata
  }
  // Map z -> (L,U)
  real il = inv_logit(z);
  real delta = L + (U - L) * il;

  // If you place a prior *on delta itself*, add the Jacobian:
  real log_jac = log(U - L) + log(il) + log1m(il);  // = log(U-L) + log σ(z) + log(1-σ(z))
}
model {
  // ... priors on p0 (e.g., eta0 normal or p0 beta) ...

  // Mixture prior on delta (on RD scale) + Jacobian
  target += log_mix(
              w, normal_lpdf(delta | mu_inf, sigma_inf),
                 normal_lpdf(delta | mu_vague, sigma_vague)
            ) + log_jac;

  // Likelihood
  for (s in 1:S) {
    a[s] ~ binomial(n1[s], p0[s] + delta);
    c[s] ~ binomial(n0[s], p0[s]);
  }
}
