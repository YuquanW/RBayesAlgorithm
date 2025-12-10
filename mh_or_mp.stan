functions {
  real fisher_nch_precomp_lpmf(
    int n11k,
    int len,
    int[] x,
    real[] log_comb,
    real tau
  ) {
    real log_Z = negative_infinity();
    real log_T = negative_infinity();

    for (j in 1:len) {
      int n11k_j = x[j];                     // candidate n11
      real t = log_comb[j] + tau * n11k_j;   // log unnormalized weight

      if (t > log_Z)
        log_Z = t + log1p_exp(log_Z - t);
      else
        log_Z = log_Z + log1p_exp(t - log_Z);

      if (n11k_j == n11k)
        log_T = t;
    }
    return log_T - log_Z;
  }
}

data {
  int<lower=1> K;                   // number of strata

  // 2x2 counts:
  int<lower=0> n11[K];              // successes, treatment
  int<lower=0> n12[K];              // failures, treatment
  int<lower=0> n21[K];              // successes, control
  int<lower=0> n22[K];              // failures, control

  // for precomputed arrays
  int<lower=1> max_len;             // max support length across strata

  // mixture prior parameters for tau (log common OR)
  real mu_inf;
  real<lower=0> sigma_inf;
  real mu_vague;
  real<lower=0> sigma_vague;

  real<lower=0, upper=1> w;
}

transformed data {
  // margins
  int n1dot[K];   // n_{1.k}
  int n2dot[K];   // n_{2.k}
  int ndot1[K];   // n_{.1k}
  int ndot2[K];   // n_{.2k} (not needed in likelihood, but OK to store)

  int L[K];
  int U[K];
  int len[K];

  // precomputed support and log combinatorial term
  int x[K, max_len];       // support values for n11k
  real log_comb[K, max_len];   // log[C(n1dot, x) C(n2dot, ndot1 - x)]

  for (k in 1:K) {
    n1dot[k] = n11[k] + n12[k];
    n2dot[k] = n21[k] + n22[k];
    ndot1[k] = n11[k] + n21[k];
    ndot2[k] = n12[k] + n22[k];

    // support bounds for n11k given margins
    L[k] = ndot1[k] - n2dot[k] > 0 ? ndot1[k] - n2dot[k] : 0;
    U[k] = n1dot[k] < ndot1[k] ? n1dot[k] : ndot1[k];
    len[k] = U[k] - L[k] + 1;

    // fill support and log combinatorial part
    for (j in 1:max_len) {
      if (j <= len[k]) {
        int n11k_j = L[k] + (j - 1);
        x[k, j] = n11k_j;
        log_comb[k, j] =
          lchoose(n1dot[k], n11k_j)
        + lchoose(n2dot[k], ndot1[k] - n11k_j);
      } else {
        x[k, j] = 0;
        log_comb[k, j] = negative_infinity(); // never used
      }
    }
  }
}

parameters {
  real tau;                           // common log odds ratio
}

model {
  // mixture prior on tau
  target += log_mix(
    w,
    normal_lpdf(tau | mu_inf,   sigma_inf),
    normal_lpdf(tau | mu_vague, sigma_vague)
  );

  // conditional CMH likelihood via precomputed FNHG pieces
  for (k in 1:K) {
    target += fisher_nch_precomp_lpmf(
      n11[k] | len[k], x[k], log_comb[k], tau
    );
  }
}

generated quantities {
  real theta = exp(tau);  // common odds ratio
}
