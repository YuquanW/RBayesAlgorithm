// Bayesian CMH: conditional (noncentral hypergeometric) likelihood
// with a mixture prior on tau = log(common OR).
functions {
  // log binomial coefficient C(n,k) via lgamma; works for ints
  real lchoose_int(int n, int k) {
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
  }

  // log normalizing constant for a single stratum
  // Z_s(tau) = sum_{k=L..U} C(n1, k) C(n0, m - k) * exp(k * tau)
  real log_Z_stratum(real tau, int n1, int n0, int m) {
    int L = m - n0 > 0 ? m - n0 : 0;
    int U = n1 < m ? n1 : m;
    int K = U - L + 1;
    vector[K] lp;
    for (k in 0:(K - 1)) {
      int kk = L + k;
      lp[k + 1] = lchoose_int(n1, kk)
                + lchoose_int(n0, m - kk)
                + kk * tau;
    }
    return log_sum_exp(lp);
  }
}
data {
  int<lower=1> S;            // number of strata
  // 2x2 per stratum: Treated successes a_s, failures b_s; Control successes c_s, failures d_s
  int<lower=0> a[S];
  int<lower=0> b[S];
  int<lower=0> c[S];
  int<lower=0> d[S];

  // Mixture prior parameters for tau
  real mu_inf;                 // informative prior mean (e.g., log MH OR from history)
  real<lower=0> sigma_inf;     // informative prior sd
  real mu_vague;               // vague prior mean (often 0)
  real<lower=0> sigma_vague;   // vague prior sd (e.g., 2–5 on log-OR)

  // Mixture weight control
  real<lower=0, upper=1> w_fixed;   // used if estimate_w == 0
  int<lower=0, upper=1> estimate_w; // 1 = learn w, 0 = fix w
  real<lower=0> w_alpha;            // Beta prior alpha for w
  real<lower=0> w_beta;             // Beta prior beta for w
}
transformed data {
  int n1[S];
  int n0[S];
  int m[S];
  for (s in 1:S) {
    n1[s] = a[s] + b[s];
    n0[s] = c[s] + d[s];
    m[s]  = a[s] + c[s];
  }
}
parameters {
  real tau;                          // common log-odds ratio
  real<lower=0, upper=1> w;          // mixture weight if estimate_w==1
}
transformed parameters {
  real w_eff = estimate_w == 1 ? w : w_fixed;
}
model {
  // Optional prior on w
  if (estimate_w == 1) w ~ beta(w_alpha, w_beta);

  // Mixture prior on tau (robust MAP / rMAP style)
  target += log_mix(
    w_eff,
    normal_lpdf(tau | mu_inf,  sigma_inf),
    normal_lpdf(tau | mu_vague, sigma_vague)
  );

  // Conditional (CMH) likelihood across strata
  for (s in 1:S) {
    target += a[s] * tau - log_Z_stratum(tau, n1[s], n0[s], m[s]);
  }
}
generated quantities {
  real theta = exp(tau);  // common odds ratio
}
