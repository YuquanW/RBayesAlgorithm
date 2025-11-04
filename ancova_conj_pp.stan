// File: ancova_conj_pp.stan
functions{
  real interpolateC(real deltaCur, int numdelta, vector deltaKnot, vector logCKnot){
  real logCest;
  for(id in 1:(numdelta-1)){
    if(deltaCur >= deltaKnot[id] && deltaCur < deltaKnot[id+1]){
      logCest = logCKnot[id]+ (deltaCur-deltaKnot[id])*(logCKnot[id+1]-logCKnot[id])/(deltaKnot[id+1]-deltaKnot[id]);
    }  // Interpolation function, given a sequence of logCKnot
  }
  return logCest;
}
}
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

  // Unnormalized power prior
  target += w_eff * normal_id_glm_lpdf(Y_hst | X_hst, 0, beta, sigma)
              + normal_id_glm_lpdf(Y | X, 0, beta, sigma);
}
