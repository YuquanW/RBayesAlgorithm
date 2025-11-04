// File: ancova_conj_pp.stan
functions{
  real interpolateC(real w, int K, vector wknots, vector lgCknots){
    real logCest;
    for(id in 1:(K-1)){
      if(w >= wknots[id] && w < wknots[id+1]){
        logCest = lgCknots[id]+ (w-wknots[id])*(lgCknots[id+1]-lgCknots[id])/(wknots[id+1]-wknots[id]);
      } 
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

  // Normalized power prior constant
  int<lower=1> K;
  vector<lower=0, upper=1>[K] wknots;
  vector[K] lgCknots;

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

  // Optional normalization
  real lgC = interpolateC(w, K, wknots, lgCknots);

  // Power prior
  target += w_eff * normal_id_glm_lpdf(Y_hst | X_hst, 0, beta, sigma)
              + normal_id_glm_lpdf(Y | X, 0, beta, sigma) - lgC;
}
