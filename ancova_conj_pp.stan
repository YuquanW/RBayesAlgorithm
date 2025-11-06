// File: ancova_conj_pp.stan
functions{
  real interpolateC(real w, int K, vector wknots, vector lgCknots){
    real logCest;
    for(id in 1:(K-1)){
      if(w >= wknots[id] && w < wknots[id+1]){
        logCest = lgCknots[id] + (w-wknots[id])*(lgCknots[id+1]-lgCknots[id])/(wknots[id+1]-wknots[id]);
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

  // Historical prior
  vector[p] beta_hst;
  matrix[p, p] L_Sigma_hst;

  // Baseline (vague) prior
  vector[p] beta_vag;
  matrix[p, p] L_Sigma_vag;

  // Power prior control
  real<lower=0, upper=1> w_fixed; // used if estimate_w==0
  int<lower=0, upper=1> estimate_w;
  int<lower=0, upper=1> normalization;
  int<lower=0, upper=1> exact_constant;

  // Normalized power prior constant
  int<lower=1> K;
  vector<lower=0, upper=1>[K] wknots;
  vector[K] lgCknots;
}
transformed data {
  matrix[p, p] Sigma_hst = L_Sigma_hst * L_Sigma_hst';
  matrix[p, p] Sigma_vag = L_Sigma_vag * L_Sigma_vag';
}
parameters {
  vector[p] beta;
  real<lower=0> sigma2;
  real<lower=0, upper=1> w;      // only used if estimate_w==1
}
transformed parameters {
  real sigma = sqrt(sigma2);
  real<lower=0, upper=1> w_eff = estimate_w == 1 ? w : w_fixed;
}
model {
  // Optional prior on w
  if (estimate_w == 1) w ~ beta(1, 1);

  // Optional normalization
  real lgC = normalization*(exact_constant*(-0.5*w_eff*p*log(2*pi())-0.5*w_eff*log(determinant(Sigma_hst))
               -0.5*p*log(w_eff)+multi_normal_lpdf(beta_hst | beta_vag, Sigma_hst/w_eff+Sigma_vag)) 
               + (1-exact_constant)*interpolateC(w_eff, K, wknots, lgCknots));

  // Power prior
  target += w_eff * multi_normal_cholesky_lpdf(beta | beta_hst, L_Sigma_hst)
             + multi_normal_cholesky_lpdf(beta | beta_vag, L_Sigma_vag)
             + normal_id_glm_lpdf(Y | X, 0, beta, sigma) - lgC - log(sigma2);
}
