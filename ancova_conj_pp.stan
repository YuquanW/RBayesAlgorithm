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
  real tau_hst;
  real<lower=0> se_hst;

  // Baseline (vague) prior
  real tau_vag;
  real<lower=0> se_vag;

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
parameters {
  vector[p] beta;
  real<lower=0> sigma2;
  real<lower=0, upper=1> w;      // only used if estimate_w==1
}
transformed parameters {
  real tau = beta[p];
  real sigma = sqrt(sigma2);
  real<lower=0, upper=1> w_eff = estimate_w == 1 ? w : w_fixed;
}
model {
  // Optional prior on w
  if (estimate_w == 1) w ~ beta(1, 1);

  // Optional normalization
  real lgC = normalization*(exact_constant*(-0.5*w*log(2*pi())-0.5*w*log(se_hst^2)
               -0.5*log(w)+normal_lpdf(tau_hst | tau_vag, sqrt(se_hst^2/w+se_vag))) 
               + (1-exact_constant)*interpolateC(w, K, wknots, lgCknots));

  // Power prior
  target += w_eff * normal_lpdf(tau | tau_hst, se_hst)
             + normal_lpdf(tau | tau_vag, se_vag)
             + normal_id_glm_lpdf(Y | X, 0, beta, sigma) - lgC - log(sigma2);
}

generated quantities {
  real ll = normal_id_glm_lpdf(Y | X, 0, beta, sigma);
}
