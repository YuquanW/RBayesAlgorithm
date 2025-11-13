rm(list = ls())
library(rstan)
library(parallel)
rstan_options(auto_write = TRUE)
options(mc.cores = 1)
source("Utils.R")
load("Historical.RData")
#load("ancova_mixture_prior.RData")
#load("ancova_power_prior.RData")
mixture_model <- build_ancova_model("mixture")
power_model <- build_ancova_model("power")

simulation <- function(i, delta) {
  set.seed((123 + i)*k)
  e <- rnorm(n_cur, sd = sqrt(1 - tau^2*0.25 - (1+gamma)^2*0.25))
  hba1c <- hba1c_baseline + alpha + (tau + delta)*treatment + gamma*hba1c_baseline + e
  data_cur <- data.frame(hba1c_baseline, treatment, hba1c)
  
  data_fmp <- build_ancova_data(data_cur, data_hst,
                                I(hba1c - hba1c_baseline) ~ hba1c_baseline + treatment,
                                w_fixed = w_eff)
  ancova_fmp_fit <- run_model(mixture_model, data_fmp,
                              seed = 123,
                              iter = 3500, burnin = 1000,
                              save_warmup = TRUE)
  
  data_fpp <- build_ancova_data(data_cur, data_hst, 
                                I(hba1c - hba1c_baseline) ~ hba1c_baseline + treatment,
                                w_fixed = w_eff)
  ancova_fpp_fit <- run_model(power_model, data_fpp,
                              seed = 123,
                              iter = 3500, burnin = 1000,
                              save_warmup = TRUE)
  
  stan_fits <- list(ancova_fmp_fit,
                    ancova_fpp_fit)
  eff_col <- "treatment"
  model_names <- c("FMP", "FPP")
  beta0 <- tau + delta
  
  write_res2(stan_fits, model_names, eff_col, beta0, delta, w_eff)
}

set.seed(123)
delta <- c(-0.6, 0)
w <- seq(0, 1, 0.1)
n <- 144
n_cur <- 44
n_hst <- n - n_cur
alpha <- 0.5
tau <- 0.6
gamma <- 0.1

treatment <- c(rep(1, n_cur/2), rep(0, n_cur/2))
hba1c_baseline <- rnorm(n_cur)
hba1c_baseline <- scale(hba1c_baseline)*0.5 + 45

r <- 1000
for (k in 1:length(delta)) {
  for (l in 1:length(w)) {
    w_eff <- w[l]
    cl <- makeCluster(100)
    clusterExport(cl, varlist = ls())
    clusterEvalQ(cl, {
      library(rstan)
    })
    parLapply(cl, 1:r, function(i) simulation(i, delta[k]))
    stopCluster(cl)
  }
}



