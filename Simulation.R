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
                                I(hba1c - hba1c_baseline) ~ hba1c_baseline + treatment)
  ancova_fmp_fit <- run_model(mixture_model, data_fmp,
                              iter = 3500, burnin = 1000,
                              save_warmup = TRUE)
  
  data_rmp <- build_ancova_data(data_cur, data_hst, 
                                I(hba1c - hba1c_baseline) ~ hba1c_baseline + treatment, 
                                estimate_w = TRUE)
  ancova_rmp_fit <- run_model(mixture_model, data_rmp,
                              iter = 3500, burnin = 1000,
                              save_warmup = TRUE)
  
  data_fpp <- build_ancova_data(data_cur, data_hst, 
                                I(hba1c - hba1c_baseline) ~ hba1c_baseline + treatment)
  ancova_fpp_fit <- run_model(power_model, data_fpp,
                              iter = 3500, burnin = 1000,
                              save_warmup = TRUE)
  
  data_upp <- build_ancova_data(data_cur, data_hst, 
                                I(hba1c - hba1c_baseline) ~ hba1c_baseline + treatment, 
                                estimate_w = TRUE, normalization = FALSE)
  ancova_upp_fit <- run_model(power_model, data_upp,
                              iter = 3500, burnin = 1000,
                              save_warmup = TRUE)
  
  data_anpp <- build_ancova_data(data_cur, data_hst, 
                                 I(hba1c - hba1c_baseline) ~ hba1c_baseline + treatment, 
                                 estimate_w = TRUE, normalization = TRUE, 
                                 exact_constant = FALSE, 
                                 wknots = knots$wknots, lgCknots = knots$lgCknots)
  ancova_anpp_fit <- run_model(power_model, data_anpp,
                               iter = 3500, burnin = 1000,
                               save_warmup = TRUE)
  
  data_npp <- build_ancova_data(data_cur, data_hst, 
                                I(hba1c - hba1c_baseline) ~ hba1c_baseline + treatment, 
                                estimate_w = TRUE, normalization = TRUE, 
                                exact_constant = TRUE)
  ancova_npp_fit <- run_model(power_model, data_npp,
                              iter = 3500, burnin = 1000,
                              save_warmup = TRUE)
  
  ancova_fit <- lm(I(hba1c - hba1c_baseline) ~ hba1c_baseline + treatment, data_cur)
  
  stan_fits <- list(ancova_fmp_fit,
                    ancova_rmp_fit,
                    ancova_fpp_fit,
                    ancova_upp_fit,
                    ancova_anpp_fit,
                    ancova_npp_fit)
  lm_fits <- list(ancova_fit)
  eff_col <- "treatment"
  model_names <- c("FMP", "RMP", "FPP", "UPP", "ANPP", "NPP", "ANCOVA")
  beta0 <- tau + delta
  
  write_res(stan_fits, lm_fits, model_names, eff_col, beta0, delta)
}

set.seed(123)
delta <- seq(-0.6, 0, 0.05)
n <- 156
n_cur <- 56
n_hst <- n - n_cur
alpha <- 0.5
tau <- 0.6
gamma <- 0.1

treatment <- c(rep(1, n_cur/2), rep(0, n_cur/2))
hba1c_baseline <- rnorm(n_cur)
hba1c_baseline <- scale(hba1c_baseline)*0.5 + 45

r <- 1000
for (k in 1:length(delta)) {
  cl <- makeCluster(100)
  clusterExport(cl, varlist = ls())
  clusterEvalQ(cl, {
    library(rstan)
  })
  parLapply(cl, 1:r, function(i) simulation(i, delta[k]))
  stopCluster(cl)
}



