rm(list = ls())
library(rstan)
library(parallel)
options(mc.cores = 4)
source("Utils.R")
load("Historical.RData")
load("ancova_mixture_prior.RData")
load("ancova_power_prior.RData")

simulation <- function(i, delta) {
  set.seed(123 + i)
  e <- rnorm(n_cur, sd = sqrt(var(hba1c_baseline + alpha + tau*treatment + gamma*hba1c_baseline)))
  hba1c <- hba1c_baseline + alpha + (tau + delta)*treatment + gamma*hba1c_baseline + e
  data_cur <- data.frame(hba1c_baseline, treatment, hba1c)
  
  data_fmp <- build_ancova_data(data_cur, data_hst, 
                                I(hba1c - hba1c_baseline) ~ hba1c_baseline + treatment)
  ancova_fmp_fit <- run_model(mixture_model, data_fmp, seed = NA, save_warmup = TRUE)
  
  data_rmp <- build_ancova_data(data_cur, data_hst, 
                                I(hba1c - hba1c_baseline) ~ hba1c_baseline + treatment, 
                                estimate_w = TRUE)
  ancova_rmp_fit <- run_model(mixture_model, data_rmp, seed = NA, save_warmup = TRUE)
  
  data_fpp <- build_ancova_data(data_cur, data_hst, 
                                I(hba1c - hba1c_baseline) ~ hba1c_baseline + treatment)
  ancova_fpp_fit <- run_model(power_model, data_fpp, seed = NA, save_warmup = TRUE)
  
  data_upp <- build_ancova_data(data_cur, data_hst, 
                                I(hba1c - hba1c_baseline) ~ hba1c_baseline + treatment, 
                                estimate_w = TRUE, normalization = FALSE)
  ancova_upp_fit <- run_model(power_model, data_upp, seed = NA, save_warmup = TRUE)
  
  data_anpp <- build_ancova_data(data_cur, data_hst, 
                                 I(hba1c - hba1c_baseline) ~ hba1c_baseline + treatment, 
                                 estimate_w = TRUE, normalization = TRUE, 
                                 exact_constant = FALSE, 
                                 wknots = knots$wknots, lgCknots = knots$lgCknots)
  ancova_anpp_fit <- run_model(power_model, data_anpp, seed = NA, save_warmup = TRUE)
  
  data_npp <- build_ancova_data(data_cur, data_hst, 
                                I(hba1c - hba1c_baseline) ~ hba1c_baseline + treatment, 
                                estimate_w = TRUE, normalization = TRUE, 
                                exact_constant = TRUE)
  ancova_npp_fit <- run_model(power_model, data_npp, seed = NA, save_warmup = TRUE)
  
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
delta <- c(-0.6, -0.3, 0, 0.3, 0.6)
n <- 200
n_cur <- 82
n_hst <- n - n_cur
alpha <- 0.5
tau <- 0.6
gamma <- 0.1

treatment <- rbinom(n_cur, 1, 0.5)
hba1c_baseline <- rnorm(n_cur, 45, 0.5)

r <- 2
simulation(1, delta[1])
for (k in 1:length(delta)) {
  cl <- makeCluster(detectCores() -1)
  clusterExport(cl, varlist = ls())
  clusterEvalQ(cl, {
    library(rstan)
  })
  parLapply(cl, 1:r, function(i) simulation(i, delta[k]))
  stopCluster(cl)
}



