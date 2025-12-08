# --- Util: Build design matrices ---
mk_design <- function(df, formula) {
  X <- model.matrix(formula, df)
  Y <- model.response(model.frame(formula, df))
  list(Y = as.numeric(Y), X = X, cols = colnames(X))
}

# --- Util: Run Stan model ---

run_model <- function(model, data,
                      iter = 5000, burnin = iter/2, chains = 4, seed = 123, cores = getOption("mc.cores", 1L), refresh = max(iter/10, 1), save_warmup = FALSE) {
  suppressWarnings({
    fit <- sampling(model, data, iter = iter, warmup = burnin, chains = chains, seed = seed, cores = cores, refresh = refresh, save_warmup = save_warmup)
  })
  return(list(fit = fit, data = data))
}

# --- Util: Path sampling ---
build_ancova_path_data <- function(df_hst, formula,
                                   tau_vag = NULL, se_vag = NULL) {
  des_hst <- mk_design(df_hst, formula)
  p <- ncol(des_hst$X)
  
  # historical informative prior (scaled by a0)
  fit_hst <- lm(formula, df_hst)
  tau_hst <- coef(fit_hst)[p]
  se_hst <- sqrt(sandwich::vcovHC(fit_hst)[p, p])
  
  # vague prior
  if (is.null(tau_vag)) {
    tau_vag <- tau_hst
  }
  if (is.null(se_vag)) {
    se_vag <- sqrt(nrow(df_hst))*se_hst
  }
  
  list(
    cols = des_hst$cols,
    p = ncol(des_hst$X),
    tau_hst = tau_hst,
    se_hst = se_hst,
    tau_vag = tau_vag,
    se_vag = se_vag
  )
}

build_ancova_path_model <- function() {
  stan_file <- "ancova_conj_path.stan"
  model <- stan_model(file = stan_file)
  return(model)
}

ancova_path_integrate <- function(model, df_hst, formula, wknots = (0:200/200)^2,
                                  tau_vag = NULL, se_vag = NULL,
                                  maxiter = 50000, chains = 1, seed = 123, 
                                  refresh = 0, save_warmup = FALSE) {
  K <- length(wknots)
  mlls <- rep(NA, K)
  ancova_path_dt <- build_ancova_path_data(df_hst, formula,
                                           tau_vag, se_vag)
  
  for (i in 1:K) {  # Consider use parLapply here
    ancova_path_dt$wknot <- wknots[i]
    iter <- max(ceiling(maxiter^(1-wknots[i])), 5000)
    fit <- run_model(model, ancova_path_dt,
                     iter, round(iter/2), chains, seed, 
                     refresh = refresh,
                     save_warmup = save_warmup)
    mlls[i] <- mean(extract(fit$fit)$ll)
    cat(sprintf("Knot %i has finished, %i MCMC samples have been drawn.\n", i, iter))
  }
  dvec <- diff(wknots)
  lgCknots <- c(0, cumsum(dvec*(mlls[1:(K-1)]+mlls[2:K])/2))
  return(list(wknots = wknots, lgCknots = lgCknots))
}

# --- Build Stan data ---
build_ancova_data <- function(df_cur, df_hst, formula,
                              w_fixed = 0.5, estimate_w = FALSE, normalization = TRUE,
                              tau_vag = NULL, se_vag = NULL,
                              wknots = c(0, 1), lgCknots = c(0, 0), exact_constant = TRUE) {
  
  des_cur <- mk_design(df_cur, formula)
  p <- ncol(des_cur$X)
  
  # historical informative prior (scaled by a0)
  fit_hst <- lm(formula, df_hst)
  tau_hst <- coef(fit_hst)[p]
  se_hst <- sqrt(sandwich::vcovHC(fit_hst)[p, p])
  
  # vague prior
  if (is.null(tau_vag)) {
    tau_vag <- tau_hst
  }
  if (is.null(se_vag)) {
    se_vag <- sqrt(nrow(df_hst))*se_hst
  }
  
  list(
    cols = des_cur$cols,
    n = nrow(des_cur$X),
    p = ncol(des_cur$X),
    Y = des_cur$Y,
    X = des_cur$X,
    tau_hst = tau_hst,
    se_hst = se_hst,
    tau_vag = tau_vag,
    se_vag = se_vag,
    w_fixed = w_fixed,
    estimate_w = estimate_w,
    normalization = normalization,
    K = length(wknots),
    wknots = wknots,
    lgCknots = lgCknots,
    exact_constant = exact_constant
  )
}

# --- Build Stan model ---
build_ancova_model <- function(prior_type = c("mixture", "power")) {
  prior_type <- match.arg(prior_type)
  stan_file <- ifelse(prior_type == "mixture", "ancova_conj_mp.stan", "ancova_conj_pp.stan")
  model <- stan_model(file = stan_file)
  return(model)
}

# --- Yield simulation results ---
yield_stan_res <- function(fit_obj, eff_col, beta0) {
  sm <- summary(fit_obj$fit, pars = "beta")$summary
  iterations <- extract(fit_obj$fit, pars = "beta")$beta
  idx <- match(eff_col, fit_obj$dat$cols)
  summary_res <- sm[idx, c("mean", "2.5%", "97.5%", "Rhat")]
  c("Estimate" = summary_res[1],
    "CI.length" = summary_res[3] - summary_res[2],
    "Coverage" = 1*(beta0 >= summary_res[2] & beta0 <= summary_res[3]),
    "Rejection" = 1*(mean(iterations[, idx]<=0) < 0.025),
    "PP" = mean(iterations[, idx]<=0),
    "Rhat" = summary_res[4])
}

yield_lm_res <- function(fit_obj, eff_col, beta0) {
  summary_res <- coef(summary(fit_obj))[eff_col, ]
  lb <- summary_res[1] - qnorm(0.975)*summary_res[2]
  ub <- summary_res[1] + qnorm(0.975)*summary_res[2]
  c("Estimate" = summary_res[1],
    "CI.length" = ub - lb,
    "Coverage" = 1*(beta0 >= lb & beta0 <= ub),
    "Rejection" = 1*(summary_res[4]/2*(summary_res[1]>0)+(1-summary_res[4]/2)*(summary_res[1]<=0) < 0.025),
    "P.value" = summary_res[4]/2*(summary_res[1]>0)+(1-summary_res[4]/2)*(summary_res[1]<=0),
    "Rhat" = NA)
}

write_res <- function(stan_fits, lm_fits, model_names, eff_col, beta0, delta) {
  n_stan <- length(stan_fits)
  n_fit <- length(lm_fits)
  if (is.null(model_names)) {
    model_names <- paste0("M", 1:(n_stan + n_fit))
  }
  res_matrix <- matrix(NA, nrow = n_stan+n_fit, ncol = 6)
  
  for (i in 1:n_stan) {
    stan_fit_i <- stan_fits[[i]]
    res_matrix[i, ] <- yield_stan_res(stan_fit_i, eff_col, beta0)
  }
  for (j in 1:n_fit) {
    lm_fit_j <- lm_fits[[j]]
    res_matrix[i+j, ] <- yield_lm_res(lm_fit_j, eff_col, beta0)
  }
  rownames(res_matrix) <- model_names
  write.table(data.frame(t(res_matrix[, 1])), 
              file = sprintf("./simres/ancova_estimate_delta%.2f.txt", delta), 
              append = file.exists(sprintf("./simres/ancova_estimate_delta%.2f.txt", delta)), 
              quote = FALSE,
              sep = "\t",
              row.names = F,
              col.names = !file.exists(sprintf("./simres/ancova_estimate_delta%.2f.txt", delta)))
  write.table(data.frame(t(res_matrix[, 2])), 
              file = sprintf("./simres/ancova_cilength_delta%.2f.txt", delta), 
              append = file.exists(sprintf("./simres/ancova_cilength_delta%.2f.txt", delta)), 
              quote = FALSE,
              sep = "\t",
              row.names = F,
              col.names = !file.exists(sprintf("./simres/ancova_cilength_delta%.2f.txt", delta)))
  write.table(data.frame(t(res_matrix[, 3])), 
              file = sprintf("./simres/ancova_coverage_delta%.2f.txt", delta), 
              append = file.exists(sprintf("./simres/ancova_coverage_delta%.2f.txt", delta)), 
              quote = FALSE,
              sep = "\t",
              row.names = F,
              col.names = !file.exists(sprintf("./simres/ancova_coverage_delta%.2f.txt", delta)))
  write.table(data.frame(t(res_matrix[, 4])), 
              file = sprintf("./simres/ancova_rejection_delta%.2f.txt", delta), 
              append = file.exists(sprintf("./simres/ancova_rejection_delta%.2f.txt", delta)), 
              quote = FALSE,
              sep = "\t",
              row.names = F,
              col.names = !file.exists(sprintf("./simres/ancova_rejection_delta%.2f.txt", delta)))
  write.table(data.frame(t(res_matrix[, 5])), 
              file = sprintf("./simres/ancova_pvalue_delta%.2f.txt", delta), 
              append = file.exists(sprintf("./simres/ancova_pvalue_delta%.2f.txt", delta)), 
              quote = FALSE,
              sep = "\t",
              row.names = F,
              col.names = !file.exists(sprintf("./simres/ancova_pvalue_delta%.2f.txt", delta)))
  write.table(data.frame(t(res_matrix[, 6])), 
              file = sprintf("./simres/ancova_rhat_delta%.2f.txt", delta), 
              append = file.exists(sprintf("./simres/ancova_rhat_delta%.2f.txt", delta)), 
              quote = FALSE,
              sep = "\t",
              row.names = F,
              col.names = !file.exists(sprintf("./simres/ancova_rhat_delta%.2f.txt", delta)))
}

# --- Yield simulation results 2 ---
yield_stan_res2 <- function(fit_obj, eff_col, beta0) {
  sm <- summary(fit_obj$fit, pars = "beta")$summary
  iterations <- extract(fit_obj$fit, pars = "beta")$beta
  idx <- match(eff_col, fit_obj$dat$cols)
  summary_res <- sm[idx, c("mean", "2.5%", "97.5%", "Rhat")]
  
  y_hat <- drop(fit_obj$dat$X%*%sm[, "mean"])
  sigma_hat <- summary(fit_obj$fit, pars = "sigma")$summary[, "mean"]
  ll_mean <- summary(fit_obj$fit, pars = "ll")$summary[, "mean"]
  dev_mean <- -2*ll_mean
  dev_plug <- -2*sum(dnorm(fit_obj$dat$Y, y_hat, sigma_hat, log = T))
  p_d <- dev_mean - dev_plug
  dic <- dev_plug + 2*p_d
  c("Rejection" = 1*(mean(iterations[, idx]<=0) < 0.025),
    "PP" = mean(iterations[, idx]<=0),
    "DIC" = dic)
}

write_res2 <- function(stan_fits, model_names, eff_col, beta0, delta, w_eff) {
  n_stan <- length(stan_fits)
  if (is.null(model_names)) {
    model_names <- paste0("M", 1:n_stan)
  }
  res_matrix <- matrix(NA, nrow = n_stan, ncol = 3)
  
  for (i in 1:n_stan) {
    stan_fit_i <- stan_fits[[i]]
    res_matrix[i, ] <- yield_stan_res2(stan_fit_i, eff_col, beta0)
  }
  rownames(res_matrix) <- model_names
  
  write.table(data.frame(t(res_matrix[, 1])), 
              file = sprintf("./simres2/ancova_rejection_delta%.2f_w%.2f.txt", delta, w_eff), 
              append = file.exists(sprintf("./simres2/ancova_rejection_delta%.2f_w%.2f.txt", delta, w_eff)), 
              quote = FALSE,
              sep = "\t",
              row.names = F,
              col.names = !file.exists(sprintf("./simres2/ancova_rejection_delta%.2f_w%.2f.txt", delta, w_eff)))
  write.table(data.frame(t(res_matrix[, 2])), 
              file = sprintf("./simres2/ancova_pvalue_delta%.2f_w%.2f.txt", delta, w_eff), 
              append = file.exists(sprintf("./simres2/ancova_pvalue_delta%.2f_w%.2f.txt", delta, w_eff)), 
              quote = FALSE,
              sep = "\t",
              row.names = F,
              col.names = !file.exists(sprintf("./simres2/ancova_pvalue_delta%.2f_w%.2f.txt", delta, w_eff)))
  write.table(data.frame(t(res_matrix[, 3])), 
              file = sprintf("./simres2/ancova_dic_delta%.2f_w%.2f.txt", delta, w_eff), 
              append = file.exists(sprintf("./simres2/ancova_dic_delta%.2f_w%.2f.txt", delta, w_eff)), 
              quote = FALSE,
              sep = "\t",
              row.names = F,
              col.names = !file.exists(sprintf("./simres2/ancova_dic_delta%.2f_w%.2f.txt", delta, w_eff)))
}

# --- Yield simulation results 3 ---
yield_stan_res3 <- function(fit_obj, eff_col, beta0) {
  sm <- summary(fit_obj$fit, pars = "beta")$summary
  iterations <- extract(fit_obj$fit, pars = "beta")$beta
  idx <- match(eff_col, fit_obj$dat$cols)
  summary_res <- sm[idx, c("mean", "2.5%", "97.5%", "Rhat")]
  
  y_hat <- drop(fit_obj$dat$X%*%sm[, "mean"])
  sigma_hat <- summary(fit_obj$fit, pars = "sigma")$summary[, "mean"]
  ll_mean <- summary(fit_obj$fit, pars = "ll")$summary[, "mean"]
  dev_mean <- -2*ll_mean
  dev_plug <- -2*sum(dnorm(fit_obj$dat$Y, y_hat, sigma_hat, log = T))
  p_d <- dev_mean - dev_plug
  dic <- dev_plug + 2*p_d
  c("Estimate" = summary_res[1],
    "LB" = summary_res[2],
    "UB" = summary_res[3],
    "DIC" = dic)
}

write_res3 <- function(stan_fits, model_names, eff_col, beta0, delta, w_eff) {
  n_stan <- length(stan_fits)
  if (is.null(model_names)) {
    model_names <- paste0("M", 1:n_stan)
  }
  res_matrix <- matrix(NA, nrow = n_stan, ncol = 4)
  
  for (i in 1:n_stan) {
    stan_fit_i <- stan_fits[[i]]
    res_matrix[i, ] <- yield_stan_res3(stan_fit_i, eff_col, beta0)
  }
  rownames(res_matrix) <- model_names
  
  write.table(data.frame(w = w_eff, t(res_matrix[, 1])), 
              file = sprintf("./simres3/ancova_estimate_delta%.2f.txt", delta), 
              append = file.exists(sprintf("./simres3/ancova_estimate_delta%.2f.txt", delta)), 
              quote = FALSE,
              sep = "\t",
              row.names = F,
              col.names = !file.exists(sprintf("./simres3/ancova_estimate_delta%.2f.txt", delta)))
  write.table(data.frame(w = w_eff, t(res_matrix[, 2])), 
              file = sprintf("./simres3/ancova_lb_delta%.2f.txt", delta), 
              append = file.exists(sprintf("./simres3/ancova_lb_delta%.2f.txt", delta)), 
              quote = FALSE,
              sep = "\t",
              row.names = F,
              col.names = !file.exists(sprintf("./simres3/ancova_lb_delta%.2f.txt", delta)))
  write.table(data.frame(w = w_eff, t(res_matrix[, 3])), 
              file = sprintf("./simres3/ancova_ub_delta%.2f.txt", delta), 
              append = file.exists(sprintf("./simres3/ancova_ub_delta%.2f.txt", delta)), 
              quote = FALSE,
              sep = "\t",
              row.names = F,
              col.names = !file.exists(sprintf("./simres3/ancova_ub_delta%.2f.txt", delta)))
  write.table(data.frame(w = w_eff, t(res_matrix[, 4])), 
              file = sprintf("./simres3/ancova_dic_delta%.2f.txt", delta), 
              append = file.exists(sprintf("./simres3/ancova_dic_delta%.2f.txt", delta)), 
              quote = FALSE,
              sep = "\t",
              row.names = F,
              col.names = !file.exists(sprintf("./simres3/ancova_dic_delta%.2f.txt", delta)))
}
