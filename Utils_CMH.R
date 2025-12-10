## Generate 2×2×K tables
mk_mh_or_tab <- function(data,
                         formula) {
  if (!requireNamespace("survival", quietly = TRUE)) {
    stop("Package 'survival' is required. Please install.packages('survival').")
  }
  
  # Evaluate formula: user should have library(survival) loaded,
  # or write survival::strata(...) in the formula.
  mf <- stats::model.frame(formula, data = data, na.action = stats::na.omit)
  
  # Response
  y <- mf[[1L]]
  
  # RHS columns
  rhs_names <- names(mf)[-1L]
  rhs_cols  <- mf[-1L]
  
  # Identify strata column by name containing 'strata('
  strata_idx <- grep("strata\\(", rhs_names)
  if (length(strata_idx) != 1L) {
    stop(
      "Could not uniquely identify strata term on RHS.\n",
      "Make sure the formula looks like: y ~ trt + strata(S1 + S2)"
    )
  }
  
  # Treatment is the other RHS term
  trt_idx <- setdiff(seq_along(rhs_names), strata_idx)
  if (length(trt_idx) != 1L) {
    stop("Formula must have exactly one non-strata predictor (treatment A).")
  }
  
  a      <- rhs_cols[[trt_idx]]
  strata <- rhs_cols[[strata_idx]]
  
  ## Outcome Y -> factor with 2 levels: succ/fail
  y_fac <- if (is.factor(y)) y else factor(y)
  
  if (nlevels(y_fac) != 2L) {
    stop("Outcome Y must have exactly 2 levels after recoding.")
  }
  levels(y_fac) <- c("succ", "fail")
  
  ## Treatment A -> factor with 2 levels: trt/ctl
  a_fac <- if (is.factor(a)) a else factor(a)
  
  if (nlevels(a_fac) != 2L) {
    stop("Treatment A must have exactly 2 levels after recoding.")
  }
  levels(a_fac) <- c("trt", "ctl")
  
  ## Strata factor: survival::strata already combines multiple vars
  strata_fac <- factor(strata)  # make sure it's a clean factor
  
  ## 2x2xK table: dim1 = A (trt/ctl), dim2 = Y (succ/fail), dim3 = strata
  tab <- stats::xtabs(~ a_fac + y_fac + strata_fac)
  
  if (!all(dim(tab)[1:2] == c(2, 2))) {
    stop("Expected a 2x2 table for A x Y in each stratum.")
  }
  
  tab
}

## Build prior
build_mh_or_prior <- function(tab_hist, conf.level = 0.95) {
  if (!is.array(tab_hist) || length(dim(tab_hist)) != 3L ||
      any(dim(tab_hist)[1:2] != c(2, 2))) {
    stop("tab_hist must be a 2x2xK array")
  }
  
  mt <- stats::mantelhaen.test(tab_hist, correct = FALSE, conf.level = conf.level)
  
  or_hat  <- unname(mt$estimate)
  tau_hat <- log(or_hat)
  
  ci    <- unname(mt$conf.int)
  alpha <- 1 - conf.level
  z     <- qnorm(1 - alpha/2)
  
  se_tau <- (log(ci[2]) - log(ci[1])) / (2 * z)
  
  list(
    mu_inf = as.numeric(tau_hat),
    sigma_inf = as.numeric(se_tau),
    mu_vague = as.numeric(tau_hat),
    sigma_vague = sqrt(sum(tab_hist))*as.numeric(se_tau)
  )
}

## Build stan data
build_mh_or_data <- function(
    data_curr,
    formula_curr,
    mu_inf = NULL,
    sigma_inf = NULL,
    mu_vague = NULL,
    sigma_vague  = NULL,
    w = 0.5
) {
  tab_curr <- mk_mh_or_tab(
    data = data_curr,
    formula = formula_curr
  )
  
  K <- dim(tab_curr)[3L]
  n11 <- as.integer(tab_curr[1, 1, ])
  n12 <- as.integer(tab_curr[1, 2, ])
  n21 <- as.integer(tab_curr[2, 1, ])
  n22 <- as.integer(tab_curr[2, 2, ])
  
  n1dot <- n11 + n12
  n2dot <- n21 + n22
  ndot1 <- n11 + n21
  
  L      <- pmax(0L, ndot1 - n2dot)
  U      <- pmin(n1dot, ndot1)
  len    <- U - L + 1L
  max_len <- max(len)
  
  if (is.null(mu_inf) || is.null(sigma_inf)) {
      if (is.null(mu_inf))    mu_inf    <- 0
      if (is.null(sigma_inf)) sigma_inf <- 2
  }
  if (is.null(mu_vague) || is.null(sigma_vague)) {
    if (is.null(mu_vague))    mu_vague    <- 0
    if (is.null(sigma_vague)) sigma_vague <- 2
  }
  
  list(
    K = K,
    n11 = n11,
    n12 = n12,
    n21 = n21,
    n22 = n22,
    max_len = max_len,
    mu_inf = mu_inf,
    sigma_inf = sigma_inf,
    mu_vague = mu_vague,
    sigma_vague = sigma_vague,
    w = w
  )
}

build_mh_model <- function(prior_type = c("mixture", "power")) {
  prior_type <- match.arg(prior_type)
  stan_file <- ifelse(prior_type == "mixture", "mh_or_mp.stan", "mh_or_pp.stan")
  model <- stan_model(file = stan_file)
  return(model)
}


# --- Util: Run Stan model ---

run_model <- function(model, data,
                      iter = 5000, burnin = iter/2, chains = 4, seed = 123, cores = getOption("mc.cores", 1L), refresh = max(iter/10, 1), save_warmup = FALSE) {
  suppressWarnings({
    fit <- sampling(model, data, iter = iter, warmup = burnin, chains = chains, seed = seed, cores = cores, refresh = refresh, save_warmup = save_warmup)
  })
  return(list(fit = fit, data = data))
}
