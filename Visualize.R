library(tidyverse)

#simres1
# === 1) 参数与文件列表 ===
dir_path <- "./simres/"   # 改成你的目录
pattern <- "^ancova_(estimate|cilength|coverage|rejection|rhat)_delta(-?\\d+(?:\\.\\d+)?)\\.txt$"
methods <- c("FMP","RMP","FPP","UPP","ANPP","NPP","ANCOVA")

files <- list.files(dir_path, pattern = pattern, full.names = TRUE)
if (length(files) == 0L) stop("未找到匹配文件：ancova_xxx_deltayyy.txt")

# === 2) 读取与整形（从文件名抽取 metric 和 delta） ===
read_one <- function(fp) {
  bn <- basename(fp)
  m <- stringr::str_match(bn, pattern)
  metric <- m[2]
  delta  <- as.numeric(m[3])
  
  # 读 1000 x 7，无列名，TAB 分隔
  dat <- readr::read_tsv(fp, col_names = TRUE, show_col_types = FALSE)
  stopifnot(ncol(dat) == length(methods))
  colnames(dat) <- methods
  
  dat |>
    tidyr::pivot_longer(everything(), names_to = "method", values_to = "value") |>
    mutate(metric = metric,
           tau = 0.60 + delta,                     # 关键：真值 tau
           tau_lab  = sprintf("tau=%.2f", tau),  # 用于图例的标签
           .before = 1)
}

long <- purrr::map_dfr(files, read_one) |>
  mutate(method = factor(method, levels = methods))  # 固定方法顺序

# === 3) 逐指标汇总 ===
# 真值 = 0.6 + delta
truth_tbl <- long |>
  distinct(tau)

# estimate -> Bias / RMSE
est_sum <- long |>
  filter(metric == "estimate") |>
  left_join(truth_tbl, by = "tau") |>
  group_by(tau, method) |>
  summarise(
    Bias = mean(value - tau),
    RMSE = sqrt(mean((value - tau)^2)),
    .groups = "drop"
  )

# cilength -> 平均 CI length
cil_sum <- long |>
  filter(metric == "cilength") |>
  group_by(tau, method) |>
  summarise(CI_length = mean(value), .groups = "drop")

# coverage -> 覆盖概率
cov_sum <- long |>
  filter(metric == "coverage") |>
  group_by(tau, method) |>
  summarise(Coverage = mean(value), .groups = "drop")

# rejection -> Power
pow_sum <- long |>
  filter(metric == "rejection") |>
  group_by(tau, method) |>
  summarise(Power = mean(value), .groups = "drop")

# rhat -> 平均 Rhat（如需最大/分位数可自行加列）
rhat_sum <- long |>
  filter(metric == "rhat") |>
  group_by(tau, method) |>
  summarise(Rhat = mean(value), .groups = "drop")

# === 4) 合并汇总表 ===
summary_tbl <- list(est_sum, cil_sum, cov_sum, pow_sum, rhat_sum) |>
  purrr::reduce(dplyr::full_join, by = c("tau","method")) |>
  arrange(tau, method)

# 便于展示的四舍五入版本
summary_disp <- summary_tbl |>
  mutate(across(c(Bias, RMSE, CI_length, Coverage, Power, Rhat), \(x) round(x, 4))) |>
  rename("Probability of rejection" = "Power")

# 输出 CSV（可选）
readr::write_csv(summary_disp, "ancova_summary.csv")

# 如果在 Rmd 里想表格展示：
# knitr::kable(summary_disp, caption = "Summary by delta & method") |>
#   kableExtra::kable_styling(full_width = FALSE)

# === 5) 画图：Bias / RMSE / Power 随 delta 变化 ===
lt <- c("solid","dashed","dotted","dotdash","longdash","twodash","F8")   # 任意7种
sh <- c(16, 17, 15, 3, 8, 4, 18)                                         # 圆/三角/方/十/星/×/菱

plot_metric <- function(df, y, ylab, ttl,
                        jitter_width = 0.001) {
  
  p <- ggplot(df, aes(x = tau, y = .data[[y]],
                      color = method, linetype = method, shape = method, group = method)) +
    geom_line(linewidth = 0.9, alpha = 0.9) +
    geom_point(size = 2.6, stroke = 0.5,
               position = position_jitter(width = jitter_width, height = 0)) +
    scale_color_brewer(palette = "Dark2") +
    scale_linetype_manual(values = lt) +
    scale_shape_manual(values = sh) +
    scale_x_continuous(breaks = sort(unique(df$tau))) +
    labs(title = ttl, x = "Current effect size", y = ylab, color = "Method",
         linetype = "Method", shape = "Method") +
    geom_vline(xintercept = 0.6, linetype = "dashed", color = "red", linewidth = 1) +
    theme_bw() +
    theme(legend.position = "bottom", legend.box = "vertical", legend.title = element_blank(),
          plot.title = element_text(face = "bold", hjust = 0.5))
  
  if (identical(tolower(y), "power")) {
    p <- p + geom_hline(yintercept = 0.025,
                        linetype = "dashed", color = "firebrick", linewidth = 0.7) +
      geom_hline(yintercept = 0.80,
                 linetype = "dotdash", color = "steelblue", linewidth = 0.7) +
      scale_y_continuous(
        breaks = sort(unique(c(c(0.025, 0.80), pretty(range(df[[y]]), n = 2)))),
        labels = scales::label_number(accuracy = 0.001)
      )
  }
  p
}

p_bias  <- plot_metric(est_sum, "Bias", "Bias",  "Bias v.s. Current effect size")
p_rmse  <- plot_metric(est_sum, "RMSE", "RMSE",  "RMSE v.s. Current effect size")
p_power <- plot_metric(pow_sum, "Power", "Probability of rejection", "Probability of rejection v.s. Current effect size")
# 显示 / 保存
print(p_bias);  print(p_rmse);  print(p_power)
save(summary_disp, p_bias, p_rmse, p_power, file = "simres1.RData")
load("simres1.RData")

# simres2
# ===== 1) 文件与方法名 =====
dir_path <- "./simres2/"   # 改成你的目录
pattern <- "^ancova_(dic|rejection)_delta(-?\\d+(?:\\.\\d+)?)_w(\\d+(?:\\.\\d+)?)\\.txt$"
methods <- c("FMP","FPP")  # 按你的两种方法名改

files <- list.files(dir_path, pattern = pattern, full.names = TRUE)
stopifnot(length(files) > 0)

# ===== 2) 读取并抽取 metric/delta/w，生成 tau_true =====
read_one <- function(fp) {
  bn <- basename(fp)
  m  <- stringr::str_match(bn, pattern)
  metric <- m[2]
  delta  <- as.numeric(m[3])
  w      <- as.numeric(m[4])
  
  dat <- readr::read_tsv(fp, col_names = TRUE, show_col_types = FALSE)
  stopifnot(ncol(dat) == 2L)
  colnames(dat) <- methods
  
  dat |>
    pivot_longer(everything(), names_to = "method", values_to = "value") |>
    mutate(metric = metric,
           delta  = delta,
           w      = w,
           tau = 0.60 + delta,                     # 关键：真值 tau
           tau_lab  = sprintf("Current effect size=%.2f", tau),  # 用于图例的标签
           .before = 1)
}

long <- purrr::map_dfr(files, read_one) |>
  mutate(method   = factor(method, levels = methods),
         # 图例按数值从小到大排序（比如 tau=0.00 在前，tau=0.60 在后）
         tau_lab = factor(tau_lab, levels = unique(tau_lab[order(tau)])))

# ===== 3) 汇总：DIC 均值 & Power（rejection 均值）=====
dic_sum <- long |>
  filter(metric == "dic") |>
  group_by(method, w, tau, tau_lab) |>
  summarise(DIC = mean(value), .groups = "drop")

pow_sum <- long |>
  filter(metric == "rejection") |>
  group_by(method, w, tau, tau_lab) |>
  summarise(Power = mean(value), .groups = "drop")

# 可选导出
readr::write_csv(dic_sum, "dic_by_w_tau.csv")
readr::write_csv(pow_sum, "power_by_w_tau.csv")

# ===== 4) 作图：以 tau_true（tau_lab）为图例曲线 =====
make_plot <- function(df, yvar, ylab, title, add_power_refs = FALSE) {
  p <- ggplot(df, aes(x = w, y = .data[[yvar]],
                      color = tau_lab, linetype = tau_lab, shape = tau_lab, group = tau_lab)) +
    geom_line(linewidth = 0.9, alpha = 0.9) +
    geom_point(size = 2.6, position = position_jitter(width = 0.003, height = 0)) +
    scale_x_continuous(breaks = sort(unique(df$w))) +
    labs(x = "w", y = ylab, title = title, color = NULL, linetype = NULL, shape = NULL) +
    facet_wrap(~ method, ncol = length(levels(df$method))) +
    theme_bw() +
    theme(legend.position = "bottom", legend.box = "vertical",
          plot.title = element_text(face = "bold", hjust = 0.5))
  
  if (add_power_refs) {
    p <- p +
      geom_hline(yintercept = 0.025, linetype = "dashed",  color = "#D55E00", linewidth = 0.7) +
      geom_hline(yintercept = 0.05,  linetype = "dashed",  color = "#E69F00", linewidth = 0.7) +
      geom_hline(yintercept = 0.10,  linetype = "dashed",  color = "#F0E442", linewidth = 0.7) +
      geom_hline(yintercept = 0.70,  linetype = "dotdash", color = "#0072B2", linewidth = 0.7) +
      geom_hline(yintercept = 0.80,  linetype = "dotdash", color = "#009E73", linewidth = 0.7) +
      scale_y_continuous(
        breaks = sort(unique(c(c(0.025, 0.05, 0.10, 0.70, 0.80), pretty(range(df[[yvar]]), n = 5)))),
        labels = scales::label_number(accuracy = 0.001)
      )
  }
  p
}

p_dic   <- make_plot(dic_sum, "DIC",   "Mean DIC", "DIC v.s. w")
p_power <- make_plot(pow_sum, "Power", "Probability of rejection",    "Probability of rejection v.s. w", add_power_refs = TRUE)

print(p_dic);print(p_power)
save(p_dic, p_power, file = "simres2.RData")

# simres3
# ===== 1) 文件与方法名 =====
dir_path <- "./simres3/"  # 修改为你的目录
n_cur <- 44
pattern <- "^ancova_(dic|estimate|lb|ub|sd)_delta(-?\\d+(?:\\.\\d+)?)\\.txt$"

files <- list.files(dir_path, pattern = pattern, full.names = TRUE)
if (length(files) == 0L) stop("未找到 ancova_xxx_deltayyy.txt 文件")

read_one <- function(fp) {
  bn <- basename(fp)
  m  <- stringr::str_match(bn, pattern)
  metric <- m[2]               # dic / estimate / lb / ub / sd
  delta  <- as.numeric(m[3])   # -0.60 / -0.30 ...
  
  dat <- readr::read_tsv(fp, show_col_types = FALSE)
  if (ncol(dat) < 3L) stop("文件列数 < 3（期待 w + 2 方法）：", bn)
  
  # 第一列是 w，后两列是两种方法
  names(dat)[1] <- "w"
  
  dat |>
    tidyr::pivot_longer(
      cols = -w,
      names_to  = "method",
      values_to = "value"
    ) |>
    mutate(
      metric = metric,
      delta  = delta,
      .before = 1
    )
}

long <- purrr::map_dfr(files, read_one)

# 确保 w 为数值，方法做成因子（保持顺序）
long <- long |>
  mutate(
    w      = as.numeric(w),
    method = factor(method),
    tau_lab = sprintf("Current effect size=%.2f", 0.6+delta),
  )

deltas <- sort(unique(long$delta))
methods <- levels(long$method)

#===== 2. 整理出 CI 用的数据（estimate + lb + ub）======
ci_df <- long |>
  filter(metric %in% c("estimate", "lb", "ub", "sd")) |>
  select(delta, w, method, metric, value) |>
  tidyr::pivot_wider(
    names_from  = metric,
    values_from = value
  )
# 得到列：delta, w, method, estimate, lb, ub


#===== 3. 整理出 DIC 用的数据 =====
grid <- 0:10
dic_df <- long |>
  filter(metric == "dic") |>
  transmute(
    delta, w, method,tau_lab,
    dic = value
  ) |>
  filter((w*10) %in% grid)


#===== 4. 画图函数：CI 图 & DIC 图 =====
# CI 图：线 + ribbon 可信域，按方法 facet（左右两个 panel）
plot_ci_one_delta <- function(d) {
  dat <- ci_df |>
    filter(delta == d)
  
  ggplot(dat,
         aes(x = w, y = estimate,
             group = method, color = method, fill = method)) +
    geom_ribbon(aes(ymin = lb, ymax = ub),
                alpha = 0.2, color = NA) +
    geom_line(linewidth = 0.8) +
    geom_hline(aes(yintercept = 0, color = method), linetype = "dashed", linewidth = 0.7) +
    labs(
      title = paste0("Sensitivity analysis (current effect size = ", 0.6+d, ")"),
      x = "w", y = "Estimate"
    ) +
    facet_wrap(~ method, nrow = 1) +
    theme_bw() +
    theme(
      legend.position = "none",
      plot.title = element_text(face = "bold", hjust = 0.5)
    )
}

#===== 5. 生成 4 张图（2 个 delta * CI / DIC） =====
# 假设 delta 只有 -0.60 和 -0.30
delta1 <- deltas[1]
delta2 <- deltas[2]
delta3 <- deltas[3]

p_ci_1   <- plot_ci_one_delta(delta1)
p_ci_2   <- plot_ci_one_delta(delta2)
p_ci_3   <- plot_ci_one_delta(delta3)
p_dic  <- make_plot(dic_df, "dic", "DIC", "DIC v.s. w")

# 显示
print(p_ci_1);print(p_ci_2);print(p_ci_3);print(p_dic)

#===== 6. 找到边界有效样本量 =====
phase_df <- ci_df |>
  arrange(delta, method, w) |>
  group_by(delta, method) |>
  filter(lb > 0) |>
  slice_head(n = 1) |>
  ungroup() |>
  transmute(
    tau = delta+0.6,
    method,
    w_phase = w,
    sd_phase = sd,
    estimate_phase = estimate,
    lb_phase = lb,
    ub_phase = ub
  )
sd_ref_df <- ci_df |>
  group_by(delta, method) |>
  arrange(w) |>
  slice(which.min(abs(w - 0))) |>
  ungroup() |>
  transmute(
    tau = delta+0.6,
    method,
    w_ref = w,
    sd_ref = sd
  )
ehss_df <- phase_df |>
  left_join(sd_ref_df, by = c("tau", "method")) |>
  mutate(
    EHSS = if_else(w_phase > 0, n_cur * (sd_ref^2 / sd_phase^2 - 1), NA)
  )
save(p_ci_1, p_ci_2, p_ci_3, p_dic, ehss_df, file = "simres3.RData")
