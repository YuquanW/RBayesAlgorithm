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
           beta = 0.60 + delta,                     # 关键：真值 beta
           beta_lab  = sprintf("beta=%.2f", beta),  # 用于图例的标签
           .before = 1)
}

long <- purrr::map_dfr(files, read_one) |>
  mutate(method = factor(method, levels = methods))  # 固定方法顺序

# === 3) 逐指标汇总 ===
# 真值 = 0.6 + delta
truth_tbl <- long |>
  distinct(beta)

# estimate -> Bias / RMSE
est_sum <- long |>
  filter(metric == "estimate") |>
  left_join(truth_tbl, by = "beta") |>
  group_by(beta, method) |>
  summarise(
    Bias = mean(value - beta),
    RMSE = sqrt(mean((value - beta)^2)),
    .groups = "drop"
  )

# cilength -> 平均 CI length
cil_sum <- long |>
  filter(metric == "cilength") |>
  group_by(beta, method) |>
  summarise(CI_length = mean(value), .groups = "drop")

# coverage -> 覆盖概率
cov_sum <- long |>
  filter(metric == "coverage") |>
  group_by(beta, method) |>
  summarise(Coverage = mean(value), .groups = "drop")

# rejection -> Power
pow_sum <- long |>
  filter(metric == "rejection") |>
  group_by(beta, method) |>
  summarise(Power = mean(value), .groups = "drop")

# rhat -> 平均 Rhat（如需最大/分位数可自行加列）
rhat_sum <- long |>
  filter(metric == "rhat") |>
  group_by(beta, method) |>
  summarise(Rhat = mean(value), .groups = "drop")

# === 4) 合并汇总表 ===
summary_tbl <- list(est_sum, cil_sum, cov_sum, pow_sum, rhat_sum) |>
  purrr::reduce(dplyr::full_join, by = c("beta","method")) |>
  arrange(beta, method)

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
  
  p <- ggplot(df, aes(x = beta, y = .data[[y]],
                      color = method, linetype = method, shape = method, group = method)) +
    geom_line(linewidth = 0.9, alpha = 0.9) +
    geom_point(size = 2.6, stroke = 0.5,
               position = position_jitter(width = jitter_width, height = 0)) +
    scale_color_brewer(palette = "Dark2") +
    scale_linetype_manual(values = lt) +
    scale_shape_manual(values = sh) +
    scale_x_continuous(breaks = sort(unique(df$beta))) +
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

p_bias  <- plot_metric(est_sum, "Bias", "Bias",  "Bias v.s. Current Effect Size")
p_rmse  <- plot_metric(est_sum, "RMSE", "RMSE",  "RMSE v.s. Current Effect Size")
p_power <- plot_metric(pow_sum, "Power", "Probability of rejection", "Probability of Rejection v.s. Current Effect Size")
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

# ===== 2) 读取并抽取 metric/delta/w，生成 beta_true =====
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
           beta = 0.60 + delta,                     # 关键：真值 beta
           beta_lab  = sprintf("Current effect size=%.2f", beta),  # 用于图例的标签
           .before = 1)
}

long <- purrr::map_dfr(files, read_one) |>
  mutate(method   = factor(method, levels = methods),
         # 图例按数值从小到大排序（比如 beta=0.00 在前，beta=0.60 在后）
         beta_lab = factor(beta_lab, levels = unique(beta_lab[order(beta)])))

# ===== 3) 汇总：DIC 均值 & Power（rejection 均值）=====
dic_sum <- long |>
  filter(metric == "dic") |>
  group_by(method, w, beta, beta_lab) |>
  summarise(DIC = mean(value), .groups = "drop")

pow_sum <- long |>
  filter(metric == "rejection") |>
  group_by(method, w, beta, beta_lab) |>
  summarise(Power = mean(value), .groups = "drop")

# 可选导出
readr::write_csv(dic_sum, "dic_by_w_beta.csv")
readr::write_csv(pow_sum, "power_by_w_beta.csv")

# ===== 4) 作图：以 beta_true（beta_lab）为图例曲线 =====
make_plot <- function(df, yvar, ylab, title, add_power_refs = FALSE) {
  p <- ggplot(df, aes(x = w, y = .data[[yvar]],
                      color = beta_lab, linetype = beta_lab, shape = beta_lab, group = beta_lab)) +
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
p_power <- make_plot(pow_sum, "Power", "Probability of rejection",    "Probability of Rejection v.s. w", add_power_refs = TRUE)

print(p_dic);print(p_power)
save(p_dic, p_power, file = "simres2.RData")

# simres3
# ===== 1) 文件与方法名 =====
dir_path <- "./simres3/"  # 修改为你的目录
pattern <- "^ancova_(dic|estimate|lb|ub)_delta(-?\\d+(?:\\.\\d+)?)\\.txt$"

files <- list.files(dir_path, pattern = pattern, full.names = TRUE)
if (length(files) == 0L) stop("未找到 ancova_xxx_deltayyy.txt 文件")

read_one <- function(fp) {
  bn <- basename(fp)
  m  <- stringr::str_match(bn, pattern)
  metric <- m[2]                 # dic / estimate / lb / ub
  delta  <- as.numeric(m[3])     # -0.60 / -0.30 等
  
  # 文件内容：100 x 3，有列名：第一列是 w，其余两列是两种方法
  dat <- readr::read_tsv(fp, show_col_types = FALSE)
  if (ncol(dat) < 3L) stop("期望 3 列( w + 2 方法 )，但文件中列数 < 3: ", bn)
  
  # 把第一列改名为 w，其余列视为方法
  names(dat)[1] <- "w"
  
  dat |>
    tidyr::pivot_longer(
      cols = -w,
      names_to = "method",
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
    method = factor(method)
  )

deltas <- sort(unique(long$delta))

#===== 2. 整理出 CI 用的数据（estimate + lb + ub）======
ci_df <- long |>
  filter(metric %in% c("estimate", "lb", "ub")) |>
  select(delta, w, method, metric, value) |>
  tidyr::pivot_wider(
    names_from  = metric,
    values_from = value
  )
# 得到列：delta, w, method, estimate, lb, ub


#===== 3. 整理出 DIC 用的数据 =====
dic_df <- long |>
  filter(metric == "dic") |>
  transmute(
    delta, w, method,
    dic = value
  )


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
    labs(
      title = paste0("Tipping-point (delta = ", d, ")"),
      x = "w", y = "估计值及 95% 区间"
    ) +
    facet_wrap(~ method, nrow = 1) +
    theme_bw() +
    theme(
      legend.position = "none"  # 已经分 panel，就不再需要图例
    )
}

# DIC 图：点 + 线，按方法 facet
plot_dic_one_delta <- function(d) {
  dat <- dic_df |>
    filter(delta == d)
  
  ggplot(dat,
         aes(x = w, y = dic,
             group = method, color = method)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2) +
    labs(
      title = paste0("DIC vs w (delta = ", d, ")"),
      x = "w", y = "DIC"
    ) +
    facet_wrap(~ method, nrow = 1) +
    theme_bw() +
    theme(
      legend.position = "none"
    )
}


#===== 5. 生成 4 张图（2 个 delta * CI / DIC） =====
# 假设 delta 只有 -0.60 和 -0.30
delta1 <- deltas[1]
delta2 <- deltas[2]

p_ci_1   <- plot_ci_one_delta(delta1)
p_ci_2   <- plot_ci_one_delta(delta2)
p_dic_1  <- plot_dic_one_delta(delta1)
p_dic_2  <- plot_dic_one_delta(delta2)

# 显示
print(p_ci_1);print(p_ci_2);print(p_dic_1);print(p_dic_2)
