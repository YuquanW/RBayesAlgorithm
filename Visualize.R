library(tidyverse)

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
    mutate(metric = metric, delta = delta, .before = 1)
}

long <- purrr::map_dfr(files, read_one) |>
  mutate(method = factor(method, levels = methods))  # 固定方法顺序

# === 3) 逐指标汇总 ===
# 真值 = 0.6 + delta
truth_tbl <- long |>
  distinct(delta) |>
  mutate(truth = 0.6 + delta)

# estimate -> Bias / RMSE
est_sum <- long |>
  filter(metric == "estimate") |>
  left_join(truth_tbl, by = "delta") |>
  group_by(delta, method) |>
  summarise(
    Bias = mean(value - truth),
    RMSE = sqrt(mean((value - truth)^2)),
    .groups = "drop"
  )

# cilength -> 平均 CI length
cil_sum <- long |>
  filter(metric == "cilength") |>
  group_by(delta, method) |>
  summarise(CI_length = mean(value), .groups = "drop")

# coverage -> 覆盖概率
cov_sum <- long |>
  filter(metric == "coverage") |>
  group_by(delta, method) |>
  summarise(Coverage = mean(value), .groups = "drop")

# rejection -> Power
pow_sum <- long |>
  filter(metric == "rejection") |>
  group_by(delta, method) |>
  summarise(Power = mean(value), .groups = "drop")

# rhat -> 平均 Rhat（如需最大/分位数可自行加列）
rhat_sum <- long |>
  filter(metric == "rhat") |>
  group_by(delta, method) |>
  summarise(Rhat = mean(value), .groups = "drop")

# === 4) 合并汇总表 ===
summary_tbl <- list(est_sum, cil_sum, cov_sum, pow_sum, rhat_sum) |>
  purrr::reduce(dplyr::full_join, by = c("delta","method")) |>
  arrange(delta, method)

# 便于展示的四舍五入版本
summary_disp <- summary_tbl |>
  mutate(across(c(Bias, RMSE, CI_length, Coverage, Power, Rhat), \(x) round(x, 4)))

# 输出 CSV（可选）
readr::write_csv(summary_disp, "ancova_summary.csv")

# 如果在 Rmd 里想表格展示：
# knitr::kable(summary_disp, caption = "Summary by delta & method") |>
#   kableExtra::kable_styling(full_width = FALSE)

# === 5) 画图：Bias / RMSE / Power 随 delta 变化 ===
lt <- c("solid","dashed","dotted","dotdash","longdash","twodash","F8")   # 任意7种
sh <- c(16, 17, 15, 3, 8, 4, 18)                                         # 圆/三角/方/十/星/×/菱

plot_metric <- function(df, y, ylab, ttl,
                        alpha_threshold = 0.025,
                        target_power = 0.8,
                        jitter_width = 0.001) {
  
  p <- ggplot(df, aes(x = delta, y = .data[[y]],
                      color = method, linetype = method, shape = method, group = method)) +
    geom_line(linewidth = 0.9, alpha = 0.9) +
    geom_point(size = 2.6, stroke = 0.5,
               position = position_jitter(width = jitter_width, height = 0)) +
    scale_color_brewer(palette = "Dark2") +
    scale_linetype_manual(values = lt) +
    scale_shape_manual(values = sh) +
    scale_x_continuous(breaks = sort(unique(df$delta))) +
    labs(title = ttl, x = "Delta", y = ylab, color = "Method",
         linetype = "Method", shape = "Method") +
    theme_bw() +
    theme(legend.position = "bottom", legend.box = "vertical", legend.title = element_blank())
  
  if (identical(tolower(y), "power")) {
    if (!is.null(alpha_threshold)) {
      p <- p + geom_hline(yintercept = alpha_threshold,
                          linetype = "dashed", color = "firebrick", linewidth = 0.7)
    }
    if (!is.null(target_power)) {
      p <- p + geom_hline(yintercept = target_power,
                          linetype = "dotdash", color = "steelblue", linewidth = 0.7)
    }
  }
  p
}

p_bias  <- plot_metric(est_sum, "Bias", "Bias",  "Bias vs Delta")
p_rmse  <- plot_metric(est_sum, "RMSE", "RMSE",  "RMSE vs Delta")
p_power <- plot_metric(pow_sum, "Power","Power", "Power vs Delta")
# 显示 / 保存
print(p_bias);  print(p_rmse);  print(p_power)
ggsave("bias_vs_delta.png", p_bias, width = 6, height = 4, dpi = 150)
ggsave("rmse_vs_delta.png", p_rmse, width = 6, height = 4, dpi = 150)
ggsave("power_vs_delta.png", p_power, width = 6, height = 4, dpi = 150)

# 可选：查看汇总表
summary_disp
