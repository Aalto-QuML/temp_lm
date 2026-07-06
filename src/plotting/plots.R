library(tidyverse)

metrics_fp  <- "file:///u/84/blohmp1/unix/Downloads/metrics_openwebtext_train_slice100000-109000_L128_bs4.csv"
baseline_fp <- "file:///u/84/blohmp1/unix/Downloads/pretrained_metrics_openwebtext_train_slice100000_109000_L128_bs4.csv"


library(mgcv)

# df must contain at least:
# group_id, source (baseline/metrics), temp_ratio, mean_logL, mean_var

k_sd   <- .05       # ribbon is +/- k_sd * SD
n_grid <- 250     # smoothness of ribbon/line (more points = smoother drawing)

# helper: fit mean + log-variance smoothers per group, then predict on a grid
smooth_mean_var <- function(d, n = 250, k_sd = 1) {
  d <- d %>%
    filter(is.finite(temp_ratio), is.finite(mean_logL), is.finite(mean_var)) %>%
    mutate(mean_var = pmax(mean_var, 1e-12))  # avoid log(0)

  if (nrow(d) < 6) return(tibble()) # too few points to fit anything stable

  xg <- tibble(temp_ratio = seq(min(d$temp_ratio), max(d$temp_ratio), length.out = n))

  # Smooth mean and smooth log-variance with the same type of smoother
  m_mu  <- mgcv::gam(mean_logL ~ s(temp_ratio, bs = "cs"), data = d)
  m_lvv <- mgcv::gam(log(mean_var) ~ s(temp_ratio, bs = "cs"), data = d)

  mu  <- predict(m_mu,  newdata = xg, type = "response")
  lvv <- predict(m_lvv, newdata = xg, type = "response")

  sd_sm <- sqrt(exp(lvv))

  xg %>%
    mutate(
      mean_sm = mu,
      var_sm  = exp(lvv),
      ymin    = mean_sm - k_sd * sd_sm,
      ymax    = mean_sm + k_sd * sd_sm
    )
}


as_local_path <- function(x) sub("^file://", "", x)

# group_id = everything up to (but excluding) the last "/"
group_until_last_slash <- function(x) {
  x2 <- str_replace_all(x, "\\\\", "/")   # normalize slashes
  x2 <- str_replace(x2, "/+$", "")       # drop trailing slash if any
  str_replace(x2, "/[^/]+$", "")         # remove last path segment
  # equivalently: str_replace(x2, "(.*)/.*?$", "\\1")
}

metrics <- readr::read_csv(as_local_path(metrics_fp), show_col_types = FALSE) %>%
  mutate(source = "metrics")

baseline <- readr::read_csv(as_local_path(baseline_fp), show_col_types = FALSE) %>%
  mutate(source = "baseline")

df <- bind_rows(metrics, baseline) %>%
  mutate(
    mean_logL = -mean_logL/128,
    group_id = case_when(
      source != "baseline" ~ model_id,                  # baseline unchanged
      TRUE ~ group_until_last_slash(model_id)           # metrics grouped by parent dir
    ),
    temperature = suppressWarnings(as.numeric(temperature))
    # filter out files where the group_id contains the string emp0.8
  ) %>% filter(!str_detect(group_id, "emp0\\.8")) %>%
  # make it so that emp[0-9]\.[0-9] is grouped together
    mutate(
      group_id = str_replace_all(group_id, "emp[0-9]\\.[0-9]", "")
    )

# Keep only requested fields
plot_wide <- df %>%
  transmute(
    source,
    group_id,
    model_id,
    temperature,
    temp_ratio = ratio_scale_mean,
    mean_var,
    mean_logL
  )


table_info <- plot_wide %>%
  group_by(source, group_id) %>%
  summarise(
    n_models = n(),
    temp_ratio_min = min(temp_ratio),
    temp_ratio_max = max(temp_ratio),
    nll_at_.5 = mean_logL[which.min(abs(temp_ratio - 0.5))],
    var_at_.5 = mean_var[which.min(abs(temp_ratio - 0.5))],
    # nll per temperature deviation from 1.0
    nll_per_temp_dev = (mean_logL[which.min(abs(temp_ratio - 1.0))] - mean_logL[which.min(abs(temp_ratio - 0.5))]) / abs(1.0 - 0.5),
    mean_logL_max = max(mean_logL),
    runtime= 0,  # placeholder
    .groups = "drop"
  )


# Optional long format for ggplot
plot_long <- plot_wide %>%
  pivot_longer(c(mean_var, mean_logL), names_to = "metric", values_to = "value")



ggplot(plot_wide, aes(x = temp_ratio, y = mean_logL, color = group_id, shape = source)) +
  geom_smooth() +
  geom_point(alpha = 0.85) +
  theme_minimal() +
  labs(x = "Temperature ratio (ratio_scale_mean)", y = "Mean Log-Likelihood", color = "Parent folder", shape = "File") + 
  ylim(1.3,3) + xlim(.4,3)



# Example plot
ggplot(plot_long, aes(x = temp_ratio, y = value, color = group_id, shape = source)) +
  geom_point(alpha = 0.85) +
  facet_wrap(~metric, scales = "free_y") +
  theme_minimal() +
  labs(x = "Temperature ratio (ratio_scale_mean)", y = NULL, color = "Parent folder", shape = "File")


# =================================== VARIANCE RIBBONS ===================================

pred <- plot_wide %>%
#   filter(source == "metrics") %>%
  group_by(group_id) %>%
  group_modify(~ smooth_mean_var(.x, n = n_grid, k_sd = k_sd)) %>%
  ungroup()
ggplot(data = pred) +
# Smoothed variance ribbon
geom_ribbon(
aes(x = temp_ratio, ymin = ymin, ymax = ymax, fill = group_id),
alpha = 0.18,
colour = NA
) +
# Smoothed mean line
geom_line(
aes(x = temp_ratio, y = mean_sm, color = group_id),
linewidth = 1
) +
# Raw points (baseline + metrics)
geom_point(
data = plot_wide,
aes(x = temp_ratio, y = mean_logL, color = group_id, shape = source),
alpha = 0.85,
size = 1.8
) +
theme_minimal() +
labs(
x = "Temperature ratio (ratio_scale_mean)",
y = "Mean Log-Likelihood",
color = "Parent folder",
fill  = "Parent folder",
shape = "File"
) + coord_cartesian(xlim = c(.4, 3), ylim = c(1,5)) +#ylim(1,5) + xlim(.4,3)
scale_x_log10()


# SAVE PLOT IN ICML FORMAT

fig_width_in  <- 4.25
fig_height_in <- fig_width_in * 9/16

paper_font_pt <- 10  # match your LaTeX main font size (often 10pt for ICML)


p <- ggplot(data = pred) +
  geom_ribbon(
    aes(x = temp_ratio, ymin = ymin, ymax = ymax, fill = group_id),
    alpha = 0.18, colour = NA
  ) +
  geom_line(
    aes(x = temp_ratio, y = mean_sm, color = group_id),
    linewidth = 0.9
  ) +
  geom_point(
    data = plot_wide,
    aes(x = temp_ratio, y = mean_logL, color = group_id),
    alpha = 0.85, size = 1.6
  ) +
  scale_x_log10(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_color_discrete(labels =c("Exact (Ours)","Efficient (Ours)", "Myopic")) +
  scale_fill_discrete(labels =c("Exact (Ours)","Efficient (Ours)", "Myopic")) +
  coord_cartesian(xlim = c(.4, 3), ylim = c(1, 4), expand = FALSE) +
  labs(
    x = "Effective Temperature",
    y = "Mean Neg. Log-Likelihood",
    color = "Method",
    fill  = "Method",
  ) +

  theme_minimal(base_size = paper_font_pt) +
  theme(
    # legend on top, horizontal layout
    legend.position = "top",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.title = element_text(size = paper_font_pt),
    legend.text  = element_text(size = paper_font_pt),
    legend.key.height = unit(0.35, "lines"),
    legend.key.width  = unit(0.9,  "lines"),

    # remove *all* outer margins/whitespace
    plot.margin = margin(0, 10, 0, 0, unit = "pt"),

    # also remove panel padding
    panel.spacing = unit(0, "pt"),

    # optional: slightly tighter axis titles
    axis.title.x = element_text(margin = margin(t = 2, unit = "pt")),
    axis.title.y = element_text(margin = margin(r = 2, unit = "pt"))
  )


p
ggsave(
  filename = "icml_ll_plot.pdf",
  plot = p,
  width = fig_width_in,
  height = fig_height_in,
  units = "in",
  dpi = 300,
  device = cairo_pdf
)


# ============================================== TABLE FOR EMP.08 EXPERIMENT ==============================================
# we want to exclude emp0.8 from the plots above, but now make a table including only emp0.8
# the table then contains the effective temperature ratio and the mean log-likelihood at that temperature ratio and the variance

df2 <- bind_rows(metrics, baseline) %>%
  mutate(
    mean_logL = -mean_logL/128,
    group_id = case_when(
      source != "baseline" ~ model_id,                  # baseline unchanged
      TRUE ~ group_until_last_slash(model_id)           # metrics grouped by parent dir
    ),
    temperature = suppressWarnings(as.numeric(temperature))
    # filter out files where the group_id contains the string emp0.8
  ) %>% filter(((str_detect(group_id, "emp0\\.8.*lr1"))) | ((temperature < 0.795) & (temperature > 0.794))) %>%
  transmute(
    source,
    group_id,
    model_id,
    temperature,
    temp_ratio = ratio_scale_mean,
    mean_var,
    mean_logL
  )

names = c("Baseline", "LHTS","ratio","elbo","elbo_ao","ratio_ao")
df2$name <- names
library(knitr)
library(kableExtra)
df2 %>% transmute(
    name,
    temp_ratio = round(temp_ratio,2),
    mean_logL = round(mean_logL,2),
    mean_var = round(mean_var,2)
) %>% knitr::kable(
      format  = "latex",
      booktabs = TRUE,
    )

sym_check <- "$\\checkmark$"
sym_x     <- "$\\times$"

tab <- df2 %>%
  # create AO flag + base name (ratio_ao -> ratio, elbo_ao -> elbo)
  mutate(
    ao   = if_else(str_detect(name, "_ao$"), sym_check, sym_x),
    name = str_remove(name, "_ao$")
  ) %>%
  # order within each name: non-AO first, AO second (so they stack nicely)
  arrange(name, desc(ao == sym_check)) %>%
  transmute(
    Name     = name,
    AO       = ao,
    Ratio    = round(temp_ratio, 2),
    ELBO     = round(mean_logL, 2),
    MeanVar  = round(mean_var, 2)
  )

tab %>%
  kable(
    format   = "latex",
    booktabs = TRUE,
    escape   = FALSE,     # needed for \checkmark / \times
    align    = "lcccc"
  ) %>%
  collapse_rows(columns = 1, valign = "middle")  # multirow Name



# ================================== Variance Plot ===================================

# read numpy array CSV (typically no header; if you *do* have a header, this still works)
rawA <- readr::read_csv("/u/84/blohmp1/unix/Downloads/ratio_model.csv", col_names = FALSE, show_col_types = FALSE) %>% select(where(is.numeric))
rawB <- readr::read_csv("/u/84/blohmp1/unix/Downloads/temp_myopic_model.csv", col_names = FALSE, show_col_types = FALSE) %>% select(where(is.numeric))


# if numpy wrote an index column, drop any non-numeric column(s)

csA <- as.matrix(rawA)
csA <- csA - csA[, ncol(csA)]
csB <- as.matrix(rawB)
csB <- csB - csB[, ncol(csB)]

mat_to_long <- function(m, label) {
  as.data.frame(m) %>%
    mutate(series = row_number(), condition = label) %>%
    pivot_longer(-c(series, condition), names_to = "t", values_to = "y") %>%
    mutate(t = as.integer(sub("^[A-Za-z]", "", t)))
}

envelope <- function(df, probs = c(.05, .5, .95)) {
  df %>%
    group_by(condition, t) %>%
    summarise(
      lo  = quantile(y, probs[1], na.rm = TRUE),
      mid = quantile(y, probs[2], na.rm = TRUE),
      hi  = quantile(y, probs[3], na.rm = TRUE),
      width = hi - lo,
      .groups = "drop"
    )
}
library(scales)
pal3 <- scales::hue_pal()(3)

method_cols <- c(
  "Fine-tuning (Ours) T=.5" = pal3[1],  # red-ish (e.g. #F8766D)
  "Myopic T=.5" = pal3[3]   # blue-ish (e.g. #619CFF)
)
df  <- bind_rows(mat_to_long(csA, "Fine-tuning (Ours) T=.5"),
                 mat_to_long(csB, "Myopic T=.5"))

env <- envelope(df, probs = c(.05, .5, .95))


ggplot() +
  geom_line(
    data = df,
    aes(t, y, group = interaction(condition, series)),
    alpha = 0.06, linewidth = 0.35
  ) +
  geom_ribbon(
    data = env,
    aes(t, ymin = lo, ymax = hi,color = condition, fill = condition),
    alpha = 0.20
  ) +
  geom_line(
    data = env,
    aes(t, mid, color = condition),
    linewidth = 0.9
  ) +
#   facet_wrap(~ condition, ncol = 1) +
  theme_minimal(base_size = 12) +
  labs(x = NULL, y = NULL) +
theme_minimal(base_size = paper_font_pt) +
  theme(
    # legend on top, horizontal layout
    legend.position = "top",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.title = element_text(size = paper_font_pt),
    legend.text  = element_text(size = paper_font_pt),
    legend.key.height = unit(0.35, "lines"),
    legend.key.width  = unit(0.9,  "lines"),

    # remove *all* outer margins/whitespace
    plot.margin = margin(0, 10, 0, 0, unit = "pt"),

    # also remove panel padding
    panel.spacing = unit(0, "pt"),

    # optional: slightly tighter axis titles
    axis.title.x = element_text(margin = margin(t = 2, unit = "pt")),
    axis.title.y = element_text(margin = margin(r = 2, unit = "pt"))
  ) +
  labs(
    x = "Number of Sequence Trajectories",
    y = "NLL Estimation Deviation",
    color = "Method",
    fill  = "Method",
  ) + coord_cartesian(ylim = c(-20, 5), expand = FALSE) +
    scale_colour_manual(values = method_cols) +
  scale_fill_manual(values = method_cols)



ggsave(
  filename = "icml_variance_plot.pdf",
#   plot = p,
  width = fig_width_in,
  height = fig_height_in,
  units = "in",
  dpi = 300,
  device = cairo_pdf
)



