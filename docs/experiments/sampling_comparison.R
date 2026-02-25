## sampling_comparison.R
##
## Fit a full-data and a downsampled+offset mgcv model on synthetic prepayment
## data, then compare their smooth curves.
##
## Requires: mgcv, data.table, arrow

suppressPackageStartupMessages({
  library(mgcv)
  library(data.table)
  library(arrow)
})

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR          <- "C:/Users/yimingz/repo/quantbullet/docs/experiments"
DATA_PATH         <- file.path(DATA_DIR, "sampling_data.parquet")
PDF_OUT           <- file.path(DATA_DIR, "sampling_comparison.pdf")
TARGET_EVENT_RATE <- 0.1
SEED              <- 42

# ── Load & preprocess ─────────────────────────────────────────────────────────
if (!file.exists(DATA_PATH)) {
  stop("File not found: ", DATA_PATH,)
}

dt <- as.data.table(arrow::read_parquet(DATA_PATH))
dt[, loan_age  := pmin(loan_age, 80)]
dt[, incentive := pmin(pmax(incentive, -0.02), 0.04)]

pi_true <- mean(dt$prepaid)
cat(sprintf("Full data:  %d rows  |  event rate = %.4f\n", nrow(dt), pi_true))

# ── Fit full model ────────────────────────────────────────────────────────────
cat("\nFitting full model...\n")
model_full <- bam(
  prepaid ~ s(loan_age) + s(incentive),
  data     = dt,
  family   = binomial(),
  method   = "fREML",
  discrete = TRUE
)
cat("  Done.\n")

# ── Downsample ────────────────────────────────────────────────────────────────
set.seed(SEED)

events    <- dt[prepaid == 1]
nonevents <- dt[prepaid == 0]
n_events  <- nrow(events)
n_nonevents_target <- as.integer(n_events * (1 - TARGET_EVENT_RATE) / TARGET_EVENT_RATE)

if (n_nonevents_target > nrow(nonevents)) {
  stop("Not enough non-events to reach target rate ", TARGET_EVENT_RATE)
}

sampled_idx <- sample.int(nrow(nonevents), n_nonevents_target)
dt_down     <- rbind(events, nonevents[sampled_idx])

pi_down <- mean(dt_down$prepaid)
cat(sprintf("\nDownsampled: %d rows  |  event rate = %.4f\n", nrow(dt_down), pi_down))

# ── KZ offset ─────────────────────────────────────────────────────────────────
kz_correction <- log((pi_true * (1 - pi_down)) / (pi_down * (1 - pi_true)))
offset_val    <- -kz_correction

cat(sprintf("KZ correction (post-hoc shift):        %+.4f\n", kz_correction))
cat(sprintf("Offset for fitting (absorb inflation): %+.4f\n", offset_val))

dt_down[, correction := offset_val]

# ── Fit downsampled model with offset ─────────────────────────────────────────
cat("\nFitting downsampled model (with offset)...\n")
model_down <- bam(
  prepaid ~ s(loan_age) + s(incentive) + offset(correction),
  data     = dt_down,
  family   = binomial(),
  method   = "fREML",
  discrete = TRUE
)
cat("  Done.\n")

# ── Helper: extract smooth curve on a shared grid ─────────────────────────────
extract_smooth <- function(model, var_name, grid_length = 200) {
  x_range <- range(model$model[[var_name]], na.rm = TRUE)
  x_grid  <- seq(x_range[1], x_range[2], length.out = grid_length)

  ref_row <- model$model[1, , drop = FALSE]
  newdata <- ref_row[rep(1, grid_length), , drop = FALSE]
  newdata[[var_name]] <- x_grid

  if ("correction" %in% names(newdata)) {
    newdata$correction <- 0
  }

  pred <- predict.gam(model, newdata = newdata, type = "terms", se.fit = TRUE)
  term_label <- paste0("s(", var_name, ")")
  fit_col <- which(colnames(pred$fit) == term_label)

  data.frame(
    x   = x_grid,
    fit = pred$fit[, fit_col],
    se  = pred$se.fit[, fit_col]
  )
}

# ── Helper: draw overlaid comparison for one variable ─────────────────────────
plot_overlay <- function(var_name) {
  curve_full <- extract_smooth(model_full, var_name)
  curve_down <- extract_smooth(model_down, var_name)

  y_all <- c(
    curve_full$fit - 1.96 * curve_full$se,
    curve_full$fit + 1.96 * curve_full$se,
    curve_down$fit - 1.96 * curve_down$se,
    curve_down$fit + 1.96 * curve_down$se
  )

  plot(NULL,
       xlim = range(c(curve_full$x, curve_down$x)),
       ylim = range(y_all, na.rm = TRUE),
       xlab = var_name, ylab = "Partial effect (log-odds)",
       main = paste0("s(", var_name, ")"))

  polygon(c(curve_full$x, rev(curve_full$x)),
          c(curve_full$fit - 1.96 * curve_full$se,
            rev(curve_full$fit + 1.96 * curve_full$se)),
          col = rgb(0.2, 0.4, 0.8, 0.2), border = NA)
  lines(curve_full$x, curve_full$fit, col = rgb(0.2, 0.4, 0.8), lwd = 2)

  polygon(c(curve_down$x, rev(curve_down$x)),
          c(curve_down$fit - 1.96 * curve_down$se,
            rev(curve_down$fit + 1.96 * curve_down$se)),
          col = rgb(0.8, 0.2, 0.2, 0.15), border = NA)
  lines(curve_down$x, curve_down$fit, col = rgb(0.8, 0.2, 0.2), lwd = 2, lty = 2)

  legend("topleft",
         legend = c("Full data", "Downsampled + KZ offset"),
         col    = c(rgb(0.2, 0.4, 0.8), rgb(0.8, 0.2, 0.2)),
         lwd = 2, lty = c(1, 2), bty = "n", cex = 0.9)
}

# ── Interactive plots (on-screen) ─────────────────────────────────────────────
cat("\n── Separate smooth plots ──\n")
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot.gam(model_full, pages = 0, main = "", select = 1); title("Full: s(loan_age)")
plot.gam(model_full, pages = 0, main = "", select = 2); title("Full: s(incentive)")
plot.gam(model_down, pages = 0, main = "", select = 1); title("Down: s(loan_age)")
plot.gam(model_down, pages = 0, main = "", select = 2); title("Down: s(incentive)")

readline("Press [Enter] for overlaid comparison...")

cat("── Overlaid comparison ──\n")
par(mfrow = c(1, 2), mar = c(4.5, 4.5, 3, 1))
plot_overlay("loan_age")
plot_overlay("incentive")

readline("Press [Enter] to save PDF and exit...")

# ── Save to PDF ───────────────────────────────────────────────────────────────
pdf(PDF_OUT, width = 12, height = 10, onefile = TRUE)

par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot.gam(model_full, pages = 0, main = "", select = 1); title("Full: s(loan_age)")
plot.gam(model_full, pages = 0, main = "", select = 2); title("Full: s(incentive)")
plot.gam(model_down, pages = 0, main = "", select = 1); title("Down: s(loan_age)")
plot.gam(model_down, pages = 0, main = "", select = 2); title("Down: s(incentive)")

par(mfrow = c(1, 2), mar = c(4.5, 4.5, 3, 1))
plot_overlay("loan_age")
plot_overlay("incentive")

dev.off()
cat(sprintf("Plots saved to %s\n", PDF_OUT))
