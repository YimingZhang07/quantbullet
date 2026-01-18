## R benchmark for mgcv.R predict performance
## Usage (in RStudio):
##   source("tests/r/run_mgcv_benchmark.R")

time_now <- function() as.numeric(Sys.time())

# Resolve script directory (works when sourced or opened in RStudio)
get_script_dir <- function() {
  # Try RStudio API first (most reliable when running from editor)
  if (requireNamespace("rstudioapi", quietly = TRUE)) {
    ctx <- tryCatch(rstudioapi::getActiveDocumentContext(), error = function(e) NULL)
    if (!is.null(ctx) && nzchar(ctx$path)) {
      return(dirname(normalizePath(ctx$path, winslash = "/", mustWork = FALSE)))
    }
  }

  # Fallback: when sourced
  if (!is.null(sys.frame(1)$ofile)) {
    return(dirname(normalizePath(sys.frame(1)$ofile, winslash = "/", mustWork = FALSE)))
  }

  # Last resort
  getwd()
}

find_upwards <- function(start_dir, rel_path, max_depth = 6) {
  dir <- normalizePath(start_dir, winslash = "/", mustWork = FALSE)
  for (i in 0:max_depth) {
    candidate <- normalizePath(file.path(dir, rel_path), winslash = "/", mustWork = FALSE)
    if (file.exists(candidate)) return(candidate)
    parent <- dirname(dir)
    if (parent == dir) break
    dir <- parent
  }
  return(NA_character_)
}

script_dir <- get_script_dir()

mgcv_path <- find_upwards(script_dir, file.path("src", "quantbullet", "r", "mgcv.R"))
parquet_path <- find_upwards(script_dir, file.path("tests", "_cache_dir", "mgcv_benchmark.parquet"))

cat(sprintf("[R benchmark] script_dir: %s\n", script_dir))
cat(sprintf("[R benchmark] mgcv.R: %s\n", mgcv_path))
cat(sprintf("[R benchmark] parquet: %s\n", parquet_path))

if (is.na(mgcv_path) || !file.exists(mgcv_path)) stop("mgcv.R not found (searching upwards from): ", script_dir)
if (is.na(parquet_path) || !file.exists(parquet_path)) stop("Parquet not found (searching upwards from): ", script_dir)

source(mgcv_path)

formula <- "y ~ s(x1) + s(x2)"

cat("[R benchmark] Reading parquet...\n")
t0 <- time_now()
dt <- data.table::as.data.table(arrow::read_parquet(parquet_path))
t_read <- time_now() - t0
cat(sprintf("[R benchmark] Read parquet: %.2fs\n", t_read))

cat("[R benchmark] Fitting model...\n")
t0 <- time_now()
res <- fit_gam_api(
  data_train = dt,
  model_formula = formula,
  family_str = "gaussian",
  num_cores = 8,
  discrete = TRUE,
  nthreads = 8
)
t_fit <- time_now() - t0
cat(sprintf("[R benchmark] Fit time: %.2fs\n", t_fit))

if (!isTRUE(res$ok)) stop("Fit failed: ", res$error_msg)
model <- res$model

cat("[R benchmark] Predicting...\n")
t0 <- time_now()
pred <- predict_gam_chunked_api(
  gam_fit = model,
  X = dt,
  type = "response",
  chunk_size = 500000L
)
t_pred <- time_now() - t0
cat(sprintf("[R benchmark] Predict time: %.2fs\n", t_pred))
cat(sprintf("[R benchmark] Pred length: %d\n", length(pred)))

cat("[R benchmark] Done.\n")

