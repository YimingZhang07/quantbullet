## Pure R test for mgcv.R fit + predict using local data.parquet

get_script_dir <- function() {
  # Method 1: commandArgs (works with Rscript from command line)
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- args[grep("^--file=", args)]
  if (length(file_arg) > 0) {
    script_path <- sub("^--file=", "", file_arg[1])
    return(dirname(normalizePath(script_path, mustWork = TRUE)))
  }
  
  # Method 2: sys.frames (works when sourced)
  for (i in seq_len(sys.nframe())) {
    if (!is.null(sys.frame(i)$ofile)) {
      return(dirname(normalizePath(sys.frame(i)$ofile, mustWork = TRUE)))
    }
  }
  
  # Method 3: rstudioapi (works in RStudio with saved file)
  if (requireNamespace("rstudioapi", quietly = TRUE)) {
    if (rstudioapi::isAvailable()) {
      context <- tryCatch(
        rstudioapi::getActiveDocumentContext(),
        error = function(e) NULL
      )
      if (!is.null(context) && !is.null(context$path) && nzchar(context$path)) {
        return(dirname(normalizePath(context$path, mustWork = TRUE)))
      }
    }
  }
  
  # Fallback: current working directory
  stop("Could not determine script directory. Please run with: Rscript test_mgcv_r.R")
}

script_dir <- get_script_dir()
data_path <- file.path(script_dir, "data.parquet")
if (!file.exists(data_path)) {
  stop("data.parquet not found at: ", data_path)
}

mgcv_path <- normalizePath(
  file.path(script_dir, "..", "..", "src", "quantbullet", "r", "mgcv.R"),
  mustWork = TRUE
)
source(mgcv_path)

cat("Loading data:", data_path, "\n")
dt <- arrow::read_parquet(data_path)
dt <- as.data.frame(dt)

if ("level" %in% names(dt)) {
  dt$level <- as.factor(dt$level)
}

formula_str <- "happiness ~ s(age) + s(income) + s(education) + level"

cat("Fitting model with formula:", formula_str, "\n")
fit_res <- fit_gam_api(
  data_train = dt,
  model_formula = formula_str,
  family_str = "gaussian",
  num_cores = 1,
  discrete = TRUE,
  nthreads = 1
)

if (!isTRUE(fit_res$ok)) {
  stop("Fit failed: ", fit_res$error_msg)
}

gam_fit <- fit_res$model
cat("Fit OK\n")

cat("Predicting with predict_bam_api...\n")
pred <- tryCatch(
  predict_bam_api(
    gam_fit = gam_fit,
    X = dt,
    type = "response",
    newdata_guaranteed = FALSE,
    discrete = TRUE,
    n_threads = 1
  ),
  error = function(e) {
    message("predict_bam_api error: ", conditionMessage(e))
    NULL
  }
)
