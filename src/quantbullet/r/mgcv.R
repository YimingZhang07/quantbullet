## ---- Locale / encoding guard (for rpy2 / Windows) ----
# print(paste("SOURCED mgcv.R @ ", Sys.time()))

try({
  Sys.setenv(LANGUAGE = "en")
  Sys.setlocale("LC_ALL", "C")
}, silent = TRUE)

suppressPackageStartupMessages({
    library(arrow)
    library(dplyr)
    library(stringr)
    library(data.table)
    library(ggplot2)
    library(gplots)
    library(mgcv)
    library(parallel)
    library(snow)
    library(doSNOW)
    library(foreach)
    library(itertools)
    library(glue)
    library(pdftools)
    library(itertools)
    library(glue)
})

# --- cluster manager (PSOCK) ---

# changed from .qb_cluster_env <- new.env(parent = emptyenv())
# to avoid overwriting existing cluster env on re-source
if (!exists(".qb_cluster_env", envir = .GlobalEnv, inherits = FALSE)) {
  .qb_cluster_env <- new.env(parent = emptyenv())
}

qb_stop_cluster <- function() {
  if (exists("cl", envir = .qb_cluster_env, inherits = FALSE)) {
    cl <- get("cl", envir = .qb_cluster_env)
    try(parallel::stopCluster(cl), silent = TRUE)
    rm("cl", envir = .qb_cluster_env)
  }
  if (exists("n", envir = .qb_cluster_env, inherits = FALSE)) {
    rm("n", envir = .qb_cluster_env)
  }
  invisible(TRUE)
}

qb_get_cluster <- function(num_cores) {
  num_cores <- as.integer(num_cores)
  if (is.na(num_cores) || num_cores <= 1L) return(NULL)

  # already have a cluster of the right size -> reuse
  if (exists("cl", envir = .qb_cluster_env, inherits = FALSE) &&
      exists("n",  envir = .qb_cluster_env, inherits = FALSE)) {
    cl <- get("cl", envir = .qb_cluster_env)
    n0 <- get("n",  envir = .qb_cluster_env)
    if (!is.null(cl) && length(cl) == n0 && n0 == num_cores){
      print(glue("Reusing existing cluster with {n0} cores."))
      return(cl)
    }

    # size mismatch or invalid -> recreate
    qb_stop_cluster()
  }

  # create new cluster (use --vanilla to avoid loading .Rprofile/site)
  cl <- parallel::makeCluster(num_cores, outfile = "", rscript_args = "--vanilla")

  # warm-up: load needed pkgs once per worker
  parallel::clusterEvalQ(cl, {
    library(mgcv)
    library(data.table)
    NULL
  })

  assign("cl", cl, envir = .qb_cluster_env)
  assign("n",  num_cores, envir = .qb_cluster_env)

  cl
}

# ---------- Pinned data registry ----------
# .qb_pin_env <- new.env(parent = emptyenv())
if (!exists(".qb_pin_env", envir = .GlobalEnv, inherits = FALSE)) {
  .qb_pin_env <- new.env(parent = emptyenv())
}

qb_pin_put <- function(name, df, as_datatable = TRUE, lock = TRUE) {
  # name: string key
  # df: data.frame coming from rpy2
  # lock: set as "locked binding" to prevent accidental overwrite

  if (!is.character(name) || length(name) != 1L) stop("name must be a single string")

  if (as_datatable) {
    df <- data.table::as.data.table(df)
  }

  # Store
  assign(name, df, envir = .qb_pin_env)

  if (isTRUE(lock)) {
    # prevent overwrite unless explicitly dropped
    lockBinding(name, .qb_pin_env)
  }

  invisible(TRUE)
}

qb_pin_get <- function(name) {
  if (!exists(name, envir = .qb_pin_env, inherits = FALSE)) {
    stop("Pinned data not found: ", name)
  }
  get(name, envir = .qb_pin_env, inherits = FALSE)
}

qb_pin_exists <- function(name) {
  exists(name, envir = .qb_pin_env, inherits = FALSE)
}

qb_pin_drop <- function(name) {
  if (exists(name, envir = .qb_pin_env, inherits = FALSE)) {
    # unlock before removal
    if (bindingIsLocked(name, .qb_pin_env)) unlockBinding(name, .qb_pin_env)
    rm(list = name, envir = .qb_pin_env)
    invisible(TRUE)
  } else {
    invisible(FALSE)
  }
}

qb_pin_drop_all <- function() {
  all_names <- ls(envir = .qb_pin_env, all.names = TRUE)
  for (name in all_names) {
    qb_pin_drop(name)
  }
  invisible(TRUE)
}

qb_pin_list <- function() {
  ls(envir = .qb_pin_env, all.names = TRUE)
}

qb_pin_info <- function(name) {
  dt <- qb_pin_get(name)
  list(
    name = name,
    nrow = nrow(dt),
    ncol = ncol(dt),
    colnames = names(dt),
    object_size = format(utils::object.size(dt), units = "auto")
  )
}

qb_pin_put_parquet <- function(data_name, parquet_path, lock = TRUE) {
  if (!requireNamespace("arrow", quietly = TRUE)) stop("arrow package not installed")
  if (!requireNamespace("data.table", quietly = TRUE)) stop("data.table package not installed")

  dt <- data.table::as.data.table(arrow::read_parquet(parquet_path))
  qb_pin_put(data_name, dt, as_datatable = TRUE, lock = lock)
  invisible(TRUE)
}


sanitize_factor_levels <- function(fit, data, set_level = FALSE, verbose = TRUE) {

  # Ensure factor levels in new data are compatible with the fitted model.
  # Any unseen factor levels are either replaced with a fallback level
  # or set to NA, to prevent prediction errors in mgcv::predict.gam().

  if (!data.table::is.data.table(data)) {
    data <- data.table::as.data.table(data)
  }

  fit_model <- fit$model
  if (is.null(fit_model) || !is.data.frame(fit_model)) {
    stop("fit$model is missing or not a data.frame. Cannot infer factor levels.")
  }

  # only check columns that exist in both fit$model and data
  common_cols <- intersect(names(fit_model), names(data))
  
  # Pre-identify factor columns only (avoid checking every column)
  factor_cols <- character(0)
  for (colname in common_cols) {
    if (is.factor(fit_model[[colname]])) {
      factor_cols <- c(factor_cols, colname)
    }
  }
  
  # Early return if no factors to sanitize
  if (length(factor_cols) == 0L) return(data)

  # Process only factor columns
  for (colname in factor_cols) {
    x_fit <- fit_model[[colname]]
    lvls <- levels(x_fit)

    # data col might not be factor; compare as character safely
    x_new <- data[[colname]]

    bad_idx <- which(!is.na(x_new) & !(as.character(x_new) %in% lvls))
    if (length(bad_idx) == 0L) next

    if (verbose) {
      message(colname)
      warning(sprintf("%s: '%s' is not in fit levels", colname, as.character(x_new[bad_idx[1]])))
    }

    if (set_level) {
      data[bad_idx, (colname) := lvls[length(lvls)]]
    } else {
      data[bad_idx, (colname) := NA]
    }

    # optional: keep as factor with training levels (often beneficial)
    # uncomment if you want strict factor typing:
    # data[, (colname) := factor(get(colname), levels = lvls)]
  }

  data
}

strip_gam_object <- function(cm, force = FALSE) {
  # Check if already stripped (avoid re-stripping)
  if (!force && !is.null(attr(cm, "qb_stripped"))) {
    return(cm)
  }
  
  if (!is.null(cm$offset) && any(cm$offset != 0, na.rm = TRUE)) {
    warning("offset is not zero (make sure this is intended).")
  }

  mb <- function(x) floor(as.numeric(object.size(x)) / 1024 / 1024)
  orig_size <- mb(cm)

  # helper: safely drop top-level fields
  drop_fields <- function(x, fields) {
    for (nm in fields) {
      if (!is.null(x[[nm]])) x[[nm]] <- NULL
    }
    x
  }

  # helper: safely drop nested fields
  drop_nested <- function(x, parent, child) {
    if (!is.null(x[[parent]]) && !is.null(x[[parent]][[child]])) {
      x[[parent]][[child]] <- NULL
    }
    x
  }

  # 1) drop big pieces that are not needed for predict()
  cm <- drop_fields(cm, c(
    "y", "residuals", "fitted.values", "effects",
    "linear.predictors", "weights", "prior.weights",
    "data"
  ))

  # model frame: keep only schema
  if (!is.null(cm$model) && is.data.frame(cm$model)) {
    cm$model <- cm$model[0, , drop = FALSE]
  }

  # qr is often huge
  cm <- drop_nested(cm, "qr", "qr")
  # if you want, you can drop more of qr:
  # if (!is.null(cm$qr)) cm$qr <- NULL

  # 2) family functions: usually not needed for prediction output
  # (keep linkinv etc. intact; just removing these is fine)
  if (!is.null(cm$family) && is.list(cm$family)) {
    cm$family <- drop_fields(cm$family, c(
      "variance", "dev.resids", "aic", "validmu", "simulate"
    ))
  }

  # 3) strip environments to reduce capture size
  # safer to use emptyenv()
  if (!is.null(cm$terms))   attr(cm$terms,   ".Environment") <- emptyenv()
  if (!is.null(cm$formula)) attr(cm$formula, ".Environment") <- emptyenv()

  # Mark as stripped
  attr(cm, "qb_stripped") <- TRUE
  
  stripped_size <- mb(cm)
  cat(sprintf("[strip_gam_object] %d MB → %d MB\n", orig_size, stripped_size))
  cm
}

predict_bam_api <- function(gam_fit, X, set_level = FALSE, type = "response",
                            chunk_size = 250000L,
                            newdata_guaranteed = FALSE,
                            discrete = TRUE,
                            n_threads = NULL,
                            gc_level = 0) {
  time_total_begin <- Sys.time()
  
  # Avoid unnecessary conversions - keep data.table if it is
  time_convert_begin <- Sys.time()
  if (!inherits(X, "data.frame")) {
    X <- as.data.frame(X)
  }
  time_convert <- as.numeric(Sys.time() - time_convert_begin)

  # Sanitize factor levels
  time_sanitize_begin <- Sys.time()
  X <- sanitize_factor_levels(gam_fit, X, set_level = set_level, verbose = FALSE)
  time_sanitize <- as.numeric(Sys.time() - time_sanitize_begin)
  
  # Strip model (this can be slow)
  time_strip_begin <- Sys.time()
  gam_fit <- strip_gam_object(gam_fit)
  time_strip <- as.numeric(Sys.time() - time_strip_begin)

  n <- nrow(X)
  if (n == 0L) return(numeric(0))

  chunk_size <- as.integer(chunk_size)
  if (is.na(chunk_size) || chunk_size <= 0L) chunk_size <- n

  # Actual prediction (predict.bam does its own internal blocking)
  time_predict_begin <- Sys.time()
  
  discrete_val <- if (is.null(discrete)) TRUE else isTRUE(discrete)
  n_threads_val <- if (is.null(n_threads)) 1L else as.integer(n_threads)

  out <- predict.bam(
    gam_fit,
    newdata = X,
    type = type,
    block.size = as.integer(chunk_size),
    newdata.guaranteed = isTRUE(newdata_guaranteed),
    discrete = discrete_val,
    n.threads = n_threads_val,
    gc.level = as.integer(gc_level)
  )
  
  time_predict <- as.numeric(Sys.time() - time_predict_begin)
  
  time_total <- as.numeric(Sys.time() - time_total_begin)
  
  # Detailed timing breakdown
  cat(sprintf("[R predict_bam] n=%d | convert=%.3fs sanitize=%.3fs strip=%.3fs predict=%.3fs | total=%.3fs\n", 
                  n, time_convert, time_sanitize, time_strip, time_predict, time_total))

  out
}

predict_bam_pinned_data_api <- function(gam_fit, data_name,
                                        set_level = FALSE, type = "response",
                                        chunk_size = 250000L,
                                        newdata_guaranteed = FALSE,
                                        discrete = TRUE,
                                        n_threads = NULL,
                                        gc_level = 0)
{
  X <- qb_pin_get(data_name)  # will error if missing

  predict_bam_api(
    gam_fit     = gam_fit,
    X           = X,
    set_level   = set_level,
    type        = type,
    chunk_size  = chunk_size,
    newdata_guaranteed = newdata_guaranteed,
    discrete    = discrete,
    n_threads   = n_threads,
    gc_level    = gc_level
  )
}

predict_gam_parallel_pinned_data_api <- function(gam_fit, data_name,
                              set_level = FALSE, type = "response",
                              num_cores_predict = 20,
                              num_split = 8)
{
  X <- qb_pin_get(data_name)  # will error if missing

  predict_gam_parallel_api(
    gam_fit            = gam_fit,
    X                  = X,
    set_level         = set_level,
    type              = type,
    num_cores_predict = num_cores_predict,
    num_split         = num_split
  )
}

predict_gam_parallel_api <- function(gam_fit, X, set_level = FALSE, type = "response",
                              num_cores_predict = 20,
                              num_split = 8)
{
  time_begin = Sys.time()
  # --- keep original semantics ---
  if (inherits(X, "data.table")) {
    X <- data.table::copy(X)
  } else {
    X <- as.data.frame(X)  # avoid surprises
  }

  X <- sanitize_factor_levels(gam_fit, X, set_level = set_level)
  gam_fit <- strip_gam_object(gam_fit)

  n <- nrow(X)
  if (n == 0L) {
    return(if (type == "terms") X[0, , drop = FALSE] else numeric(0))
  }

  # Decide chunks: don't oversplit unnecessarily
  num_cores_predict <- max(1L, as.integer(num_cores_predict))
  num_split <- max(1L, as.integer(num_split))
  num_split <- min(num_split, n)

  # split indices (cheap) instead of splitting data (expensive)
  # roughly equal sized blocks
  idx <- split(seq_len(n), cut(seq_len(n), breaks = num_split, labels = FALSE))

  time_cluster_begin <- Sys.time()
  cat(sprintf("[R parallel] Predict with %d cores.\n", num_cores_predict))

  cl <- qb_get_cluster(num_cores_predict)

  # Export only what we need
  parallel::clusterExport(
    cl,
    varlist = c("gam_fit", "X", "type"),
    envir = environment()
  )

  time_cluster_finish <- Sys.time()

  # worker function: subset by indices, then predict
  worker_fun <- function(ii) {
    # subset with drop=FALSE to preserve data.frame behavior
    dat <- X[ii, , drop = FALSE]

    if (type == "terms") {
      predict.gam(gam_fit, newdata = dat, type = type)
    } else {
      as.numeric(predict.gam(gam_fit, newdata = dat, type = type))
    }
  }

  # Load-balanced apply tends to be better when chunk cost varies
  pieces <- parallel::parLapplyLB(cl, idx, worker_fun)

  predictions <- if (type == "terms") {
    do.call(rbind, pieces)
  } else {
    unlist(pieces, use.names = FALSE)
  }

  time_end <- Sys.time()

  print(glue("predict_gam_parallel_api::prepare {time_cluster_begin - time_begin}"))
  print(glue("predict_gam_parallel_api::cluster {time_cluster_finish - time_cluster_begin}"))
  print(glue("predict_gam_parallel_api::predict  {time_end - time_cluster_finish}"))
  print(glue("predict_gam_parallel_api::total    {time_end - time_begin}"))
  return(predictions)
}

fit_gam_pinned_data_api <- function(
  data_name,
  model_formula,
  weights_col = "weight",
  family_str = c("gaussian", "binomial"),
  num_cores = 4,
  maxit = 100,
  scale = -1,
  min_sp = NULL,
  coef_init = NULL,
  discrete = TRUE,
  nthreads = 1
) {
  data_train <- qb_pin_get(data_name)  # will error if missing

  fit_gam_api(
    data_train     = data_train,
    model_formula  = model_formula,
    weights_col    = weights_col,
    family_str     = family_str,
    num_cores      = num_cores,
    maxit          = maxit,
    scale          = scale,
    min_sp         = min_sp,
    coef_init      = coef_init,
    discrete       = discrete,
    nthreads      = nthreads
  )
}

fit_gam_api <- function(
  data_train,
  model_formula,
  weights_col = "weight",
  family_str = c("gaussian", "binomial"),
  num_cores = 4,
  maxit = 100,
  scale = -1,
  min_sp = NULL,
  coef_init = NULL,
  discrete = TRUE,
  nthreads = 1
) {
  out <- list(ok = FALSE, model = NULL, error_msg = NULL)

  tryCatch({
    dt <- data.table::as.data.table(data_train)

    if (!weights_col %in% names(dt)) {
      dt[, (weights_col) := 1]
    }

    w <- dt[[weights_col]]
    if (anyNA(w)) stop("weights contain NA")
    if (any(w < 0)) stop("weights must be non-negative")

    if (is.character(model_formula)) {
      model_formula <- stats::as.formula(model_formula)
    } else if (!inherits(model_formula, "formula")) {
      stop("model_formula must be a formula or a character string.")
    }

    family_str <- match.arg(family_str)
    family <- switch(
      family_str,
      gaussian = stats::gaussian(),
      binomial = stats::binomial()
    )

    # below logic has been improved to reuse clusters
    # cl <- qb_get_cluster(num_cores)
    # when discrete=TRUE, clusters are not used at all so avoid creating them
    if (isTRUE(discrete)) {
      cl <- NULL
    } else {
      cl <- qb_get_cluster(num_cores)
    }
    ctrl <- mgcv::gam.control(trace = FALSE, maxit = as.integer(maxit))

    # we have removed the as.data.frame(dt) conversion to reduce overhead
    # use fREML, and discrete=TRUE for large data
    method_val <- if (isTRUE(discrete)) "fREML" else "REML"
    
    model_obj <- mgcv::bam(
      formula = model_formula,
      data = dt,
      weights = w,
      family = family,
      method = method_val,
      cluster = cl,
      control = ctrl,
      scale = scale,
      min.sp = min_sp,
      coef = coef_init,
      discrete = discrete,
      nthreads = as.integer(max(1, nthreads))
    )

    out$ok <- TRUE
    out$model <- model_obj
    out
  }, error = function(e) {
    out$error_msg <- conditionMessage(e)
    out
  })
}

plot_gam_smooth_pages_api <- function(
  model,
  pages  = 1,
  rug    = FALSE,
  scheme = 1,
  scale  = FALSE,
  ...
) {
  scale_val <- if (is.logical(scale)) {
    ifelse(scale, 0, -1)
  } else {
    as.numeric(scale)
  }

  plot.gam(
    model,
    pages = pages,
    rug = rug,
    scheme = scheme,
    scale = scale_val,
    ...
  )
  invisible(TRUE)
}

plot_gam_smooth_api <- function(
  model,
  fpath,
  width  = 1200,
  height = 800,
  dpi    = 150,
  pages  = 1,
  rug    = FALSE,
  scheme = 1,
  scale  = FALSE,
  ...
) {
  ext <- tolower(tools::file_ext(fpath))

  if (!nzchar(ext)) {
    stop("fpath must have a file extension (png, pdf, svg, ...)")
  }

  # open device based on extension
  if (ext %in% c("png", "jpeg", "jpg", "tiff", "tif")) {
    dev <- if (ext == "jpg") "jpeg" else if (ext == "tif") "tiff" else ext
    fn <- get(dev, envir = asNamespace("grDevices"))
    fn(
      filename = fpath,
      width = width,
      height = height,
      res = dpi,
      ...
    )

  } else if (ext == "pdf") {
    grDevices::pdf(
      file = fpath,
      width  = width  / dpi,
      height = height / dpi,
      onefile = TRUE,
      ...
    )

  } else if (ext == "svg") {
    grDevices::svg(
      filename = fpath,
      width  = width  / dpi,
      height = height / dpi,
      ...
    )

  } else {
    stop("Unsupported file extension: .", ext)
  }

  on.exit(grDevices::dev.off(), add = TRUE)

  plot_gam_smooth_pages_api(
    model = model,
    pages = pages,
    rug = rug,
    scheme = scheme,
    scale = scale,
    ...
  )
  invisible(TRUE)
}

model_summary_text_api <- function(model, include_header = TRUE) {
  smy <- capture.output(summary(model))
  if (isTRUE(include_header)) {
    smy <- c("Model Summary", "", smy)
  }
  smy
}

model_report_pdf_api <- function(
  model,
  fpath,
  width  = 1200,
  height = 800,
  dpi    = 150,
  pages  = 1,
  rug    = FALSE,
  scheme = 1,
  scale  = FALSE,
  include_header = TRUE
) {
  ext <- tolower(tools::file_ext(fpath))
  if (ext != "pdf") {
    stop("model_report_pdf_api only supports .pdf output.")
  }

  grDevices::pdf(
    file = fpath,
    width  = width  / dpi,
    height = height / dpi,
    onefile = TRUE
  )
  on.exit(grDevices::dev.off(), add = TRUE)

  # Smooth pages first
  plot_gam_smooth_pages_api(
    model = model,
    pages = pages,
    rug = rug,
    scheme = scheme,
    scale = scale
  )

  # Summary page last
  smy <- model_summary_text_api(model, include_header = include_header)
  gplots::textplot(smy, valign = "top", cex = 0.8)

  invisible(TRUE)
}