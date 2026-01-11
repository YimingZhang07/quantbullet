## ---- Locale / encoding guard (for rpy2 / Windows) ----
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
})

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

  for (colname in common_cols) {
    x_fit <- fit_model[[colname]]
    if (!is.factor(x_fit)) next

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

strip_gam_object <- function(cm) {
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

  stripped_size <- mb(cm)
  message(sprintf("from size %d MB to %d MB", orig_size, stripped_size))
  cm
}

predict_gam_parallel_api <- function(gam_fit, X, set_level = FALSE, type = "response",
                              num_cores_predict = 20,
                              num_split = 8)
{
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

  start_time <- Sys.time()
  message(sprintf("Predict with %d cores.", num_cores_predict))

  cl <- parallel::makeCluster(num_cores_predict)
  on.exit({
    try(parallel::stopCluster(cl), silent = TRUE)
  }, add = TRUE)

  # Ensure mgcv is available on workers (and data.table if ed is data.table)
  parallel::clusterEvalQ(cl, {
    library(mgcv)
    NULL
  })

  # Export only what we need
  parallel::clusterExport(
    cl,
    varlist = c("gam_fit", "X", "type"),
    envir = environment()
  )

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

  end_time <- Sys.time()
  print(end_time - start_time)
  return(predictions)
}

# fit_bam_api <- function(
#   data_train,
#   model_formula,
#   weights_col = "weight",
#   family_str = "gaussian",
#   num_cores = 4,
#   maxit = 100,
#   scale = -1,
#   min_sp = NULL,
#   coef_init = NULL
# ) {
#   data_train <- data.table::as.data.table(data_train)

#   if (!weights_col %in% names(data_train)) {
#     data_train[, (weights_col) := 1]
#   }

#   if (is.character(model_formula)) {
#     model_formula <- as.formula(model_formula)
#   }

#   family <- switch(
#     family_str,
#     "gaussian" = gaussian(),
#     "binomial" = binomial(),
#     stop("Unsupported family_str: ", family_str)
#   )

#   cl <- NULL
#   if (num_cores > 1) {
#     cl <- parallel::makeCluster(num_cores)
#     on.exit(try(parallel::stopCluster(cl), silent = TRUE), add = TRUE)
#   }

#   ctrl <- mgcv::gam.control(trace = FALSE, maxit = maxit)

#   model_obj <- mgcv::bam(
#     formula = model_formula,
#     data = as.data.frame(data_train),
#     weights = data_train[[weights_col]],
#     family = family,
#     method = "REML",
#     cluster = cl,
#     control = ctrl,
#     scale = scale,
#     min.sp = min_sp,
#     coef = coef_init
#   )

#   list(ok = TRUE, model = model_obj, error = NULL)
# }

fit_gam_api <- function(
  data_train,
  model_formula,
  weights_col = "weight",
  family_str = c("gaussian", "binomial"),
  num_cores = 4,
  maxit = 100,
  scale = -1,
  min_sp = NULL,
  coef_init = NULL
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

    cl <- NULL
    if (isTRUE(num_cores > 1)) {
      cl <- parallel::makeCluster(as.integer(num_cores))
      on.exit(try(parallel::stopCluster(cl), silent = TRUE), add = TRUE)

      # optional: keep libs consistent across workers (can help on Windows)
      # parallel::clusterEvalQ(cl, .libPaths(.libPaths()))
    }

    ctrl <- mgcv::gam.control(trace = FALSE, maxit = as.integer(maxit))

    model_obj <- mgcv::bam(
      formula = model_formula,
      data = as.data.frame(dt),
      weights = w,
      family = family,
      method = "REML",
      cluster = cl,
      control = ctrl,
      scale = scale,
      min.sp = min_sp,
      coef = coef_init
    )

    out$ok <- TRUE
    out$model <- model_obj
    out
  }, error = function(e) {
    out$error_msg <- conditionMessage(e)
    out
  })
}

plot_gam_smooth_api <- function(
  model,
  fpath,
  width  = 1200,
  height = 800,
  dpi    = 150,
  pages  = 1,
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

  plot.gam(model, pages = pages, ...)
  invisible(TRUE)
}

