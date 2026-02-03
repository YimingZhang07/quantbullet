## ---- Report generation combining model and diagnostic plots ----
## Orchestrates mgcv.R (model) and plots.R (diagnostics) into unified reports
##
## Usage:
##   source("mgcv.R")   # model fitting and smooth plots
##   source("plots.R")  # diagnostic plots
##   source("reports.R") # this file - combined reports

# Check dependencies are loaded
.check_plots_dep <- function() {
  if (!exists("render_diagnostic_plots", mode = "function") ||
      !exists("build_diag_configs", mode = "function")) {
    stop("plots.R must be sourced before reports.R")
  }
  invisible(TRUE)
}

.check_mgcv_dep <- function() {
  if (!exists("plot_gam_smooth_pages_api", mode = "function")) {
    stop("mgcv.R must be sourced before using model report functions")
  }
  invisible(TRUE)
}

.coalesce <- function(x, default) {
  if (is.null(x)) default else x
}

.write_pdf_pages <- function(fpath, width, height, dpi, plot_fn) {
  grDevices::pdf(
    file = fpath,
    width = width / dpi,
    height = height / dpi,
    onefile = TRUE
  )
  on.exit(grDevices::dev.off(), add = TRUE)
  plot_fn()
  invisible(TRUE)
}

# ============================================================================
# Combined Model + Diagnostic Report
# ============================================================================

#' Create comprehensive model report PDF
#'
#' Combines GAM smooth plots, summary text, and diagnostics into a single PDF.
#' Sequence: smooth curves -> summary text -> diagnostics.
#'
#' @param model Fitted GAM/BAM model object
#' @param df Data frame used for fitting (with actual and predicted values)
#' @param diag_configs List of diagnostic plot configs (required for diagnostics).
#' @param fpath Output PDF file path
#' @param width Default page width in pixels (default 1200)
#' @param height Default page height in pixels (default 800)
#' @param dpi Default resolution (default 150)
#' @param smooth_width Smooth section page width in pixels (optional)
#' @param smooth_height Smooth section page height in pixels (optional)
#' @param smooth_dpi Smooth section DPI (optional)
#' @param summary_width Summary page width in pixels (optional)
#' @param summary_height Summary page height in pixels (optional)
#' @param summary_dpi Summary page DPI (optional)
#' @param diag_width Diagnostics section page width in pixels (optional)
#' @param diag_height Diagnostics section page height in pixels (optional)
#' @param diag_dpi Diagnostics section DPI (optional)
#' @param smooth_pages Number of smooth terms per page (default 1)
#' @param include_smooths Include GAM smooth plots (default TRUE)
#' @param include_diagnostics Include diagnostic plots (default TRUE)
#' @param include_summary Include model summary page (default TRUE)
#' @param rug Show rug on smooth plots (default FALSE)
#' @param scheme Color scheme for smooth plots (default 1)
#' @return invisible(TRUE)
#'
#' @examples
#' \dontrun{
#' fit <- bam(y ~ s(x1) + s(x2) + factor_var, data = df)
#' df$pred <- predict(fit)
#' 
#' combined_model_report(
#'   model = fit,
#'   df = df,
#'   diag_configs = my_diag_configs,
#'   fpath = "model_report.pdf"
#' )
#' }
combined_model_report <- function(
  model,
  df,
  fpath,
  diag_configs = NULL,
  width = 1200,
  height = 800,
  dpi = 150,
  smooth_width = NULL,
  smooth_height = NULL,
  smooth_dpi = NULL,
  summary_width = NULL,
  summary_height = NULL,
  summary_dpi = NULL,
  diag_width = NULL,
  diag_height = NULL,
  diag_dpi = NULL,
  smooth_pages = 1,
  include_smooths = TRUE,
  include_diagnostics = TRUE,
  include_summary = TRUE,
  rug = FALSE,
  scheme = 1
) {
  .check_plots_dep()
  .check_mgcv_dep()
  
  ext <- tolower(tools::file_ext(fpath))
  if (ext != "pdf") {
    stop("combined_model_report only supports .pdf output.")
  }
  
  # Build diagnostic configs if needed
  if (include_diagnostics && is.null(diag_configs)) {
    stop("diag_configs is required when include_diagnostics=TRUE")
  }

  smooth_size <- list(
    width = if (!is.null(smooth_width)) smooth_width else width,
    height = if (!is.null(smooth_height)) smooth_height else height,
    dpi = if (!is.null(smooth_dpi)) smooth_dpi else dpi
  )
  summary_size <- list(
    width = if (!is.null(summary_width)) summary_width else width,
    height = if (!is.null(summary_height)) summary_height else height,
    dpi = if (!is.null(summary_dpi)) summary_dpi else dpi
  )
  diag_size <- list(
    width = if (!is.null(diag_width)) diag_width else width,
    height = if (!is.null(diag_height)) diag_height else height,
    dpi = if (!is.null(diag_dpi)) diag_dpi else dpi
  )

  smooth_pages_fn <- function() {
    plot_gam_smooth_pages_api(
      model = model,
      pages = smooth_pages,
      rug = rug,
      scheme = scheme,
      scale = FALSE
    )
  }

  summary_pages_fn <- function() {
    smy <- model_summary_text_api(model, include_header = TRUE)
    gplots::textplot(smy, valign = "top", cex = 0.8)
  }

  diag_pages_fn <- function() {
    render_diagnostic_plots(df, diag_configs, defaults = list())
  }

  has_smooth <- isTRUE(include_smooths)
  has_summary <- isTRUE(include_summary)
  has_diag <- isTRUE(include_diagnostics)

  sections <- list()
  if (has_smooth) {
    sections <- append(sections, list(list(
      name = "smooth",
      size = smooth_size,
      fn = smooth_pages_fn
    )))
  }
  if (has_summary) {
    sections <- append(sections, list(list(
      name = "summary",
      size = summary_size,
      fn = summary_pages_fn
    )))
  }
  if (has_diag) {
    sections <- append(sections, list(list(
      name = "diagnostics",
      size = diag_size,
      fn = diag_pages_fn
    )))
  }

  if (length(sections) == 0) {
    warning("No report pages to render.")
    return(invisible(TRUE))
  }

  if (length(sections) == 1) {
    sec <- sections[[1]]
    .write_pdf_pages(fpath, sec$size$width, sec$size$height, sec$size$dpi, sec$fn)
    return(invisible(TRUE))
  }

  # Always render each section separately and combine
  tmp_paths <- c()
  for (sec in sections) {
    tmp <- tempfile(fileext = ".pdf")
    .write_pdf_pages(tmp, sec$size$width, sec$size$height, sec$size$dpi, sec$fn)
    tmp_paths <- c(tmp_paths, tmp)
  }
  pdftools::pdf_combine(tmp_paths, output = fpath)
  unlink(tmp_paths)
  
  invisible(TRUE)
}


