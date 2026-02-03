## ---- Plotting utilities for model diagnostics ----
## Mirrors key functions from quantbullet/plot/binned_plots.py
## Designed to integrate with mgcv.R model reports

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tidyr)
  library(scales)
})

# ============================================================================
# Color Palette (Economist-inspired)
# ============================================================================

ECONOMIST_COLORS <- c(
  "#E3120B",  # Red
  "#336699",  # Blue
  "#29A329",  # Green
  "#9932CC",  # Purple
  "#FF8C00",  # Orange
  "#008B8B",  # Teal
  "#DC143C",  # Crimson
  "#4169E1"   # Royal Blue
)

LONDON_GRAY <- "#7F7F7F"

# ============================================================================
# Data Preparation
# ============================================================================

#' Prepare binned data for actual vs predicted plots
#'
#' @param df Data frame with x, actual, and prediction columns
#' @param x_col Name of x-axis column
#' @param act_col Name of actual values column
#' @param pred_cols Character vector of prediction column names
#' @param facet_col Optional faceting column name
#' @param weight_col Optional weight column name
#' @param bins Binning strategy: NULL (quantile), FALSE/"discrete" (exact values), or numeric vector (custom breaks)
#' @param n_bins Number of quantile bins (default 10)
#' @param min_size Minimum scatter point size (default 2)
#' @param max_size Maximum scatter point size (default 8)
#' @return Data frame with aggregated binned data
prepare_binned_data <- function(
  df,
  x_col,
  act_col,
  pred_cols,
  facet_col = NULL,
  weight_col = NULL,
  bins = NULL,
  n_bins = 10,
  min_size = 2,
  max_size = 8
) {
  pred_cols <- as.character(pred_cols)

  # Setup weights
  if (!is.null(weight_col) && weight_col %in% names(df)) {
    weights <- df[[weight_col]]
  } else {
    weights <- rep(1, nrow(df))
  }

  # Build working data frame
  tmp <- data.frame(
    x = df[[x_col]],
    act = df[[act_col]],
    weight = weights,
    stringsAsFactors = FALSE
  )
  for (pred_col in pred_cols) {
    tmp[[pred_col]] <- df[[pred_col]]
  }
  if (!is.null(facet_col)) {
    tmp$facet <- df[[facet_col]]
  }

  # Binning logic
  if (isFALSE(bins) || identical(bins, "discrete")) {
    # Discrete mode: group by exact x values
    tmp$bin_val <- tmp$x
  } else if (is.null(bins)) {
    # Quantile binning (default)
    breaks <- unique(quantile(tmp$x, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE))
    tmp$bin <- cut(tmp$x, breaks = breaks, include.lowest = TRUE)
    # Use right edge of interval as bin_val
    tmp$bin_val <- as.numeric(sub(".*,\\s*([^]]+)\\]", "\\1", as.character(tmp$bin)))
  } else {
    # Custom bins
    tmp$bin <- cut(tmp$x, breaks = bins, include.lowest = TRUE)
    tmp$bin_val <- as.numeric(sub(".*,\\s*([^]]+)\\]", "\\1", as.character(tmp$bin)))
  }

  # Aggregation
  group_cols <- if (!is.null(facet_col)) c("facet", "bin_val") else "bin_val"

  agg <- tmp %>%
    filter(!is.na(bin_val)) %>%
    group_by(across(all_of(group_cols))) %>%
    summarise(
      act_mean = weighted.mean(act, weight, na.rm = TRUE),
      count = n(),
      across(all_of(pred_cols), ~ weighted.mean(.x, weight, na.rm = TRUE), .names = "pred_mean__{.col}"),
      .groups = "drop"
    )

  agg
}

# ============================================================================
# Plotting Functions
# ============================================================================

#' Plot binned actual vs predicted with facets (ggplot2)
#'
#' @param df Data frame
#' @param x_col X-axis column name
#' @param act_col Actual values column name
#' @param pred_col Prediction column name(s) - single string or character vector
#' @param facet_col Optional faceting column name
#' @param bins Binning strategy (NULL, FALSE/"discrete", or custom breaks)
#' @param n_bins Number of quantile bins
#' @param n_cols Number of facet columns
#' @param min_size Minimum size for points
#' @param max_size Maximum size for points
#' @param pred_colors Colors for prediction lines
#' @param y_transform Optional function to transform y values
#' @param title Optional plot title
#' @param ... Additional arguments passed to prepare_binned_data
#' @return ggplot object
plot_binned_actual_vs_pred <- function(
  df,
  x_col,
  act_col,
  pred_col,
  facet_col = NULL,
  weight_col = NULL,
  bins = NULL,
  n_bins = 10,
  n_cols = 3,
  min_size = 2,
  max_size = 8,
  pred_colors = NULL,
  y_transform = NULL,
  title = NULL,
  ...
) {
  pred_cols <- if (is.character(pred_col)) pred_col else as.character(pred_col)
  if (is.null(pred_colors)) pred_colors <- ECONOMIST_COLORS

  agg <- prepare_binned_data(
    df, x_col, act_col, pred_cols, facet_col,
    weight_col = weight_col, bins = bins, n_bins = n_bins, min_size = min_size, max_size = max_size, ...
  )
  pred_mean_cols <- setNames(paste0("pred_mean__", pred_cols), pred_cols)

  if (nrow(agg) == 0) {
    return(ggplot() + theme_void() + ggtitle("No data"))
  }

  # Apply y-transform if provided
  if (!is.null(y_transform)) {
    agg$act_mean <- y_transform(agg$act_mean)
    for (col in meta$pred_mean_cols) {
      agg[[col]] <- y_transform(agg[[col]])
    }
  }

  # Reshape predictions to long format for multi-line plotting
  id_vars <- c("bin_val", "count")
  if (!is.null(facet_col)) id_vars <- c(id_vars, "facet")

  lines_df <- agg %>%
    pivot_longer(
      cols = all_of(unname(pred_mean_cols)),
      names_to = "pred_mean_col",
      values_to = "pred_mean"
    ) %>%
    mutate(pred_col = names(pred_mean_cols)[match(pred_mean_col, pred_mean_cols)]) %>%
    filter(!is.na(pred_mean)) %>%
    arrange(bin_val)

  # Build plot
  p <- ggplot() +
    # Actual points (gray)
    geom_point(
      data = agg,
      aes(x = bin_val, y = act_mean, size = count),
      color = LONDON_GRAY,
      alpha = 0.6
    ) +
    # Prediction lines
    geom_line(
      data = lines_df,
      aes(x = bin_val, y = pred_mean, color = pred_col, group = pred_col),
      linewidth = 1
    ) +
    scale_size_area(max_size = max_size, name = "Count", n.breaks = 3) +
    scale_color_manual(values = setNames(pred_colors[seq_along(pred_cols)], pred_cols), name = "Model") +
    labs(
      title = if (!is.null(title)) title else paste(act_col, "vs", paste(pred_cols, collapse = ", ")),
      x = x_col,
      y = act_col
    ) +
    theme_bw(base_size = 11) +
    theme(
      axis.title = element_text(size = 10, face = "bold"),
      axis.text = element_text(size = 9),
      panel.grid.major = element_line(color = "#e0e0e0", linewidth = 0.3),
      panel.grid.minor = element_line(color = "#f0f0f0", linewidth = 0.2),
      legend.position = "right",
      legend.title = element_text(size = 9, face = "bold"),
      legend.text = element_text(size = 8),
      plot.title = element_text(size = 11, face = "bold", hjust = 0),
      strip.background = element_rect(fill = "white", color = "#4d4d4d"),
      strip.text = element_text(size = 9, face = "bold", color = "#333333")
    )

  # Add faceting if specified
  if (!is.null(facet_col)) {
    p <- p + facet_wrap(~ facet, ncol = n_cols)
  }

  p
}


#' Plot binned actual vs predicted with overlay (single panel, colored by group)
#'
#' @param df Data frame
#' @param x_col X-axis column name
#' @param act_col Actual values column name
#' @param pred_col Prediction column name(s)
#' @param facet_col Grouping column for overlay (required for multiple groups)
#' @param bins Binning strategy
#' @param n_bins Number of quantile bins
#' @param pred_colors Colors for groups
#' @param min_size Minimum size for points
#' @param max_size Maximum size for points
#' @param y_transform Optional function to transform y values
#' @param title Optional plot title
#' @param ... Additional arguments passed to prepare_binned_data
#' @return ggplot object
plot_binned_actual_vs_pred_overlay <- function(
  df,
  x_col,
  act_col,
  pred_col,
  facet_col = NULL,
  weight_col = NULL,
  bins = NULL,
  n_bins = 10,
  min_size = 2,
  max_size = 8,
  pred_colors = NULL,
  y_transform = NULL,
  title = NULL,
  ...
) {
  pred_cols <- if (is.character(pred_col)) pred_col else as.character(pred_col)
  if (is.null(pred_colors)) pred_colors <- ECONOMIST_COLORS

  # Create dummy facet if none provided
  if (is.null(facet_col)) {
    df <- df
    df$`_all_group` <- "All"
    facet_col <- "_all_group"
  }

  agg <- prepare_binned_data(
    df, x_col, act_col, pred_cols, facet_col,
    weight_col = weight_col, bins = bins, n_bins = n_bins, min_size = min_size, max_size = max_size, ...
  )
  pred_mean_cols <- setNames(paste0("pred_mean__", pred_cols), pred_cols)

  if (nrow(agg) == 0) {
    return(ggplot() + theme_void() + ggtitle("No data"))
  }

  # Apply y-transform if provided
  if (!is.null(y_transform)) {
    agg$act_mean <- y_transform(agg$act_mean)
    for (col in meta$pred_mean_cols) {
      agg[[col]] <- y_transform(agg[[col]])
    }
  }

  # Reshape predictions to long format
  lines_df <- agg %>%
    pivot_longer(
      cols = all_of(unname(pred_mean_cols)),
      names_to = "pred_mean_col",
      values_to = "pred_mean"
    ) %>%
    mutate(pred_col = names(pred_mean_cols)[match(pred_mean_col, pred_mean_cols)]) %>%
    filter(!is.na(pred_mean)) %>%
    arrange(facet, bin_val)

  # Color map by facet level
  facet_levels <- unique(agg$facet)
  color_map <- setNames(pred_colors[seq_along(facet_levels)], facet_levels)

  # Build plot - points and lines share color by group
  p <- ggplot() +
    geom_point(
      data = agg,
      aes(x = bin_val, y = act_mean, size = count, color = facet),
      alpha = 0.6
    ) +
    geom_line(
      data = lines_df,
      aes(
        x = bin_val,
        y = pred_mean,
        color = facet,
        linetype = if (length(pred_cols) > 1) pred_col else NULL,
        group = interaction(facet, pred_col)
      ),
      linewidth = 1
    ) +
    scale_size_area(max_size = max_size, name = "Count", n.breaks = 3) +
    scale_color_manual(values = color_map, name = facet_col) +
    labs(
      title = if (!is.null(title)) title else paste(act_col, "vs", paste(pred_cols, collapse = ", ")),
      x = x_col,
      y = act_col
    ) +
    theme_bw(base_size = 11) +
    theme(
      axis.title = element_text(size = 10, face = "bold"),
      axis.text = element_text(size = 9),
      panel.grid.major = element_line(color = "#e0e0e0", linewidth = 0.3),
      panel.grid.minor = element_line(color = "#f0f0f0", linewidth = 0.2),
      legend.position = "right",
      legend.title = element_text(size = 9, face = "bold"),
      legend.text = element_text(size = 8),
      plot.title = element_text(size = 11, face = "bold", hjust = 0)
    )

  if (length(pred_cols) > 1) {
    p <- p + scale_linetype_discrete(name = "Model")
  }

  p
}


# ============================================================================
# Diagnostic Plot Helper (config-driven)
# ============================================================================

#' Merge config with defaults (defaults only fill in missing keys)
#'
#' @param defaults Default values (used only for keys not present in cfg)
#' @param cfg User-provided config (takes precedence, even if value is NULL)
#' @return Merged config list
.merge_config <- function(defaults, cfg) {
  # Start with cfg - user's explicit config takes full precedence
  out <- if (is.null(cfg)) list() else cfg

  # Only fill in defaults for keys NOT present in cfg
  if (length(defaults)) {
    cfg_names <- names(out)
    for (nm in names(defaults)) {
      if (!(nm %in% cfg_names)) {
        out[[nm]] <- defaults[[nm]]
      }
    }
  }
  out
}

#' Normalize diagnostic config with defaults
#'
#' Defaults are only used for keys not present in cfg.
#' Config values (even NULL) take precedence over defaults.
#'
#' @param cfg User-provided config
#' @param defaults Default values for missing keys
#' @return Normalized config list
.normalize_diag_config <- function(cfg, defaults) {
  if (is.null(cfg)) cfg <- list()
  out <- .merge_config(defaults, cfg)

  # Handle overlay shorthand
  if (!is.null(out$overlay) && isTRUE(out$overlay)) {
    out$type <- "overlay"
  }

  # Default type if not specified (check key existence, not just NULL)
  if (!("type" %in% names(out)) || is.null(out$type)) {
    out$type <- "facet"
  }
  out$type <- tolower(out$type)

  # Validate type

  if (!(out$type %in% c("facet", "overlay"))) {
    stop(paste0("Invalid diag config type '", out$type, "'. Must be 'facet' or 'overlay'."))
  }

  # Required fields validation
  if (is.null(out$x_col)) stop("diag config missing x_col")
  if (is.null(out$act_col)) stop("diag config missing act_col")
  if (is.null(out$pred_col)) stop("diag config missing pred_col")

  out$pred_col <- as.character(out$pred_col)

  # Auto-generate title only if not provided (check key existence)
  if (!("title" %in% names(out)) || is.null(out$title)) {
    base_title <- paste(out$act_col, "vs", paste(out$pred_col, collapse = ", "), "by", out$x_col)
    if (!is.null(out$facet_col) && nzchar(out$facet_col)) {
      # Use appropriate label based on plot type
      group_label <- if (out$type == "overlay") "color:" else "facet:"
      out$title <- paste0(base_title, " (", group_label, " ", out$facet_col, ")")
    } else {
      out$title <- base_title
    }
  }

  out
}

#' Build diagnostic plot configs from x columns
#'
#' @param x_cols Character vector of x columns
#' @param act_col Actual values column name
#' @param pred_col Prediction column name(s)
#' @param plot_type "facet" or "overlay"
#' @param facet_col Optional faceting column
#' @param weight_col Optional weight column
#' @param bins Binning strategy
#' @param n_bins Number of bins
#' @param n_cols Number of facet columns
#' @param pred_colors Colors for predictions
#' @param min_size Minimum size for points
#' @param max_size Maximum size for points
#' @param y_transform Optional y transform
#' @param title_prefix Optional title prefix
#' @return list of config lists
build_diag_configs <- function(
  x_cols,
  act_col,
  pred_col,
  plot_type = "facet",
  facet_col = NULL,
  weight_col = NULL,
  bins = NULL,
  n_bins = 10,
  n_cols = 3,
  min_size = 2,
  max_size = 8,
  pred_colors = NULL,
  y_transform = NULL,
  title_prefix = ""
) {
  if (is.null(x_cols) || length(x_cols) == 0) {
    return(list())
  }

  lapply(x_cols, function(x_col) {
    title <- if (nzchar(title_prefix)) {
      paste(title_prefix, "-", act_col, "by", x_col)
    } else {
      NULL
    }
    list(
      type = plot_type,
      x_col = x_col,
      act_col = act_col,
      pred_col = pred_col,
      facet_col = facet_col,
      weight_col = weight_col,
      bins = bins,
      n_bins = n_bins,
      n_cols = n_cols,
      min_size = min_size,
      max_size = max_size,
      pred_colors = pred_colors,
      y_transform = y_transform,
      title = title
    )
  })
}

#' Render diagnostic plots from configs
#'
#' Each config can include:
#'   type ("facet" or "overlay"), x_col, act_col, pred_col,
#'   facet_col, weight_col, bins, n_bins, n_cols, pred_colors, y_transform, title.
#'
#' @param df Data frame with actual and predicted values
#' @param configs List of config lists (or a single config list)
#' @param defaults Optional defaults merged into each config
#' @return invisible(TRUE)
render_diagnostic_plots <- function(df, configs, defaults = list()) {
  if (is.null(configs) || length(configs) == 0) {
    return(invisible(FALSE))
  }

  # allow single config list
  if (!is.null(configs$x_col) || !is.null(configs$pred_col)) {
    configs <- list(configs)
  }

  for (cfg in configs) {
    cfg <- .normalize_diag_config(cfg, defaults)

    if (cfg$type == "overlay") {
      p <- plot_binned_actual_vs_pred_overlay(
        df,
        x_col = cfg$x_col,
        act_col = cfg$act_col,
        pred_col = cfg$pred_col,
        facet_col = cfg$facet_col,
        weight_col = cfg$weight_col,
        bins = cfg$bins,
        n_bins = cfg$n_bins,
        min_size = cfg$min_size,
        max_size = cfg$max_size,
        pred_colors = cfg$pred_colors,
        y_transform = cfg$y_transform,
        title = cfg$title
      )
    } else {
      p <- plot_binned_actual_vs_pred(
        df,
        x_col = cfg$x_col,
        act_col = cfg$act_col,
        pred_col = cfg$pred_col,
        facet_col = cfg$facet_col,
        weight_col = cfg$weight_col,
        bins = cfg$bins,
        n_bins = cfg$n_bins,
        n_cols = cfg$n_cols,
        min_size = cfg$min_size,
        max_size = cfg$max_size,
        pred_colors = cfg$pred_colors,
        y_transform = cfg$y_transform,
        title = cfg$title
      )
    }

    print(p)
  }

  invisible(TRUE)
}
