"""
Python wrapper for R plotting functions (plots.R).

Provides binned actual vs predicted plots for model diagnostics.
Can display inline in Jupyter or save to files.
"""
import logging
import tempfile
import os
from pathlib import Path
from typing import Optional, Union, List

import pandas as pd

from .r_session import get_r
from .rpy_convert import py_df_to_r, py_obj_to_r

logger = logging.getLogger(__name__)


def _with_temp_png(plot_fn, width: int, height: int, dpi: int) -> None:
    """Execute plot function with temp PNG, display inline, then cleanup."""
    try:
        from IPython.display import Image, display
    except ImportError as e:
        raise ImportError("IPython required for inline plotting. Use *_to_file() instead.") from e

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        plot_fn(tmp_path, width, height, dpi)
        display(Image(filename=str(tmp_path)))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {tmp_path}: {e}")


class RPlots:
    """
    Python wrapper for R diagnostic plotting functions.

    Wraps plots.R to provide:
    - Binned actual vs predicted plots (faceted or overlay)
    - Config-driven diagnostic plots
    """

    def __init__(self):
        """Initialize wrapper and source R plotting functions."""
        r = get_r()
        self.r_ = r

        # Source plots.R (sits next to this file)
        r_file = Path(__file__).resolve().with_name("plots.R")
        if not r_file.exists():
            raise FileNotFoundError(f"plots.R not found at: {r_file}")

        r.ro.r(f'source("{r_file.as_posix()}")')

        # Bind R functions
        self._plot_binned = r.ro.globalenv["plot_binned_actual_vs_pred"]
        self._plot_binned_overlay = r.ro.globalenv["plot_binned_actual_vs_pred_overlay"]
        self._render_diagnostic_plots = r.ro.globalenv["render_diagnostic_plots"]
        self._build_diag_configs = r.ro.globalenv["build_diag_configs"]

        # R's ggsave for saving ggplot objects
        r.ro.r("library(ggplot2)")
        self._ggsave = r.ro.r["ggsave"]

        self._dev_off = r.ro.r["dev.off"]

    def _save_ggplot(self, plot_obj, path: Path, width: int, height: int, dpi: int) -> None:
        """Save ggplot object to file."""
        self._ggsave(
            filename=str(path),
            plot=plot_obj,
            width=width / dpi,
            height=height / dpi,
            dpi=dpi,
            units="in",
        )

    # =========================================================================
    # Binned Actual vs Predicted (Faceted)
    # =========================================================================

    def plot_binned_to_file(
        self,
        df: pd.DataFrame,
        x_col: str,
        act_col: str,
        pred_col: Union[str, List[str]],
        out_path: Union[str, Path],
        facet_col: Optional[str] = None,
        weight_col: Optional[str] = None,
        n_bins: int = 10,
        n_cols: int = 3,
        width: int = 1200,
        height: int = 800,
        dpi: int = 150,
        title: Optional[str] = None,
    ) -> str:
        """Plot binned actual vs predicted and save to file."""
        out_path = Path(out_path)
        df_r = py_df_to_r(df, r=self.r_)
        pred_col_r = py_obj_to_r(pred_col, r=self.r_)

        kwargs = {
            "df": df_r,
            "x_col": x_col,
            "act_col": act_col,
            "pred_col": pred_col_r,
            "n_bins": n_bins,
            "n_cols": n_cols,
        }
        if facet_col:
            kwargs["facet_col"] = facet_col
        if weight_col:
            kwargs["weight_col"] = weight_col
        if title:
            kwargs["title"] = title

        plot_obj = self._plot_binned(**kwargs)
        self._save_ggplot(plot_obj, out_path, width, height, dpi)

        return str(out_path)

    def plot_binned(
        self,
        df: pd.DataFrame,
        x_col: str,
        act_col: str,
        pred_col: Union[str, List[str]],
        facet_col: Optional[str] = None,
        weight_col: Optional[str] = None,
        n_bins: int = 10,
        n_cols: int = 3,
        width: int = 1200,
        height: int = 800,
        dpi: int = 150,
        title: Optional[str] = None,
    ) -> None:
        """Plot binned actual vs predicted inline in Jupyter."""

        def do_plot(path, w, h, d):
            self.plot_binned_to_file(
                df, x_col, act_col, pred_col, path,
                facet_col=facet_col,
                weight_col=weight_col,
                n_bins=n_bins, n_cols=n_cols,
                width=w, height=h, dpi=d, title=title
            )

        _with_temp_png(do_plot, width, height, dpi)

    # =========================================================================
    # Binned Actual vs Predicted (Overlay)
    # =========================================================================

    def plot_binned_overlay_to_file(
        self,
        df: pd.DataFrame,
        x_col: str,
        act_col: str,
        pred_col: Union[str, List[str]],
        out_path: Union[str, Path],
        facet_col: Optional[str] = None,
        weight_col: Optional[str] = None,
        n_bins: int = 10,
        width: int = 1200,
        height: int = 800,
        dpi: int = 150,
        title: Optional[str] = None,
    ) -> str:
        """Plot binned overlay and save to file."""
        out_path = Path(out_path)
        df_r = py_df_to_r(df, r=self.r_)
        pred_col_r = py_obj_to_r(pred_col, r=self.r_)

        kwargs = {
            "df": df_r,
            "x_col": x_col,
            "act_col": act_col,
            "pred_col": pred_col_r,
            "n_bins": n_bins,
        }
        if facet_col:
            kwargs["facet_col"] = facet_col
        if weight_col:
            kwargs["weight_col"] = weight_col
        if title:
            kwargs["title"] = title

        plot_obj = self._plot_binned_overlay(**kwargs)
        self._save_ggplot(plot_obj, out_path, width, height, dpi)

        return str(out_path)

    def plot_binned_overlay(
        self,
        df: pd.DataFrame,
        x_col: str,
        act_col: str,
        pred_col: Union[str, List[str]],
        facet_col: Optional[str] = None,
        weight_col: Optional[str] = None,
        n_bins: int = 10,
        width: int = 1200,
        height: int = 800,
        dpi: int = 150,
        title: Optional[str] = None,
    ) -> None:
        """Plot binned overlay inline in Jupyter."""

        def do_plot(path, w, h, d):
            self.plot_binned_overlay_to_file(
                df, x_col, act_col, pred_col, path,
                facet_col=facet_col,
                weight_col=weight_col,
                n_bins=n_bins,
                width=w, height=h, dpi=d, title=title
            )

        _with_temp_png(do_plot, width, height, dpi)

    # =========================================================================
    # Multi-plot diagnostics
    # =========================================================================

    def diagnostic_plots_to_pdf(
        self,
        df: pd.DataFrame,
        out_path: Union[str, Path],
        configs: Optional[List[dict]] = None,
        defaults: Optional[dict] = None,
        x_cols: Optional[List[str]] = None,
        act_col: Optional[str] = None,
        pred_col: Optional[Union[str, List[str]]] = None,
        facet_col: Optional[str] = None,
        weight_col: Optional[str] = None,
        n_bins: int = 10,
        width: int = 1200,
        height: int = 800,
        dpi: int = 150,
    ) -> str:
        """Create multi-page PDF with binned diagnostic plots."""
        out_path = Path(out_path)
        if out_path.suffix.lower() != ".pdf":
            raise ValueError("diagnostic_plots_to_pdf only supports .pdf output")

        df_r = py_df_to_r(df, r=self.r_)

        if configs is None:
            if x_cols is None or act_col is None or pred_col is None:
                raise ValueError("Provide configs or (x_cols, act_col, pred_col)")
            x_cols_r = py_obj_to_r(x_cols, r=self.r_)
            plot_type = None
            bins = None
            n_cols = None
            min_size = None
            max_size = None
            pred_colors = None
            y_transform = None
            title_prefix = None
            if defaults and isinstance(defaults, dict):
                plot_type = defaults.get("plot_type") or defaults.get("type")
                if plot_type is None and defaults.get("overlay") is True:
                    plot_type = "overlay"
                bins = defaults.get("bins")
                n_cols = defaults.get("n_cols")
                min_size = defaults.get("min_size")
                max_size = defaults.get("max_size")
                pred_colors = defaults.get("pred_colors")
                y_transform = defaults.get("y_transform")
                title_prefix = defaults.get("title_prefix")
            configs_r = self._build_diag_configs(
                x_cols=x_cols_r,
                act_col=act_col,
                pred_col=py_obj_to_r(pred_col, r=self.r_),
                facet_col=facet_col,
                weight_col=weight_col,
                bins=bins,
                n_bins=n_bins,
                n_cols=n_cols or 3,
                min_size=min_size if min_size is not None else 2,
                max_size=max_size if max_size is not None else 8,
                pred_colors=pred_colors,
                y_transform=y_transform,
                title_prefix=title_prefix or "",
                plot_type=plot_type or "binned",
            )
        else:
            configs_r = py_obj_to_r(configs, r=self.r_)

        defaults_r = py_obj_to_r(defaults or {}, r=self.r_)

        # Open PDF device
        self.r_.ro.r["pdf"](
            file=str(out_path),
            width=width / dpi,
            height=height / dpi,
            onefile=True
        )

        try:
            self._render_diagnostic_plots(df=df_r, configs=configs_r, defaults=defaults_r)
        finally:
            self._dev_off()

        return str(out_path)
