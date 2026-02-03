"""
Python wrapper for R report generation (reports.R).

Combines model smooth plots with diagnostic plots into unified PDF reports.

Example:
    >>> from quantbullet.r.reports_wrapper import combined_model_report
    >>> combined_model_report(model_r, df, "y", "report.pdf", diag_configs=diag_configs)
"""
import logging
from pathlib import Path
from typing import Optional, Union, List

import pandas as pd

from .r_session import get_r
from .rpy_convert import py_df_to_r, py_obj_to_r

logger = logging.getLogger(__name__)


class RReports:
    """
    Python wrapper for R report generation functions.
    
    Wraps reports.R to provide combined model + diagnostic reports.
    """
    
    def __init__(self):
        """Initialize wrapper and source R report functions."""
        r = get_r()
        self.r_ = r
        
        # Source dependencies in order
        r_dir = Path(__file__).resolve().parent
        
        mgcv_file = r_dir / "mgcv.R"
        plots_file = r_dir / "plots.R"
        reports_file = r_dir / "reports.R"
        
        for f in [mgcv_file, plots_file, reports_file]:
            if not f.exists():
                raise FileNotFoundError(f"{f.name} not found at: {f}")
            r.ro.r(f'source("{f.as_posix()}")')
        
        # Bind R functions
        self._combined_model_report = r.ro.globalenv["combined_model_report"]


    def combined_report(
        self,
        model_r,
        df: pd.DataFrame,
        out_path: Union[str, Path],
        diag_configs: Optional[List[dict]] = None,
        diag_defaults: Optional[dict] = None,
        width: int = 1200,
        height: int = 800,
        dpi: int = 150,
        smooth_width: Optional[int] = None,
        smooth_height: Optional[int] = None,
        smooth_dpi: Optional[int] = None,
        summary_width: Optional[int] = None,
        summary_height: Optional[int] = None,
        summary_dpi: Optional[int] = None,
        diag_width: Optional[int] = None,
        diag_height: Optional[int] = None,
        diag_dpi: Optional[int] = None,
        smooth_pages: int = 1,
        include_smooths: bool = True,
        include_diagnostics: bool = True,
        include_summary: bool = True,
        rug: bool = False,
        scheme: int = 1,
    ) -> str:
        """
        Create combined model + diagnostic PDF report.
        
        Includes GAM smooth plots, diagnostic plots, and model summary.
        
        Args:
            model_r: Fitted R GAM/BAM model object
            df: DataFrame used for fitting
            out_path: Output PDF path
            diag_configs: List of diagnostic plot configs (preferred)
            width: Page width in pixels
            height: Page height in pixels
            dpi: Resolution
            smooth_width: Smooth section page width in pixels (optional)
            smooth_height: Smooth section page height in pixels (optional)
            smooth_dpi: Smooth section DPI (optional)
            summary_width: Summary page width in pixels (optional)
            summary_height: Summary page height in pixels (optional)
            summary_dpi: Summary page DPI (optional)
            diag_width: Diagnostics section page width in pixels (optional)
            diag_height: Diagnostics section page height in pixels (optional)
            diag_dpi: Diagnostics section DPI (optional)
            smooth_pages: Number of smooth terms per page
            include_smooths: Include GAM smooth plots
            include_diagnostics: Include diagnostic plots
            include_summary: Include model summary page
            rug: Show rug on smooth plots
            scheme: Color scheme for smooth plots
            
        Returns:
            Output file path as string
        """
        out_path = Path(out_path)
        if out_path.suffix.lower() != ".pdf":
            raise ValueError("combined_report only supports .pdf output")
        
        df_r = py_df_to_r(df, r=self.r_)
        
        kwargs = {
            "model": model_r,
            "df": df_r,
            "fpath": str(out_path),
            "width": width,
            "height": height,
            "dpi": dpi,
            "smooth_pages": smooth_pages,
            "include_smooths": include_smooths,
            "include_diagnostics": include_diagnostics,
            "include_summary": include_summary,
            "rug": rug,
            "scheme": scheme,
        }
        if diag_configs is not None:
            kwargs["diag_configs"] = py_obj_to_r(diag_configs, r=self.r_)
        if diag_defaults is not None:
            kwargs["diag_defaults"] = py_obj_to_r(diag_defaults, r=self.r_)
        if smooth_width is not None:
            kwargs["smooth_width"] = smooth_width
        if smooth_height is not None:
            kwargs["smooth_height"] = smooth_height
        if smooth_dpi is not None:
            kwargs["smooth_dpi"] = smooth_dpi
        if summary_width is not None:
            kwargs["summary_width"] = summary_width
        if summary_height is not None:
            kwargs["summary_height"] = summary_height
        if summary_dpi is not None:
            kwargs["summary_dpi"] = summary_dpi
        if diag_width is not None:
            kwargs["diag_width"] = diag_width
        if diag_height is not None:
            kwargs["diag_height"] = diag_height
        if diag_dpi is not None:
            kwargs["diag_dpi"] = diag_dpi
        
        self._combined_model_report(**kwargs)
        logger.info(f"Combined report saved to: {out_path}")
        
        return str(out_path)


# =============================================================================
# Standalone function (preferred API)
# =============================================================================

_reports_instance: Optional[RReports] = None

def _get_reports() -> RReports:
    """Get or create singleton RReports instance."""
    global _reports_instance
    if _reports_instance is None:
        _reports_instance = RReports()
    return _reports_instance


def combined_model_report(
    model_r,
    df: pd.DataFrame,
    out_path: Union[str, Path],
    diag_configs: Optional[List[dict]] = None,
    diag_defaults: Optional[dict] = None,
    width: int = 1200,
    height: int = 800,
    dpi: int = 150,
    smooth_width: Optional[int] = None,
    smooth_height: Optional[int] = None,
    smooth_dpi: Optional[int] = None,
    summary_width: Optional[int] = None,
    summary_height: Optional[int] = None,
    summary_dpi: Optional[int] = None,
    diag_width: Optional[int] = None,
    diag_height: Optional[int] = None,
    diag_dpi: Optional[int] = None,
    smooth_pages: int = 1,
    include_smooths: bool = True,
    include_diagnostics: bool = True,
    include_summary: bool = True,
    rug: bool = False,
    scheme: int = 1,
) -> str:
    """
    Create combined model + diagnostic PDF report.
    
    Includes GAM smooth plots, diagnostic plots, and model summary in one PDF.
    
    Args:
        model_r: Fitted R GAM/BAM model object
        df: DataFrame used for fitting
        out_path: Output PDF path
        diag_configs: List of diagnostic plot configs (preferred)
        width: Page width in pixels
        height: Page height in pixels
        dpi: Resolution
        smooth_width: Smooth section page width in pixels (optional)
        smooth_height: Smooth section page height in pixels (optional)
        smooth_dpi: Smooth section DPI (optional)
        summary_width: Summary page width in pixels (optional)
        summary_height: Summary page height in pixels (optional)
        summary_dpi: Summary page DPI (optional)
        diag_width: Diagnostics section page width in pixels (optional)
        diag_height: Diagnostics section page height in pixels (optional)
        diag_dpi: Diagnostics section DPI (optional)
        smooth_pages: Number of smooth terms per page
        include_smooths: Include GAM smooth plots
        include_diagnostics: Include diagnostic plots
        include_summary: Include model summary page
        rug: Show rug on smooth plots
        scheme: Color scheme for smooth plots
        
    Returns:
        Output file path as string
        
    Example:
        >>> from quantbullet.r.mgcv_bam import MgcvBamWrapper
        >>> from quantbullet.r.reports_wrapper import combined_model_report
        >>> 
        >>> wrapper = MgcvBamWrapper()
        >>> wrapper.fit(df, "y ~ s(x1) + s(x2)")
        >>> df["pred"] = wrapper.predict(df)
        >>> 
        >>> combined_model_report(
        ...     wrapper.model_r_, df, "report.pdf",
        ...     diag_configs=diag_configs
        ... )
    """
    return _get_reports().combined_report(
        model_r=model_r,
        df=df,
        out_path=out_path,
        diag_configs=diag_configs,
        diag_defaults=diag_defaults,
        width=width,
        height=height,
        dpi=dpi,
        smooth_width=smooth_width,
        smooth_height=smooth_height,
        smooth_dpi=smooth_dpi,
        summary_width=summary_width,
        summary_height=summary_height,
        summary_dpi=summary_dpi,
        diag_width=diag_width,
        diag_height=diag_height,
        diag_dpi=diag_dpi,
        smooth_pages=smooth_pages,
        include_smooths=include_smooths,
        include_diagnostics=include_diagnostics,
        include_summary=include_summary,
        rug=rug,
        scheme=scheme,
    )


