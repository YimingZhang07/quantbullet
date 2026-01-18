import time
import logging
from pathlib import Path
from typing import Optional, Union, Literal
import pandas as pd
import numpy as np

from .r_session import get_r
from .rpy_convert import py_df_to_r, r_array_to_py, r_generic_types_to_py

logger = logging.getLogger(__name__)


class MgcvBamWrapper:
    """
    Python wrapper for R's mgcv::bam() function for fitting Generalized Additive Models.
    
    Provides a Pythonic interface to R's mgcv package with support for:
    - Large-scale GAM fitting with parallel processing
    - Data pinning to avoid repeated Python-R conversions
    - Chunked prediction for memory efficiency
    - Model visualization
    
    Example:
        >>> wrapper = MgcvBamWrapper()
        >>> wrapper.fit(df, formula='y ~ s(x1) + s(x2)', num_cores=8)
        >>> predictions = wrapper.predict(df_new)
    """
    
    def __init__(self):
        """Initialize the wrapper and source R backend functions."""
        r = get_r()
        self.r_ = r

        # mgcv.R sits next to this file (adjust if your layout differs)
        r_file = Path(__file__).resolve().with_name("mgcv.R")
        if not r_file.exists():
            raise FileNotFoundError(f"mgcv.R not found at: {r_file}")

        # Use forward slashes for R on Windows
        r_path = r_file.as_posix()
        r.ro.r(f'source("{r_path}")')

        self._bam_fit               = r.ro.globalenv["fit_gam_api"]
        self._bam_fit_pinned_data   = r.ro.globalenv["fit_gam_pinned_data_api"]

        self._bam_predict           = r.ro.globalenv["predict_gam_chunked_api"]
        self._bam_predict_pinned_data   = r.ro.globalenv["predict_gam_chunked_pinned_data_api"]
        
        self._plot_fn               = r.ro.globalenv["plot_gam_smooth_api"]
        self._stop_cluster          = r.ro.globalenv["qb_stop_cluster"]

        # Pinning functions
        self._qb_pin_put            = r.ro.globalenv["qb_pin_put"]
        self._qb_pin_drop           = r.ro.globalenv["qb_pin_drop"]
        self._qb_pin_drop_all       = r.ro.globalenv["qb_pin_drop_all"]
        self._qb_pin_parquet        = r.ro.globalenv["qb_pin_put_parquet"]
        self._qb_pin_list           = r.ro.globalenv["qb_pin_list"]

        _R_GET_VARS = r.ro.r("""
            function(formula_str) {
            f <- as.formula(formula_str)
            unique(all.vars(f))
            }
            """)
        
        def r_formula_vars(formula_str: str):
            vars_r = _R_GET_VARS(formula_str)
            return list(vars_r)
        
        self._get_formula_vars = r_formula_vars

        self.model_r_ = None
        self.formula_ = None

    def pin_put(self, name: str, df: pd.DataFrame, as_datatable: bool = True, lock: bool = True) -> None:
        """
        Pin a DataFrame to R's memory to avoid repeated Python-R conversions.
        
        Args:
            name: Unique identifier for the pinned data
            df: DataFrame to pin
            as_datatable: Convert to data.table in R (recommended for performance)
            lock: Prevent accidental overwrite of pinned data
            
        Raises:
            ValueError: If name is invalid or df is empty
        """
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")
        if df is None or len(df) == 0:
            raise ValueError("df must be a non-empty DataFrame")
            
        df_r = py_df_to_r(df, r=self.r_)
        self._qb_pin_put(name=name, df=df_r, as_datatable=as_datatable, lock=lock)
        logger.debug(f"Pinned data '{name}' with shape {df.shape}")

    def pin_drop(self, name: str) -> bool:
        """
        Remove a pinned DataFrame from R's memory.
        
        Args:
            name: Name of the pinned data to remove
            
        Returns:
            True if data was removed, False if it didn't exist
        """
        result = self._qb_pin_drop(name)
        removed = r_generic_types_to_py(result) if result is not None else False
        if removed:
            logger.debug(f"Dropped pinned data '{name}'")
        return bool(removed)

    def pin_drop_all(self) -> None:
        """Remove all pinned DataFrames from R's memory."""
        self._qb_pin_drop_all()
        logger.debug("Dropped all pinned data")
    
    def pin_list(self) -> list:
        """
        List all currently pinned data names.
        
        Returns:
            List of pinned data names
        """
        result = self._qb_pin_list()
        return list(result) if result is not None else []

    def _validate_fit_params(self, family: str, num_cores: int, nthreads: int) -> None:
        """Validate common fitting parameters."""
        if family not in ["gaussian", "binomial"]:
            raise ValueError(f"family must be 'gaussian' or 'binomial', got '{family}'")
        if num_cores < 1:
            raise ValueError(f"num_cores must be >= 1, got {num_cores}")
        if nthreads < 1:
            raise ValueError(f"nthreads must be >= 1, got {nthreads}")
    
    def _handle_fit_result(self, res, data_source: str, timing: dict) -> None:
        """Process fit result and update model state."""
        ok = r_generic_types_to_py(res.rx2('ok'))
        error = r_generic_types_to_py(res.rx2('error_msg'))
        if not ok:
            raise ValueError(f"mgcv bam model fitting failed in R: {error}")
        
        self.model_r_ = res.rx2('model')
        
        # Log timing info
        timing_str = " | ".join(f"{k}={v:.2f}s" for k, v in timing.items())
        print(f"Model fitted from {data_source}: {timing_str}")
    
    def _ensure_fitted(self) -> None:
        """Ensure model is fitted before prediction/plotting."""
        if self.model_r_ is None:
            raise ValueError("Model is not fitted yet. Call fit() or fit_pinned_data() first.")

    def select_columns_for_formula(self, df: pd.DataFrame, formula: str, extra_cols: Optional[list] = None) -> pd.DataFrame:
        """
        Extract only the columns needed for a formula from a DataFrame.
        
        Args:
            df: Input DataFrame
            formula: R-style formula string (e.g., 'y ~ s(x1) + s(x2)')
            extra_cols: Additional columns to include (e.g., ['weight'])
            
        Returns:
            DataFrame with only the necessary columns
            
        Raises:
            ValueError: If formula is invalid or required columns are missing
        """
        if not formula or not isinstance(formula, str):
            raise ValueError("formula must be a non-empty string")
            
        extra_cols = extra_cols or []
        try:
            formula_vars = self._get_formula_vars(formula)
        except Exception as e:
            raise ValueError(f"Invalid formula '{formula}': {e}") from e
            
        cols = set(formula_vars) | set(extra_cols)
        cols = [c for c in cols if c in df.columns]
        
        missing = set(formula_vars) - set(df.columns)
        if missing:
            raise ValueError(f"Formula requires columns not in DataFrame: {missing}")
            
        return df.loc[:, cols]
    
    def pin_df_to_parquet(self, df: pd.DataFrame, name: str, formula: Optional[str] = None, fpath: Optional[Union[str, Path]] = None) -> None:
        """
        Save DataFrame to Parquet and pin it to R's memory.
        
        This is more efficient than pin_put() for large datasets as it:
        1. Writes to disk first (avoiding full in-memory copy)
        2. Loads directly in R using Arrow
        
        Args:
            df: DataFrame to pin
            name: Unique identifier for the pinned data
            formula: Optional formula to select only necessary columns
            fpath: Path to save parquet file (default: ./{name}.parquet)
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")
        if df is None or len(df) == 0:
            raise ValueError("df must be a non-empty DataFrame")
            
        # if fpath is None, just save in the current directory with the name
        if fpath is None:
            fpath = f"./{name}.parquet"

        if formula is None:
            df_sub = df
        else:
            df_sub = self.select_columns_for_formula(df, formula, extra_cols=["weight"])
            
        df_sub.to_parquet(fpath, index=False)
        r_fpath = str(Path(fpath).as_posix())  # Use forward slashes for R on Windows
        self._qb_pin_parquet(data_name=name, parquet_path=r_fpath, lock=True)
        logger.debug(f"Pinned parquet '{name}' from {fpath} with shape {df_sub.shape}")

    def fit(
        self,
        df: Optional[pd.DataFrame] = None,
        formula: Optional[str] = None,
        *,
        data_name: Optional[str] = None,
        family: Literal["gaussian", "binomial"] = "gaussian", 
        num_cores: int = 20, 
        discrete: bool = True, 
        nthreads: int = 1
    ) -> "MgcvBamWrapper":
        """
        Fit a GAM model using R's mgcv::bam().
        
        Supports both direct DataFrame input and previously pinned data.
        Use pinned data for large datasets to avoid Python-R conversion overhead.
        
        Args:
            df: Training data DataFrame (use either df or data_name, not both).
                For backwards compatibility, can be passed as first positional arg.
            formula: R-style formula (e.g., 'y ~ s(x1, k=10) + s(x2)').
                For backwards compatibility, can be passed as second positional arg.
            data_name: Name of pinned data (use either df or data_name, not both).
                Must be passed as keyword argument.
            family: Distribution family ('gaussian' or 'binomial')
            num_cores: Number of cores for fitting (ignored if discrete=True)
            discrete: Use discrete fitting (faster for large data, recommended)
            nthreads: Number of threads when discrete=True
            
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If inputs are invalid or fitting fails
            
        Note:
            For large datasets (>100k rows), use discrete=True with nthreads > 1
            For smaller datasets, use discrete=False with num_cores > 1
            
        Examples:
            >>> # Using DataFrame directly (backwards compatible)
            >>> wrapper.fit(train_df, 'y ~ s(x)')  # positional args (old style)
            >>> wrapper.fit(df=train_df, formula='y ~ s(x)')  # keyword args
            
            >>> # Using pinned data (more efficient for large data)
            >>> wrapper.pin_put('train', train_df)
            >>> wrapper.fit(formula='y ~ s(x)', data_name='train')
        """
        # Validate inputs
        if df is not None and data_name is not None:
            raise ValueError("Provide either 'df' or 'data_name', not both")
        if df is None and data_name is None:
            raise ValueError("Must provide either 'df' or 'data_name'")
        if formula is None:
            raise ValueError("formula is required")
        
        # Common parameter validation
        self._validate_fit_params(family, num_cores, nthreads)
        self.formula_ = formula
        
        # Route to appropriate implementation
        if df is not None:
            # Direct DataFrame path
            if len(df) == 0:
                raise ValueError("df must be a non-empty DataFrame")
            
            df_sub = self.select_columns_for_formula(df, formula, extra_cols=["weight"])
            
            t0 = time.perf_counter()
            df_r = py_df_to_r(df_sub, r=self.r_)
            t_py_to_r = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            res = self._bam_fit(
                data_train=df_r, 
                model_formula=formula, 
                family_str=family, 
                num_cores=num_cores,
                discrete=discrete,
                nthreads=nthreads
            )
            t_bam_fit = time.perf_counter() - t0
            
            self._handle_fit_result(res, "DataFrame", {"py_df_to_r": t_py_to_r, "bam_fit": t_bam_fit})
        else:
            # Pinned data path
            if not isinstance(data_name, str) or not data_name:
                raise ValueError("data_name must be a non-empty string")
            
            t0 = time.perf_counter()
            res = self._bam_fit_pinned_data(
                data_name=data_name, 
                model_formula=formula, 
                family_str=family, 
                num_cores=num_cores,
                discrete=discrete,
                nthreads=nthreads
            )
            t_bam_fit = time.perf_counter() - t0
            
            self._handle_fit_result(res, f"pinned data '{data_name}'", {"bam_fit": t_bam_fit})
        
        return self
    
    def fit_pinned_data(
        self, 
        data_name: str, 
        formula: str, 
        family: Literal["gaussian", "binomial"] = "gaussian", 
        num_cores: int = 20, 
        discrete: bool = True, 
        nthreads: int = 1
    ) -> "MgcvBamWrapper":
        """
        Fit a GAM model using previously pinned data.
        
        This is a convenience method that calls fit(data_name=...).
        Consider using fit(data_name=...) directly for a unified interface.
        
        Args:
            data_name: Name of the pinned data (from pin_put or pin_df_to_parquet)
            formula: R-style formula (e.g., 'y ~ s(x1, k=10) + s(x2)')
            family: Distribution family ('gaussian' or 'binomial')
            num_cores: Number of cores for fitting (ignored if discrete=True)
            discrete: Use discrete fitting (faster for large data, recommended)
            nthreads: Number of threads when discrete=True
            
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If data_name doesn't exist or fitting fails
        """
        return self.fit(
            formula=formula,
            data_name=data_name,
            family=family,
            num_cores=num_cores,
            discrete=discrete,
            nthreads=nthreads
        )
        
    def predict(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        data_name: Optional[str] = None,
        type: str = "response", 
        chunk_size: int = 250000
    ) -> np.ndarray:
        """
        Make predictions on new data using the fitted model.
        
        Supports both direct DataFrame input and previously pinned data.
        Use pinned data for large datasets to avoid Python-R conversion overhead.
        
        Args:
            df: DataFrame to predict on (use either df or data_name, not both).
                For backwards compatibility, can be passed as first positional arg.
            data_name: Name of pinned data (use either df or data_name, not both).
                Must be passed as keyword argument.
            type: Prediction type ('response', 'link', or 'terms')
            chunk_size: Size of chunks for memory-efficient prediction (default: 250000)
            
        Returns:
            Array of predictions
            
        Raises:
            ValueError: If model is not fitted or inputs are invalid
            
        Examples:
            >>> # Using DataFrame directly (backwards compatible)
            >>> predictions = wrapper.predict(test_df)  # positional arg (old style)
            >>> predictions = wrapper.predict(df=test_df)  # keyword arg
            
            >>> # Using pinned data (more efficient for large data)
            >>> wrapper.pin_put('test', test_df)
            >>> predictions = wrapper.predict(data_name='test')
        """
        t_total_begin = time.perf_counter()
        
        # Ensure model is fitted
        self._ensure_fitted()
        
        # Validate input: exactly one of df or data_name must be provided
        if df is not None and data_name is not None:
            raise ValueError("Provide either 'df' or 'data_name', not both")
        if df is None and data_name is None:
            raise ValueError("Must provide either 'df' or 'data_name'")
        
        # Initialize timing dict
        timing = {}
        
        # Route to appropriate implementation
        if df is not None:
            # Direct DataFrame path
            if len(df) == 0:
                raise ValueError("df must be a non-empty DataFrame")
            
            # Time: column selection
            t0 = time.perf_counter()
            df_sub = self.select_columns_for_formula(df, self.formula_, extra_cols=["weight"])
            timing['select_cols'] = time.perf_counter() - t0
            
            # Time: Python -> R conversion
            t0 = time.perf_counter()
            df_r = py_df_to_r(df_sub)
            timing['py_to_r'] = time.perf_counter() - t0
            
            # Time: R prediction (includes internal R timing)
            t0 = time.perf_counter()
            pred_r = self._bam_predict(
                self.model_r_,
                df_r,
                type=type,
                chunk_size=chunk_size
            )
            timing['r_predict'] = time.perf_counter() - t0
            
            data_source = f"DataFrame[{len(df)}]"
        else:
            # Pinned data path
            if not isinstance(data_name, str) or not data_name:
                raise ValueError("data_name must be a non-empty string")
            
            # Time: R prediction with pinned data (includes internal R timing)
            t0 = time.perf_counter()
            pred_r = self._bam_predict_pinned_data(
                self.model_r_,
                data_name,
                type=type,
                chunk_size=chunk_size
            )
            timing['r_predict'] = time.perf_counter() - t0
            
            data_source = f"pinned '{data_name}'"
        
        # Time: R -> Python conversion
        t0 = time.perf_counter()
        pred = r_array_to_py(pred_r)
        timing['r_to_py'] = time.perf_counter() - t0
        
        # Total time
        timing['total'] = time.perf_counter() - t_total_begin
        
        # Log detailed timing
        timing_str = " | ".join(f"{k}={v:.2f}s" for k, v in timing.items())
        print(f"Predictions made from {data_source}: {timing_str}")
        
        return pred
    
    def predict_pinned_data(
        self, 
        data_name: str, 
        type: str = "response", 
        chunk_size: int = 250000
    ) -> np.ndarray:
        """
        Make predictions using previously pinned data.
        
        This is a convenience method that calls predict(data_name=...).
        Consider using predict(data_name=...) directly for a unified interface.
        
        Args:
            data_name: Name of the pinned data (from pin_put or pin_df_to_parquet)
            type: Prediction type ('response', 'link', or 'terms')
            chunk_size: Size of chunks for memory-efficient prediction (default: 250000)
            
        Returns:
            Array of predictions
            
        Raises:
            ValueError: If model is not fitted or data_name doesn't exist
        """
        return self.predict(
            data_name=data_name,
            type=type,
            chunk_size=chunk_size
        )
    
    def plot_to_file(
        self,
        out_fpath: Union[str, Path],
        pages: int = 1,
        width: int = 3200,
        height: int = 2400,
        dpi: int = 300,
    ) -> str:
        """
        Plot model's smooth terms and save to file.
        
        Args:
            out_fpath: Output file path (supports .png, .pdf, .svg)
            pages: Number of pages for multi-page plots (1 for single page)
            width: Width in pixels (for raster) or device units (for vector)
            height: Height in pixels (for raster) or device units (for vector)
            dpi: Resolution for raster formats
            
        Returns:
            Path to the saved plot
            
        Raises:
            ValueError: If model is not fitted or parameters are invalid
        """
        self._ensure_fitted()
        
        if pages < 1:
            raise ValueError(f"pages must be >= 1, got {pages}")
        if width < 100 or height < 100:
            raise ValueError(f"width and height must be >= 100, got {width}x{height}")

        out_fpath = str(out_fpath)

        self._plot_fn(
            self.model_r_,
            fpath=out_fpath,
            pages=pages,
            width=width,
            height=height,
            dpi=dpi,
        )

        logger.debug(f"Saved plot to {out_fpath}")
        return out_fpath
    
    def plot(
        self,
        width: int = 3200,
        height: int = 2400,
        dpi: int = 300,
    ) -> None:
        """
        Plot model's smooth terms inline in Jupyter notebook.
        
        Creates a temporary PNG file for display. The temp file is automatically
        cleaned up after display.
        
        Args:
            width: Width in pixels
            height: Height in pixels
            dpi: Resolution
            
        Raises:
            ValueError: If model is not fitted
            ImportError: If IPython is not available
        """
        self._ensure_fitted()
        
        try:
            from IPython.display import Image, display
        except ImportError as e:
            raise ImportError("IPython is required for inline plotting. Use plot_to_file() instead.") from e

        import tempfile
        import os
        
        # Create temp file with automatic cleanup
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            tmp_path = Path(tmpfile.name)

        try:
            # Since we are doing an inline display, can only use 1 page
            self.plot_to_file(
                out_fpath=tmp_path,
                pages=1,
                width=width,
                height=height,
                dpi=dpi,
            )

            # Display inline in Jupyter
            display(Image(filename=str(tmp_path)))
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary plot file {tmp_path}: {e}")

    def stop_cluster(self) -> None:
        """
        Stop the R parallel cluster and free resources.
        
        Should be called when done with parallel operations to clean up properly.
        """
        self._stop_cluster()
        logger.debug("Stopped R cluster")
    
    @property
    def is_fitted(self) -> bool:
        """Check if a model has been fitted."""
        return self.model_r_ is not None
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "fitted" if self.is_fitted else "not fitted"
        formula = f", formula='{self.formula_}'" if self.formula_ else ""
        return f"MgcvBamWrapper({status}{formula})"