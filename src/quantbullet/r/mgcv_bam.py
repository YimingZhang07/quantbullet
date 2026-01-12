import time
from pathlib import Path
from .r_session import get_r
from .rpy_convert import py_df_to_r, r_array_to_py, r_generic_types_to_py

class MgcvBamWrapper:
    def __init__(self):
        r = get_r()

        # mgcv.R sits next to this file (adjust if your layout differs)
        r_file = Path(__file__).resolve().with_name("mgcv.R")
        if not r_file.exists():
            raise FileNotFoundError(f"mgcv.R not found at: {r_file}")

        # Use forward slashes for R on Windows
        r_path = r_file.as_posix()
        r.ro.r(f'source("{r_path}")')

        self._bam_fit = r.ro.globalenv["fit_gam_api"]
        # self._bam_fit = r.ro.globalenv["fit_gam_no_cluster_api"]
        self._bam_predict = r.ro.globalenv["predict_gam_parallel_api"]
        # self._bam_predict = r.ro.globalenv["bam_predict_no_cluster_api"]
        self._plot_fn = r.ro.globalenv["plot_gam_smooth_api"]

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

    def select_columns_for_formula(self, df, formula: str, extra_cols=None):
        extra_cols = extra_cols or []
        cols = set(self._get_formula_vars(formula)) | set(extra_cols)
        cols = [c for c in cols if c in df.columns]
        return df.loc[:, cols]

    def fit( self, df, formula: str, family: str = "gaussian", num_cores: int = 20):
        self.formula_ = formula
        df_sub = self.select_columns_for_formula(df, formula, extra_cols=("weight",))

        t0 = time.perf_counter()
        df_r = py_df_to_r(df_sub)
        t_py_to_r = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        res = self._bam_fit( data_train = df_r, 
                             model_formula = formula, 
                             family_str = family, 
                             num_cores = num_cores)
        t_bam_fit = time.perf_counter() - t0

        ok = r_generic_types_to_py(res.rx2('ok'))
        error = r_generic_types_to_py(res.rx2('error_msg'))
        if not ok:
            raise ValueError(f"mgcv bam model fitting failed in R: {error}")
        model_r = res.rx2('model')
        self.model_r_ = model_r

        print(f"[mgcv.fit] py_df_to_r={t_py_to_r:.2f}s | bam_fit={t_bam_fit:.2f}s")
        return self
        
    def predict( self, df, type: str = "response", num_cores_predict: int = 20, num_split: int = 8):
        if self.model_r_ is None:
            raise ValueError("Model is not fitted yet. Call fit() before predict().")
        
        df_sub = self.select_columns_for_formula(df, self.formula_, extra_cols=("weight",))
        df_r = py_df_to_r(df_sub)
        pred_r = self._bam_predict( self.model_r_,
                                    df_r,
                                    type = type,
                                    num_cores_predict = num_cores_predict,
                                    num_split = num_split)
        pred = r_array_to_py(pred_r)
        return pred
    
    def plot_to_file(
        self,
        out_fpath: str | Path,
        pages: int = 1,
        width: int = 3200,
        height: int = 2400,
        dpi: int = 300,
    ):
        
        if self.model_r_ is None:
            raise ValueError("Model is not fitted yet. Call fit() before plot().")

        out_fpath = str(out_fpath)

        self._plot_fn(
            self.model_r_,
            fpath=out_fpath,
            pages=pages,
            width=width,
            height=height,
            dpi=dpi,
        )

        return out_fpath
    
    def plot(
        self,
        width: int = 3200,
        height: int = 2400,
        dpi: int = 300,
    ):
        """
        This function should plot the model's smooth terms in png and show inline in Jupyter.
        Uses a temporary file for the png output.
        """
        import tempfile
        from IPython.display import Image, display

        if self.model_r_ is None:
            raise ValueError("Model is not fitted yet. Call fit() before plot().")


        with tempfile.NamedTemporaryFile(
            suffix=".png", delete= False
        ) as tmpfile:
            tmp_path = Path(tmpfile.name)

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