import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from quantbullet.model.feature import DataType
from pygam import LinearGAM, s, f, te
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, Union, Tuple, Optional, Any

from .terms import (
    GAMTermData,
    SplineTermData,
    SplineByGroupTermData,
    TensorTermData,
    FactorTermData,
    format_term_name,
)
from .utils import (
    dump_partial_dependence_json,
    center_partial_dependence,
)
from .plot import plot_partial_dependence


class WrapperGAM:
    """A wrapper around pygam's LinearGAM to integrate with FeatureSpec and provide additional functionality.
    
    Attributes
    ----------
    feature_term_map_ : dict
        A mapping from feature names to their corresponding pygam terms.

    category_levels_ : dict
        A mapping from categorical feature names to their levels (categories) observed during training.
    """
    __slots__ = [
        'feature_spec', 'gam_', 'formula_', 'feature_term_map_',
        'category_levels_', 'design_columns_', 'by_dummy_info_',
        '_centered_intercept_cache',
    ]

    def __init__( self, feature_spec ):
        self.feature_spec = feature_spec
        self.gam_ = None
        self.formula_ = None
        self.feature_term_map_ = {} # feature_name -> term
        self.category_levels_ = {} # col_name -> categories
        self.design_columns_ = None
        self.by_dummy_info_ = {}  # (x_name, by_cat) -> list of dummy col names
        self._centered_intercept_cache = None

    def _prepare_design_matrix_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select inputs, lock category levels, and expand dummy columns for FLOAT-by-CATEGORY."""
        Xd = X[self.feature_spec.all_inputs].copy()

        # lock categories
        for col in self.feature_spec.sec_x_cat + self.feature_spec.x_cat:
            Xd[col] = Xd[col].astype('category')
            self.category_levels_[col] = Xd[col].cat.categories

        # expand dummies for FLOAT-by-CATEGORY
        self.by_dummy_info_.clear()
        for x_name in self.feature_spec.x:
            feat = self.feature_spec[x_name]
            if feat.dtype != DataType.FLOAT:
                continue
            by = (feat.specs or {}).get("by")
            if not by:
                continue

            by_feat = self.feature_spec[by]
            if by_feat.dtype == DataType.CATEGORY:
                # one-hot using locked categories
                cat = pd.Categorical(Xd[by], categories=self.category_levels_[by])
                dummies = pd.get_dummies(cat, prefix=f"{by}__", dtype=float)

                # attach + record column names
                dummy_cols = list(dummies.columns)
                for c in dummy_cols:
                    Xd[c] = dummies[c].values

                self.by_dummy_info_[(x_name, by)] = dummy_cols

        # finally, encode original categorical columns to codes (for f() terms if used)
        for col in self.feature_spec.sec_x_cat + self.feature_spec.x_cat:
            Xd[col] = pd.Categorical(Xd[col], categories=self.category_levels_[col]).codes

        self.design_columns_ = list(Xd.columns)
        return Xd
    
    def _prepare_design_matrix_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build design matrix with same columns as training."""
        if self.design_columns_ is None:
            raise ValueError("Model is not fit yet; design_columns_ is missing.")

        Xd = X[self.feature_spec.all_inputs].copy()

        # enforce categories and create dummy cols identically
        for col in self.feature_spec.sec_x_cat + self.feature_spec.x_cat:
            Xd[col] = pd.Categorical(Xd[col], categories=self.category_levels_[col])

        # rebuild dummies for each recorded FLOAT-by-CATEGORY interaction
        for (x_name, by), dummy_cols in self.by_dummy_info_.items():
            cat = Xd[by]
            dummies = pd.get_dummies(cat, prefix=f"{by}__", dtype=float)
            # ensure all dummy_cols exist
            for c in dummy_cols:
                Xd[c] = dummies[c].values if c in dummies.columns else 0.0

        # encode original categoricals to codes
        for col in self.feature_spec.sec_x_cat + self.feature_spec.x_cat:
            Xd[col] = Xd[col].cat.codes

        # add any missing training columns (shouldn't happen often, but safe)
        for c in self.design_columns_:
            if c not in Xd.columns:
                Xd[c] = 0.0

        # reorder and drop extras
        Xd = Xd[self.design_columns_]
        return Xd

    def _build_formula_from_design(self, design_cols: list[str]):
        """Build a pygam formula using final design column ordering."""
        col_to_idx = {c: i for i, c in enumerate(design_cols)}
        terms = []
        self.feature_term_map_.clear()

        for x_name in self.feature_spec.x:
            feat = self.feature_spec[x_name]
            specs = feat.specs or {}

            # Below logic is pretty simple;
            # feat -> FLOAT
            #  -> by = None         : s(...)
            #  -> by = FLOAT       : te(...)
            #  -> by = CATEGORY    : multiple s(..., by=dummy)
            # feat -> CATEGORY
            #  -> f(...)

            if feat.dtype == DataType.FLOAT:
                by = specs.get("by")
                kwargs = {k: v for k, v in specs.items()
                          if v is not None and k in ['spline_order', 'n_splines', 'lam', 'constraints']}
                if by is None:
                    t = s(col_to_idx[x_name], **kwargs)
                    terms.append(t)
                    self.feature_term_map_[x_name] = [t]
                else:
                    by_feat = self.feature_spec[by]
                    if by_feat.dtype == DataType.FLOAT:
                        # continuous-by-continuous: keep TE
                        t = te(col_to_idx[x_name], col_to_idx[by], **kwargs)
                        terms.append(t)
                        self.feature_term_map_[x_name] = [t]
                    elif by_feat.dtype == DataType.CATEGORY:
                        # continuous-by-categorical: expand into multiple s(..., by=dummy)
                        dummy_cols = self.by_dummy_info_.get((x_name, by))
                        if not dummy_cols:
                            raise ValueError(f"Missing dummy columns for ({x_name}, by={by}).")
                        t_list = []
                        for dc in dummy_cols:
                            t_k = s(col_to_idx[x_name], by=col_to_idx[dc], **kwargs)
                            terms.append(t_k)
                            t_list.append(t_k)
                        self.feature_term_map_[x_name] = t_list
                    else:
                        raise ValueError(f"Unsupported by dtype: {by_feat.dtype}")

            elif feat.dtype == DataType.CATEGORY:
                # optional: include main effect of the categorical itself if you want
                t = f(col_to_idx[x_name])
                terms.append(t)
                self.feature_term_map_[x_name] = [t]
            else:
                raise ValueError(f"Unsupported dtype: {feat.dtype}")

        if not terms:
            raise ValueError("No terms to combine")

        formula = terms[0]
        for term in terms[1:]:
            formula = formula + term

        self.formula_ = formula
        return formula
    
    def fit(self, X, y, weights=None, fit_intercept=True):
        Xd = self._prepare_design_matrix_fit(X)
        formula = self._build_formula_from_design(list(Xd.columns))

        self.gam_ = LinearGAM(formula, fit_intercept=fit_intercept).fit(np.asarray(Xd), np.asarray(y), weights=weights)
        self._centered_intercept_cache = None
        return self

    def predict(self, X):
        Xd = self._prepare_design_matrix_predict(X)
        return self.gam_.predict(np.asarray(Xd))
    
    def __repr__(self):
        return f"WrapperGAM({self.gam_})"
    
    def _get_term_indices_for_feature(self, feature_name: str):
        """Map stored term objects to their term indices in self.gam_.terms."""
        term_list = self.feature_term_map_.get(feature_name, [])
        if not isinstance(term_list, (list, tuple)):
            term_list = [term_list]

        idxs = []
        for ti, t in enumerate(self.gam_.terms):
            # match by object identity
            for t_ref in term_list:
                if t is t_ref:
                    idxs.append(ti)
                    break
        return idxs
    
    def get_partial_dependence_data(self, grid_size=200, width=0.95, center=False) -> Dict[Union[str, Tuple[str, str]], GAMTermData]:
        """
        Extract partial dependence data for all features using structured Dataclasses.

        Parameters
        ----------
        grid_size : int
            Number of grid points per smooth curve (default: 200).
        width : float
            Confidence interval width (default: 0.95 for 95% CI).
        center : bool
            If True, shift each smooth curve by its mean so that the curve
            averages to ~0, absorbing the offsets into the intercept.
            By-group offsets are folded into the corresponding FactorTermData.

        Returns
        -------
        dict
            A dictionary where keys are feature names (str) or interaction tuples (str, str).
            Values are instances of:
            - SplineTermData
            - SplineByGroupTermData
            - TensorTermData
            - FactorTermData
        """
        if self.gam_ is None:
            raise ValueError("Model not fit yet.")

        data = {}
        feature_names = list(self.feature_term_map_.keys())

        # Helpers
        def _get_colname(col_idx: int) -> str:
            return self.design_columns_[col_idx]

        def _is_dummy_of(by_name: str, colname: str) -> bool:
            return colname.startswith(f"{by_name}___")

        def _dummy_label(by_name: str, colname: str) -> str:
            prefix = f"{by_name}___"
            return colname[len(prefix):] if colname.startswith(prefix) else colname

        for feature_name in feature_names:
            idxs = self._get_term_indices_for_feature(feature_name)
            if not idxs:
                continue

            term0 = self.gam_.terms[idxs[0]]
            term0_name = getattr(term0, "_name", "")

            # ----------------------------
            # Case A: spline_term
            # ----------------------------
            if term0_name == "spline_term":
                # Check if this feature is "FLOAT by CATEGORY" expanded:
                # heuristics:
                # - multiple spline terms
                # - each has a non-None 'by'
                # - dummy column names share prefix "{by}__"
                specs = getattr(self.feature_spec[feature_name], "specs", None) or {}
                by_name = specs.get("by", None)

                is_group_by_cat = False
                if by_name and len(idxs) > 1:
                    all_have_by = True
                    all_look_like_dummies = True
                    for ti in idxs:
                        t = self.gam_.terms[ti]
                        if getattr(t, "by", None) is None:
                            all_have_by = False
                            break
                        by_col_idx = int(t.by)
                        by_colname = _get_colname(by_col_idx)
                        if not _is_dummy_of(by_name, by_colname):
                            all_look_like_dummies = False
                            break
                    is_group_by_cat = all_have_by and all_look_like_dummies

                if is_group_by_cat:
                    key = (feature_name, by_name)
                    curves = {}
                    
                    x_col_idx = int(self.gam_.terms[idxs[0]].feature)
                    Xg = self.gam_.generate_X_grid(term=idxs[0])
                    x_vals = Xg[:, x_col_idx]
                    x_min, x_max = np.min(x_vals), np.max(x_vals)
                    x_grid = np.linspace(x_min, x_max, grid_size)
                    
                    for ti in idxs:
                        t = self.gam_.terms[ti]
                        by_col_idx = int(t.by)
                        by_colname = _get_colname(by_col_idx)
                        label = _dummy_label(by_name, by_colname)
                        
                        XX = np.zeros((len(x_grid), len(self.design_columns_)))
                        XX[:, x_col_idx] = x_grid
                        XX[:, by_col_idx] = 1.0
                        
                        pdep, confi = self.gam_.partial_dependence(term=ti, X=XX, width=width)
                        curves[label] = {
                            'x': x_grid, 
                            'y': np.asarray(pdep).flatten(),
                            'conf_lower': confi[:, 0],
                            'conf_upper': confi[:, 1]
                        }
                        
                    data[key] = SplineByGroupTermData(
                        feature=feature_name,
                        by_feature=by_name,
                        group_curves=curves
                    )
                
                else:
                    # Simple spline
                    ti = idxs[0]
                    t = self.gam_.terms[ti]
                    x_col_idx = int(t.feature)
                    
                    Xg = self.gam_.generate_X_grid(term=ti, n=grid_size)
                    pdep, confi = self.gam_.partial_dependence(term=ti, X=Xg, width=width)
                    
                    data[feature_name] = SplineTermData(
                        feature=feature_name,
                        x=Xg[:, x_col_idx],
                        y=np.asarray(pdep).flatten(),
                        conf_lower=confi[:, 0],
                        conf_upper=confi[:, 1]
                    )

            # ----------------------------
            # Case B: tensor_term
            # ----------------------------
            elif term0_name == "tensor_term":
                specs = getattr(self.feature_spec[feature_name], "specs", None) or {}
                by_name = specs.get("by", None)
                key = (feature_name, by_name) if by_name else feature_name
                features = [feature_name, by_name] if by_name else [feature_name]
                
                x_grid, y_grid, Z = self.get_tensor_term_grid(feature_name, grid_size=grid_size)
                data[key] = TensorTermData(
                    feature_x=features[0],
                    feature_y=features[1] if len(features) > 1 else "unknown",
                    x=x_grid,
                    y=y_grid,
                    z=Z
                )

            # ----------------------------
            # Case C: factor_term
            # ----------------------------
            elif term0_name == "factor_term":
                ti = idxs[0]
                t = self.gam_.terms[ti]
                cat_col_idx = int(t.feature)
                
                cat_colname = _get_colname(cat_col_idx)
                labels = self.category_levels_.get(feature_name, None)
                if labels is None and cat_colname in self.category_levels_:
                    labels = self.category_levels_[cat_colname]
                
                if labels is None:
                    Xg = self.gam_.generate_X_grid(term=ti)
                    codes = sorted(list(np.unique(Xg[:, cat_col_idx])))
                    labels = [str(c) for c in codes]
                else:
                    codes = list(range(len(labels)))
                    
                XX = np.zeros((len(codes), len(self.design_columns_)))
                XX[:, cat_col_idx] = codes
                
                pdep, confi = self.gam_.partial_dependence(term=ti, X=XX, width=width)
                pdep = np.asarray(pdep).flatten()
                
                data[feature_name] = FactorTermData(
                    feature=feature_name,
                    categories=list(labels),
                    values=pdep,
                    conf_lower=confi[:, 0],
                    conf_upper=confi[:, 1]
                )

        if center:
            data, _ = center_partial_dependence(data, self.intercept_)
        return data

    def export_partial_dependence_json(
        self,
        path: str,
        grid_size: int = 200,
        width: float = 0.95,
        include_intercept: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        center: bool = False,
    ) -> Dict[str, Any]:
        term_data = self.get_partial_dependence_data(
            grid_size=grid_size, width=width, center=False,
        )
        intercept = self.intercept_ if include_intercept else 0.0
        if center:
            term_data, intercept = center_partial_dependence(term_data, intercept)
        return dump_partial_dependence_json(
            term_data=term_data,
            path=path,
            intercept=intercept,
            metadata=metadata,
        )

    def plot_partial_dependence(self, n_cols=3, suptitle=None, scale_y_axis=True, te_plot_style="heatmap", width=5, height=4, center=False):
        """Plot partial dependence for each feature.

        Delegates to the module-level :func:`plot_partial_dependence`.
        """
        if self.gam_ is None:
            raise ValueError("Model not fit yet. Call fit() before plotting.")

        pdep_data = self.get_partial_dependence_data(center=center)
        return plot_partial_dependence(
            pdep_data,
            n_cols=n_cols,
            suptitle=suptitle,
            scale_y_axis=scale_y_axis,
            te_plot_style=te_plot_style,
            width=width,
            height=height,
        )

    def __getattr__(self, name):
        """
        Delegate attribute/method access to the underlying GAM model, but avoid
        recursion if `gam_` is not yet set (e.g., during unpickling).
        """
        try:
            gam = object.__getattribute__(self, "gam_")
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from None

        try:
            return getattr(gam, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from None

    def __getitem__(self, key):
        """Delegate indexing to the underlying GAM model safely."""
        gam = object.__getattribute__(self, "gam_")
        return gam[key]
    
    def __getstate__(self):
        return {
            "feature_spec"      : self.feature_spec,
            "gam_"              : self.gam_,
            "formula_"          : self.formula_,
            "feature_term_map_" : self.feature_term_map_,
            "category_levels_"  : self.category_levels_,
            "design_columns_"   : self.design_columns_,
            "by_dummy_info_"    : self.by_dummy_info_,
        }

    def __setstate__(self, state):
        self.feature_spec       = state["feature_spec"]
        self.gam_               = state["gam_"]
        self.formula_           = state["formula_"]
        self.feature_term_map_  = state["feature_term_map_"]
        self.category_levels_   = state["category_levels_"]
        self.design_columns_    = state["design_columns_"]
        self.by_dummy_info_     = state["by_dummy_info_"]
        self._centered_intercept_cache = None

    @property
    def intercept_(self):
        """Get the intercept term of the fitted GAM model."""
        if self.gam_ is None:
            raise ValueError("Model not fit yet. Call fit() before accessing intercept.")
        if not getattr(self.gam_, "fit_intercept", True):
            return 0.0
        return self.gam_.coef_[-1]

    @property
    def centered_intercept_(self):
        """Intercept adjusted for partial-dependence centering.

        Computed lazily on first access and cached until the next ``fit()`` call.
        """
        if self._centered_intercept_cache is None:
            raw_data = self.get_partial_dependence_data(center=False)
            _, adj = center_partial_dependence(raw_data, self.intercept_)
            self._centered_intercept_cache = adj
        return self._centered_intercept_cache
    
    # ##########Below codes are for tensor term surface extraction and plotting ##########
    # Instead of plotting the full surface, we provide utilities to extract the surface and plot slices in 2D.

    def get_tensor_term_grid(self, feature_name: str, grid_size: int = 100):
        """Get grid values for a tensor term corresponding to the given feature name.

        Parameters
        ----------
        feature_name : str
            The name of the feature associated with the tensor term.
        grid_size : int
            The number of points along each axis in the grid.
        """
        idxs = self._get_term_indices_for_feature(feature_name)
        if not idxs:
            raise ValueError(f"No term found for feature: {feature_name}")

        term0 = self.gam_.terms[idxs[0]]
        term0_name = getattr(term0, "_name", "")

        if term0_name != "tensor_term":
            raise ValueError(f"Feature '{feature_name}' is not associated with a tensor term.")

        ti = idxs[0]
        t = self.gam_.terms[ti]

        feats = t.feature
        if feats is None or len(feats) != 2:
            raise ValueError(f"Tensor term for feature '{feature_name}' does not have exactly two features.")

        # generate meshgrid
        XX = self.gam_.generate_X_grid(term=ti, meshgrid=True, n=grid_size)
        Z = self.gam_.partial_dependence(term=ti, X=XX, meshgrid=True)

        X1, X2 = XX[0], XX[1]
        x1_grid = X1[:, 0]
        x2_grid = X2[0, :]
        return x1_grid, x2_grid, Z
    
    def make_tensor_term_surface(self, feature_name: str, grid_size: int = 100):
        """Get a callable surface function for a tensor term corresponding to the given feature name."""
        x1_grid, x2_grid, Z = self.get_tensor_term_grid(feature_name, grid_size)
        interp = RegularGridInterpolator(
            (x1_grid, x2_grid),
            Z,
            bounds_error=False,
            fill_value=np.nan
        )

        def surface(x1, x2):
            x1 = np.asarray(x1)
            x2 = np.asarray(x2)
            x1b, x2b = np.broadcast_arrays(x1, x2)
            pts = np.column_stack([x1b.ravel(), x2b.ravel()])
            return interp(pts).reshape(x1b.shape)
        return surface
    
    def plot_tensor_term_slices( self, feature_name: str, cross_term_vals=None, ax=None, grid_size: int = 100 ):
        if ax is None:
            _, ax = plt.subplots()

        x1_grid, x2_grid, _ = self.get_tensor_term_grid(feature_name, grid_size=grid_size)
        surface = self.make_tensor_term_surface(feature_name, grid_size=grid_size)

        if cross_term_vals is None:
            slice_vals = np.quantile(x2_grid, [0.25, 0.5, 0.75])
        else:
            slice_vals = np.asarray(cross_term_vals, dtype=float)

        x1_dense = np.linspace(x1_grid.min(), x1_grid.max(), grid_size)

        lines = []
        for v in slice_vals:
            y = surface(x1_dense, np.full_like(x1_dense, v))
            line, = ax.plot(x1_dense, y)
            lines.append(line)

        cross_term_name = self.feature_spec[feature_name].specs['by']

        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel("Partial Dependence", fontsize=12)
        ax.set_title(f"Slices of tensor term for {feature_name} by {cross_term_name}", fontsize=12)

        for line, v in zip(lines, slice_vals):
            line.set_label(f"{cross_term_name}={v:.1f}")

        ax.legend()
        return ax
    
    # ########## Prediction Decomposition ##########
    def decompose( self, X: pd.DataFrame ):
        if self.gam_ is None:
            raise ValueError("Model not fit yet.")

        Xd = self._prepare_design_matrix_predict(X)
        Xd_np = np.asarray(Xd)

        pred = self.gam_.predict(Xd_np)
        
        term_contribs = []
        term_names = []
        used_term_indices = []

        for ti, t in enumerate(self.gam_.terms):
            if getattr(t, "isintercept", False):
                continue
            c = self.gam_.partial_dependence(term=ti, X=Xd_np)
            term_contribs.append(np.asarray(c).reshape(-1))
            term_names.append(self._format_term_name(ti))
            used_term_indices.append(ti)

        if term_contribs:
            term_contribs = np.column_stack(term_contribs)
        else:
            term_contribs = np.zeros((len(Xd_np), 0))

        intercept = self.intercept_
        term_contribs_df = pd.DataFrame(term_contribs, columns=term_names, index=X.index)
        term_contribs_df['intercept'] = intercept
        term_contribs_df['pred'] = pred

        out = {
            "pred": pred,
            "intercept": intercept,
            "term_contrib": term_contribs_df,
            "term_indices": used_term_indices,
        }
        return out
        
    def _format_term_name(self, ti: int) -> str:
        """
        Format term name using standard convention: {type}__{feature}[__{by_feature}__{by_level}]
        
        Examples:
            s__age, s__age__level__B, f__level, te__x1__x2
        """
        t = self.gam_.terms[ti]
        tname = getattr(t, "_name", "term")

        # spline_term
        if tname == "spline_term":
            feat_idx = int(getattr(t, "feature", -1))
            feat = self.design_columns_[feat_idx] if feat_idx >= 0 else "unknown"
            by = getattr(t, "by", None)
            if by is None:
                return format_term_name("s", feat)
            by_idx = int(by)
            by_col_name = self.design_columns_[by_idx]
            if "___" in by_col_name:
                by_feature, by_level = by_col_name.split("___", 1)
            else:
                by_feature, by_level = by_col_name, "unknown"
            return format_term_name("s", feat, by_feature=by_feature, by_level=by_level)

        # tensor_term
        if tname == "tensor_term":
            feats = getattr(t, "feature", None)
            if feats is not None and len(feats) == 2:
                a = self.design_columns_[int(feats[0])]
                b = self.design_columns_[int(feats[1])]
                return format_term_name("te", a, feature2=b)
            return "te__unknown"

        # factor_term
        if tname == "factor_term":
            feat_idx = int(getattr(t, "feature", -1))
            feat = self.design_columns_[feat_idx] if feat_idx >= 0 else "unknown"
            return format_term_name("f", feat)

        return f"{tname}__{ti}"

    def get_model_summary_text(self) -> str:
        """Get a formatted text summary of the model including feature mapping and GAM summary.
        
        Returns
        -------
        str
            Multi-line summary text with:
            - Model formula
            - Feature index to name mapping
            - GAM statistical summary (from pygam)
        """
        if self.gam_ is None:
            raise ValueError("Model not fit yet. Call fit() before getting summary.")
        
        import io
        from contextlib import redirect_stdout
        
        lines = []
        
        # Formula section
        lines.append("=" * 60)
        lines.append("Model Formula")
        lines.append("=" * 60)
        lines.append(self.get_formula_string(max_line_length=70, multiline=True))
        lines.append("")
        
        # Feature mapping section
        lines.append("=" * 60)
        lines.append("Feature Index Mapping")
        lines.append("=" * 60)
        for idx, feat in enumerate(self.feature_spec.x + self.feature_spec.sec_x):
            lines.append(f"Feature {idx}: {feat}")
        lines.append("")
        
        # GAM summary section
        lines.append("=" * 60)
        lines.append("GAM Model Summary")
        lines.append("=" * 60)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.gam_.summary()
        lines.append(buf.getvalue())
        
        return "\n".join(lines)
    
    def get_formula_string(self, max_line_length: int = 80, multiline: bool = False) -> str:
        """Generate a human-readable formula string for the fitted GAM model.
        
        Returns a formula in R-style notation showing the model structure, e.g.:
        "target ~ s(feature1) + s(feature2, by=category) + te(feature3, feature4) + f(category)"
        
        Parameters
        ----------
        max_line_length : int, optional
            Maximum line length before wrapping to next line, by default 80.
            Only used if multiline=True.
        multiline : bool, optional
            If True, wrap formula across multiple lines for better readability, by default False.
        
        Returns
        -------
        str
            Human-readable formula string describing the model structure.
            
        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if self.gam_ is None:
            raise ValueError("Model not fit yet. Call fit() before getting formula string.")
        
        target_name = self.feature_spec.y
        
        term_strings = []
        
        for feature_name in self.feature_spec.x:
            feat = self.feature_spec[feature_name]
            specs = feat.specs or {}
            
            if feat.dtype == DataType.FLOAT:
                by = specs.get("by")
                
                if by is None:
                    term_strings.append(f"s({feature_name})")
                    
                else:
                    by_feat = self.feature_spec[by]
                    
                    if by_feat.dtype == DataType.FLOAT:
                        term_strings.append(f"te({feature_name}, {by})")
                        
                    elif by_feat.dtype == DataType.CATEGORY:
                        term_strings.append(f"s({feature_name}, by={by})")
                        
            elif feat.dtype == DataType.CATEGORY:
                term_strings.append(f"f({feature_name})")
        
        if not multiline:
            formula = f"{target_name} ~ " + " + ".join(term_strings)
        else:
            lines = [f"{target_name} ~"]
            current_line = "    "
            
            for i, term in enumerate(term_strings):
                prefix = " + " if i > 0 else ""
                term_with_prefix = prefix + term
                
                if len(current_line + term_with_prefix) > max_line_length and current_line.strip():
                    lines.append(current_line.rstrip())
                    current_line = "    " + term
                else:
                    current_line += term_with_prefix
            
            if current_line.strip():
                lines.append(current_line.rstrip())
            
            formula = "\n".join(lines)
        
        return formula
