import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from quantbullet.model.feature import DataType
from pygam import LinearGAM, s, f, te
from quantbullet.plot import (
    EconomistBrandColor,
    get_grid_fig_axes
)
from quantbullet.plot.utils import close_unused_axes
from quantbullet.plot.cycles import use_economist_cycle
import numpy as np
import matplotlib.pyplot as plt

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
        'category_levels_', 'design_columns_', 'by_dummy_info_'
    ]

    def __init__( self, feature_spec ):
        self.feature_spec = feature_spec
        self.gam_ = None
        self.formula_ = None
        self.feature_term_map_ = {} # feature_name -> term
        self.category_levels_ = {} # col_name -> categories
        self.design_columns_ = None
        self.by_dummy_info_ = {}  # (x_name, by_cat) -> list of dummy col names

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
    
    def fit(self, X, y, weights=None):
        Xd = self._prepare_design_matrix_fit(X)
        formula = self._build_formula_from_design(list(Xd.columns))

        self.gam_ = LinearGAM(formula).fit(np.asarray(Xd), np.asarray(y), weights=weights)
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
    
    def plot_partial_dependence(self, n_cols=3, suptitle=None, scale_y_axis=True, te_plot_style="heatmap"):
        """
        Plot partial dependence for each feature.

        Works with:
        - spline_term: s(feature)
        - spline_term with by: s(feature, by=indicator)  (including expanded by-categorical dummies)
        - tensor_term: te(...)
        - factor_term: f(category)
        """
        if self.gam_ is None:
            raise ValueError("Model not fit yet. Call fit() before plotting.")

        # ---------- helpers ----------
        def _get_colname(col_idx: int) -> str:
            return self.design_columns_[col_idx]

        def _is_dummy_of(by_name: str, colname: str) -> bool:
            return colname.startswith(f"{by_name}___")

        def _dummy_label(by_name: str, colname: str) -> str:
            # "level__A" -> "A"
            prefix = f"{by_name}___"
            return colname[len(prefix):] if colname.startswith(prefix) else colname

        # ---------- create axes ----------
        feature_names = list(self.feature_term_map_.keys())

        with use_economist_cycle():
            fig, axes = get_grid_fig_axes(n_charts=len(feature_names), n_cols=n_cols)
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        continuous_axes = []

        # ---------- main loop ----------
        for i_feat, feature_name in enumerate(feature_names):
            ax = axes.flat[i_feat]
            idxs = self._get_term_indices_for_feature(feature_name)

            if not idxs:
                ax.set_title(f"{feature_name} (no term found)")
                ax.axis("off")
                continue

            # Grab the first term type to decide plotting mode
            term0 = self.gam_.terms[idxs[0]]
            term0_name = getattr(term0, "_name", "")

            # ----------------------------
            # Case A: spline_term (possibly expanded by-categorical)
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
                    # Plot multiple curves on same axis, one per dummy level
                    # All share the same x feature column index:
                    x_col_idx = int(self.gam_.terms[idxs[0]].feature)

                    # use X_grid from the first term just to get x range
                    Xg = self.gam_.generate_X_grid(term=idxs[0])
                    x_vals = Xg[:, x_col_idx]
                    x_min, x_max = np.min(x_vals), np.max(x_vals)
                    x_grid = np.linspace(x_min, x_max, 200)

                    for ti in idxs:
                        # if we have term 1 cross with four dummies, then we have s(0, by=1), s(0, by=2), s(0, by=3), s(0, by=4)
                        # say we are at term s(0, by=3)
                        # t = 2, we are at the third term of the formula
                        # x_col_idx = 0, the main continuous feature is still at column 0
                        # by_col_idx = 3, the dummy column for this term is at column 3
                        t = self.gam_.terms[ti]
                        x_col_idx = int(t.feature)
                        by_col_idx = int(t.by)

                        by_colname = _get_colname(by_col_idx)
                        label = _dummy_label(by_name, by_colname)

                        # Build a full X for partial dependence evaluation
                        XX = np.zeros((len(x_grid), len(self.design_columns_)))
                        XX[:, x_col_idx] = x_grid
                        XX[:, by_col_idx] = 1.0  # turn on this dummy

                        pdep, confi = self.gam_.partial_dependence(term=ti, X=XX, width=0.95)
                        ax.plot(x_grid, pdep, label=label)
                        ax.fill_between(x_grid, confi[:, 0], confi[:, 1], alpha=0.15)

                    ax.set_xlabel(f"{feature_name} (by {by_name})", fontdict={'fontsize': 12})
                    ax.set_ylabel("Partial Dependence", fontdict={'fontsize': 12})
                    ax.legend(title=by_name)
                    continuous_axes.append(ax)

                else:
                    # Single spline term
                    ti = idxs[0]
                    t = self.gam_.terms[ti]
                    x_col_idx = int(t.feature)

                    Xg = self.gam_.generate_X_grid(term=ti)
                    pdep, confi = self.gam_.partial_dependence(term=ti, X=Xg, width=0.95)

                    ax.plot(Xg[:, x_col_idx], pdep, color=EconomistBrandColor.CHICAGO_45)
                    ax.fill_between(Xg[:, x_col_idx], confi[:, 0], confi[:, 1],
                                    alpha=0.2, color=EconomistBrandColor.CHICAGO_45)
                    ax.set_xlabel(feature_name, fontdict={'fontsize': 12})
                    ax.set_ylabel("Partial Dependence", fontdict={'fontsize': 12})
                    continuous_axes.append(ax)

            # ----------------------------
            # Case B: tensor_term
            # ----------------------------
            elif term0_name == "tensor_term":
                ti = idxs[0]
                t = self.gam_.terms[ti]

                feats = t.feature
                if feats is None or len(feats) != 2:
                    ax.set_title(f"{feature_name} (tensor: invalid)")
                    ax.axis("off")
                    continue

                x_col_idx = int(feats[0])
                z_col_idx = int(feats[1])
                x_col_name = _get_colname(x_col_idx)
                z_col_name = _get_colname(z_col_idx)

                # generate meshgrid
                XX = self.gam_.generate_X_grid(term=ti, meshgrid=True)
                Z = self.gam_.partial_dependence(term=ti, X=XX, meshgrid=True)

                X1, X2 = XX[0], XX[1]

                plot_tensor(ax, X1, X2, Z, style=te_plot_style)

                ax.set_xlabel(x_col_name, fontsize=12)
                ax.set_ylabel(z_col_name, fontsize=12)
                ax.set_title(f"{x_col_name} x {z_col_name} (tensor surface)", fontsize=12)

            # ----------------------------
            # Case C: factor_term (categorical main effect)
            # ----------------------------
            elif term0_name == "factor_term":
                # due to the by terms, the term indices may be multiple for the same feature
                # ti is the index for the factor_term
                # cat_col_idx is the column index of the categorical feature
                # if feature A has three by terms ( term indices 0, 1, 2 ), and term B is a categorical,
                # then ti = 3, but cat_col_index = 2.
                ti = idxs[0]
                t = self.gam_.terms[ti]
                cat_col_idx = int(t.feature)

                cat_colname = _get_colname(cat_col_idx)
                # Prefer feature_name if it is actually that cat feature
                labels = self.category_levels_.get(feature_name, None)
                if labels is None and cat_colname in self.category_levels_:
                    labels = self.category_levels_[cat_colname]

                if labels is None:
                    # fallback if unknown
                    # just plot codes 0..max seen
                    Xg = self.gam_.generate_X_grid(term=ti)
                    codes = sorted(list(np.unique(Xg[:, cat_col_idx])))
                    labels = [str(c) for c in codes]
                else:
                    codes = list(range(len(labels)))

                XX = np.zeros((len(codes), len(self.design_columns_)))
                XX[:, cat_col_idx] = codes

                pdep, confi = self.gam_.partial_dependence(term=ti, X=XX, width=0.95)

                ax.errorbar(labels, pdep,
                            yerr=[pdep - confi[:, 0], confi[:, 1] - pdep],
                            fmt='o', capsize=5,
                            color=EconomistBrandColor.CHICAGO_45)
                ax.axhline(0, color='gray', linestyle='--', linewidth=1)
                ax.set_xlabel(feature_name, fontdict={'fontsize': 12})
                ax.set_ylabel("Partial Dependence", fontdict={'fontsize': 12})

            else:
                ax.set_title(f"{feature_name} ({term0_name}: not handled)")
                ax.axis("off")

        # ---------- scale y axis across continuous plots ----------
        if scale_y_axis and continuous_axes:
            y_mins = [ax.get_ylim()[0] for ax in continuous_axes]
            y_maxs = [ax.get_ylim()[1] for ax in continuous_axes]
            global_y_min, global_y_max = min(y_mins), max(y_maxs)
            for ax in continuous_axes:
                ax.set_ylim(global_y_min, global_y_max)

        if suptitle:
            plt.suptitle(suptitle, fontsize=14)

        close_unused_axes(axes)
        return fig, axes

    def __getattr__(self, name):
        """
        Delegate attribute/method access to the underlying GAM model, but avoid
        recursion if `gam_` is not yet set (e.g., during unpickling).
        """
        # Try to fetch gam_ without invoking __getattr__ again
        try:
            gam = object.__getattribute__(self, "gam_")
        except AttributeError:
            # `gam_` isn't available yet; behave like a normal missing attribute
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from None

        # Delegate to the underlying model
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

    def intercept_(self):
        """Get the intercept term of the fitted GAM model."""
        if self.gam_ is None:
            raise ValueError("Model not fit yet. Call fit() before accessing intercept.")
        return self.gam_.intercept_

def plot_tensor(
    ax,
    X1,
    X2,
    Z,
    style="contour",
    levels=10,
    cmap="viridis",
    alpha=0.6,
    show_labels=True,
    colorbar=False,
    colorbar_label=None,
):
    """
    Plot a tensor term (2D smooth) on the given axes.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    X1, X2 : 2D arrays
        Meshgrid arrays for the two dimensions.
    Z : 2D array
        Partial dependence values.
    style : {"contour", "contourf", "heatmap"}
        Plotting style.
    levels : int
        Number of contour levels.
    cmap : str or Colormap
        Colormap to use. 'viridis' is good for screen, 'Greys' for print.
    alpha : float
        Transparency for filled plots.
    show_labels : bool
        Whether to label contour lines.
    colorbar : bool
        Whether to add a colorbar.
    colorbar_label : str
        Label for the colorbar.
    """

    if style not in {"contour", "contourf", "heatmap"}:
        raise ValueError(f"Unknown style: {style}")

    mappable = None  # for optional colorbar

    # -------------------------
    # 1) Contour only (best for print)
    # -------------------------
    if style == "contour":
        cs = ax.contour(
            X1, X2, Z,
            levels=levels,
            colors="black",
            linewidths=1.0,
        )
        if show_labels:
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

        mappable = cs

    # -------------------------
    # 2) Contour + light fill
    # -------------------------
    elif style == "contourf":
        cf = ax.contourf(
            X1, X2, Z,
            levels=levels,
            cmap=cmap,
            alpha=alpha,
        )
        cs = ax.contour(
            X1, X2, Z,
            levels=levels,
            colors="black",
            linewidths=0.8,
        )
        if show_labels:
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

        mappable = cf

    # -------------------------
    # 3) Heatmap (screen only)
    # -------------------------
    elif style == "heatmap":
        mappable = ax.pcolormesh(
            X1, X2, Z,
            shading="auto",
            cmap=cmap,
        )

    # -------------------------
    # Optional colorbar
    # -------------------------
    if colorbar and mappable is not None:
        cb = plt.colorbar(mappable, ax=ax)
        cb.set_label(colorbar_label)

    return ax
