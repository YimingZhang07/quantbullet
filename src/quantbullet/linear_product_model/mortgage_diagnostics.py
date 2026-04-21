from dataclasses import dataclass, field

import pandas as pd
import polars as pl
from matplotlib import ticker as mticker

from quantbullet.plot.binned_plots import plot_binned_actual_vs_pred

NAMED_TRANSFORMS = {
    'smm_to_cpr': lambda smm: 1 - (1 - smm) ** 12,
    'annualize':  lambda x: x * 12,
}


@dataclass
class MortgageColnames:
    """Maps dataset column names to standardized roles for mortgage diagnostics.

    Only ``response`` is required.  All other fields default to ``None``
    and are checked lazily when a plot method needs them.
    """
    response    : str
    model_preds : dict[str, str] = field(default_factory=dict)
    incentive   : str | None = None
    cltv        : str | None = None
    age         : str | None = None
    orig_dt     : str | None = None
    factor_dt   : str | None = None
    weight      : str | None = None


class MortgageDiagnostics:
    """Mortgage model diagnostic plots.

    Each plot method validates its own prerequisites and delegates to
    the generic ``plot_binned_actual_vs_pred`` utility.  The caller's
    input DataFrame is never mutated.

    Internally the class always stores a polars DataFrame: pandas inputs
    are converted via ``pl.from_pandas`` at construction (Arrow-backed,
    near-zero copy for numeric and datetime columns).  This keeps the
    implementation single-path and lets the vectorized polars aggregation
    in ``plot_binned_actual_vs_pred`` handle ~10M+ row inputs.

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        The evaluation dataset.  The caller's frame is never modified.
    colnames : MortgageColnames
        Column-name mapping for standardised roles.
    bin_config : dict, optional
        Per-column binning strategy.  Keys are column *roles* (e.g.
        ``'age'``, ``'incentive'``), values are ``'discrete'`` or a
        numeric rounding unit.  Columns not listed use quantile binning.
    y_transform : callable or str, optional
        Applied to both actual and predicted y-values before plotting.
        Can be a callable or a string key from ``NAMED_TRANSFORMS``
        (e.g. ``'smm_to_cpr'``, ``'annualize'``).
    y_as_percent : bool
        If True (default), format the y-axis as percentages.
    """

    def __init__(
        self,
        df: pl.DataFrame | pd.DataFrame,
        colnames: MortgageColnames,
        bin_config: dict | None = None,
        y_transform=None,
        y_as_percent: bool = True,
    ):
        if not isinstance(df, pl.DataFrame):
            try:
                df = pl.from_pandas(df)
            except Exception as e:
                raise TypeError(
                    f"MortgageDiagnostics expects a polars or pandas DataFrame; "
                    f"got {type(df).__name__}. pl.from_pandas failed: {e}"
                ) from e
        self.df = df
        self.colnames = colnames
        self.bin_config: dict = bin_config or {}
        if isinstance(y_transform, str):
            if y_transform not in NAMED_TRANSFORMS:
                raise ValueError(
                    f"Unknown y_transform '{y_transform}'. "
                    f"Available: {list(NAMED_TRANSFORMS.keys())}"
                )
            y_transform = NAMED_TRANSFORMS[y_transform]
        self.y_transform = y_transform
        self.y_as_percent = y_as_percent

    def _require(self, *fields):
        """Raise if any of the named column mappings are ``None``."""
        for f in fields:
            if getattr(self.colnames, f, None) is None:
                raise ValueError(
                    f"Column mapping '{f}' is required for this plot "
                    f"but not set in MortgageColnames."
                )

    def _vintage_year(self) -> pl.Series:
        """Derive vintage year from ``orig_dt`` (no mutation)."""
        self._require('orig_dt')
        col = self.df.get_column(self.colnames.orig_dt)
        if col.dtype in (pl.Date, pl.Datetime):
            return col.dt.year()
        return col.cast(pl.Utf8).str.to_date().dt.year()

    def _build_plot_df(self, x_role: str) -> tuple[pl.DataFrame, str | None]:
        """Assemble a slim plot-frame for one x-axis, applying bin_config rounding.

        Returns ``(plot_df, bins)`` where *bins* is the value to pass to
        ``plot_binned_actual_vs_pred``'s ``bins`` parameter.
        """
        self._require(x_role)
        x_col_name = getattr(self.colnames, x_role)

        strategy = self.bin_config.get(x_role)
        round_step = strategy if isinstance(strategy, (int, float)) else None
        bins = 'discrete' if (round_step is not None or strategy == 'discrete') else None

        x_expr = pl.col(x_col_name)
        if round_step is not None:
            x_expr = (x_expr / round_step).round() * round_step

        select_exprs: list[pl.Expr] = [
            x_expr.alias(x_role),
            pl.col(self.colnames.response).alias('actual'),
        ]
        for name, orig_col in self.colnames.model_preds.items():
            select_exprs.append(pl.col(orig_col).alias(name))
        if self.colnames.weight is not None:
            select_exprs.append(pl.col(self.colnames.weight))

        return self.df.select(select_exprs), bins

    def _attach_facet(
        self,
        plot_df: pl.DataFrame,
        facet_col: str,
        facet_series=None,
    ) -> pl.DataFrame:
        """Attach a facet column to ``plot_df``.

        If ``facet_series`` is supplied it is used verbatim; otherwise the
        column of the same name is pulled from ``self.df``.
        """
        if facet_series is None:
            return plot_df.with_columns(
                self.df.get_column(facet_col).alias(facet_col)
            )
        if not isinstance(facet_series, pl.Series):
            facet_series = pl.Series(facet_col, facet_series)
        return plot_df.with_columns(facet_series.alias(facet_col))

    def _plot(self, x_role: str, facet_col: str | None = None,
              facet_series=None, **kwargs):
        """Shared plotting logic: build data, apply instance-level defaults, delegate."""
        plot_df, bins = self._build_plot_df(x_role)
        if facet_col is not None:
            plot_df = self._attach_facet(plot_df, facet_col, facet_series)

        kwargs.setdefault('y_transform', self.y_transform)

        fig, axes = plot_binned_actual_vs_pred(
            plot_df,
            x_col=x_role,
            act_col='actual',
            pred_col=list(self.colnames.model_preds.keys()),
            facet_col=facet_col,
            weight_col=self.colnames.weight,
            bins=bins,
            **kwargs,
        )

        if self.y_as_percent:
            for ax in axes:
                if ax.get_visible():
                    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        return fig, axes

    def incentive_plot(self, facet_col: str | None = None, **kwargs):
        """Actual vs predicted by incentive, optionally faceted."""
        self._require('incentive')
        return self._plot('incentive', facet_col=facet_col, **kwargs)

    def incentive_by_vintage_year_plots(self, **kwargs):
        """Actual vs predicted by incentive, faceted by origination vintage year."""
        self._require('incentive', 'orig_dt')
        return self._plot('incentive', facet_col='vintage_year',
                          facet_series=self._vintage_year(), **kwargs)

    def age_plot(self, facet_col: str | None = None, **kwargs):
        """Actual vs predicted by loan age, optionally faceted."""
        self._require('age')
        return self._plot('age', facet_col=facet_col, **kwargs)
