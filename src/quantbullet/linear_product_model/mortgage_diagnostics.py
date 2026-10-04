import warnings
from dataclasses import dataclass, field, fields, replace

import numpy as np
import pandas as pd
import polars as pl
from matplotlib import ticker as mticker

from quantbullet.plot.grouped_data import BinSpec, summarize_grouped_means
from quantbullet.plot.grouped_means import draw_grouped_means
from quantbullet.plot.theme import MINIMAL_THEME

NAMED_TRANSFORMS = {
    'smm_to_cpr': lambda smm: 1 - (1 - smm) ** 12,
    'annualize':  lambda x: x * 12,
}

# Pretty x-axis labels per role (overridable by passing ``x_label=...`` to plot methods).
_ROLE_LABELS: dict[str, str] = {
    'incentive':      'Incentive (rate − (PMMS_low + LLPA))',
    'age':            'Loan age (months)',
    'cltv':           'Updated CLTV',
    'current_factor': 'Current factor',
    'burnout':        'Burnout',
    'sato':           'SATO',
    'fico':           'FICO',
    'orig_dt':        'Origination date',
    'factor_dt':      'Factor date',
}

# y-axis labels inferred from the named ``y_transform``.
_Y_TRANSFORM_LABELS: dict[str, str] = {
    'smm_to_cpr': 'CPR',
    'annualize':  'Annualized rate',
}


@dataclass
class MortgageColnames:
    """Maps dataset column names to standardized roles for mortgage diagnostics.

    Only ``response`` is required.  All other fields default to ``None``
    and are checked lazily when a plot method needs them.
    """
    response       : str
    model_preds    : dict[str, str] = field(default_factory=dict)
    incentive      : str | None = None
    cltv           : str | None = None
    age            : str | None = None
    current_factor : str | None = None
    burnout        : str | None = None
    sato           : str | None = None
    fico           : str | None = None
    orig_dt        : str | None = None
    factor_dt      : str | None = None
    weight         : str | None = None


# Roles usable as the x axis; response, predictions and weight are not.
_X_ROLES = frozenset(f.name for f in fields(MortgageColnames)) - {'response', 'model_preds', 'weight'}


class MortgageDiagnostics:
    """Mortgage model diagnostic plots.

    ``plot`` maps a role or source column to grouped-data dimensions,
    aggregates through Polars, then draws curves and Count bars with
    grouped-means; the role methods (``incentive_plot`` etc.) wrap it.
    The caller's input DataFrame is never mutated. Pandas categorical order
    is retained, and neither input path requires PyArrow.

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        The evaluation dataset.  The caller's frame is never modified.
    colnames : MortgageColnames
        Column-name mapping for standardised roles.
    bin_config : dict, optional
        Per-column binning strategy.  Keys are column *roles* (e.g.
        ``'age'``, ``'incentive'``) or source column names, values are
        ``'discrete'``, a numeric rounding unit or a ``BinSpec``.  Columns
        not listed use quantile binning.
    y_transform : callable or str, optional
        Applied to aggregated actual and predicted means before plotting.
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
        if not isinstance(df, (pl.DataFrame, pd.DataFrame)):
            raise TypeError("MortgageDiagnostics expects a polars or pandas DataFrame")
        self.df = df
        self.colnames = colnames
        self.bin_config: dict = bin_config or {}

        # Store the named transform separately so we can derive a y-axis label
        # when the plot methods auto-fill defaults.
        self.y_transform_name: str | None = None
        if isinstance(y_transform, str):
            if y_transform not in NAMED_TRANSFORMS:
                raise ValueError(
                    f"Unknown y_transform '{y_transform}'. "
                    f"Available: {list(NAMED_TRANSFORMS.keys())}"
                )
            self.y_transform_name = y_transform
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

    def _default_x_label(self, x_role: str) -> str:
        return _ROLE_LABELS.get(x_role, x_role.replace('_', ' ').capitalize())

    def _default_y_label(self) -> str:
        if self.y_transform_name is not None:
            base = _Y_TRANSFORM_LABELS.get(self.y_transform_name, 'Rate')
        elif self.y_transform is None:
            base = 'Rate'
        else:
            base = 'Transformed rate'
        return f"{base} (%)" if self.y_as_percent else base

    def _vintage_year(self) -> pl.Series | pd.Series:
        """Derive vintage year from ``orig_dt`` (no mutation)."""
        self._require('orig_dt')
        if isinstance(self.df, pd.DataFrame):
            return pd.to_datetime(self.df[self.colnames.orig_dt]).dt.year
        col = self.df.get_column(self.colnames.orig_dt)
        if col.dtype in (pl.Date, pl.Datetime):
            return col.dt.year()
        return col.cast(pl.Utf8).str.to_date().dt.year()

    def _source_column(self, x: str) -> str:
        """Resolve a ``MortgageColnames`` role or a source column name."""
        if x in _X_ROLES:
            self._require(x)
            return getattr(self.colnames, x)
        if x not in self.df.columns:
            raise ValueError(f"{x!r} is neither a MortgageColnames role nor a column of the frame")
        return x

    def _bin_spec(self, x: str, column: str, bins, n_bins: int) -> BinSpec | None:
        """``bins`` overrides ``bin_config`` (keyed by role or column)."""
        strategy = bins if bins is not None else self.bin_config.get(x, self.bin_config.get(column))
        if isinstance(strategy, BinSpec):
            return strategy
        if strategy is None:
            return BinSpec.quantile(n_bins)
        if isinstance(strategy, str) and strategy == 'discrete':
            return None
        if isinstance(strategy, (int, float)) and not isinstance(strategy, bool):
            return BinSpec.round(strategy)
        raise ValueError(f"Unknown bin configuration for {x}: {strategy!r}")

    def _with_column(self, name: str, values):
        """Attach a derived facet column without copying the other columns."""
        values = values.to_numpy() if hasattr(values, 'to_numpy') else np.asarray(values)
        if isinstance(self.df, pd.DataFrame):
            return self.df.assign(**{name: values})
        return self.df.with_columns(pl.Series(name, values))

    def plot(self, x: str, *, bins=None, facet_col: str | None = None,
             facet_series=None, **kwargs):
        """Actual vs predicted by a mortgage role or any source column.

        ``x`` is a ``MortgageColnames`` role (e.g. ``'incentive'``) or a column
        of the frame. ``bins`` overrides ``bin_config``: ``'discrete'``, a
        rounding unit, or a ``BinSpec``; otherwise quantile bins
        (``n_bins``, default 10) are used. Weighted means are aggregated first;
        ``y_transform`` (e.g. SMM -> CPR) and ``min_count`` then apply to the
        bin-level values. Count bars show rows and share one scale across facets.
        ``figsize`` is per panel. Returns ``(fig, primary_axes)``.
        """
        column = self._source_column(x)
        transform = kwargs.pop('y_transform', self.y_transform)
        xlabel = kwargs.pop('x_label', self._default_x_label(x))
        ylabel = kwargs.pop('y_label', self._default_y_label())
        min_count = kwargs.pop('min_count', 0)
        n_bins = kwargs.pop('n_bins', 10)
        n_cols = kwargs.pop('n_cols', 3)
        figsize = kwargs.pop('figsize', (6, 4))
        align_ylim = kwargs.pop('align_ylim', False)
        theme = kwargs.pop('theme', None) or MINIMAL_THEME
        pred_colors = kwargs.pop('pred_colors', None)
        close_unused = kwargs.pop('close_unused', True)
        for legacy in ('min_size', 'max_size'):
            if kwargs.pop(legacy, None) is not None:
                warnings.warn(f"{legacy} is ignored; counts are drawn as bars", DeprecationWarning, stacklevel=2)
        if pred_colors is not None:
            theme = replace(theme, palette=(theme.palette[0], *pred_colors))

        frame = self.df if facet_series is None else self._with_column(facet_col, facet_series)
        spec = self._bin_spec(x, column, bins, n_bins)
        response, preds = self.colnames.response, self.colnames.model_preds
        data = summarize_grouped_means(
            frame, x=column, y=[response, *preds.values()], weight=self.colnames.weight,
            col=facet_col, bins={column: spec} if spec is not None else None,
        )
        if transform is not None:
            data = data.map_means(transform)
        data = data.mask_support(min_count)
        labels = {column: xlabel, response: 'Actual', **{source: name for name, source in preds.items()}}
        title = f"Actual vs {', '.join(preds)}" if preds else "Actual"
        result = draw_grouped_means(
            data, wrap=n_cols if facet_col is not None else None,
            panel_size=figsize, share_y=align_ylim, share_count_y=True,
            labels=labels, ylabel=ylabel, theme=theme,
            title=None if facet_col is not None else title,
            **kwargs,
        )
        fig, axes = result.fig, list(result.axes.flat)
        if not close_unused:
            for ax in axes:
                ax.set_visible(True)

        if self.y_as_percent:
            for ax in axes:
                if ax.get_visible():
                    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        return fig, axes

    # ---- single-feature plot methods -------------------------------------
    def incentive_plot(self, facet_col: str | None = None, **kwargs):
        """Actual vs predicted by incentive, optionally faceted."""
        self._require('incentive')
        return self.plot('incentive', facet_col=facet_col, **kwargs)

    def age_plot(self, facet_col: str | None = None, **kwargs):
        """Actual vs predicted by loan age, optionally faceted."""
        self._require('age')
        return self.plot('age', facet_col=facet_col, **kwargs)

    def cltv_plot(self, facet_col: str | None = None, **kwargs):
        """Actual vs predicted by updated CLTV, optionally faceted."""
        self._require('cltv')
        return self.plot('cltv', facet_col=facet_col, **kwargs)

    def current_factor_plot(self, facet_col: str | None = None, **kwargs):
        """Actual vs predicted by current factor (remaining UPB ratio), optionally faceted."""
        self._require('current_factor')
        return self.plot('current_factor', facet_col=facet_col, **kwargs)

    def burnout_plot(self, facet_col: str | None = None, **kwargs):
        """Actual vs predicted by burnout, optionally faceted."""
        self._require('burnout')
        return self.plot('burnout', facet_col=facet_col, **kwargs)

    def sato_plot(self, facet_col: str | None = None, **kwargs):
        """Actual vs predicted by spread-at-origination, optionally faceted."""
        self._require('sato')
        return self.plot('sato', facet_col=facet_col, **kwargs)

    def fico_plot(self, facet_col: str | None = None, **kwargs):
        """Actual vs predicted by borrower FICO, optionally faceted."""
        self._require('fico')
        return self.plot('fico', facet_col=facet_col, **kwargs)

    def factor_date_plot(self, facet_col: str | None = None, **kwargs):
        """Actual vs predicted across factor date (monthly time series)."""
        self._require('factor_dt')
        return self.plot('factor_dt', facet_col=facet_col, **kwargs)

    # ---- vintage-year helper (used by multiple features) ----------------
    def by_vintage_year(self, x_role: str, **kwargs):
        """Plot ``x_role`` faceted by origination vintage year.

        Vintage year is derived on-the-fly from ``orig_dt`` and is *not*
        required to exist as a column on the input frame.
        """
        self._require('orig_dt')
        return self.plot(x_role, facet_col='vintage_year',
                         facet_series=self._vintage_year(), **kwargs)

    def incentive_by_vintage_year_plots(self, **kwargs):
        """Back-compat alias for ``by_vintage_year('incentive', **kwargs)``."""
        return self.by_vintage_year('incentive', **kwargs)
