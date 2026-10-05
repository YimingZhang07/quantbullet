import warnings
from dataclasses import dataclass, field, fields, replace

import numpy as np
import pandas as pd
import polars as pl

from quantbullet.plot.formatter import StepPercentFormatter
from quantbullet.plot.grouped_data import BinSpec, GroupedMeansData, summarize_grouped_means
from quantbullet.plot.grouped_means import GroupedMeansStyle, draw_grouped_means
from quantbullet.plot.panels import PanelSet
from quantbullet.plot.theme import MINIMAL_THEME, PlotTheme

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
    'orig_balance':   'Original balance',
    'current_balance':'Current balance',
    'orig_dt':        'Origination date',
    'factor_dt':      'Factor date',
}

# y-axis labels inferred from the named ``y_transform``.
_Y_TRANSFORM_LABELS: dict[str, str] = {
    'smm_to_cpr': 'CPR',
    'annualize':  'Annualized rate',
}


def _padded_range(values: np.ndarray, margin: float = .05) -> tuple[float, float] | None:
    """Finite min/max widened by ``margin`` of the span, as matplotlib pads autoscaled axes."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    low, high = float(finite.min()), float(finite.max())
    pad = (high - low) * margin or abs(high) * margin or margin
    return low - pad, high + pad


def _column_and_bin(role: str, value):
    """A role is a column name, or ``(column, bin)``."""
    if value is None or isinstance(value, str):
        return value, None
    strategy = value[1] if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str) else None
    if strategy is not None and (
        isinstance(strategy, BinSpec) or strategy == "discrete"
        or (isinstance(strategy, (int, float)) and not isinstance(strategy, bool))
    ):
        return value[0], strategy
    raise ValueError(f"Column mapping {role!r} must be a column name or (column, bin), got {value!r}")


@dataclass
class MortgageColnames:
    """Maps dataset column names to standardized roles for mortgage diagnostics.

    Only ``response`` is required.  All other fields default to ``None``
    and are checked lazily when a plot method needs them.  An x-axis role
    may be ``(column, bin)``; after initialization the attribute is the
    column name and the bin is stored in ``bins``.
    """
    response       : str
    model_preds    : dict[str, str] = field(default_factory=dict)
    incentive      : str | tuple | None = None
    cltv           : str | tuple | None = None
    age            : str | tuple | None = None
    current_factor : str | tuple | None = None
    burnout        : str | tuple | None = None
    sato           : str | tuple | None = None
    fico           : str | tuple | None = None
    orig_balance   : str | tuple | None = None
    current_balance: str | tuple | None = None
    orig_dt        : str | tuple | None = None
    factor_dt      : str | tuple | None = None
    weight         : str | None = None
    bins           : dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        collected = {}
        for role in _X_ROLES:
            column, strategy = _column_and_bin(role, getattr(self, role))
            setattr(self, role, column)
            if strategy is not None:
                collected[role] = strategy
        self.bins = collected


# Roles usable as the x axis; response, predictions and weight are not.
_X_ROLES = frozenset(f.name for f in fields(MortgageColnames)) - {'response', 'model_preds', 'weight', 'bins'}


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
        Extra per-column binning strategy.  Keys are column *roles* (e.g.
        ``'age'``, ``'incentive'``) or source column names, values are
        ``'discrete'``, a numeric rounding unit or a ``BinSpec``.  These
        entries override bins declared on ``colnames``.  Columns not listed
        use quantile binning.
    y_transform : callable or str, optional
        Applied to aggregated actual and predicted means before plotting.
        Can be a callable or a string key from ``NAMED_TRANSFORMS``
        (e.g. ``'smm_to_cpr'``, ``'annualize'``).
    y_as_percent : bool
        If True (default), format the y-axis as percentages.
    theme, style : optional
        Defaults for every plot (e.g. ``PRINT_THEME`` and
        ``PRINT_GROUPED_MEANS_STYLE`` in PDF reports); a ``theme`` or
        ``style`` passed to a plot method wins.
    """

    def __init__(
        self,
        df: pl.DataFrame | pd.DataFrame,
        colnames: MortgageColnames,
        bin_config: dict | None = None,
        y_transform=None,
        y_as_percent: bool = True,
        theme: PlotTheme | None = None,
        style: GroupedMeansStyle | None = None,
    ):
        if not isinstance(df, (pl.DataFrame, pd.DataFrame)):
            raise TypeError("MortgageDiagnostics expects a polars or pandas DataFrame")
        self.df = df
        self.colnames = colnames
        self.bin_config: dict = {**colnames.bins, **(bin_config or {})}

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
        self.theme = theme or MINIMAL_THEME
        self.style = style

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
        bin-level values. Count bars show rows. Scale and label arguments are
        ``draw_grouped_means``'s; here ``y_scale`` defaults to ``'free'`` and
        ``count_scale`` to ``'shared'``. ``figsize`` is per panel. ``title=None`` drops the default figure title,
        e.g. under a report heading. Returns ``(fig, primary_axes)``.
        """
        n_cols = kwargs.pop('n_cols', 3)
        figsize = kwargs.pop('figsize', (6, 4))
        close_unused = kwargs.pop('close_unused', True)
        preds = self.colnames.model_preds
        default_title = None if facet_col is not None else (f"Actual vs {', '.join(preds)}" if preds else "Actual")
        title = kwargs.pop('title', default_title)
        data, options = self._prepare(x, bins, facet_col, facet_series, kwargs)
        result = draw_grouped_means(
            data, wrap=n_cols if facet_col is not None else None,
            panel_size=figsize, title=title, **options,
        )
        fig, axes = result.fig, list(result.axes.flat)
        if not close_unused:
            for ax in axes:
                ax.set_visible(True)
        self._format_y(axes)
        return fig, axes

    def facet_panels(self, x: str, facet_col: str, *, bins=None, facet_series=None, **kwargs) -> PanelSet:
        """Faceted ``plot`` as a ``PanelSet``: a report layout sizes and pages it.

        Aggregates once. ``facet_label`` names the facet in panel titles
        (default: the column name). Takes ``plot``'s options except
        ``figsize`` and ``close_unused``. A ``'shared'`` scale here means one
        range on every page, fixed from the full aggregation: counts default
        to it, ``y_scale`` to ``'free'``. Tick labels default to ``'outer'``
        on scales that are not free, and every axis title to ``'outer'``.
        """
        for name in ('figsize', 'close_unused'):
            if name in kwargs:
                raise TypeError(f"facet_panels takes its panel size from the layout; {name!r} is not accepted")
        n_cols = kwargs.pop('n_cols', 3)
        data, options = self._prepare(x, bins, facet_col, facet_series, kwargs)
        if options['count_scale'] == 'shared':
            totals = data.summary.groupby(['col', 'x'], observed=True)['count'].sum()
            options['count_scale'] = max(float(totals.max()) if len(totals) else 0., 1.) * 1.05
        if options['y_scale'] == 'shared':
            means = data.summary[[f"{metric}__mean" for metric in data.metrics]].to_numpy(dtype=float)
            options['y_scale'] = _padded_range(means)
        options.setdefault('y_ticks', 'all' if options['y_scale'] == 'free' else 'outer')
        options.setdefault('count_ticks', 'all' if options['count_scale'] == 'free' else 'outer')
        for titles in ('y_titles', 'count_titles', 'x_titles'):
            options.setdefault(titles, 'outer')

        def render(panels, n_cols, panel_size):
            result = draw_grouped_means(
                data.select('col', panels), wrap=n_cols, compact_cols=False, panel_size=panel_size, **options,
            )
            self._format_y(result.axes.flat)
            return result.fig

        return PanelSet(data.levels['col'], render, n_cols)

    def _prepare(self, x, bins, facet_col, facet_series, kwargs) -> tuple[GroupedMeansData, dict]:
        """Aggregate ``x`` and turn the remaining ``kwargs`` into drawing options."""
        column = self._source_column(x)
        transform = kwargs.pop('y_transform', self.y_transform)
        xlabel = kwargs.pop('x_label', self._default_x_label(x))
        ylabel = kwargs.pop('y_label', self._default_y_label())
        min_count = kwargs.pop('min_count', 0)
        n_bins = kwargs.pop('n_bins', 10)
        theme = kwargs.pop('theme', None) or self.theme
        pred_colors = kwargs.pop('pred_colors', None)
        facet_label = kwargs.pop('facet_label', None)
        for legacy in ('min_size', 'max_size'):
            if kwargs.pop(legacy, None) is not None:
                warnings.warn(f"{legacy} is ignored; counts are drawn as bars", DeprecationWarning, stacklevel=3)
        if pred_colors is not None:
            theme = replace(theme, palette=(theme.palette[0], *pred_colors))
        if self.style is not None:
            kwargs.setdefault('style', self.style)

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
        if facet_col is not None and facet_label is not None:
            labels[facet_col] = facet_label
        options = dict(labels=labels, ylabel=ylabel, theme=theme, **kwargs)
        options.setdefault('y_scale', 'free')
        options.setdefault('count_scale', 'shared')
        return data, options

    def _format_y(self, axes):
        if self.y_as_percent:
            for ax in axes:
                if ax.get_visible():
                    ax.yaxis.set_major_formatter(StepPercentFormatter())

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

    def orig_balance_plot(self, facet_col: str | None = None, **kwargs):
        """Actual vs predicted by original balance, optionally faceted."""
        self._require('orig_balance')
        return self.plot('orig_balance', facet_col=facet_col, **kwargs)

    def current_balance_plot(self, facet_col: str | None = None, **kwargs):
        """Actual vs predicted by current balance, optionally faceted."""
        self._require('current_balance')
        return self.plot('current_balance', facet_col=facet_col, **kwargs)

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
