from dataclasses import dataclass, field
import numpy as np
import pandas as pd
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
    the generic ``plot_binned_actual_vs_pred`` utility.  The input
    DataFrame is never mutated.

    Parameters
    ----------
    df : pd.DataFrame
        The evaluation dataset (not modified).
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
        df: pd.DataFrame,
        colnames: MortgageColnames,
        bin_config: dict | None = None,
        y_transform=None,
        y_as_percent: bool = True,
    ):
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

    def _vintage_year(self) -> pd.Categorical:
        """Derive vintage year from ``orig_dt`` (returns a Series, no mutation)."""
        self._require('orig_dt')
        years = pd.to_datetime(self.df[self.colnames.orig_dt]).dt.year
        return pd.Categorical(years, categories=sorted(years.unique()), ordered=True)

    def _build_plot_df(self, x_role: str) -> tuple[pd.DataFrame, str]:
        """Assemble a local DataFrame for one plot, applying bin_config rounding.

        Returns ``(plot_df, bins)`` where *bins* is the value to pass to
        ``plot_binned_actual_vs_pred``'s ``bins`` parameter.
        """
        self._require(x_role)
        x_col_name = getattr(self.colnames, x_role)

        data = {
            x_role: self.df[x_col_name].copy(),
            'actual': self.df[self.colnames.response],
        }
        for name, orig_col in self.colnames.model_preds.items():
            data[name] = self.df[orig_col]
        if self.colnames.weight is not None:
            data[self.colnames.weight] = self.df[self.colnames.weight]

        plot_df = pd.DataFrame(data)

        strategy = self.bin_config.get(x_role)
        if isinstance(strategy, (int, float)):
            plot_df[x_role] = (plot_df[x_role] / strategy).round() * strategy
            bins = 'discrete'
        elif strategy == 'discrete':
            bins = 'discrete'
        else:
            bins = None

        return plot_df, bins

    def _plot(self, x_role: str, facet_col: str | None = None,
              facet_series=None, **kwargs):
        """Shared plotting logic: build data, apply instance-level defaults, delegate."""
        plot_df, bins = self._build_plot_df(x_role)
        if facet_col is not None:
            if facet_series is not None:
                plot_df[facet_col] = facet_series
            else:
                plot_df[facet_col] = self.df[facet_col].values

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