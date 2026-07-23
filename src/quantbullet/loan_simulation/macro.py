from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import pandas as pd


class MacroFeatureProvider(ABC):
    """Base class for calendar-date macro feature lookup."""

    @abstractmethod
    def features_for_date(self, period_date: Any) -> Mapping[str, Any]:
        """Return macro features for a simulation period date.

        ``period_date`` may be a date-like string, ``datetime.date``,
        ``datetime.datetime``, ``pandas.Timestamp``, or ``pandas.Period``.
        Implementations define how the date is aligned to their data frequency.
        """
        raise NotImplementedError


class DataFrameMacroFeatureProvider(MacroFeatureProvider):
    """Macro feature provider backed by a pandas DataFrame.

    The DataFrame index must be date-like. It is converted to a monthly
    PeriodIndex by default so simulations can match features by calendar month.
    ``features_for_date`` accepts date-like strings, Python date/datetime
    objects, pandas Timestamps, or pandas Periods.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        *,
        frequency: str = "M",
        method: str = "exact",
    ) -> None:
        if method not in {"exact", "ffill"}:
            raise ValueError("method must be 'exact' or 'ffill'")
        if features.empty:
            raise ValueError("features must be non-empty")

        self.frequency = frequency
        self.method = method
        self._features = _prepare_feature_frame(features, frequency=frequency)

    def features_for_date(self, period_date: Any) -> Mapping[str, Any]:
        period = _to_period(period_date, frequency=self.frequency)

        if self.method == "exact":
            if period not in self._features.index:
                raise KeyError(f"No macro features available for period {period}")
            return self._features.loc[period].to_dict()

        eligible = self._features.loc[self._features.index <= period]
        if eligible.empty:
            raise KeyError(f"No macro features available on or before period {period}")
        return eligible.iloc[-1].to_dict()


def _prepare_feature_frame(
    features: pd.DataFrame,
    *,
    frequency: str,
) -> pd.DataFrame:
    frame = features.copy()
    if isinstance(frame.index, pd.PeriodIndex):
        frame.index = frame.index.asfreq(frequency)
    else:
        frame.index = pd.to_datetime(frame.index).to_period(frequency)

    if frame.index.has_duplicates:
        duplicated = sorted(str(value) for value in frame.index[frame.index.duplicated()])
        raise ValueError(f"features index contains duplicate periods: {duplicated}")

    return frame.sort_index()


def _to_period(period_date: Any, *, frequency: str) -> pd.Period:
    if isinstance(period_date, pd.Period):
        return period_date.asfreq(frequency)
    return pd.Period(period_date, freq=frequency)
