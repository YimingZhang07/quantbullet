from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .entities import PeriodCashflow
from .status import StatusConfig


@dataclass
class PathFeatureTracker:
    """Track compact path-dependent delinquency history features."""

    ever_delinquent: bool = False
    months_since_last_delinquency: int | None = None
    consecutive_delinquent_months: int = 0
    times_delinquent: int = 0

    def features(self) -> dict[str, Any]:
        return {
            "ever_delinquent": self.ever_delinquent,
            "months_since_last_delinquency": self.months_since_last_delinquency,
            "consecutive_delinquent_months": self.consecutive_delinquent_months,
            "times_delinquent": self.times_delinquent,
        }

    def update(
        self,
        cashflow: PeriodCashflow,
        status_config: StatusConfig,
    ) -> None:
        if status_config.is_delinquent(cashflow.end_status):
            self.ever_delinquent = True
            self.months_since_last_delinquency = 0
            self.consecutive_delinquent_months += 1
            self.times_delinquent += 1
            return

        self.consecutive_delinquent_months = 0
        if self.months_since_last_delinquency is not None:
            self.months_since_last_delinquency += 1
