from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .entities import Loan, LoanState


@dataclass(frozen=True)
class FeatureContext:
    """Inputs available to model-backed transition probability functions."""

    loan: Loan
    current_state: LoanState
    macro_features: Mapping[str, Any] = field(default_factory=dict)
    path_features: Mapping[str, Any] = field(default_factory=dict)


EdgeSpec = float | Callable[[FeatureContext], float]


def _evaluate_edge_probability(edge: EdgeSpec, context: FeatureContext) -> float:
    probability = edge(context) if callable(edge) else edge
    probability = float(probability)
    if probability < 0 or probability > 1:
        raise ValueError(f"Edge probability must be between 0 and 1; got {probability}")
    return probability
