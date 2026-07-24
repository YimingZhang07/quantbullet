from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .entities import Loan, LoanState
from .status import DEFAULT_STATUS_CONFIG, StatusConfig
from .transition import TransitionModel


@dataclass(frozen=True)
class FeatureContext:
    """Inputs available to model-backed transition probability functions."""

    loan: Loan
    current_state: LoanState
    macro_features: Mapping[str, Any] = field(default_factory=dict)
    path_features: Mapping[str, Any] = field(default_factory=dict)


EdgeSpec = float | Callable[[FeatureContext], float]


class CompositeTransitionModel(TransitionModel):
    """Assemble status transition rows from edge-level probability specs."""

    def __init__(
        self,
        edges: Mapping[str, Mapping[str, EdgeSpec]],
        *,
        status_config: StatusConfig | None = None,
        probability_tolerance: float = 1e-12,
    ) -> None:
        if probability_tolerance < 0:
            raise ValueError("probability_tolerance must be non-negative")

        self.status_config = status_config or DEFAULT_STATUS_CONFIG
        self.probability_tolerance = probability_tolerance
        self._edges = _freeze_edges(edges, status_config=self.status_config)

    @property
    def edges(self) -> Mapping[str, Mapping[str, EdgeSpec]]:
        return self._edges

    def predict(
        self,
        loan: Loan,
        current_state: LoanState,
        macro_features: Mapping[str, Any] | None = None,
        path_features: Mapping[str, Any] | None = None,
    ) -> Mapping[str, float]:
        from_status = self.status_config.require_valid_status(current_state.status)
        if self.status_config.is_terminal(from_status):
            return MappingProxyType({from_status: 1.0})

        context = FeatureContext(
            loan=loan,
            current_state=current_state,
            macro_features=macro_features or {},
            path_features=path_features or {},
        )
        probabilities: dict[str, float] = {}
        for to_status, edge in self._edges[from_status].items():
            try:
                probabilities[to_status] = _evaluate_edge_probability(edge, context)
            except ValueError as exc:
                raise ValueError(
                    "Invalid edge probability "
                    f"loan_id={loan.loan_id!r}, period={current_state.period}, "
                    f"from_status={from_status!r}, to_status={to_status!r}: {exc}"
                ) from exc

        edge_sum = sum(probabilities.values())
        if edge_sum > 1.0 + self.probability_tolerance:
            raise ValueError(
                "Composite transition probabilities exceed 1 "
                f"loan_id={loan.loan_id!r}, period={current_state.period}, "
                f"from_status={from_status!r}, probabilities={probabilities}, "
                f"sum={edge_sum}"
            )

        probabilities[from_status] = max(0.0, 1.0 - edge_sum)
        return MappingProxyType(probabilities)


def _evaluate_edge_probability(edge: EdgeSpec, context: FeatureContext) -> float:
    probability = edge(context) if callable(edge) else edge
    probability = float(probability)
    if probability < 0 or probability > 1:
        raise ValueError(f"Edge probability must be between 0 and 1; got {probability}")
    return probability


def _freeze_edges(
    edges: Mapping[str, Mapping[str, EdgeSpec]],
    *,
    status_config: StatusConfig,
) -> Mapping[str, Mapping[str, EdgeSpec]]:
    terminal_statuses = status_config.terminal_statuses
    non_terminal_statuses = status_config.valid_statuses - terminal_statuses

    normalized_edges: dict[str, Mapping[str, EdgeSpec]] = {}
    for from_status, row in edges.items():
        normalized_from_status = status_config.require_valid_status(from_status)
        if normalized_from_status in terminal_statuses:
            raise ValueError(
                f"Terminal status {normalized_from_status!r} cannot define transition edges"
            )
        normalized_edges[normalized_from_status] = _freeze_edge_row(
            normalized_from_status,
            row,
            status_config=status_config,
        )

    missing_statuses = non_terminal_statuses - frozenset(normalized_edges)
    if missing_statuses:
        raise ValueError(
            "Composite transition edges missing non-terminal statuses: "
            f"{sorted(missing_statuses)}"
        )

    return MappingProxyType(normalized_edges)


def _freeze_edge_row(
    from_status: str,
    row: Mapping[str, EdgeSpec],
    *,
    status_config: StatusConfig,
) -> Mapping[str, EdgeSpec]:
    normalized: dict[str, EdgeSpec] = {}
    for to_status, edge in row.items():
        normalized_to_status = status_config.require_valid_status(to_status)
        if normalized_to_status == from_status:
            raise ValueError(
                f"Stay probability for {from_status!r} is residual and cannot be configured"
            )
        if not callable(edge):
            _evaluate_constant_edge_probability(edge, from_status, normalized_to_status)
        normalized[normalized_to_status] = edge

    return MappingProxyType(normalized)


def _evaluate_constant_edge_probability(
    edge: EdgeSpec,
    from_status: str,
    to_status: str,
) -> float:
    try:
        probability = float(edge)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Constant edge probability {from_status!r}->{to_status!r} must be numeric"
        ) from exc
    if probability < 0 or probability > 1:
        raise ValueError(
            f"Constant edge probability {from_status!r}->{to_status!r} "
            f"must be between 0 and 1; got {probability}"
        )
    return probability
