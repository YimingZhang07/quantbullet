from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .entities import Loan, LoanState
from .status import StatusConfig
from .transition import TransitionModel


@dataclass(frozen=True)
class FeatureContext:
    """Inputs available to model-backed transition probability functions."""

    loan: Loan
    current_state: LoanState
    macro_features: Mapping[str, Any] = field(default_factory=dict)
    path_features: Mapping[str, Any] = field(default_factory=dict)
    model_features: Mapping[str, Any] = field(default_factory=dict)


EdgeSpec = float | Callable[[FeatureContext], float]

# Same shape as EdgeSpec, but values are multinomial-logit scores rather than
# probabilities. Kept as a separate alias because the semantics differ.
LogitSpec = float | Callable[[FeatureContext], float]

# Same shape as EdgeSpec, but values are independent binary event probabilities
# that will compete through odds normalization rather than residual stay.
ProbabilitySpec = float | Callable[[FeatureContext], float]


class CompositeTransitionModel(TransitionModel):
    """Assemble status transition rows from edge-level probability specs."""

    def __init__(
        self,
        edges: Mapping[str, Mapping[str, EdgeSpec]],
        *,
        status_config: StatusConfig,
        probability_tolerance: float = 1e-12,
    ) -> None:
        if probability_tolerance < 0:
            raise ValueError("probability_tolerance must be non-negative")

        self.status_config = status_config
        self.probability_tolerance = probability_tolerance
        self._edges = _freeze_edge_table(
            edges,
            status_config=self.status_config,
            value_label="probability",
            validate_constant=_validate_constant_edge_probability,
        )

    @property
    def edges(self) -> Mapping[str, Mapping[str, EdgeSpec]]:
        return self._edges

    def __getstate__(self) -> dict[str, Any]:
        return {
            "edges": _plain_edge_table(self._edges),
            "status_config": self.status_config,
            "probability_tolerance": self.probability_tolerance,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__init__(
            state["edges"],
            status_config=state["status_config"],
            probability_tolerance=state["probability_tolerance"],
        )

    def predict(
        self,
        loan: Loan,
        current_state: LoanState,
        macro_features: Mapping[str, Any] | None = None,
        path_features: Mapping[str, Any] | None = None,
        model_features: Mapping[str, Any] | None = None,
    ) -> Mapping[str, float]:
        from_status = self.status_config.require_valid_status(current_state.status)
        if self.status_config.is_terminal(from_status):
            return MappingProxyType({from_status: 1.0})

        context = FeatureContext(
            loan=loan,
            current_state=current_state,
            macro_features=macro_features or {},
            path_features=path_features or {},
            model_features=model_features or {},
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


class SoftmaxTransitionModel(TransitionModel):
    """Assemble transition rows from edge logits via a stay-based softmax.

    Stay is the base category with an implicit logit of 0:

        P(to_i) = exp(z_i) / (1 + sum_j exp(z_j))
        P(stay) = 1        / (1 + sum_j exp(z_j))

    Scores are shifted by ``max(0, max(logits))`` before exponentiating so
    large finite model outputs cannot overflow ``math.exp``. The shift cancels
    in the ratio and leaves probabilities unchanged.
    """

    def __init__(
        self,
        logits: Mapping[str, Mapping[str, LogitSpec]],
        *,
        status_config: StatusConfig,
    ) -> None:
        self.status_config = status_config
        self._logits = _freeze_edge_table(
            logits,
            status_config=self.status_config,
            value_label="logit",
            validate_constant=_validate_constant_logit,
        )

    @property
    def logits(self) -> Mapping[str, Mapping[str, LogitSpec]]:
        return self._logits

    def __getstate__(self) -> dict[str, Any]:
        return {
            "logits": _plain_edge_table(self._logits),
            "status_config": self.status_config,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__init__(state["logits"], status_config=state["status_config"])

    def predict(
        self,
        loan: Loan,
        current_state: LoanState,
        macro_features: Mapping[str, Any] | None = None,
        path_features: Mapping[str, Any] | None = None,
        model_features: Mapping[str, Any] | None = None,
    ) -> Mapping[str, float]:
        from_status = self.status_config.require_valid_status(current_state.status)
        if self.status_config.is_terminal(from_status):
            return MappingProxyType({from_status: 1.0})

        context = FeatureContext(
            loan=loan,
            current_state=current_state,
            macro_features=macro_features or {},
            path_features=path_features or {},
            model_features=model_features or {},
        )
        scores: dict[str, float] = {}
        for to_status, logit in self._logits[from_status].items():
            value = float(logit(context) if callable(logit) else logit)
            if not math.isfinite(value):
                raise ValueError(
                    "Non-finite transition logit "
                    f"loan_id={loan.loan_id!r}, period={current_state.period}, "
                    f"from_status={from_status!r}, to_status={to_status!r}: {value}"
                )
            scores[to_status] = value

        shift = max(0.0, max(scores.values())) if scores else 0.0
        stay_weight = math.exp(-shift)
        weights = {
            to_status: math.exp(value - shift) for to_status, value in scores.items()
        }
        denominator = stay_weight + sum(weights.values())

        probabilities = {
            to_status: weight / denominator for to_status, weight in weights.items()
        }
        probabilities[from_status] = stay_weight / denominator
        return MappingProxyType(probabilities)


class ProbabilitySoftmaxTransitionModel(TransitionModel):
    """Compete independent binary edge probabilities through odds normalization.

    Each edge spec returns an independent event probability ``p``. The model
    converts it to odds ``p / (1 - p)`` and normalizes those odds against stay
    odds of 1. This is useful when separate binary models estimate different
    mutually-exclusive transition events.
    """

    def __init__(
        self,
        probabilities: Mapping[str, Mapping[str, ProbabilitySpec]],
        *,
        status_config: StatusConfig,
    ) -> None:
        self.status_config = status_config
        self._probabilities = _freeze_edge_table(
            probabilities,
            status_config=self.status_config,
            value_label="competing probability",
            validate_constant=_validate_constant_competing_probability,
        )

    @property
    def probabilities(self) -> Mapping[str, Mapping[str, ProbabilitySpec]]:
        return self._probabilities

    def __getstate__(self) -> dict[str, Any]:
        return {
            "probabilities": _plain_edge_table(self._probabilities),
            "status_config": self.status_config,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__init__(state["probabilities"], status_config=state["status_config"])

    def predict(
        self,
        loan: Loan,
        current_state: LoanState,
        macro_features: Mapping[str, Any] | None = None,
        path_features: Mapping[str, Any] | None = None,
        model_features: Mapping[str, Any] | None = None,
    ) -> Mapping[str, float]:
        from_status = self.status_config.require_valid_status(current_state.status)
        if self.status_config.is_terminal(from_status):
            return MappingProxyType({from_status: 1.0})

        context = FeatureContext(
            loan=loan,
            current_state=current_state,
            macro_features=macro_features or {},
            path_features=path_features or {},
            model_features=model_features or {},
        )
        odds: dict[str, float] = {}
        for to_status, probability_spec in self._probabilities[from_status].items():
            try:
                probability = _evaluate_competing_probability(probability_spec, context)
            except ValueError as exc:
                raise ValueError(
                    "Invalid competing edge probability "
                    f"loan_id={loan.loan_id!r}, period={current_state.period}, "
                    f"from_status={from_status!r}, to_status={to_status!r}: {exc}"
                ) from exc
            odds[to_status] = probability / (1.0 - probability)

        denominator = 1.0 + sum(odds.values())
        probabilities = {
            to_status: value / denominator for to_status, value in odds.items()
        }
        probabilities[from_status] = 1.0 / denominator
        return MappingProxyType(probabilities)


def _evaluate_edge_probability(edge: EdgeSpec, context: FeatureContext) -> float:
    probability = edge(context) if callable(edge) else edge
    probability = float(probability)
    if probability < 0 or probability > 1:
        raise ValueError(f"Edge probability must be between 0 and 1; got {probability}")
    return probability


def _plain_edge_table(table: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {from_status: dict(row) for from_status, row in table.items()}


def _evaluate_competing_probability(
    probability_spec: ProbabilitySpec,
    context: FeatureContext,
) -> float:
    probability = (
        probability_spec(context) if callable(probability_spec) else probability_spec
    )
    probability = float(probability)
    if probability < 0 or probability >= 1:
        raise ValueError(
            "Competing edge probability must be greater than or equal to 0 "
            f"and less than 1; got {probability}"
        )
    return probability


def _freeze_edge_table(
    table: Mapping[str, Mapping[str, Any]],
    *,
    status_config: StatusConfig,
    value_label: str,
    validate_constant: Callable[[Any, str, str], float],
) -> Mapping[str, Mapping[str, Any]]:
    """Validate and freeze a from-status -> to-status edge value table.

    Structural rules are shared by direct-probability and logit models; only
    the constant leaf validation differs, so it is injected by the caller.
    """
    terminal_statuses = status_config.terminal_statuses
    non_terminal_statuses = status_config.valid_statuses - terminal_statuses

    normalized_table: dict[str, Mapping[str, Any]] = {}
    for from_status, row in table.items():
        normalized_from_status = status_config.require_valid_status(from_status)
        if normalized_from_status in terminal_statuses:
            raise ValueError(
                f"Terminal status {normalized_from_status!r} cannot define transition edges"
            )
        normalized_table[normalized_from_status] = _freeze_edge_table_row(
            normalized_from_status,
            row,
            status_config=status_config,
            value_label=value_label,
            validate_constant=validate_constant,
        )

    missing_statuses = non_terminal_statuses - frozenset(normalized_table)
    if missing_statuses:
        raise ValueError(
            f"Transition {value_label} table missing non-terminal statuses: "
            f"{sorted(missing_statuses)}"
        )

    return MappingProxyType(normalized_table)


def _freeze_edge_table_row(
    from_status: str,
    row: Mapping[str, Any],
    *,
    status_config: StatusConfig,
    value_label: str,
    validate_constant: Callable[[Any, str, str], float],
) -> Mapping[str, Any]:
    normalized: dict[str, Any] = {}
    for to_status, value in row.items():
        normalized_to_status = status_config.require_valid_status(to_status)
        if normalized_to_status == from_status:
            raise ValueError(
                f"Stay {value_label} for {from_status!r} cannot be configured; "
                "it is derived automatically"
            )
        if not callable(value):
            validate_constant(value, from_status, normalized_to_status)
        normalized[normalized_to_status] = value

    return MappingProxyType(normalized)


def _validate_constant_edge_probability(
    edge: Any,
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


def _validate_constant_logit(
    logit: Any,
    from_status: str,
    to_status: str,
) -> float:
    try:
        value = float(logit)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Constant edge logit {from_status!r}->{to_status!r} must be numeric"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Constant edge logit {from_status!r}->{to_status!r} "
            f"must be finite; got {value}"
        )
    return value


def _validate_constant_competing_probability(
    probability: Any,
    from_status: str,
    to_status: str,
) -> float:
    try:
        value = float(probability)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Constant competing probability {from_status!r}->{to_status!r} "
            "must be numeric"
        ) from exc
    if value < 0 or value >= 1:
        raise ValueError(
            f"Constant competing probability {from_status!r}->{to_status!r} "
            f"must be >= 0 and < 1; got {value}"
        )
    return value
