from __future__ import annotations

import random
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, Mapping

from .entities import Loan, LoanState
from .status import StatusConfig


class TransitionModel(ABC):
    """Base class for one-period loan status transition probabilities."""

    @abstractmethod
    def predict(
        self,
        loan: Loan,
        current_state: LoanState,
        macro_features: Mapping[str, Any] | None = None,
        path_features: Mapping[str, Any] | None = None,
        model_features: Mapping[str, Any] | None = None,
    ) -> Mapping[str, float]:
        """Return next-status probabilities for one loan path.

        Projection period, loan age, balance, and status are read from
        ``current_state`` so the simulator has a single source of truth.
        """
        raise NotImplementedError


class ConstantTransitionModel(TransitionModel):
    """Status transition model backed by fixed probability tables."""

    def __init__(
        self,
        transitions: Mapping[str, Mapping[str, float]],
        *,
        status_config: StatusConfig,
        probability_tolerance: float = 1e-9,
    ) -> None:
        self.status_config = status_config
        self.probability_tolerance = probability_tolerance
        self._transitions = _freeze_transition_table(
            transitions,
            status_config=self.status_config,
            probability_tolerance=probability_tolerance,
        )

    @property
    def transitions(self) -> Mapping[str, Mapping[str, float]]:
        return self._transitions

    def __getstate__(self) -> dict[str, Any]:
        return {
            "transitions": {
                status: dict(probabilities)
                for status, probabilities in self._transitions.items()
            },
            "status_config": self.status_config,
            "probability_tolerance": self.probability_tolerance,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.status_config = state["status_config"]
        self.probability_tolerance = state["probability_tolerance"]
        self._transitions = _freeze_transition_table(
            state["transitions"],
            status_config=self.status_config,
            probability_tolerance=self.probability_tolerance,
        )

    def predict(
        self,
        loan: Loan,
        current_state: LoanState,
        macro_features: Mapping[str, Any] | None = None,
        path_features: Mapping[str, Any] | None = None,
        model_features: Mapping[str, Any] | None = None,
    ) -> Mapping[str, float]:
        current_status = self.status_config.require_valid_status(current_state.status)
        probabilities = self._transitions.get(current_status)
        if probabilities is not None:
            return probabilities

        raise KeyError(f"No transition probabilities configured for {current_status!r}")


def sample_next_status(
    probabilities: Mapping[str, float],
    rng: random.Random | None = None,
) -> str:
    """Sample one status from a probability mapping."""
    if not probabilities:
        raise ValueError("probabilities must be non-empty")

    generator = rng or random.Random()
    draw = generator.random()
    cumulative = 0.0
    last_status = None

    for status, probability in probabilities.items():
        last_status = status
        cumulative += probability
        if draw <= cumulative:
            return status

    # Guard against tiny floating point gaps when probabilities sum to 1.
    if last_status is not None:
        return last_status

    raise ValueError("probabilities must be non-empty")


def _freeze_transition_table(
    transitions: Mapping[str, Mapping[str, float]],
    *,
    status_config: StatusConfig,
    probability_tolerance: float,
) -> Mapping[str, Mapping[str, float]]:
    if not transitions:
        raise ValueError("transitions must be non-empty")
    if probability_tolerance < 0:
        raise ValueError("probability_tolerance must be non-negative")

    normalized: dict[str, Mapping[str, float]] = {}
    for from_status, probabilities in transitions.items():
        normalized_from_status = status_config.require_valid_status(from_status)
        normalized[normalized_from_status] = _freeze_probability_row(
            probabilities,
            status_config=status_config,
            probability_tolerance=probability_tolerance,
        )

    missing_statuses = status_config.valid_statuses - frozenset(normalized)
    if missing_statuses:
        raise ValueError(
            "Transition table is missing rows for statuses: "
            f"{sorted(missing_statuses)}"
        )

    return MappingProxyType(normalized)


def _freeze_probability_row(
    probabilities: Mapping[str, float],
    *,
    status_config: StatusConfig,
    probability_tolerance: float,
) -> Mapping[str, float]:
    if not probabilities:
        raise ValueError("transition probability rows must be non-empty")

    total_probability = 0.0
    normalized: dict[str, float] = {}
    for to_status, probability in probabilities.items():
        normalized_to_status = status_config.require_valid_status(to_status)
        probability = float(probability)
        if probability < 0:
            raise ValueError(
                f"Transition probability for {normalized_to_status!r} cannot be negative"
            )
        normalized[normalized_to_status] = probability
        total_probability += probability

    if abs(total_probability - 1.0) > probability_tolerance:
        raise ValueError(
            "Transition probabilities must sum to 1.0; "
            f"got {total_probability:.12g}"
        )

    return MappingProxyType(normalized)
