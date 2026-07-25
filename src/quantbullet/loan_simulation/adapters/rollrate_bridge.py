"""Bridge parsed roll-rate GAM edge models into transition models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from quantbullet.model.gam_replay import GAMReplayModel
from quantbullet.loan_simulation.model_transition import (
    FeatureContext,
    LogitSpec,
    SoftmaxTransitionModel,
)
from quantbullet.loan_simulation.status import StatusConfig


FeatureBuilder = Callable[[FeatureContext], Mapping[str, Any]]


def replay_model_logit(
    edge_model: GAMReplayModel,
    feature_builder: FeatureBuilder,
) -> LogitSpec:
    """Wrap a GAM replay edge model as a ``FeatureContext -> logit`` callable."""
    if not isinstance(edge_model, GAMReplayModel):
        raise TypeError(
            "edge_model must be a quantbullet.model.gam_replay.GAMReplayModel"
        )

    def logit(context: FeatureContext) -> float:
        features = pd.DataFrame([dict(feature_builder(context))])
        predictions = edge_model.predict(features)
        if len(predictions) != 1:
            raise ValueError(
                "GAM replay edge model must return exactly one prediction for "
                f"one feature row; got {len(predictions)}"
            )
        return float(predictions[0])

    return logit


def build_softmax_transition_model(
    edge_models: Mapping[str, Mapping[str, GAMReplayModel]],
    *,
    feature_builder: FeatureBuilder,
    status_config: StatusConfig,
) -> SoftmaxTransitionModel:
    """Build a stay-based softmax transition model from parsed edge models.

    ``edge_models`` is keyed by ``from_status -> to_status -> replay model``.
    Missing non-terminal source statuses are filled with empty rows so the
    resulting ``SoftmaxTransitionModel`` satisfies its explicit-row contract
    without inventing any dataset-specific behavior.
    """
    logits: dict[str, dict[str, LogitSpec]] = {
        status: {}
        for status in status_config.valid_statuses
        if status not in status_config.terminal_statuses
    }

    for from_status, row in edge_models.items():
        normalized_from_status = status_config.require_valid_status(from_status)
        if status_config.is_terminal(normalized_from_status):
            raise ValueError(
                f"Terminal status {normalized_from_status!r} cannot define edge models"
            )
        logits[normalized_from_status] = {
            status_config.require_valid_status(to_status): replay_model_logit(
                edge_model,
                feature_builder,
            )
            for to_status, edge_model in row.items()
        }

    return SoftmaxTransitionModel(logits=logits, status_config=status_config)
