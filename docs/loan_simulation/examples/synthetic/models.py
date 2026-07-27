"""Probability models and transition graph for the synthetic example."""

from __future__ import annotations

from dataclasses import dataclass

from feature_builder import build_feature_dict
from quantbullet.loan_simulation import (
    FeatureContext,
    ProbabilitySoftmaxTransitionModel,
    StatusConfig,
)


CURRENT = "C"
DELINQUENT_1 = "D1"
DELINQUENT_2 = "D2"
PREPAID = "PIF"
CHARGED_OFF = "CO"


@dataclass(frozen=True)
class BoundedLinearProbabilityModel:
    """Score a bounded binary probability from age, incentive, and HPI."""

    intercept: float
    age_coefficient: float
    incentive_coefficient: float
    hpi_coefficient: float
    minimum: float
    maximum: float
    hpi_center: float = 100.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum <= self.maximum < 1.0:
            raise ValueError("Probability bounds must satisfy 0 <= min <= max < 1")

    def __call__(self, context: FeatureContext) -> float:
        features = build_feature_dict(context)
        probability = (
            self.intercept
            + self.age_coefficient * features["age"]
            + self.incentive_coefficient * features["incentive"]
            + self.hpi_coefficient * (features["hpi"] - self.hpi_center)
        )
        return min(max(probability, self.minimum), self.maximum)


C_TO_D1_MODEL = BoundedLinearProbabilityModel(
    intercept=0.045,
    age_coefficient=0.00015,
    incentive_coefficient=0.40,
    hpi_coefficient=-0.0005,
    minimum=0.02,
    maximum=0.12,
)

C_TO_PIF_MODEL = BoundedLinearProbabilityModel(
    intercept=0.025,
    age_coefficient=0.00025,
    incentive_coefficient=1.20,
    hpi_coefficient=0.0003,
    minimum=0.02,
    maximum=0.18,
)


def build_status_config() -> StatusConfig:
    return StatusConfig(
        valid_statuses={
            CURRENT,
            DELINQUENT_1,
            DELINQUENT_2,
            PREPAID,
            CHARGED_OFF,
        },
        terminal_statuses={PREPAID, CHARGED_OFF},
        prepay_statuses={PREPAID},
        default_statuses={CHARGED_OFF},
        delinquency_buckets={
            DELINQUENT_1: "dq30_balance",
            DELINQUENT_2: "dq60_balance",
        },
    )


def build_transition_model(
    status_config: StatusConfig | None = None,
) -> ProbabilitySoftmaxTransitionModel:
    config = status_config or build_status_config()
    return ProbabilitySoftmaxTransitionModel(
        probabilities={
            CURRENT: {
                DELINQUENT_1: C_TO_D1_MODEL,
                PREPAID: C_TO_PIF_MODEL,
            },
            DELINQUENT_1: {
                CURRENT: 0.30,
                DELINQUENT_2: 0.25,
                PREPAID: 0.03,
                CHARGED_OFF: 0.02,
            },
            DELINQUENT_2: {
                CURRENT: 0.10,
                DELINQUENT_1: 0.20,
                PREPAID: 0.02,
                CHARGED_OFF: 0.25,
            },
        },
        status_config=config,
    )
