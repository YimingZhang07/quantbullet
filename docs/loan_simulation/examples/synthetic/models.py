"""Probability models and transition graph for the synthetic example."""

from __future__ import annotations

from dataclasses import dataclass

from feature_builder import build_feature_dict
from quantbullet.loan_simulation import (
    FeatureContext,
    LoanStatus,
    ProbabilitySoftmaxTransitionModel,
    StatusConfig,
)


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


CURRENT_TO_DQ30_MODEL = BoundedLinearProbabilityModel(
    intercept=0.045,
    age_coefficient=0.00015,
    incentive_coefficient=0.40,
    hpi_coefficient=-0.0005,
    minimum=0.02,
    maximum=0.12,
)

CURRENT_TO_PREPAID_MODEL = BoundedLinearProbabilityModel(
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
            LoanStatus.CURRENT,
            LoanStatus.DQ30,
            LoanStatus.DQ60,
            LoanStatus.PREPAID,
            LoanStatus.CHARGED_OFF,
        },
        terminal_statuses={LoanStatus.PREPAID, LoanStatus.CHARGED_OFF},
        prepay_statuses={LoanStatus.PREPAID},
        default_statuses={LoanStatus.CHARGED_OFF},
        delinquency_buckets={
            LoanStatus.DQ30: "dq30_balance",
            LoanStatus.DQ60: "dq60_balance",
        },
    )


def build_transition_model(
    status_config: StatusConfig,
) -> ProbabilitySoftmaxTransitionModel:
    return ProbabilitySoftmaxTransitionModel(
        probabilities={
            LoanStatus.CURRENT: {
                LoanStatus.DQ30: CURRENT_TO_DQ30_MODEL,
                LoanStatus.PREPAID: CURRENT_TO_PREPAID_MODEL,
            },
            LoanStatus.DQ30: {
                LoanStatus.CURRENT: 0.30,
                LoanStatus.DQ60: 0.25,
                LoanStatus.PREPAID: 0.03,
                LoanStatus.CHARGED_OFF: 0.02,
            },
            LoanStatus.DQ60: {
                LoanStatus.CURRENT: 0.10,
                LoanStatus.DQ30: 0.20,
                LoanStatus.PREPAID: 0.02,
                LoanStatus.CHARGED_OFF: 0.25,
            },
        },
        status_config=status_config,
    )
