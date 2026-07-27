"""Feature provider for the PAR_2025_1 + GENERIC_v4 QuantBullet demo."""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd

from quantbullet.loan_simulation import (
    Loan,
    LoanState,
    PeriodCashflow,
    RuntimeFeatureProvider,
)

MISSING = object()
FeatureAdvance = Callable[["FeatureState", Mapping[str, Any]], Any]
FeatureKind = Literal[
    "static_model",
    "dynamic_model",
    "static_context",
    "dynamic_context",
]

def build_feature_dict(context) -> Mapping[str, Any]:
    return context.model_features

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: FeatureKind
    deps: tuple[str, ...]
    advance: FeatureAdvance | None = None
    carry_forward: bool = False

class FeatureEvaluationError(ValueError):
    pass

def advance_c_age_pct(state: "FeatureState", env: Mapping[str, Any]) -> float:
    return float(state.age_months) / float(state.term) if state.term else 0.0

def advance_cpi_inflator_12(state: "FeatureState", env: Mapping[str, Any]) -> Any:
    return cpi_inflator(
        report_period=state.report_period,
        cpi_lookup=env["cpi_lookup"],
        rounding_digits=env.get("rounding_digits"),
        lookback_months=12,
    )

def advance_cpi_inflator_36(state: "FeatureState", env: Mapping[str, Any]) -> Any:
    return cpi_inflator(
        report_period=state.report_period,
        cpi_lookup=env["cpi_lookup"],
        rounding_digits=env.get("rounding_digits"),
        lookback_months=36,
    )

def advance_days_to_month_end(state: "FeatureState", env: Mapping[str, Any]) -> int:
    return derive_days_to_month_end(state.report_period, state.pmt_day)

def advance_month(state: "FeatureState", env: Mapping[str, Any]) -> str:
    return state.report_period.strftime("%B")

def advance_month_group(state: "FeatureState", env: Mapping[str, Any]) -> str:
    return derive_month_group(state.days_to_month_end)

def advance_rate_incentive_all(state: "FeatureState", env: Mapping[str, Any]) -> Any:
    if state.coupon_at_vintage is None or not state.fico_bkt:
        return MISSING

    coupon_at_report_date = env["fico_coupon_lookup"].get(
        f"{state.report_period.year:04d}{state.report_period.month:02d}|{state.fico_bkt}"
    )
    if coupon_at_report_date is None:
        return MISSING

    return round_model_value(
        float(coupon_at_report_date) - float(state.coupon_at_vintage),
        rounding_digits=env.get("rounding_digits"),
    )

@dataclass
class FeatureState:
    FEATURE_SPECS: ClassVar[tuple[FeatureSpec, ...]] = ()
    MODEL_FEATURE_NAMES: ClassVar[tuple[str, ...]] = ()
    ADVANCED_FEATURE_NAMES: ClassVar[tuple[str, ...]] = ()
    PROVIDER_FEATURE_NAMES: ClassVar[tuple[str, ...]] = ()

    # model-facing fields
    adj_balance_cpi: float = field(metadata={"kind": "static_model"})
    c_age_pct: float = field(
        metadata={
            "kind": "dynamic_model",
            "deps": ("age_months", "term"),
            "advance": advance_c_age_pct,
        }
    )
    c_credit_age: float = field(metadata={"kind": "static_model"})
    cpi_inflator_12: float = field(
        metadata={
            "kind": "dynamic_model",
            "deps": ("report_period",),
            "advance": advance_cpi_inflator_12,
            "carry_forward": True,
        }
    )
    cpi_inflator_36: float = field(
        metadata={
            "kind": "dynamic_model",
            "deps": ("report_period",),
            "advance": advance_cpi_inflator_36,
            "carry_forward": True,
        }
    )
    credit_age: float = field(metadata={"kind": "static_model"})
    days_to_month_end: int = field(
        metadata={
            "kind": "dynamic_model",
            "deps": ("report_period", "pmt_day"),
            "advance": advance_days_to_month_end,
        }
    )
    employed_f: str = field(metadata={"kind": "static_model"})
    hm_owner: str = field(metadata={"kind": "static_model"})
    lending_environment: float = field(metadata={"kind": "static_model"})
    month: str = field(
        metadata={
            "kind": "dynamic_model",
            "deps": ("report_period",),
            "advance": advance_month,
        }
    )
    month_group: str = field(
        metadata={
            "kind": "dynamic_model",
            "deps": ("days_to_month_end",),
            "advance": advance_month_group,
        }
    )
    ofico: float = field(metadata={"kind": "static_model"})
    opti: float = field(metadata={"kind": "static_model"})
    oterm_f: str = field(metadata={"kind": "static_model"})
    purpose: str = field(metadata={"kind": "static_model"})
    rate_incentive_ALL: float = field(
        metadata={
            "kind": "dynamic_model",
            "deps": ("report_period", "fico_bkt", "coupon_at_vintage"),
            "advance": advance_rate_incentive_all,
            "carry_forward": True,
        }
    )
    rel_fico_ratio_ALL: float = field(metadata={"kind": "static_model"})
    v_credit_age: float = field(metadata={"kind": "static_model"})
    v_ofico: float = field(metadata={"kind": "static_model"})
    v_opti: float = field(metadata={"kind": "static_model"})
    v_rate_incentive_ALL: float = field(metadata={"kind": "static_model"})
    v_rel_fico_ratio_ALL: float = field(metadata={"kind": "static_model"})

    # provider-only context for advancing model-facing fields
    report_period: pd.Period = field(metadata={"kind": "dynamic_context"})
    term: int = field(metadata={"kind": "static_context"})
    pmt_day: int = field(metadata={"kind": "static_context"})
    fico_bkt: str = field(metadata={"kind": "static_context"})
    coupon_at_vintage: float | None = field(metadata={"kind": "static_context"})
    age_months: int = field(metadata={"kind": "dynamic_context"})
    period: int = field(default=0, metadata={"kind": "dynamic_context"})

def build_feature_specs() -> tuple[FeatureSpec, ...]:
    specs = []
    for dataclass_field in fields(FeatureState):
        metadata = dataclass_field.metadata
        specs.append(
            FeatureSpec(
                name=dataclass_field.name,
                kind=metadata["kind"],
                deps=tuple(metadata.get("deps", ())),
                advance=metadata.get("advance"),
                carry_forward=bool(metadata.get("carry_forward", False)),
            )
        )
    return tuple(specs)

FeatureState.FEATURE_SPECS = build_feature_specs()
FeatureState.MODEL_FEATURE_NAMES = tuple(
    spec.name
    for spec in FeatureState.FEATURE_SPECS
    if spec.kind in {"static_model", "dynamic_model"}
)
FeatureState.ADVANCED_FEATURE_NAMES = tuple(
    spec.name for spec in FeatureState.FEATURE_SPECS if spec.advance is not None
)
FeatureState.PROVIDER_FEATURE_NAMES = tuple(
    spec.name
    for spec in FeatureState.FEATURE_SPECS
    if spec.kind in {"static_context", "dynamic_context"}
)

def derive_days_to_month_end(report_period: pd.Period, pmt_day: int) -> int:
    return report_period.days_in_month - min(pmt_day, report_period.days_in_month)

def derive_month_group(days_to_month_end: int) -> str:
    return "30_Day" if days_to_month_end <= 28 else "31_Day"

def cpi_inflator(
    *,
    report_period: pd.Period,
    cpi_lookup: Mapping[pd.Period, float],
    rounding_digits: int | None,
    lookback_months: int,
) -> Any:
    cpi_now = cpi_lookup.get(report_period)
    cpi_prior = cpi_lookup.get(report_period - lookback_months)
    if cpi_now is None or cpi_prior is None or cpi_prior <= 0:
        return MISSING
    return round_model_value(
        float(cpi_now) / float(cpi_prior) - 1.0,
        rounding_digits=rounding_digits,
    )

def round_model_value(value: float, *, rounding_digits: int | None) -> float:
    return value if rounding_digits is None else round(value, rounding_digits)

def load_cpi_lookup(
    path: Path,
    *,
    extend_months: int = 120,
) -> dict[pd.Period, float]:
    lookup: dict[pd.Period, float] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            lookup[pd.Period(row["DATE"], freq="M")] = float(row["CPIAUCNS"])

    last_period = max(lookup)
    last_cpi = lookup[last_period]
    cpi_12_months_ago = lookup.get(last_period - 12)
    annual_rate = (
        last_cpi / cpi_12_months_ago - 1.0
        if cpi_12_months_ago and cpi_12_months_ago > 0
        else 0.025
    )
    monthly_rate = (1.0 + annual_rate) ** (1.0 / 12) - 1.0
    cpi = last_cpi
    for offset in range(1, extend_months + 1):
        cpi *= 1.0 + monthly_rate
        lookup.setdefault(last_period + offset, round(cpi, 3))
    return lookup

def load_fico_coupon_lookup(path: Path) -> dict[str, float]:
    lookup: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            lookup[f"{row['vint_moyy']}|{row['fico_bkt']}"] = float(
                row["fico_bkt_coupon"]
            )
    return lookup

class GenericV4FeatureProvider(RuntimeFeatureProvider):
    def __init__(
        self,
        *,
        cpi_lookup: Mapping[pd.Period, float] | None = None,
        fico_coupon_lookup: Mapping[str, float] | None = None,
        rounding_digits: int | None = 4,
    ) -> None:
        self.cpi_lookup = dict(cpi_lookup or {})
        self.fico_coupon_lookup = dict(fico_coupon_lookup or {})
        self.rounding_digits = rounding_digits

    @classmethod
    def from_input_dir(cls, input_dir: Path) -> "GenericV4FeatureProvider":
        macro_dir = Path(input_dir) / "macro"
        cpi_path = macro_dir / "CPIAUCNS.csv"
        fico_coupon_path = macro_dir / "FICO_BKT_COUPON.csv"
        missing = [path for path in (cpi_path, fico_coupon_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing runtime feature input(s): "
                + ", ".join(str(path) for path in missing)
            )
        return cls(
            cpi_lookup=load_cpi_lookup(cpi_path),
            fico_coupon_lookup=load_fico_coupon_lookup(fico_coupon_path),
        )

    @property
    def inputs(self) -> dict[str, Any]:
        return {
            "cpi_lookup": self.cpi_lookup,
            "fico_coupon_lookup": self.fico_coupon_lookup,
            "rounding_digits": self.rounding_digits,
        }

    def initialize_path_state(self, loan: Loan, start_period: pd.Period) -> FeatureState:
        features = loan.metadata
        report_period = pd.Period(features["r_dt"], freq="M")
        pmt_day = int(features.get("pmt_day", 15))
        days_to_month_end = derive_days_to_month_end(report_period, pmt_day)
        coupon_at_vintage = features.get("_coupon_at_vintage")
        return FeatureState(
            adj_balance_cpi=float(features["adj_balance_cpi"]),
            c_age_pct=float(features["c_age_pct"]),
            c_credit_age=float(features["c_credit_age"]),
            cpi_inflator_12=float(features["cpi_inflator_12"]),
            cpi_inflator_36=float(features["cpi_inflator_36"]),
            credit_age=float(features["credit_age"]),
            days_to_month_end=days_to_month_end,
            employed_f=str(features["employed_f"]),
            hm_owner=str(features["hm_owner"]),
            lending_environment=float(features["lending_environment"]),
            month=str(features["month"]),
            month_group=derive_month_group(days_to_month_end),
            ofico=float(features["ofico"]),
            opti=float(features["opti"]),
            oterm_f=str(features["oterm_f"]),
            purpose=str(features["purpose"]),
            rate_incentive_ALL=float(features["rate_incentive_ALL"]),
            rel_fico_ratio_ALL=float(features["rel_fico_ratio_ALL"]),
            v_credit_age=float(features["v_credit_age"]),
            v_ofico=float(features["v_ofico"]),
            v_opti=float(features["v_opti"]),
            v_rate_incentive_ALL=float(features["v_rate_incentive_ALL"]),
            v_rel_fico_ratio_ALL=float(features["v_rel_fico_ratio_ALL"]),
            report_period=report_period,
            term=int(features["term"]),
            pmt_day=pmt_day,
            fico_bkt=str(features["_fico_bkt"]),
            coupon_at_vintage=(
                None if coupon_at_vintage is None else float(coupon_at_vintage)
            ),
            age_months=loan.age_months,
        )

    def model_features_for_period(self, loan: Loan, current_state: LoanState, period_date: pd.Period, macro_features: Mapping[str, Any], path_features: Mapping[str, Any], feature_state: Any) -> Mapping[str, Any]:
        features = {
            name: getattr(feature_state, name)
            for name in FeatureState.MODEL_FEATURE_NAMES
        }
        return features

    def advance_path_state(
        self,
        feature_state: FeatureState,
        cashflow: PeriodCashflow,
        next_state: LoanState,
    ) -> None:
        feature_state.period = next_state.period
        feature_state.age_months = next_state.age_months
        feature_state.report_period = feature_state.report_period + 1

        for spec in FeatureState.FEATURE_SPECS:
            if spec.advance is None:
                continue

            value = spec.advance(feature_state, self.inputs)
            if value is MISSING:
                if spec.carry_forward:
                    continue
                raise FeatureEvaluationError(
                    f"feature {spec.name!r} is missing for period {feature_state.period}"
                )

            setattr(feature_state, spec.name, value)