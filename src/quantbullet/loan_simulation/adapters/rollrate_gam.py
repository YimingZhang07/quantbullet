"""Adapters for the TSV GAM coefficient format used by roll-rate-model."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quantbullet.model.gam_replay import GAMReplayModel


_REQUIRED_COLUMNS = frozenset(
    {"model", "var_name1", "var_val1", "var_name2", "var_val2", "value"}
)
_SMOOTH_MIN_ROWS = 6


@dataclass(frozen=True)
class _CoefficientRow:
    model: str
    var_name1: str
    var_val1: str
    var_name2: str
    var_val2: str
    value: float
    line_number: int


def parse_rollrate_coefficients(path: str | Path) -> dict[str, GAMReplayModel]:
    """Parse one roll-rate coefficient TSV into replayable edge-logit models.

    The returned mapping is keyed by the TSV ``model`` column, which is the
    destination status for the source-status file. The parser supports the term
    shapes used by the first synthetic adapter tests and by real roll-rate
    coefficient dumps: intercepts, categorical lookups, one-dimensional
    smooths, factor-by smooths, and numeric-by smooths.
    """
    rows_by_model: dict[str, list[_CoefficientRow]] = defaultdict(list)
    for row in _read_coefficient_rows(path):
        rows_by_model[row.model].append(row)

    return {
        model_name: _build_replay_model(model_name, rows)
        for model_name, rows in rows_by_model.items()
    }


def _read_coefficient_rows(path: str | Path) -> list[_CoefficientRow]:
    coefficient_path = Path(path)
    with coefficient_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        fieldnames = set(reader.fieldnames or ())
        missing_columns = _REQUIRED_COLUMNS - fieldnames
        if missing_columns:
            raise ValueError(
                f"Coefficient file {coefficient_path} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        rows = [
            _parse_row(raw_row, line_number=line_number, path=coefficient_path)
            for line_number, raw_row in enumerate(reader, start=2)
        ]

    if not rows:
        raise ValueError(f"Coefficient file {coefficient_path} contains no rows")
    return rows


def _parse_row(
    raw_row: dict[str, str | None],
    *,
    line_number: int,
    path: Path,
) -> _CoefficientRow:
    if raw_row.get(None):
        raise ValueError(
            f"Coefficient file {path} line {line_number} has more fields than its header"
        )

    model = _required_text(raw_row, "model", line_number=line_number, path=path)
    var_name1 = _required_text(
        raw_row,
        "var_name1",
        line_number=line_number,
        path=path,
    )
    var_val1 = _required_text(
        raw_row,
        "var_val1",
        line_number=line_number,
        path=path,
    )
    var_name2 = (raw_row.get("var_name2") or "").strip()
    var_val2 = (raw_row.get("var_val2") or "").strip()
    if not var_name2 and var_val2:
        raise ValueError(
            f"Coefficient file {path} line {line_number} has var_val2 without var_name2"
        )

    raw_value = _required_text(raw_row, "value", line_number=line_number, path=path)
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Coefficient file {path} line {line_number} has non-numeric value "
            f"{raw_value!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Coefficient file {path} line {line_number} has non-finite value "
            f"{raw_value!r}"
        )

    return _CoefficientRow(
        model=model,
        var_name1=var_name1,
        var_val1=var_val1,
        var_name2=var_name2,
        var_val2=var_val2,
        value=value,
        line_number=line_number,
    )


def _required_text(
    row: dict[str, str | None],
    column: str,
    *,
    line_number: int,
    path: Path,
) -> str:
    value = (row.get(column) or "").strip()
    if not value:
        raise ValueError(
            f"Coefficient file {path} line {line_number} requires {column!r}"
        )
    return value


def _build_replay_model(
    model_name: str,
    rows: list[_CoefficientRow],
) -> GAMReplayModel:
    from quantbullet.model.gam.terms import (
        FactorTermData,
        SplineByGroupTermData,
        SplineByNumericTermData,
        SplineTermData,
    )
    from quantbullet.model.gam_replay import GAMReplayModel

    rows_by_term: dict[tuple[str, str], list[_CoefficientRow]] = defaultdict(list)
    for row in rows:
        rows_by_term[(row.var_name1, row.var_name2)].append(row)

    intercept: float | None = None
    term_data = {}
    for (var_name1, var_name2), term_rows in rows_by_term.items():
        if var_name1 == "intercept":
            intercept = _parse_intercept(
                term_rows,
                model_name=model_name,
                var_name2=var_name2,
            )
        elif not var_name2:
            term_data[var_name1] = _build_one_dimensional_term(
                term_rows,
                model_name=model_name,
                feature=var_name1,
                factor_type=FactorTermData,
                spline_type=SplineTermData,
            )
        else:
            term_data[(var_name1, var_name2)] = _build_by_term(
                term_rows,
                model_name=model_name,
                feature=var_name1,
                by_feature=var_name2,
                group_spline_type=SplineByGroupTermData,
                numeric_spline_type=SplineByNumericTermData,
            )

    if intercept is None:
        raise ValueError(f"Model {model_name!r} does not contain an intercept")
    return GAMReplayModel(term_data=term_data, intercept=intercept)


def _parse_intercept(
    rows: list[_CoefficientRow],
    *,
    model_name: str,
    var_name2: str,
) -> float:
    if var_name2:
        raise ValueError(
            f"Model {model_name!r} has an unsupported by-variable on its intercept"
        )
    if len(rows) != 1 or rows[0].var_val1 != "intercept":
        raise ValueError(
            f"Model {model_name!r} must contain exactly one "
            "'intercept\\tintercept' row"
        )
    return rows[0].value


def _build_one_dimensional_term(
    rows: list[_CoefficientRow],
    *,
    model_name: str,
    feature: str,
    factor_type,
    spline_type,
):
    numeric_flags = [_is_finite_number(row.var_val1) for row in rows]
    term_label = _term_label(model_name, feature)
    if all(numeric_flags):
        if len(rows) < _SMOOTH_MIN_ROWS:
            raise ValueError(
                f"{term_label} has {len(rows)} numeric rows; "
                f"at least {_SMOOTH_MIN_ROWS} are required for a smooth"
            )
        x, y = _sorted_curve(rows, term_label=term_label)
        return spline_type(feature=feature, x=x, y=y, interpolation="linear")
    if any(numeric_flags):
        raise ValueError(
            f"{term_label} mixes numeric grid values and categorical levels"
        )

    _require_unique_values(rows, term_label=term_label)
    return factor_type(
        feature=feature,
        categories=[row.var_val1 for row in rows],
        values=np.asarray([row.value for row in rows], dtype=float),
    )


def _build_by_term(
    rows: list[_CoefficientRow],
    *,
    model_name: str,
    feature: str,
    by_feature: str,
    group_spline_type,
    numeric_spline_type,
):
    term_label = _term_label(model_name, feature, by_feature)
    if not all(_is_finite_number(row.var_val1) for row in rows):
        raise ValueError(
            f"{term_label} is a categorical interaction, which is not supported"
        )

    group_labels = {row.var_val2 for row in rows}
    if group_labels == {""}:
        x, y = _sorted_curve(rows, term_label=term_label)
        return numeric_spline_type(
            feature=feature,
            multiplier_feature=by_feature,
            x=x,
            y=y,
            interpolation="linear",
        )
    if "" in group_labels:
        raise ValueError(
            f"{term_label} mixes blank and categorical by-level values"
        )
    if len(group_labels) == 1:
        # roll-rate's own parsers classify this shape as a numeric-by smooth
        # (multiplying by the loan's value of the by-variable), while reading
        # the label as a one-level factor-by is equally plausible. Refuse to
        # guess so the divergence surfaces at parse time instead of as silent
        # logit differences in a tie-out.
        only_label = next(iter(group_labels))
        raise ValueError(
            f"{term_label} has a single by-level {only_label!r}; this is "
            "ambiguous between a factor-by smooth and a numeric-by multiplier "
            "and must be resolved manually"
        )

    curves = {}
    for group_label in sorted(group_labels):
        group_rows = [row for row in rows if row.var_val2 == group_label]
        x, y = _sorted_curve(
            group_rows,
            term_label=f"{term_label}, level={group_label!r}",
        )
        curves[group_label] = {"x": x, "y": y}
    return group_spline_type(
        feature=feature,
        by_feature=by_feature,
        group_curves=curves,
        interpolation="linear",
    )


def _sorted_curve(
    rows: list[_CoefficientRow],
    *,
    term_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    if len(rows) < _SMOOTH_MIN_ROWS:
        raise ValueError(
            f"{term_label} has {len(rows)} numeric rows; "
            f"at least {_SMOOTH_MIN_ROWS} are required for a smooth"
        )

    points = sorted((float(row.var_val1), row.value) for row in rows)
    x = np.asarray([point[0] for point in points], dtype=float)
    if not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError(
            f"{term_label} grid values must be finite and strictly increasing"
        )
    return x, np.asarray([point[1] for point in points], dtype=float)


def _require_unique_values(
    rows: list[_CoefficientRow],
    *,
    term_label: str,
) -> None:
    values = [row.var_val1 for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"{term_label} contains duplicate categorical levels")


def _is_finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def _term_label(model_name: str, feature: str, by_feature: str = "") -> str:
    if by_feature:
        return f"Model {model_name!r}, term ({feature!r}, {by_feature!r})"
    return f"Model {model_name!r}, term {feature!r}"
