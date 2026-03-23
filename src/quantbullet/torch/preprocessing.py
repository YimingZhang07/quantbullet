"""Generic, configurable data preprocessing pipeline.

The ``DataPreprocessor`` learns thresholds (e.g. quantile-based clips) during
``fit_transform`` and reuses them on new data via ``transform``.  This ensures
training and prediction apply identical transformations.

All domain-specific logic (what to filter, how to clip, what features to
derive) is expressed through configuration objects -- the class itself has
zero domain knowledge.

Filters use the tuple format from ``quantbullet.dfutils.filter``::

    ("col", "operator", value)

Supported operators: ``==``, ``!=``, ``>``, ``<``, ``>=``, ``<=``,
``in``, ``not in``, ``between``, ``isnull``, ``notnull``, ``f`` (callable).

Two types of filters:

- **training_filters**: applied during ``fit_transform`` only.  These define
  what the model should learn from (e.g. exclude BWIC, outlier spreads).
- **scoring_rules**: checked during ``predict`` via ``check()``.  These define
  what the model *can* score (e.g. required features not null).  Rows that
  fail are flagged, not dropped.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from quantbullet.dfutils.filter import filter_df, apply_condition


@dataclass
class ClipRule:
    """Specification for clipping a numeric column.

    Fixed bounds (``lower`` / ``upper``) are used as-is.  Quantile-based
    bounds (``quantile_lower`` / ``quantile_upper``) are resolved from
    training data during ``fit_transform`` and stored for reuse.

    ``output_col``: if set, write the clipped values to a new column
    instead of overwriting the original.
    """

    lower: float | None = None
    upper: float | None = None
    quantile_lower: float | None = None
    quantile_upper: float | None = None
    output_col: str | None = None

    resolved_lower: float | None = None
    resolved_upper: float | None = None


@dataclass
class DerivedFeature:
    """A feature derived from existing columns via a callable.

    The callable receives the full DataFrame and returns a Series.
    """

    output_col: str
    func: Callable[[pd.DataFrame], pd.Series]


@dataclass
class CategoricalMapping:
    """Group a string column into ordered categories and integer-encode.

    ``mapping_func`` maps raw values to category names.
    ``categories`` defines the canonical ordering (index 0, 1, 2, ...).
    The integer-encoded column is named ``{output_col}_idx``.
    """

    input_col: str
    output_col: str
    mapping_func: Callable[[str], str]
    categories: list[str]


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class DataPreprocessor:
    """Configurable, domain-agnostic data-cleaning pipeline.

    Parameters
    ----------
    training_filters : list of filter tuples ``(col, operator, value)``
        Applied during ``fit_transform`` only.  Define what the model learns
        from (e.g. exclude BWIC, outlier targets, low-quality rows).
    scoring_rules : list of filter tuples ``(col, operator, value)``
        Checked at prediction time via ``check()``.  Define what the model
        *can* score.  Rows that fail get ``_scorable=False`` with a reason.
        Also applied during ``fit_transform`` (training data must be scorable).
    clip_rules : dict mapping column name to ``ClipRule``.
    derived_features : list of ``DerivedFeature``.
    categorical_mappings : list of ``CategoricalMapping``.
    """

    def __init__(
        self,
        training_filters: list[tuple] | None = None,
        scoring_rules: list[tuple] | None = None,
        clip_rules: dict[str, ClipRule] | None = None,
        derived_features: list[DerivedFeature] | None = None,
        categorical_mappings: list[CategoricalMapping] | None = None,
    ):
        self.training_filters = training_filters or []
        self.scoring_rules = scoring_rules or []
        self.clip_rules = clip_rules or {}
        self.derived_features = derived_features or []
        self.categorical_mappings = categorical_mappings or []
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Learn thresholds from *df*, apply all transforms.  For training.

        Applies both ``training_filters`` and ``scoring_rules`` (drops rows).
        """
        df = df.copy()
        df = self._apply_derived_features(df)
        all_filters = self.training_filters + self.scoring_rules
        df = self._apply_filters(df, all_filters)
        self._resolve_clip_thresholds(df)
        df = self._apply_clips(df)
        df = self._apply_categorical_mappings(df)
        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply derived features, clips, and categorical mappings.

        Does NOT apply training_filters or scoring_rules.  Use ``check()``
        to evaluate scoring_rules and tag non-scorable rows.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform()")
        df = df.copy()
        df = self._apply_derived_features(df)
        df = self._apply_clips(df)
        df = self._apply_categorical_mappings(df)
        return df

    def check(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate scoring_rules and add ``_scorable`` + ``_skip_reason`` columns.

        Does not drop any rows.  Returns the same DataFrame with two new columns:

        - ``_scorable``: bool, True if the row passes all scoring rules.
        - ``_skip_reason``: str, comma-separated reasons for non-scorable rows.
        """
        reasons = pd.Series("", index=df.index, dtype=str)
        scorable = pd.Series(True, index=df.index, dtype=bool)

        for rule in self.scoring_rules:
            col, op, val = rule
            if col not in df.columns:
                continue
            try:
                mask = apply_condition(df, col, op, val)
            except Exception as e:
                scorable[:] = False
                reasons = reasons + f"rule error ({col} {op}): {e}, "
                continue
            failed = ~mask
            if failed.any():
                scorable[failed] = False
                desc = f"{col} {op} {val}" if val is not None else f"{col} {op}"
                reasons[failed] = reasons[failed] + desc + ", "

        reasons = reasons.str.rstrip(", ")
        df["_scorable"] = scorable
        df["_skip_reason"] = reasons
        return df

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _apply_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for feat in self.derived_features:
            df[feat.output_col] = feat.func(df)
        return df

    def _apply_filters(self, df: pd.DataFrame, filters: list[tuple]) -> pd.DataFrame:
        if not filters:
            return df
        n_before = len(df)
        df = filter_df(df, filters)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"Filters: dropped {n_dropped} rows ({n_dropped / n_before * 100:.1f}%)")
        return df.copy()

    def _resolve_clip_thresholds(self, df: pd.DataFrame) -> None:
        for col, rule in self.clip_rules.items():
            if rule.quantile_lower is not None:
                rule.resolved_lower = float(df[col].quantile(rule.quantile_lower))
            else:
                rule.resolved_lower = rule.lower
            if rule.quantile_upper is not None:
                rule.resolved_upper = float(df[col].quantile(rule.quantile_upper))
            else:
                rule.resolved_upper = rule.upper

    def _apply_clips(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, rule in self.clip_rules.items():
            out = rule.output_col or col
            df[out] = df[col].clip(lower=rule.resolved_lower, upper=rule.resolved_upper)
        return df

    def _apply_categorical_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        for cm in self.categorical_mappings:
            df[cm.output_col] = df[cm.input_col].map(cm.mapping_func)
            cat_to_idx = {c: i for i, c in enumerate(cm.categories)}
            df[cm.output_col + "_idx"] = df[cm.output_col].map(cat_to_idx)
        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump({
                "training_filters": self.training_filters,
                "scoring_rules": self.scoring_rules,
                "clip_rules": {col: asdict(rule) for col, rule in self.clip_rules.items()},
                "derived_features": self.derived_features,
                "categorical_mappings": self.categorical_mappings,
                "fitted": self._fitted,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "DataPreprocessor":
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        clip_rules = {
            col: ClipRule(**rule_dict)
            for col, rule_dict in data["clip_rules"].items()
        }
        obj = cls(
            training_filters=data["training_filters"],
            scoring_rules=data["scoring_rules"],
            clip_rules=clip_rules,
            derived_features=data["derived_features"],
            categorical_mappings=data["categorical_mappings"],
        )
        obj._fitted = data["fitted"]
        return obj
