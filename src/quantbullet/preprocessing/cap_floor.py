from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Mapping, Sequence, Any, Union
import pandas as pd

W = "*"

@dataclass(frozen=True)
class Bounds:
    floor: Optional[float] = None
    cap: Optional[float] = None


# feature -> depends_on -> table
# Example:
# RULES["MVOC"][("currency","rating")] = { ("USD","BBB"): Bounds(...), ("USD","*"): ..., ("*","*"): ... }
Rules = Dict[str, Dict[Tuple[str, ...], Dict[Tuple[str, ...], Bounds]]]

def _key(values: Sequence[Any]) -> Tuple[str, ...]:
    return tuple(W if v is None else str(v) for v in values)

def get_bounds(
    rules: Rules,
    feature: str,
    depends_on: Tuple[str, ...],
    *values: Any,
) -> Bounds:
    """
    Explicit resolver for a given dependency shape.
    For 1 dim: (x) -> (*) 
    For 2 dims: (x,y) -> (x,*) -> (*,y) -> (*,*)
    For >2 dims: we keep it strict (exact or global) to avoid magic.
    """
    table = rules[feature][depends_on]
    vals = _key(values)

    if len(depends_on) == 0:
        return table[( )]  # store under empty tuple

    if len(depends_on) == 1:
        (a,) = vals
        return table.get((a,), table[(W,)])

    if len(depends_on) == 2:
        a, b = vals
        return (
            table.get((a, b))
            or table.get((a, W))
            or table.get((W, b))
            or table[(W, W)]
        )

    # Keep >2 dims explicit to avoid “mystery resolution”
    # (you can extend later if you truly need it)
    return table.get(vals, table[tuple([W] * len(depends_on))])

def cap_floor_scalar(x: Optional[float], b: Bounds) -> Optional[float]:
    if x is None:
        return None
    v = float(x)
    if b.floor is not None and v < b.floor:
        v = b.floor
    if b.cap is not None and v > b.cap:
        v = b.cap
    return v

# helper: do it in one line
def cap_floor_scalar_by_rules(x, rules: Rules, feature: str, depends_on: Tuple[str, ...], *values):
    b = get_bounds(rules, feature, depends_on, *values)
    return cap_floor_scalar(x, b)

def cap_floor_df(
    df: pd.DataFrame,
    rules: Rules,
    feature: str,
    value_col: str,
    depends_on: Tuple[str, ...],          # e.g. ("currency","rating")
    dim_cols: Mapping[str, str],          # e.g. {"currency":"Currency", "rating":"Rating"}
    out_col: Optional[str] = None,
) -> pd.DataFrame:
    out_col = out_col or value_col
    df = df.copy()

    group_cols = [dim_cols[d] for d in depends_on]
    if not group_cols:  # global
        b = get_bounds(rules, feature, depends_on)
        df[out_col] = df[value_col].clip(lower=b.floor, upper=b.cap)
        return df

    def _apply_group(g: pd.DataFrame) -> pd.DataFrame:
        vals = [g[dim_cols[d]].iloc[0] for d in depends_on]
        b = get_bounds(rules, feature, depends_on, *vals)
        g[out_col] = g[value_col].clip(lower=b.floor, upper=b.cap)
        return g

    return df.groupby(group_cols, dropna=False, sort=False, group_keys=False).apply(_apply_group)
