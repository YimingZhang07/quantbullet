from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Mapping, Sequence, Any, Union
import pandas as pd

W = "*"

@dataclass(frozen=True)
class Bounds:
    floor: Optional[float] = None
    cap: Optional[float] = None

Rules = Dict[str, Dict[Tuple[str, ...], Dict[Tuple[str, ...], Bounds]]]


def _dep_key(dims: Mapping[str, Any] | None) -> Tuple[str, ...]:
    """Dependency key = sorted dimension names (stable)."""
    if not dims:
        return ()
    return tuple(sorted(dims.keys()))


def _val_tuple(depends_on: Tuple[str, ...], dims: Mapping[str, Any]) -> Tuple[str, ...]:
    """Value tuple in depends_on order; missing/None -> wildcard."""
    out = []
    for d in depends_on:
        v = dims.get(d, None)
        out.append(W if v is None else str(v))
    return tuple(out)


def get_bounds(rules: Rules, feature: str, dims: Mapping[str, Any] | None = None) -> Bounds:
    depends_on = _dep_key(dims)
    table = rules[feature][depends_on]

    if not depends_on:
        return table[()]

    vals = _val_tuple(depends_on, dims or {})

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

    # >2 dims: exact-or-global default
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


def cap_floor_scalar_by_rules(
    x: Optional[float],
    rules: Rules,
    feature: str,
    dims: Mapping[str, Any] | None = None,
) -> Optional[float]:
    return cap_floor_scalar(x, get_bounds(rules, feature, dims=dims))

def cap_floor_df(
    df: pd.DataFrame,
    rules: Rules,
    feature: str,
    value_col: str,
    *,
    dims: Tuple[str, ...] = (),               # which df columns drive the rule
    dim_cols: Mapping[str, str] | None = None, # map dim name -> df column (optional)
    out_col: Optional[str] = None,
) -> pd.DataFrame:
    out_col = out_col or value_col
    df = df.copy()
    dim_cols = dim_cols or {d: d for d in dims}

    if not dims:
        b = get_bounds(rules, feature, dims=None)
        df[out_col] = df[value_col].clip(lower=b.floor, upper=b.cap)
        return df

    group_cols = [dim_cols[d] for d in dims]

    def _apply_group(g: pd.DataFrame) -> pd.DataFrame:
        dim_values = {d: g[dim_cols[d]].iloc[0] for d in dims}
        b = get_bounds(rules, feature, dims=dim_values)
        g[out_col] = g[value_col].clip(lower=b.floor, upper=b.cap)
        return g

    return df.groupby(group_cols, dropna=False, sort=False, group_keys=False).apply(_apply_group)