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