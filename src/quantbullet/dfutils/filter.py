import pandas as pd
from functools import reduce
from pandas import DataFrame
from typing import List, Any

class Criteria:
    def __init__(self, column: str, operator: str, value):
        self.column = column
        self.operator = operator
        self.value = value

    def to_tuple(self):
        return (self.column, self.operator, self.value)

    def describe(self):
        return f"{self.column} {self.operator} {self.value}"

def apply_condition(df: DataFrame, col: str, op: str, val: Any):
    series = df[col]
    if op == "==":
        return series == val
    elif op == "!=":
        return series != val
    elif op == ">":
        return series > val
    elif op == "<":
        return series < val
    elif op == ">=":
        return series >= val
    elif op == "<=":
        return series <= val
    elif op == "in":
        return series.isin(val)
    elif op == "between":
        return (series >= val[0]) & (series <= val[1])
    elif op == "isnull":
        return series.isnull()
    elif op == "notnull":
        return series.notnull()
    elif op == "f":
        return val(series)
    else:
        raise ValueError(f"Unsupported operator: {op}")

def parse_filters(df: DataFrame, filters: List[Any]):
    masks = []

    for f in filters:
        if isinstance(f, list):
            and_masks = [apply_condition(df, *cond) for cond in f]
            mask = reduce(lambda a, b: a & b, and_masks)
        else:
            mask = apply_condition(df, *f)
        masks.append(mask)

    return reduce(lambda a, b: a & b, masks)

def filter_df(df: DataFrame, filters: List[Any], sort_by: List[str] = None, ascending: bool | List[bool] = True) -> DataFrame:
    mask = parse_filters(df, filters)
    result = df[mask]

    if sort_by:
        result = result.sort_values(by=sort_by, ascending=ascending)

    return result
