from sklearn.base import BaseEstimator, TransformerMixin
from dataclasses import dataclass
from typing import Callable, List, Union, Optional
import numpy as np
import pandas as pd


@dataclass
class FeatureGeneratingRule:
    """
    A rule for generating features from a given input.

    Attributes
    ----------
    name : str
        The name of the feature.
    func : Callable
        The function to apply to the input data.
    """
    name: str
    func: Callable
    input_columns: List[str]
    output_columns: List[str]

class FeatureGenerator(BaseEstimator, TransformerMixin):
    """
    A transformer that applies a set of feature generating rules to input data.

    Attributes
    ----------
    rules : List[FeatureGeneratingRule]
        A list of rules to apply for feature generation.
    """
    
    def __init__(self, rules: List[FeatureGeneratingRule]):
        self.rules = rules

    def fit(self, X, y=None):
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        X = X.copy()
        for rule in self.rules:
            input_data = X[rule.input_columns]
            derived_data = rule.func(input_data)

            if isinstance(derived_data, pd.Series):
                if len(rule.output_columns) != 1:
                    raise ValueError(f"Expected one output column for rule '{rule.name}', got {len(rule.output_columns)}.")
                X[ rule.output_columns[0] ] = derived_data
            elif isinstance(derived_data, pd.DataFrame):
                if len(rule.output_columns) != derived_data.shape[1]:
                    raise ValueError(f"Expected {derived_data.shape[1]} output columns for rule '{rule.name}', got {len(rule.output_columns)}.")
                colnames = rule.output_columns
                derived_data.columns = colnames
                X = pd.concat([X, derived_data], axis=1)
            else:
                raise ValueError(f"Unexpected derived data type: {type(derived_data)}")
        return X