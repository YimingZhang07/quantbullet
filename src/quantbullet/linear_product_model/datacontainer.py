"""
Motivation:

The product model takes not only piecewise linear components, but also other smoother components.
A single dataframe expanded by different transformers may not be the best way to store the data.
Therefore, we need a data container that stores the original data and the expanded data at the same time.
This deals with the problem at run time, different submodels may need different features.
"""

import pandas as pd

class ProductModelDataContainer:
    """
    Guidelines:
    To adapt the existing framework, we need this container to by default operates the same way as the expanded dataframe.
    """
    def __init__( self, orig_df: pd.DataFrame, expanded_df: pd.DataFrame, feature_groups: dict=None ):
        self.orig_df = orig_df
        self.expanded_df = expanded_df
        self.feature_groups = feature_groups

    def __getattr__( self, attr: str ):
        return getattr(self.expanded_df, attr)

    def __getitem__( self, key: str ):
        return self.expanded_df[key]

    def __len__( self ):
        return len(self.expanded_df)

    def orig( self ):
        return self.orig_df

    def expanded( self ):
        return self.expanded_df

    def get_terms_from_expanded( self, feature_group_name: str ):
        if self.feature_groups is not None:
            return self.expanded_df[ self.feature_groups[feature_group_name] ]