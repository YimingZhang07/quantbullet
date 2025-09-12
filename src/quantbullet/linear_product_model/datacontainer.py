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
    def __init__( self, orig_df: pd.DataFrame, expanded_df: pd.DataFrame=None, response=None, feature_groups: dict=None ):
        self.orig_df        = orig_df
        self.expanded_df    = expanded_df
        self.response       = response
        self.feature_groups = feature_groups
        # all the keys in the feature_groups dict must be in orig_df
        if self.feature_groups is not None:
            if not self._cols_in_df(self.orig_df, list(self.feature_groups.keys())):
                raise ValueError("All the keys in the feature_groups dict must be in orig_df.")

            # all the values in the feature_groups dict must be in orig_df
            if not all(self._cols_in_df(self.expanded_df, cols) for cols in self.feature_groups.values()):
                raise ValueError("All the values in the feature_groups dict must be in expanded_df.")
        else:
            # if no feature groups is given, we assume that there is no groups
            # each col in the orig_df should be a standalone group
            self.feature_groups = { col: {} for col in orig_df.columns }

    def __getattr__( self, attr: str ):
        return getattr(self.expanded_df, attr)

    def __getitem__( self, key: str ):
        return self.expanded_df[key]

    def __len__( self ):
        return len(self.expanded_df)

    @property
    def orig( self ):
        return self.orig_df

    @property
    def expanded( self ):
        return self.expanded_df

    def _cols_in_df( self, df: pd.DataFrame, cols: list[str] ):
        return all(col in df.columns for col in cols)

    def get_expanded_terms_for_feature_group( self, feature_group_name: str ):
        """Get the expanded terms for a feature group name."""
        return self.expanded_df[ self.feature_groups[feature_group_name] ].values

    def get_expanded_terms_dict( self, feature_group_names: list[str] ):
        """Get the expanded terms for a list of feature group names."""
        return { name: self.get_expanded_terms_for_feature_group(name) for name in feature_group_names }

    def get_container_for_feature_group( self, feature_group_name: str ):
        """Get the container for a feature group name."""
        orig_subset = self.orig_df[ [ feature_group_name ] ]

        # expanded is only returned if they are not None
        expanded_subset = (
            self.expanded_df[ self.feature_groups[ feature_group_name ] ]
            if self.expanded_df is not None and self.feature_groups is not None
            else None
        )

        return ProductModelDataContainer(
            orig_df         = orig_subset,
            expanded_df     = expanded_subset,
            response        = self.response,
            feature_groups  = { feature_group_name: self.feature_groups[ feature_group_name ] } if self.feature_groups else None
        )

    def get_containers_dict( self, feature_group_names: list[str] ):
        """Get the container for a list of feature group names.
        
        Returns
        -------
        dict
            A dictionary of containers for the feature group names.
        """
        return { name: self.get_container_for_feature_group(name) for name in feature_group_names }

    def sample( self, frac: float ):
        """Return a new container sampled by fraction, preserving row alignment.
        
        Parameters
        ----------
        frac : float
            Fraction of rows to sample (0, 1].
        """
        if not 0 < frac <= 1:
            raise ValueError("frac must be in (0, 1].")

        # sample indices from the expanded dataframe to preserve alignment
        sampled_idx = self.expanded_df.sample(frac=frac, random_state=42).index

        orig_df_sampled = self.orig_df.loc[sampled_idx]
        expanded_df_sampled = self.expanded_df.loc[sampled_idx]
        if self.response is not None:
            try:
                response_sampled = self.response.loc[sampled_idx]
            except AttributeError:
                # response might be a numpy array
                response_sampled = self.response[sampled_idx]
        else:
            response_sampled = None

        return ProductModelDataContainer(
            orig_df=orig_df_sampled,
            expanded_df=expanded_df_sampled,
            response=response_sampled,
            feature_groups=self.feature_groups,
        )