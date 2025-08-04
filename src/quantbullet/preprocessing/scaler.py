from sklearn.preprocessing import StandardScaler
import numpy as np

def standard_scaler_from_feature_dict(feature_params: dict) -> StandardScaler:
    """
    Create a StandardScaler object using a dictionary keyed by features,
    each containing its own mean and scale.

    Parameters:
    -----------
    feature_params: dict
        A dictionary with feature names as keys and dictionaries with keys 'mean' and 'scale' as values.
        Example: {'feature1': {'mean': 10.0, 'scale': 2.0}, 'feature2': {'mean': 20.0, 'scale': 5.0}}

    Returns:
    --------
    scaler: StandardScaler
        A fitted StandardScaler instance with specified mean and scale.
    """
    feature_names = list(feature_params.keys())
    mean = np.array([feature_params[feat]['mean'] for feat in feature_names])
    scale = np.array([feature_params[feat]['scale'] for feat in feature_names])

    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.var_ = scale ** 2
    scaler.n_features_in_ = len(mean)

    return scaler