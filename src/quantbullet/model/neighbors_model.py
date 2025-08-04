from .core import FeatureSpec, Model
from .neighbors import FeatureScaledKNNRegressor


class KNNModel( Model ):
    def __init__( self, 
                 feature_spec: FeatureSpec,
                 n_neighbors: int = 5,
                 metrics: str = "euclidean",
                 weights: str = "uniform",
                 feature_weights: tuple | list | None = None ):
        super().__init__( feature_spec )
        self.knn_models_ = None
        self.n_neighbors = n_neighbors
        self.metrics = metrics
        self.weights = weights
        self.feature_weights = feature_weights
        # check the length of feature_weights matches the input features
        if len(feature_weights) != len(feature_spec.x):
            raise ValueError(
                f"Length of feature_weights ({len(feature_weights)}) "
                f"must match the number of input features ({len(feature_spec.x)})"
            )

    def fit( self, X, y=None ):
        self.X_train_ = X.copy()
        X_selected = X.loc[ :, self.feature_spec.x ]
        y_selected = X.loc[ :, self.feature_spec.y ]
        self.knn_model = FeatureScaledKNNRegressor(
            n_neighbors=self.n_neighbors,
            metrics=self.metrics,
            weights=self.weights,
            feature_weights=self.feature_weights
        )
        self.knn_model.fit(X_selected, y_selected)
        return self
            
    def predict(self, X):
        pass

    def get_neighbors( self, X ):
        X_selected = X.loc[ :, self.feature_spec.x ]
        res = self.knn_model.predict_with_neighbors( X_selected )
        neighbor_indices = res[ '_neighbor_index' ].tolist()
        distances = res[ '_distance' ].tolist()
        neighbors = self.X_train_.iloc[ neighbor_indices, : ].copy()
        neighbors['Distance'] = distances
        return neighbors
    
    def _get_clone_args(self):
        base_args = super()._get_clone_args()
        base_args.update({
            "n_neighbors": self.n_neighbors,
            "metrics": self.metrics,
            "weights": self.weights,
            "feature_weights": self.feature_weights
        })
        return base_args
    
    @classmethod
    def from_model_config(cls, model_config: dict):
        """Create a KNNModel instance from a model configuration dictionary. This requires every init parameter to be present in the model_config dictionary."""
        return cls(
            feature_spec    = model_config["feature_spec"],
            n_neighbors     = model_config["n_neighbors"],
            metrics         = model_config["metrics"],
            weights         = model_config["weights"],
            feature_weights = model_config["feature_weights"]
        )