import sklearn

class ModelMetricsConsts:
    @classmethod
    @property
    def xgboost_objectives(cls):
        """XGBoost objective functions
        
        https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters

        Returns:
            list: List of XGBoost objective functions
        """
        return [
            "reg:squarederror",
            "reg:absoluteerror",
        ]

    def xgboost_objective_scorer_mapping(self, objective: str):
        """Mapping of XGBoost objective functions to scikit-learn scorers"""
        mapping = {
            "reg:squarederror": "neg_mean_squared_error",
            "reg:absoluteerror": "neg_mean_absolute_error",
        }
        return mapping.get(objective, None)
    
    def xgboost_scorer_objective_mapping(self, scorer: str):
        """Mapping of scikit-learn scorers to XGBoost objective functions"""
        mapping = {
            "neg_mean_squared_error": "reg:squarederror",
            "neg_mean_absolute_error": "reg:absoluteerror",
        }
        return mapping.get(scorer, None)

    def scikit_learn_scorer_names(self):
        """Scikit-learn scorer names
        
        scores higher are better
        """
        return sklearn.metrics.get_scorer_names()
    
class TypicalHyperparameterRanges:
    """Typical hyperparameter ranges for model selection"""

    @classmethod
    @property
    def xgboost(cls):
        return {
            "max_depth": [2, 3, 5, 7, 9],
            "learning_rate": [0.01, 0.1, 0.2, 0.5],
            "n_estimators": [10, 20, 30, 50, 80, 100],
            "subsample": [0.5, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0.001, 0.1, 1, 10, 100],
            "min_child_weight": [3, 5, 10],
        }
