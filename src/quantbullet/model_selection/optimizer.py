import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

class BaseSearch:
    def __init__(self, estimator, param_grid, scoring='neg_mean_squared_error'):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring

    def fit(self, X, y): raise NotImplementedError
    def summary(self): raise NotImplementedError
    def evaluate(self, X_test, y_test): raise NotImplementedError

class GridSearch(BaseSearch):
    def __init__(self, estimator, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1):
        """
        Parameters
        ----------
        estimator : object
            The estimator to be optimized.
        param_grid : dict
            Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
        scoring : str, callable
            A single string or a callable to evaluate the predictions on the test set.
            Please check https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names
        cv : int, cross-validation generator or an iterable, default=5
        """
        super().__init__(estimator, param_grid, scoring)
        self.cv = cv
        self.verbose = verbose

    def fit(self, X, y):
        self.search_ = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=self.cv,
            verbose=self.verbose,
            return_train_score=True
        )
        self.search_.fit(X, y)
        return self

    def summary(self):
        df = pd.DataFrame(self.search_.cv_results_)
        summary_df = pd.json_normalize(df['params']).assign(
            mean_test_score=df['mean_test_score'],
            std_test_score=df['std_test_score'],
            rank=df['rank_test_score']
        ).sort_values(by='mean_test_score', ascending=False)
        return summary_df

    def best_model(self):
        return self.search_.best_estimator_

    def evaluate(self, X, y):
        y_pred = self.search_.best_estimator_.predict(X)
        return {'mse': mean_squared_error(y, y_pred)}
