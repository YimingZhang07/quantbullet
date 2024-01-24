import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from typing import List
import sklearn

from ..utils.validation import validate_range_index

class OptunaStudyResult:
    """Class to wrap the results of an Optuna study."""
    def __init__(self, study) -> None:
        self.study = study

    def _ensure_param_names(self,
                        param_names: List[str],
                        prefix='params_') -> List[str]:
        """Ensure that the parameter names are prefixed with 'params_'."""
        return [prefix + p if not p.startswith(prefix) else p for p in param_names]
    
    def partial_cv_scores(self, param_names: list = None):
        """Plot the test scores partialled over parameters.
        
        Parameters
        ----------
        param_names : list, optional
            List of parameter names to partial over, by default None
        """
        param_names = param_names or list(self.study.best_params.keys())
        if not isinstance(param_names, list):
            raise TypeError("metric_names must be a list.")
        param_names = self._ensure_param_names(param_names)

        num_params = len(param_names)
        num_cols = 2
        num_rows = (num_params + num_cols - 1) // num_cols

        cv_results_df = self.study.trials_dataframe()

        with plt.style.context("ggplot"):
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
            axes = axes.flatten()
            for i, param in enumerate(param_names):
                if param not in cv_results_df:
                    continue  # Skip if the parameter is not in the DataFrame

                # Group data and calculate mean test score
                grouped_data = cv_results_df.groupby(param)['value'].mean().reset_index()

                # Plotting using Matplotlib
                ax = axes[i]
                ax.plot(grouped_data[param], grouped_data['value'], marker='o')
                ax.set_title(f'Mean Test Score vs {param}')
                ax.set_ylabel('Mean Test Score')
                ax.set_xlabel(param)
                ax.grid(True)

            plt.tight_layout()
            plt.show()

    def trials_dataframe(self):
        """Return a dataframe of Optuna trials sorted by value."""
        df = self.study.trials_dataframe()
        df = df.sort_values(by='value', ascending=False)
        return df
                
class OptunaCVOptimizer:
    """Class to wrap the optimization of a model using Optuna."""
    def __init__(self, X, y, model, cv, objective: str=None, scoring: str=None):
        self.X = X
        self.y = y
        self.model = model
        self.cv = cv
        self.objective = objective
        self.scoring = scoring

    @property
    def best_params(self):
        """Return the best parameters found by Optuna."""
        if not hasattr(self, 'study'):
            raise ValueError("Please optimize the model first.")
        return self.study.best_params
    
    @property
    def best_model(self):
        """Return the best model found by Optuna."""
        return self.model.set_params(**self.best_params)

    def setup_objective(self, fixed_params: dict = None, tuning_params: dict = None):
        """Setup the objective function for Optuna.

        Parameters
        ----------
        fixed_params : dict, optional
            Fixed parameters for the model, by default None
        tuning_params : dict, optional
            Tuning parameters for the model, by default None

        Returns
        -------
        callable
            Objective function for Optuna.
        """
        if fixed_params is None:
            params = {}
        else:
            params = fixed_params.copy()
        if tuning_params is None:
            tuning_params = {}
        else:
            tuning_params = tuning_params.copy()
        # Check types
        if not isinstance(params, dict):
            raise TypeError("fixed_params must be a dictionary.")
        if not isinstance(tuning_params, dict):
            raise TypeError("tuning_params must be a dictionary.")
        if len(tuning_params) > 0:
            for v in tuning_params.values():
                if not isinstance(v, list):
                    raise TypeError("tuning_params values must be lists.")
                
        def objective(trial):
            params.update({k: trial.suggest_categorical(k, v) for k, v in tuning_params.items()})
            self.model.set_params(objective=self.objective, **params)
            score = sklearn.model_selection.cross_val_score(self.model, self.X, self.y, cv = self.cv, scoring=self.scoring).mean()
            return score
        return objective
    
    def optimize(self, fixed_params: dict = None, tuning_params: dict = None, n_trials: int = 100):
        """Optimize the model using Optuna.
        
        Parameters
        ----------
        fixed_params : dict, optional
            Fixed parameters for the model, by default None
        tuning_params : dict, optional
            Tuning parameters for the model, by default None
        n_trials : int, optional
            Number of trials to run, by default 100
        
        Returns
        -------
        OptunaStudyResult
            Optuna study result wrapper.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        objective = self.setup_objective(fixed_params, tuning_params)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.study = study
        return OptunaStudyResult(study)

class CrossValidationResult:
    def __init__(self, cv_results_: dict) -> None:
        if not isinstance(cv_results_, dict):
            raise TypeError("cv_results_ must be a dictionary.")
        self.cv_results_ = cv_results_.copy()
        self.n_params = len(self.cv_results_["params"][0])
        self.param_names = [k for k in self.cv_results_["params"][0].keys()]

    def cv_scores_df(self) -> pd.DataFrame:
        """Return a dataframe with the cv parameters and scores."""
        df = pd.DataFrame(self.cv_results_["params"])
        df['mean_test_score'] = self.cv_results_["mean_test_score"]
        df['mean_train_score'] = self.cv_results_["mean_train_score"]
        df['std_test_score'] = self.cv_results_["std_test_score"]
        df['std_train_score'] = self.cv_results_["std_train_score"]
        return df
    
    def partial_cv_scores(self, param_names: str = None):
        """Plot the train and test scores partialled over parameters.
        
        Parameters
        ----------
        param_names : str, optional
            List of parameter names to partial over, by default None

        Returns
        -------
        None
        """
        param_names = param_names or self.param_names
        if not isinstance(param_names, list):
            raise TypeError("metric_names must be a list.")
        if not all([m in self.param_names for m in param_names]):
            raise ValueError("metric_names must be a subset of the cv_results_ keys.")
        cv_results_df = self.cv_scores_df()
        num_params = len(param_names)
        num_cols = 2
        num_rows = (num_params + num_cols - 1) // num_cols

        with plt.style.context("ggplot"):
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))

            for i, param in enumerate(param_names):
                if param not in cv_results_df:
                    continue  # Skip if the parameter is not in the DataFrame

                # Group data and calculate mean test score
                grouped_data = cv_results_df.groupby(param)['mean_test_score'].mean().reset_index()
                grouped_data_std = cv_results_df.groupby(param)['std_test_score'].mean().reset_index()

                # Plotting using Matplotlib
                ax = axes[i // num_cols, i % num_cols]
                ax.plot(grouped_data[param], grouped_data['mean_test_score'], marker='o')
                ax.plot(grouped_data[param], grouped_data['mean_test_score'] + grouped_data_std['std_test_score'],
                        linestyle='--', color='grey')
                ax.plot(grouped_data[param], grouped_data['mean_test_score'] - grouped_data_std['std_test_score'],
                        linestyle='--', color='grey')
                ax.set_title(f'Mean Test Score vs {param}')
                ax.set_ylabel('Mean Test Score')
                ax.set_xlabel(param)
                ax.grid(True)

            plt.tight_layout()
            plt.show()

    def plot_single_param_cv_scores(self, param_name: str):
        """Plot the train and test scores for a given metric.
        
        The cv result should be one dimensional over this metric.

        Parameters
        ----------
        param_name : str
            Name of the metric to plot.
        """
        if not self.n_metrics == 1:
            raise ValueError("cv_results_ must be one dimensional over params.")
        param_name = "param_" + param_name
        if param_name not in self.cv_results_:
            raise ValueError("param_name must be a key in cv_results_.")
        param_grid = self.cv_results_[param_name].data.tolist()

        test_scores = self.cv_results_["mean_test_score"]
        test_scores_std = self.cv_results_["std_test_score"]
        train_scores = self.cv_results_["mean_train_score"]
        # train_scores_std = self.cv_results_["std_train_score"]
        with plt.style.context("ggplot"):
            plt.plot(param_grid, test_scores, label="Test score")
            plt.plot(param_grid, train_scores, label="Train score")
            plt.plot(
                param_grid, test_scores + test_scores_std, linestyle="--", color="grey"
            )
            plt.plot(
                param_grid, test_scores - test_scores_std, linestyle="--", color="grey"
            )
            plt.fill_between(
                param_grid,
                test_scores + test_scores_std,
                test_scores - test_scores_std,
                alpha=0.2,
                color="grey",
            )

            plt.xlabel(param_name)
            plt.ylabel("Score")
            plt.legend(loc="best")
            plt.title("Cross validation scores")
            plt.show()

def mean_absolute_error_ignore_nan(y_true, y_pred):
    """Calculate the Mean Absolute Error between two arrays, ignoring any NaN values."""
    # Calculate absolute differences
    abs_diff = np.abs(y_true - y_pred)

    # Calculate mean while ignoring NaN values
    mae = np.nanmean(abs_diff)
    return mae

def time_series_cv_predict(model, X, y, cv):
    """Perform time series cross validation and return the predictions.

    NOTE : currently, X and y must be pandas objects for easy indexing.
    
    Parameters
    ----------
    model : sklearn estimator
        Model to fit.
    X : array-like
        Features.
    y : array-like
        Target.
    cv : list of (train, test) tuples
        Cross validation splits.
    scoring : str
        Scoring function.

    Returns
    -------
    np.array
        Predictions.
    """
    if isinstance(X, pd.DataFrame):
        validate_range_index(X)
    if isinstance(y, pd.Series):
        validate_range_index(y)
    if isinstance(X, np.ndarray) or isinstance(y, np.ndarray):
        raise TypeError("X and y must be pandas objects.")
    y_pred = np.full(len(y), np.nan)
    for train_idx, test_idx in cv:
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred[test_idx] = model.predict(X.iloc[test_idx])
    return y_pred
