import pandas as pd
import matplotlib.pyplot as plt


class CrossValidationResult:
    def __init__(self, cv_results_: dict) -> None:
        if not isinstance(cv_results_, dict):
            raise TypeError("cv_results_ must be a dictionary.")
        self.cv_results_ = cv_results_
        self.n_params = len(self.cv_results_["params"][0])
        self.param_names = [k for k in self.cv_results_["params"][0].keys()]

    def cv_scores_df(self):
        """Return a dataframe with the cv parameters and scores."""
        df = pd.DataFrame(self.cv_results_["params"])
        df['mean_test_score'] = self.cv_results_["mean_test_score"]
        df['mean_train_score'] = self.cv_results_["mean_train_score"]
        df['std_test_score'] = self.cv_results_["std_test_score"]
        df['std_train_score'] = self.cv_results_["std_train_score"]
        return df
    
    def partial_cv_scores(self, param_names: str = None):
        param_names = param_names or self.param_names
        if not isinstance(param_names, list):
            raise TypeError("metric_names must be a list.")
        if not all([m in self.param_names for m in param_names]):
            raise ValueError("metric_names must be a subset of the cv_results_ keys.")
        cv_results_df = self.cv_scores_df()
        for param in param_names:
            if param not in cv_results_df:
                continue  # Skip if the parameter is not in the DataFrame

            # Group data and calculate mean test score
            grouped_data = cv_results_df.groupby(param)['mean_test_score'].mean().reset_index()
            grouped_data_std = cv_results_df.groupby(param)['std_test_score'].mean().reset_index()

            # Plotting using Matplotlib
            with plt.style.context("ggplot"):
                plt.figure(figsize=(6, 4))
                plt.plot(grouped_data[param], grouped_data['mean_test_score'], marker='o')
                plt.plot(grouped_data[param], 
                         grouped_data['mean_test_score'] + grouped_data_std['std_test_score'],
                         linestyle='--', 
                         color='grey')
                plt.plot(grouped_data[param], 
                         grouped_data['mean_test_score'] - grouped_data_std['std_test_score'], 
                         linestyle='--', 
                         color='grey')
                plt.title(f'Mean Test Score vs {param}')
                plt.ylabel('Mean Test Score')
                plt.xlabel(param)
                plt.grid(True)
                plt.show()

    def plot_single_metric_cv_scores(self, metric_name: str):
        """Plot the train and test scores for a given metric.
        
        The cv result should be one dimensional over this metric.
        """
        if not self.n_metrics == 1:
            raise ValueError("cv_results_ must be one dimensional over params.")
        metric_name = "param_" + metric_name
        if metric_name not in self.cv_results_:
            raise ValueError("metric_name must be a key in cv_results_.")
        metrics = self.cv_results_[metric_name].data.tolist()

        test_scores = self.cv_results_["mean_test_score"]
        test_scores_std = self.cv_results_["std_test_score"]
        train_scores = self.cv_results_["mean_train_score"]
        # train_scores_std = self.cv_results_["std_train_score"]
        with plt.style.context("ggplot"):
            plt.plot(metrics, test_scores, label="Test score")
            plt.plot(metrics, train_scores, label="Train score")
            plt.plot(
                metrics, test_scores + test_scores_std, linestyle="--", color="grey"
            )
            plt.plot(
                metrics, test_scores - test_scores_std, linestyle="--", color="grey"
            )
            plt.fill_between(
                metrics,
                test_scores + test_scores_std,
                test_scores - test_scores_std,
                alpha=0.2,
                color="grey",
            )

            plt.xlabel(metric_name)
            plt.ylabel("Score")
            plt.legend(loc="best")
            plt.title("Cross validation scores")
            plt.show()
