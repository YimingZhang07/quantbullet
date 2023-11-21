import matplotlib.pyplot as plt


class CrossValidationResult:
    def __init__(self, cv_results_: dict) -> None:
        if not isinstance(cv_results_, dict):
            raise TypeError("cv_results_ must be a dictionary.")
        self.cv_results_ = cv_results_

    def plot_train_test_scores(self, metric_name: str):
        """Plot the train and test scores for a given metric.
        
        The cv result should be one dimensional over this metric.
        """
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
