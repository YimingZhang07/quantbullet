"""
Module for statistical jump models
"""
import stat
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter
from operator import itemgetter
from sklearn import preprocessing
from sklearn.metrics import balanced_accuracy_score
from ..log_config import setup_logger

logger = setup_logger(__name__)


class DiscreteJumpModel:
    """
    Statistical Jump Model with Discrete States
    """

    def __init__(self) -> None:
        pass

    # Non-vectorized version
    # def fixed_states_optimize(self, y, s, theta_guess=None, k=2):
    #     """
    #     Optimize the parameters of a discrete jump model with fixed states

    #     Args:
    #         y (np.ndarray): observed data (T x n_features)
    #         s (np.ndarray): state sequence (T x 1)
    #         theta_guess (np.ndarray): initial guess for theta (k x n_features)
    #         k (int): number of states

    #     Returns:
    #         theta (np.ndarray): optimized parameters
    #         prob.value (float): optimal value of the objective function
    #     """
    #     if not isinstance(y, np.ndarray) or not isinstance(s, np.ndarray):
    #         raise TypeError("y and s must be numpy arrays")
    #     T, n_features = y.shape
    #     # initialize variables from guess
    #     theta = cp.Variable((k, n_features))
    #     if theta_guess is not None:
    #         theta.value = theta_guess

    #     # solve optimization problem; essentially the k-means problem
    #     diff = [0.5 * cp.norm2(y[i, :] - theta[s[i], :]) ** 2 for i in range(T)]
    #     objective = cp.Minimize(cp.sum(diff))
    #     prob = cp.Problem(objective)
    #     prob.solve()
    #     return theta.value, prob.value

    def fixed_states_optimize(self, y, s, k=2):
        """
        Optimize the parameters of a discrete jump model with states fixed first.

        Args:
            y (np.ndarray): Observed data of shape (T x n_features).
            s (np.ndarray): State sequence of shape (T x 1).
            theta_guess (np.ndarray): Initial guess for theta of shape (k x n_features).
            k (int): Number of states.

        Returns:
            tuple:
                - np.ndarray: Optimized parameters of shape (k x n_features).
                - float: Optimal value of the objective function.
        """
        if not isinstance(y, np.ndarray) or not isinstance(s, np.ndarray):
            raise TypeError("y and s must be numpy arrays")

        T, n_features = y.shape
        theta = np.zeros((k, n_features))

        # Compute the optimal centroids (theta) for each state
        for state in range(k):
            assigned_data_points = y[s == state]
            if assigned_data_points.size > 0:
                theta[state] = assigned_data_points.mean(axis=0)

        # Compute the objective value
        objective_value = 0.5 * np.sum(
            [np.linalg.norm(y[i, :] - theta[s[i], :]) ** 2 for i in range(T)]
        )

        return theta, objective_value

    # Non-vectorized version
    # def generate_loss_matrix(self, y, theta, k=2):
    #     """
    #     Generate the loss matrix for a discrete jump model for fixed theta

    #     Args:
    #         y (np.ndarray): observed data (T x n_features)
    #         theta (np.ndarray): parameters (k x n_features)
    #         k (int): number of states

    #     Returns:
    #         loss (np.ndarray): loss matrix (T x k)
    #     """
    #     T = y.shape[0]
    #     loss = np.zeros((T, k))
    #     for i in range(T):
    #         for j in range(k):
    #             # norm is the L2 norm by default
    #             loss[i, j] = 0.5 * np.linalg.norm(y[i, :] - theta[j, :]) ** 2
    #     return loss

    def generate_loss_matrix(self, y, theta):
        """
        Generate the loss matrix for a discrete jump model for fixed theta

        Args:
            y (np.ndarray): observed data (T x n_features)
            theta (np.ndarray): parameters (k x n_features)
            k (int): number of states

        Returns:
            loss (np.ndarray): loss matrix (T x k)
        """

        # Expand dimensions to broadcast subtraction across y and theta
        diff = y[:, np.newaxis, :] - theta[np.newaxis, :, :]

        # Compute squared L2 norm along the last axis (n_features)
        loss = 0.5 * np.sum(diff**2, axis=-1)

        return loss

    # Non-vectorized version
    # def fixed_theta_optimize(self, lossMatrix, lambda_, k=2):
    #     """
    #     Optimize the state sequence of a discrete jump model with fixed parameters

    #     Args:
    #         lossMatrix (np.ndarray): loss matrix (T x k)
    #         lambda_ (float): regularization parameter
    #         k (int): number of states

    #     Returns:
    #         s (np.ndarray): optimal state sequence (T x 1)
    #         v (float): optimal value of the objective function
    #     """
    #     state_choices = list(range(k))
    #     T, n_states = lossMatrix.shape
    #     V = np.zeros((T, n_states))
    #     V[0, :] = lossMatrix[0, :]
    #     for t in range(1, T):
    #         for k in range(n_states):
    #             V[t, k] = lossMatrix[t, k] + min(
    #                 V[t - 1, :]
    #                 + lambda_ * np.abs(state_choices[k] - np.array(state_choices))
    #             )

    #     # backtrack to find optimal state sequence
    #     v = V[-1, :].min()
    #     s = np.zeros(T, dtype=int)
    #     s[-1] = state_choices[V[-1, :].argmin()]
    #     for t in range(T - 2, -1, -1):
    #         s[t] = state_choices[
    #             np.argmin(V[t, :] + lambda_ * np.abs(state_choices - s[t + 1]))
    #         ]
    #     return s, v

    def fixed_theta_optimize(self, lossMatrix, lambda_):
        """
        Optimize the state sequence of a discrete jump model with fixed parameters

        Args:
            lossMatrix (np.ndarray): loss matrix (T x k)
            lambda_ (float): regularization parameter

        Returns:
            s (np.ndarray): optimal state sequence (T,)
            v (float): optimal value of the objective function
        """
        T, n_states = lossMatrix.shape
        V = np.zeros((T, n_states))
        V[0, :] = lossMatrix[0, :]

        state_choices = np.arange(n_states)

        for t in range(1, T):
            # Using broadcasting to compute the regularization term for all states at once
            regularization = lambda_ * np.abs(
                state_choices[:, np.newaxis] - state_choices
            )
            V[t, :] = lossMatrix[t, :] + (V[t - 1, :] + regularization).min(axis=1)

        # backtrack to find optimal state sequence
        v = V[-1, :].min()
        s = np.zeros(T, dtype=int)
        s[-1] = state_choices[V[-1, :].argmin()]
        for t in range(T - 2, -1, -1):
            s[t] = state_choices[
                np.argmin(V[t, :] + lambda_ * np.abs(state_choices - s[t + 1]))
            ]
        return s, v

    def initialize_kmeans_plusplus(self, data, k):
        """
        Initialize the centroids using the k-means++ method.

        Args:
            data: ndarray of shape (n_samples, n_features)
            k: number of clusters

        Returns:
            centroids: ndarray of shape (k, n_features)
        """
        initial_idx = np.random.choice(data.shape[0], 1)
        centroids = [data[initial_idx]]

        for _ in range(k - 1):
            squared_distances = np.min(
                [np.sum((data - centroid) ** 2, axis=1) for centroid in centroids],
                axis=0,
            )
            prob = squared_distances / squared_distances.sum()
            next_centroid_idx = np.random.choice(data.shape[0], 1, p=prob)
            centroids.append(data[next_centroid_idx])

        return np.array(centroids)

    def classify_data_to_states(self, data, centroids):
        """
        Classify data points to the states based on the centroids.

        Args:
            data: ndarray of shape (n_samples, n_features)
            centroids: centroids or means of the states, ndarray of shape (k, n_features)

        Returns:
            state_assignments: ndarray of shape (n_samples,), indices of the states \
                to which each data point is assigned
        """
        distances = np.array(
            [np.sum((data - centroid) ** 2, axis=1) for centroid in centroids]
        )
        state_assignments = np.argmin(distances, axis=0)
        return state_assignments

    def infer_states_stats(self, ts_returns, states):
        """
        Compute the mean and standard deviation of returns for each state

        Args:
            ts_returns (np.ndarray): observed returns (T x 1)
            states (np.ndarray): state sequence (T x 1)

        Returns:
            state_features (dict): mean and standard deviation of returns for each state
        """
        if not isinstance(ts_returns, np.ndarray) or not isinstance(states, np.ndarray):
            ts_returns = np.array(ts_returns)
            states = np.array(states, dtype=int)

        stats = dict()
        for state in set(states):
            idx = np.where(states == state)[0]
            stats[state] = (np.mean(ts_returns[idx]), np.std(ts_returns[idx]))
        return stats

    def remapResults(self, optimized_s, optimized_theta, ts_returns):
        """
        Remap the results of the optimization.

        We would like the states to be in increasing order of the volatility of returns.
        This is because vol has smaller variance than returns, a warning is triggered if \
            the states identified by volatility and returns are different.
        """
        res_s = list()
        res_theta = list()
        n = len(optimized_s)
        for i in range(n):
            # idx = np.argsort(optimized_theta[i][:, 0])[::-1]
            # whichever has the lowest volatility is assigned to state 0
            states_features = self.infer_states_stats(ts_returns, optimized_s[i])
            idx_vol = np.argsort(
                [states_features[state][1] for state in states_features]
            )
            idx_ret = np.argsort(
                [states_features[state][0] for state in states_features]
            )[::-1]
            if not np.array_equal(idx_vol, idx_ret):
                logger.warning(
                    "States identified by volatility ranks and returns ranks are different!"
                )
            # remap the states
            idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(idx_vol)}

            # if only one state, no need to remap
            if len(idx_mapping) == 1:
                remapped_s = optimized_s[i]
                remapped_theta = optimized_theta[i]

            else:
                remapped_s = [idx_mapping[_] for _ in optimized_s[i]]
                remapped_theta = optimized_theta[i][idx_vol, :]

            # append the remapped results
            res_s.append(remapped_s)
            res_theta.append(remapped_theta)

        return res_s, res_theta

    def cleanResults(self, raw_result, ts_returns, rearrange=False):
        """
        Clean the results of the optimization.

        This extracts the best results from the ten trials based on the loss.
        """
        optimized_s = list(map(itemgetter(0), raw_result))
        optimized_loss = list(map(itemgetter(1), raw_result))
        optimized_theta = list(map(itemgetter(2), raw_result))
        if rearrange:
            optimized_s, optimized_theta = self.remapResults(
                optimized_s, optimized_theta, ts_returns
            )
        idx = int(np.array(optimized_loss).argmin())
        best_s = optimized_s[idx]
        best_loss = optimized_loss[idx]
        best_theta = optimized_theta[idx]

        res_dict = {
            "best_state_sequence": best_s,
            "best_loss": best_loss,
            "best_theta": best_theta,
            "all_state_sequence": optimized_s,
            "all_loss": optimized_loss,
            "all_theta": optimized_theta,
        }

        return res_dict

    def single_run(self, y, k, lambda_):
        """
        Run a single trial of the optimization. Each trial uses a different \
            initialization of the centroids.

        Args:
            y (np.ndarray): observed data (T x n_features)
            k (int): number of states
            lambda_ (float): regularization parameter

        Returns:
            cur_s (np.ndarray): optimal state sequence (T x 1)
            loss (float): optimal value of the objective function
            cur_theta (np.ndarray): optimal parameters (k x n_features)
        """
        # initialize centroids using k-means++
        centroids = self.initialize_kmeans_plusplus(y, k)
        cur_s = self.classify_data_to_states(y, centroids)
        hist_s = [cur_s]
        # the coordinate descent algorithm
        for i in range(30):
            cur_theta, _ = self.fixed_states_optimize(y, cur_s, k=k)
            lossMatrix = self.generate_loss_matrix(y, cur_theta)
            cur_s, loss = self.fixed_theta_optimize(lossMatrix, lambda_)
            if cur_s.tolist() == hist_s[-1].tolist():
                break
            else:
                hist_s.append(cur_s)

        logger.debug(f"Single run completes after {i} iterations with loss {loss}")
        return cur_s, loss, cur_theta

    def fit(self, y, k=2, lambda_=100, rearrange=False, n_trials=10):
        """
        fit discrete jump model

        NOTE:
            A multiprocessing implementation is used to speed up the optimization
            Ten trials with k means++ initialization are ran

        Args:
            y (np.ndarray): observed data (T x n_features)
            k (int): number of states
            lambda_ (float): regularization parameter
            rearrange (bool): whether to rearrange the states in increasing order of \
                volatility

        Returns:
            best_s (np.ndarray): optimal state sequence (T x 1)
            best_loss (float): optimal value of the objective function
            best_theta (np.ndarray): optimal parameters (k x n_features)
            optimized_s (list): state sequences from all trials (10 x T)
            optimized_loss (list): objective function values from all trials (10 x 1)
            optimized_theta (list): parameters from all trials (10 x k x n_features)
        """
        args = [(y, k, lambda_)] * n_trials
        # mp.cpu_count()
        pool = mp.Pool(n_trials)
        res = pool.starmap(self.single_run, args)
        pool.close()

        res = self.cleanResults(res, y[:, 0], rearrange)
        states_stats = self.infer_states_stats(y[:, 0], res["best_state_sequence"])
        logger.info(f"Mean and Volatility by inferred states:\n {states_stats}")
        return res

    def evaluate(self, true, pred, plot=False):
        """
        Evaluate the model using balanced accuracy score

        Args:
            true (np.ndarray): true state sequence (T x 1)
            pred (np.ndarray): predicted state sequence (T x 1)
            plot (bool): whether to plot the true and predicted state sequences

        Returns:
            res (dict): evaluation results
        """
        true_len = len(true)
        pred_len = len(pred)
        if plot:
            plt.plot(true, label="True")
            plt.plot(pred, label="Pred")
            plt.title("True and Predicted State Sequences")
            plt.xlabel("Time")
            plt.ylabel("State")
            plt.legend()
            plt.show()
        res = {"BAC": balanced_accuracy_score(true[true_len - pred_len :], pred)}
        return res


class ContinuousJumpModel(DiscreteJumpModel):
    """
    Continuous Jump Model with Soft State Assignments
    """

    def fixed_states_optimize(self, y, s, k=None):
        """
        Optimize theta given fixed states

        Args:
            y: (T, n_features) array of observations
            s: (T, k) array of state assignments

        Returns:
            theta: (k, n_features) array of optimal parameters

        Note:
            s is assumed to have each row sum to 1
        """
        _, n_features = y.shape
        _, k = s.shape
        theta = np.zeros((k, n_features))

        for state in range(k):
            weights = s[:, state]
            theta[state] = np.sum(y * weights[:, np.newaxis], axis=0) / np.sum(weights)

        return theta

    def generate_loss_matrix(self, y, theta):
        """Identical to the loss function in the discrete case"""
        diff = y[:, np.newaxis, :] - theta[np.newaxis, :, :]
        loss = 0.5 * np.sum(diff**2, axis=-1)
        return loss

    def generate_C(self, k, grid_size=0.05):
        """Uniformly sample of states distributed on a grid

        Args:
            k (int): number of states

        Returns:
            matrix (np.ndarray): K x N matrix of states
        """
        N = int(1 / grid_size) ** k
        matrix = np.random.rand(k, N)
        matrix /= matrix.sum(axis=0)
        return matrix

    def fixed_theta_optimize(self, lossMatrix, lambda_, C):
        """Optimize the state sequence of a continuous jump model with fixed parameters

        Args:
            lossMatrix (np.ndarray): loss matrix (T x K)
            C (np.ndarray): K x N matrix of states
            lambda_ (float): regularization parameter

        Returns:
            s_hat (np.ndarray): optimal state sequence with probability dist (T x K)
            v_hat (float): loss value
        """
        T, K = lossMatrix.shape
        _, N = C.shape

        L_tilde = lossMatrix @ C
        Lambda = np.array(
            [
                [lambda_ / 4 * np.linalg.norm(c_i - c_j, ord=1) ** 2 for c_j in C.T]
                for c_i in C.T
            ]
        )

        V_tilde = np.zeros((T, N))
        V_tilde[0, :] = L_tilde[0, :]

        for t in range(1, T):
            for i in range(N):
                V_tilde[t, i] = L_tilde[t, i] + np.min(V_tilde[t - 1, :] + Lambda[:, i])

        s_hat = np.zeros((T, K))
        i_hat = np.argmin(V_tilde[-1, :])
        v_hat = np.min(V_tilde[-1, :])
        s_hat[-1] = C[:, i_hat]

        for t in range(T - 2, -1, -1):
            i_hat = np.argmin(V_tilde[t, :] + Lambda[:, i_hat])
            s_hat[t] = C[:, i_hat]

        return s_hat, v_hat

    def fit(self, y, k=2, lambda_=100, rearrange=False, n_trials=10, max_iter=20):
        if rearrange:
            raise NotImplementedError(
                "The rearrange function has not been \
                                      implemented."
            )
        # lists to keep best loss and state sequence across trials
        best_trial_loss = list()
        best_trial_states = list()
        best_trial_theta = list()

        for _ in range(n_trials):
            centroids = self.initialize_kmeans_plusplus(y, k)
            cur_s = self.classify_data_to_states(y, centroids)
            second_col = 1 - cur_s
            cur_s = np.column_stack((cur_s, second_col))
            cur_loss = float("inf")
            best_states = cur_s
            best_loss = cur_loss
            best_theta = None
            no_improvement_counter = 0
            for _ in range(max_iter):
                cur_theta = self.fixed_states_optimize(y, cur_s, k)  # Assuming 2 states
                lossMatrix = self.generate_loss_matrix(y, cur_theta)
                C = self.generate_C(k)
                cur_s, cur_loss = self.fixed_theta_optimize(
                    lossMatrix, lambda_=lambda_, C=C
                )

                # Check if the current solution is better than the best known solution
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    best_states = cur_s
                    best_theta = cur_theta
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1

                # Check for convergence
                if no_improvement_counter >= 3:
                    best_trial_loss.append(best_loss)
                    best_trial_states.append(best_states)
                    best_trial_theta.append(best_theta)
                    break

        final_best_loss = min(best_trial_loss)
        final_best_states = best_trial_states[best_trial_loss.index(final_best_loss)]
        final_best_theta = best_trial_theta[best_trial_loss.index(final_best_loss)]
        return final_best_states, final_best_loss, final_best_theta
    
    def predict(self, y, theta, lambda_=100):
        """
        Predict the state probabilities for a given time series using the learned parameters.
        """
        lossMatrix = self.generate_loss_matrix(y, theta)
        C = self.generate_C(theta.shape[0])
        state_probs, _ = self.fixed_theta_optimize(lossMatrix, lambda_=lambda_, C=C)
        return state_probs


def arrange_state_prob_by_volatility(returns, state_probs, thetas, threshold=0.5):
    """
    Arrange the state probabilities by volatility. Only two states are considered.
    We hope the second column of state_probs means the probability of high volatility.

    Parameters
    ----------
    returns : np.ndarray
        The returns with shape (T,).
    state_probs : np.ndarray
        The state probabilities with shape (T, 2).
    thetas : np.ndarray
        The parameters with shape (2, n_features).
    threshold : float, optional
        The threshold for volatility. The default is 0.5.

    Returns
    -------
    remapped_states : np.ndarray
        The remapped state probabilities.
    remapped_thetas : np.ndarray
        The remapped parameters.
    """
    low_vol_subset = state_probs[:, 1] < threshold
    high_vol_subset = ~low_vol_subset

    low_vol_returns = returns[low_vol_subset]
    high_vol_returns = returns[high_vol_subset]

    flip = False
    if np.std(low_vol_returns) > np.std(high_vol_returns): flip = True

    if flip:
        remapped_states = 1 - state_probs
        remapped_thetas = thetas[::-1]
    else:
        remapped_states = state_probs
        remapped_thetas = thetas

    return remapped_states, remapped_thetas

class FeatureGenerator:
    """
    Enrich univaraite time series with features
    """

    def __init__(self) -> None:
        pass

    def enrich_features(self, time_series):
        """
        Enrich a single time series with features

        Args:
            time_series (np.ndarray): time series (T x 1)

        Returns:
            features (np.ndarray): features (T x n_features)
        """
        df = pd.DataFrame({"ts": time_series})

        # Features 1-3
        df["observation"] = df["ts"]
        df["abs_change"] = df["ts"].diff().abs()
        df["prev_abs_change"] = df["abs_change"].shift()

        # Features 4-9 for w=6 and w=14
        for w in [6, 14]:
            roll = df["ts"].rolling(window=w)
            df[f"centered_mean_{w}"] = roll.mean()
            df[f"centered_std_{w}"] = roll.std()

            half_w = w // 2
            df[f"left_mean_{w}"] = df["ts"].rolling(window=half_w).mean().shift(half_w)
            df[f"left_std_{w}"] = df["ts"].rolling(window=half_w).std().shift(half_w)

            df[f"right_mean_{w}"] = df["ts"].rolling(window=half_w).mean()
            df[f"right_std_{w}"] = df["ts"].rolling(window=half_w).std()

        # Drop the original time series column
        df = df.drop(columns=["ts"])
        # Drop the first w rows where features are NaN
        # df = df[~np.isnan(df).any(axis=1)]
        df = df.dropna(how="any")

        return df.values

    def standarize_features(self, X):
        """
        Standarize features using sklearn's StandardScaler
        """
        return preprocessing.StandardScaler().fit_transform(X)


class SimulationGenerator:
    """
    Generate simulated returns that follows a Hidden Markov process.
    """

    def __init__(self) -> None:
        pass

    def stationary_distribution(self, transition_matrix):
        """
        Computes the stationary distribution for a given Markov transition matrix.

        Args:
            transition_matrix (numpy array): The Markov transition matrix.

        Returns:
            numpy array: The stationary distribution.
        """
        size = len(transition_matrix)
        # Create a matrix subtracted from the identity matrix
        Q = np.eye(size) - transition_matrix.T
        # Append a ones row to handle the constraint sum(pi) = 1
        Q = np.vstack([Q, np.ones(size)])
        # Create the target matrix (last entry is 1 for the sum(pi) = 1 constraint)
        b = np.zeros(size + 1)
        b[-1] = 1
        # Solve the linear system
        pi = np.linalg.lstsq(Q, b, rcond=None)[0]
        return pi

    def simulate_markov(self, transition_matrix, initial_distribution, steps):
        """
        Simulates a Markov process.

        Args:
            transition_matrix (numpy array): The Markov transition matrix.
            initial_distribution (numpy array): The initial state distribution.
            steps (int): The number of steps to simulate.

        Returns:
            states (list): The states at each step.
        """
        state = np.random.choice(len(initial_distribution), p=initial_distribution)
        states = [state]

        for _ in range(steps):
            state = np.random.choice(
                len(transition_matrix[state]), p=transition_matrix[state]
            )
            states.append(state)

        return states

    def generate_conditional_data(self, states, parameters):
        """
        Generate data using normal distribution conditional on the states.

        Args:
            states (list): The list of states
            parameters (dict): Parameters for each state with means and \
                standard deviations

        Returns:
            data (list): Simulated data conditional on the states.
        """
        data = []
        for state in states:
            mu, sigma = parameters[state]
            value = np.random.normal(mu, sigma)
            data.append(value)
        return data

    def run(self, steps, transition_matrix, norm_params):
        """
        Run the simulation, return the simulated states and conditional data

        NOTE:
            States are forced to cover all states, if not, re-run the simulation

        Args:
            steps (int): number of steps to simulate
            transition_matrix (np.ndarray): transition matrix (k x k)
            norm_params (dict): parameters for the normal distribution for each state

        Returns:
            simulated_states (list): simulated states
            simulated_data (list): simulated data conditional on states
        """
        initial_distribution = self.stationary_distribution(transition_matrix)
        logger.info(
            f"Step 1: Initial (stationary) distribution: {initial_distribution}"
        )
        simulated_states = self.simulate_markov(
            transition_matrix, initial_distribution, steps
        )

        # sanity check for the simulated states
        count_states = Counter(simulated_states)
        if len(count_states) != len(transition_matrix):
            logger.info("The simulated states do not cover all states. Re-run.")
            return self.run(steps, transition_matrix, norm_params)
        logger.info(f"Step 2: Simulated states: {count_states}")
        simulated_data = self.generate_conditional_data(simulated_states, norm_params)
        logger.info("Step 3: Generate simulated return data conditional on states.")
        return simulated_states, simulated_data


class TestingUtils:
    """
    Parameters and plotting functions for testing
    """

    def __init__(self) -> None:
        pass

    def daily(self):
        """
        Parameters for simulated daily return data, sourced from the paper
        """
        transition_matrix = np.array(
            [
                [0.99788413, 0.00211587],  # From state 0 to states 0 and 1
                [0.01198743, 0.98801257],  # From state 1 to states 0 and 1
            ]
        )
        norm_parameters = {
            0: (0.000615, 0.007759155881924271),  # mu1, sigma1
            1: (-0.000785, 0.017396608864948364),  # mu2, sigma2
        }

        return transition_matrix, norm_parameters

    def plot_returns(self, returns, shade_list=None):
        """
        Plot both the cumulative returns and returns on separate subplots sharing the x-axis.

        Args:
            returns (np.ndarray): An array of returns.
        """
        if not isinstance(returns, np.ndarray):
            returns = np.array(returns)

        # Create a color palette
        palette = plt.get_cmap("Set1")

        # Compute cumulative returns
        cumulative_returns = np.cumprod(1 + returns) - 1

        # Create subplots sharing the x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        # Plot cumulative returns on the first subplot
        ax1.plot(
            cumulative_returns,
            label="Cumulative Returns",
            color=palette(1),
            linestyle="-",
        )
        ax1.set_ylabel("Cumulative Returns")
        ax1.set_title("Cumulative Returns")

        # Plot returns on the second subplot
        ax2.plot(returns, label="Returns", color=palette(2))
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Returns")
        ax2.set_title("Returns")

        if shade_list is not None:
            # Shade regions corresponding to clusters of 1s in the shade_list
            start_shade = len(returns) - len(shade_list)
            start_region = None
            for i, val in enumerate(shade_list):
                if val == 1 and start_region is None:
                    start_region = start_shade + i
                elif val == 0 and start_region is not None:
                    ax1.axvspan(start_region, start_shade + i, color="gray", alpha=0.3)
                    ax2.axvspan(start_region, start_shade + i, color="gray", alpha=0.3)
                    start_region = None
            # If the last cluster extends to the end of the shade_list
            if start_region is not None:
                ax1.axvspan(
                    start_region, start_shade + len(shade_list), color="gray", alpha=0.3
                )
                ax2.axvspan(
                    start_region, start_shade + len(shade_list), color="gray", alpha=0.3
                )

        fig.tight_layout()
        plt.show()

    def plot_state_probs(self, states, prices):
        """plot the state probabilities and stock prices on the same plot
        Args:
            states (np.ndarray): An n x k array of state probabilities.
            prices (pd.DataFrame): A series of prices, indexed by date.
        """
        if not isinstance(states, np.ndarray):
            states = np.array(states)

        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("The index of prices must be a DatetimeIndex.")

        fig, ax = plt.subplots()
        ax.plot(
            prices.index[len(prices) - len(states) :],
            states[:, 1],
            label="State Probability",
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("State Probability")

        ax2 = ax.twinx()
        ax2.plot(prices.index, prices.values, color="green", label="Stock Price")
        ax2.set_ylabel("Price")

        # Use AutoDateLocator and DateFormatter for x-axis labels
        locator = mdates.AutoDateLocator()
        formatter = mdates.DateFormatter("%Y-%b")
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()  # Rotate and align the tick labels

        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, loc="upper right")

    def plot_averages(self, data_dict):
        """
        Plot the average of numbers for each key in the dictionary \
            using a line plot with x-axis labels in the form of 10^x.

        Args:
            data_dict (dict): A dictionary where keys are labels \
                    and values are lists of numbers.
        """
        # Compute averages
        labels = sorted(
            data_dict.keys(), key=float
        )  # Sort the keys by their float value
        averages = [
            sum(values) / len(values) for key in labels for values in [data_dict[key]]
        ]

        # Plot
        plt.plot(labels, averages, marker="o")
        plt.xlabel("Lambda")
        plt.ylabel("Balanced Accuracy")
        plt.title("Lambda vs. Average BAC")

        # Adjust x-axis labels to only show integer powers
        int_powers = [label for label in labels if np.log10(float(label)).is_integer()]
        plt.xticks(
            int_powers,
            [f"$10^{{{int(np.log10(float(label)))}}}$" for label in int_powers],
        )

        plt.grid(True, which="both", ls="--", c="0.8")
        plt.show()
