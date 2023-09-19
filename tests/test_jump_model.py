from quantbullet.research import DiscreteJumpModel
import numpy as np


def test_fixed_states_optimize_result():
    """
    Test the result of the fixed_states_optimize function
    """
    y = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    s = np.array([0, 1, 0, 1, 1])
    theta_expected = np.array([[3, 4], [19 / 3, 22 / 3]])
    theta, _ = DiscreteJumpModel().fixed_states_optimize(y, s)
    # Check if the optimized theta is close to the expected result
    # This is a basic check and might need adjustments based on the function's behavior
    assert np.subtract(theta, theta_expected).max() < 1e-5


def test_generate_loss_matrix():
    """
    Test the generate_loss_matrix function
    """
    model = DiscreteJumpModel()
    # Test 1
    y = np.array([[1, 2], [3, 4], [5, 6]])
    theta = np.array([[1, 2], [5, 6]])
    expected_loss = np.array([[0.0, 16.0], [4.0, 4.0], [16.0, 0.0]])
    assert np.subtract(model.generate_loss_matrix(y, theta), expected_loss).max() < 1e-5


def test_fixed_theta_optimize():
    """
    Test the fixed_theta_optimize function
    """
    model = DiscreteJumpModel()
    # Test 1
    lossMatrix = np.array([[2, 8], [6, 4], [5, 1], [7, 3], [9, 0]])
    seq, loss = model.fixed_theta_optimize(lossMatrix, lambda_=1)
    np.testing.assert_array_equal(seq, np.array([0, 1, 1, 1, 1]))
    assert loss == 11.0

    # Test 2
    lossMatrix = np.array([[2, 8], [6, 4], [5, 1], [7, 3], [9, 0]])
    seq, loss = model.fixed_theta_optimize(lossMatrix, lambda_=10)
    np.testing.assert_array_equal(seq, np.array([1, 1, 1, 1, 1]))
    assert loss == 16.0

    # Test 3
    lossMatrix = np.array([[2, 8], [6, 5], [5, 8], [7, 6], [9, 5]])
    seq, loss = model.fixed_theta_optimize(lossMatrix, lambda_=2)
    np.testing.assert_array_equal(seq, np.array([0, 0, 0, 1, 1]))
    assert loss == 26.0
