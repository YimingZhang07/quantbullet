from quantbullet.research import DiscreteJumpModel
import numpy as np
import pytest

y = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
s = np.array([0, 1, 0, 1, 1])
theta_expected = np.array([[3, 4], [19/3, 22/3]])

def test_fixed_states_optimize_result():
    theta, _ = DiscreteJumpModel().fixed_states_optimize(y, s)
    # Check if the optimized theta is close to the expected result
    # This is a basic check and might need adjustments based on the function's behavior
    np.testing.assert_array_almost_equal(theta, theta_expected, decimal=1)