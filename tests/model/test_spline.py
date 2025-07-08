import unittest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression
from quantbullet.preprocessing import FlatRampTransformer

class TestLinearSpline(unittest.TestCase):
    def setUp(self):
        pass

    def test_linear_spline(self):
        x = np.linspace(-1, 11, 100).reshape(-1, 1)
        y = np.sin(x) + np.random.normal(scale=0.1, size=x.shape)

        # create a linear spline with knots at 0, 2, 4, 6, 8, 10
        # for the bias / intercept term, we set include_bias=False as we will fit a linear regression afterwards with an intercept term
        spline = SplineTransformer(degree=1, knots = np.array([0, 2, 4, 6, 8, 10]).reshape(-1, 1), include_bias=False, extrapolation='constant')
        spline.fit(x)
        x_basis = spline.transform(x)

        # use the transformed basis to fit a linear regression
        r = LinearRegression()
        r.fit(x_basis, y)

        # plt.plot( x, x_basis @ r.coef_.T + r.intercept_)
        # plt.scatter(x, y, s=10, alpha=0.5)


class TestFlatRampTransformer(unittest.TestCase):
    def setUp(self):
        pass

    def test_flat_ramp_transformer( self ):
        x = np.linspace(-1, 11, 100).reshape(-1, 1)
        y = np.sin(x) + np.random.normal(scale=0.1, size=x.shape)

        transformer = FlatRampTransformer(knots=[0, 2, 4, 6, 8, 10], include_bias=False)
        basis = transformer.transform(x.reshape(-1, 1))

        # use the transformed basis to fit a linear regression
        r = LinearRegression()
        r.fit(basis, y)

        # plt.plot(x, basis @ r.coef_.T + r.intercept_)
        # plt.scatter(x, y, s=10, alpha=0.5)

    def test_control_slope( self ):
        transformer = FlatRampTransformer(knots=[0, 2, 4, 6, 8, 10], include_bias=False)
        betas = np.array( [1, 2, 3, 4, 5] ).reshape(1, -1)
        # plt.plot( x, basis @ betas.T.ravel() )