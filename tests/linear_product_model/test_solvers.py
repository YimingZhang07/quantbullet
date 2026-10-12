"""BCDSolver: which forms it fits, the solve options it passes on, and what it leaves on the model."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantbullet.linear_product_model import LinearProductRegressorBCD, ModelForm
from quantbullet.linear_product_model.datacontainer import ProductModelDataContainer
from quantbullet.linear_product_model.forms import IdentityMSE
from quantbullet.linear_product_model.solvers import BCDSolver


def make_data(n=2000):
    rng = np.random.default_rng(5)
    age, purpose = rng.uniform(1., 120., n), rng.integers(0, 3, n)
    matrices = {"purpose": np.eye(3)[purpose],
                "age": np.column_stack((np.ones(n), np.minimum(age, 36.) / 36., np.maximum(age - 36., 0.) / 84.))}
    groups = {name: [f"{name}_{i}" for i in range(X.shape[1])] for name, X in matrices.items()}
    expanded = pd.DataFrame(np.column_stack(list(matrices.values())), columns=sum(groups.values(), []))
    y = 1. + .5 * np.minimum(age, 36.) / 36. + .2 * purpose + rng.normal(0., .1, n)
    container = ProductModelDataContainer(pd.DataFrame({"age": age, "purpose": purpose}), expanded,
                                          response=y, feature_groups=groups)
    return container, groups


def test_bcd_needs_an_identity_form_with_block_step():
    class LogMSE(IdentityMSE):
        link = "log"

    class NoStep(ModelForm):
        link, loss_name, loss_label = "identity", "custom", "Custom"

        def factor(self, X, coef):
            return X @ coef

        def loss(self, y, y_hat, weights=None):
            return 0.

        def implied_actual_terms(self, y, m, p, w):
            return y, m

    for form in (LogMSE(), NoStep()):
        with pytest.raises(ValueError, match=f"BCD needs an identity form with block_step; got {form.name}"):
            BCDSolver(SimpleNamespace(form_=form))


@pytest.mark.parametrize("option", ["cache_qr_decomp", "use_svd"])
def test_weighted_qr_and_svd_are_rejected_for_mse_and_ignored_for_poisson(option):
    data, groups = make_data()
    weights = np.ones(len(data))
    with pytest.raises(NotImplementedError, match="Weighted least squares"):
        LinearProductRegressorBCD().fit(data, groups, loss="mse", weights=weights, verbose=0, **{option: True})
    options = dict(loss="poisson", weights=weights, n_iterations=3, verbose=0)
    with_option = LinearProductRegressorBCD().fit(data, groups, **options, **{option: True})
    without = LinearProductRegressorBCD().fit(data, groups, **options)
    np.testing.assert_array_equal(with_option.predict(data), without.predict(data))


@pytest.mark.parametrize("option, expected", [("cache_qr_decomp", ["qr"]), ("use_svd", ["use_svd"])])
def test_mse_solve_options_reach_the_form(option, expected):
    data, groups = make_data()
    seen = []

    class Recording(BCDSolver):
        def _step_options(self, group, X_basis):
            options = super()._step_options(group, X_basis)
            seen.append(sorted(options))
            return options

    class Model(LinearProductRegressorBCD):
        bcd_solver = Recording

    Model().fit(data, groups, loss="mse", n_iterations=2, verbose=0, **{option: True})
    assert seen and all(options == expected for options in seen)


def test_fit_leaves_no_solver_or_per_row_arrays_on_the_model():
    data, groups = make_data()
    model = LinearProductRegressorBCD().fit(data, groups, interactions={"age": "purpose"}, loss="poisson",
                                            n_iterations=3, verbose=0)
    assert not any(isinstance(value, BCDSolver) for value in vars(model).values())
    assert not any(isinstance(value, np.ndarray) and value.shape == (len(data),) for value in vars(model).values())
