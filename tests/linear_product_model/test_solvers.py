"""Solvers: which forms they fit, the options they take, what they leave on the model, and Newton's results."""
import pickle
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import PoissonRegressor

from quantbullet.linear_product_model import LinearProductRegressor, ModelForm, resolve_form
from quantbullet.linear_product_model.datacontainer import ProductModelDataContainer
from quantbullet.linear_product_model.forms import IdentityMSE
from quantbullet.linear_product_model.solvers import BCDSolver, NewtonSolver


class FrozenCurve:
    def __init__(self, coef):
        self.coef = np.asarray(coef, dtype=float)

    def predict(self, X):
        return X @ self.coef


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


def poisson_data(n=6000, seed=3, drop=()):
    """Counts with a log-linear age ramp, purpose and state effects; ``drop`` leaves groups out."""
    rng = np.random.default_rng(seed)
    age, purpose, state = rng.uniform(1., 120., n), rng.integers(0, 3, n), rng.integers(0, 4, n)
    ramp = np.column_stack((np.minimum(age, 36.) / 36., np.maximum(age - 36., 0.) / 84.))
    rate = .05 * np.exp(.6 * ramp[:, 0] - .3 * ramp[:, 1]) * np.array([.8, 1.2, 1.])[purpose] * (1. + .1 * state)
    matrices = {"purpose": np.eye(3)[purpose], "state": np.eye(4)[state], "age": np.column_stack((np.ones(n), ramp))}
    matrices = {name: X for name, X in matrices.items() if name not in drop}
    groups = {name: [f"{name}_{i}" for i in range(X.shape[1])] for name, X in matrices.items()}
    expanded = pd.DataFrame(np.column_stack(list(matrices.values())), columns=sum(groups.values(), []))
    orig = pd.DataFrame({"age": age, "purpose": pd.Categorical(np.array(["C", "N", "P"])[purpose]),
                         "state": pd.Categorical(state)})
    container = ProductModelDataContainer(orig, expanded, response=rng.poisson(rate).astype(float),
                                          feature_groups=groups)
    parts = SimpleNamespace(ramp=ramp, purpose=np.eye(3)[purpose], state=np.eye(4)[state])
    return container, groups, rng.uniform(.5, 2., n), parts


def sklearn_poisson(Z, y, w, offset=None):
    """Reference log-link Poisson predictions; an offset enters through the exposure trick."""
    exposure = np.ones(len(y)) if offset is None else np.exp(offset)
    model = PoissonRegressor(alpha=0., solver="newton-cholesky", tol=1e-12, max_iter=200)
    model.fit(Z, y / exposure, sample_weight=(np.ones(len(y)) if w is None else w) * exposure)
    return exposure * model.predict(Z)


def fit_log(data, groups, **options):
    return LinearProductRegressor().fit(data, groups, loss="poisson", block_link="log", verbose=0, **options)


# ----------------------------------------------------------------------
# BCD
# ----------------------------------------------------------------------

def test_bcd_needs_an_identity_form_with_block_step():
    class LogMSE(IdentityMSE):
        link = "log"

    class NoStep(ModelForm):
        link, loss_name, loss_label = "identity", "custom", "Custom"

        def factor(self, X, coef):
            return X @ coef

        def loss_terms(self, y, y_hat):
            return 0. * y

        def implied_actual_terms(self, y, m, p, w):
            return y, m

    for form in (LogMSE(), NoStep()):
        with pytest.raises(ValueError, match=f"BCD needs an identity form with block_step; got {form.name}"):
            BCDSolver(SimpleNamespace(form_=form))
    with pytest.raises(ValueError, match="Newton needs a form with newton_terms; got identity/poisson"):
        NewtonSolver(SimpleNamespace(form_=resolve_form("identity", "poisson")))


@pytest.mark.parametrize("option", ["cache_qr_decomp", "use_svd"])
def test_weighted_qr_and_svd_are_rejected_for_mse_and_ignored_for_poisson(option):
    data, groups = make_data()
    weights = np.ones(len(data))
    with pytest.raises(NotImplementedError, match="Weighted least squares"):
        LinearProductRegressor().fit(data, groups, loss="mse", weights=weights, verbose=0, **{option: True})
    options = dict(loss="poisson", weights=weights, n_iterations=3, verbose=0)
    with_option = LinearProductRegressor().fit(data, groups, **options, **{option: True})
    without = LinearProductRegressor().fit(data, groups, **options)
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

    class Model(LinearProductRegressor):
        bcd_solver = Recording

    Model().fit(data, groups, loss="mse", n_iterations=2, verbose=0, **{option: True})
    assert seen and all(options == expected for options in seen)


@pytest.mark.parametrize("block_link", ["identity", "log"])
def test_fit_leaves_no_solver_or_per_row_arrays_on_the_model(block_link):
    data, groups, _, _ = poisson_data()
    model = LinearProductRegressor().fit(data, groups, interactions={"age": "purpose"}, loss="poisson",
                                            block_link=block_link, verbose=0)
    assert model.solver_name_ == model.fit_metadata_.solver == ("bcd" if block_link == "identity" else "newton")
    assert not any(isinstance(value, (BCDSolver, NewtonSolver)) for value in vars(model).values())
    assert not any(isinstance(value, np.ndarray) and value.shape == (len(data),) for value in vars(model).values())


# ----------------------------------------------------------------------
# Newton
# ----------------------------------------------------------------------

@pytest.mark.parametrize("interaction", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_newton_matches_sklearn_and_normalizes_blocks(interaction, weighted):
    data, groups, w, parts = poisson_data()
    w = w if weighted else None
    model = fit_log(data, groups, weights=w, interactions={"age": "purpose"} if interaction else None)
    ramp = ((parts.ramp[:, :, None] * parts.purpose[:, None, :]).reshape(len(data), -1) if interaction
            else parts.ramp)
    reference = sklearn_poisson(np.column_stack([ramp, parts.purpose[:, 1:], parts.state[:, 1:]]),
                                data.response, w)
    np.testing.assert_allclose(model.predict(data), reference, rtol=1e-6)
    assert model.form_ is resolve_form("log", "poisson") and model.convergence_info_["reason"] == "converged"
    for group in groups:
        assert model.block_means_[group] == pytest.approx(1., abs=1e-10)
    if interaction:
        X_age = data.get_expanded_array_for_feature_group("age")
        for label, coef in model.coef_["age"].categories.items():
            rows = (data.orig["purpose"] == label).values
            curve = model.block_factor("age", X_age[rows], coef=coef)
            assert np.average(curve, weights=None if w is None else w[rows]) == pytest.approx(1., abs=1e-10)


def test_frozen_submodel_enters_as_an_offset():
    data, groups, w, parts = poisson_data()
    frozen = FrozenCurve([1., 1.1, 1.2, 1.3])
    model = fit_log(data, groups, weights=w, submodels={"state": frozen})
    offset = np.log(frozen.predict(parts.state))
    reference = sklearn_poisson(np.column_stack([parts.ramp, parts.purpose[:, 1:]]), data.response, w, offset)
    np.testing.assert_allclose(model.predict(data), reference, rtol=1e-6)
    assert np.isnan(model.coef_["state"]).all()
    with pytest.raises(ValueError, match="must predict finite positive values"):
        fit_log(data, groups, submodels={"state": FrozenCurve([1., 0., 1., 1.])})


def test_identity_and_log_forms_agree_when_every_block_is_one_hot():
    data, groups, w, _ = poisson_data(drop=("age",))
    log = fit_log(data, groups, weights=w)
    identity = LinearProductRegressor().fit(data, groups, loss="poisson", weights=w, n_iterations=300,
                                               ftol=None, early_stopping_rounds=None, verbose=0)
    assert identity.loss_history_[-1] == pytest.approx(log.loss_history_[-1], rel=1e-9)
    np.testing.assert_allclose(identity.predict(data), log.predict(data), rtol=1e-5)


def test_log_implied_actual_equals_each_level_factor_at_the_optimum():
    data, groups, w, _ = poisson_data()
    model = fit_log(data, groups, weights=w, interactions={"age": "purpose"})
    for group in ("purpose", "state"):
        p = model.single_feature_group_predict(group, data)
        m = model.leave_out_feature_group_predict(group, data)
        numer, denom = model.form_.implied_actual_terms(data.response, m, p, w)
        for level in np.unique(data.orig[group]):
            rows = (data.orig[group] == level).values
            assert numer[rows].sum() / denom[rows].sum() == pytest.approx(p[rows][0], rel=1e-6)


def test_design_cache_modes_agree():
    data, groups, w, _ = poisson_data()
    cached = fit_log(data, groups, weights=w, solver_options={"design_cache": True})
    streamed = fit_log(data, groups, weights=w, solver_options={"design_cache": False})
    automatic = fit_log(data, groups, weights=w, solver_options={"max_cache_gb": 1e-9})
    # Both stop within the objective tolerance; only the Hessian precision differs, so the paths do too.
    assert cached.convergence_info_["deviance"] == pytest.approx(streamed.convergence_info_["deviance"], rel=1e-10)
    np.testing.assert_allclose(cached.predict(data), streamed.predict(data), rtol=1e-6)
    assert cached.fit_timing_["design_cache"] == "float32"
    assert streamed.fit_timing_["design_cache"] == automatic.fit_timing_["design_cache"] == "stream"
    assert cached.convergence_info_["hessian"] in ("float32", "float32, then float64")
    assert streamed.convergence_info_["hessian"] == "float64"


def test_fit_arguments_are_checked_against_the_solver_before_the_model_changes():
    data, groups, w, _ = poisson_data()
    model = fit_log(data, groups, weights=w)
    before = model.predict(data)
    cases = [
        (dict(block_link="log", n_iterations=5), ValueError, "n_iterations do not apply to the newton solver"),
        (dict(block_link="log", interaction_submodels={"age": {}}), ValueError, "interaction_submodels do not apply"),
        (dict(block_link="identity", solver_options={"max_iter": 3}), ValueError,
         "solver_options do not apply to the bcd solver of form identity/poisson"),
        (dict(block_link="log", solver_options={"steps": 3}), ValueError, "Unknown solver_options \\['steps'\\]"),
        (dict(block_link="log", solver_options={"design_cache": "yes"}), ValueError, "design_cache must be"),
        (dict(block_link="log", loss="mse"), ValueError, "Unsupported model form log/mse"),
    ]
    for arguments, error, message in cases:
        with pytest.raises(error, match=message):
            model.fit(data, groups, **{"loss": "poisson", "verbose": 0, **arguments})
        np.testing.assert_array_equal(model.predict(data), before)
    with pytest.raises(NotImplementedError, match="by-block 'purpose' is a trainable full one-hot block"):
        fit_log(data, groups, interactions={"age": "purpose"}, submodels={"purpose": FrozenCurve(np.ones(3))})


def test_log_model_pickles():
    data, groups, w, _ = poisson_data()
    model = fit_log(data, groups, weights=w, interactions={"age": "purpose"})
    restored = pickle.loads(pickle.dumps(model))
    assert restored.form_ is resolve_form("log", "poisson") and restored.block_link_ == "log"
    assert restored.fit_metadata_.block_link == "log" and restored.fit_metadata_.solver == "newton"
    np.testing.assert_array_equal(restored.predict(data), model.predict(data))
    assert restored.form_.label == "log-linear factors (log-link GLM), Poisson loss"
    # Standard errors are not part of either form until they are built properly; fitted blocks
    # are already normalized, so there are no separate normalized coefficient views.
    assert not any(hasattr(model, name) for name in ("se_", "calculate_se", "calculate_feature_group_se", "summary",
                                                     "normalized_coef_", "bias_one_coef_"))


def test_former_class_name_still_unpickles():
    from quantbullet.linear_product_model import LinearProductRegressor, LinearProductRegressorBCD, regressor
    # Saved models name ``regressor.LinearProductRegressorBCD``; unpickling looks the name up on the module.
    assert LinearProductRegressorBCD is LinearProductRegressor is getattr(regressor, "LinearProductRegressorBCD")
    data, groups, _, _ = poisson_data()
    model = LinearProductRegressorBCD().fit(data, groups, loss="poisson", verbose=0)
    # Protocol 2 writes class references as plain "module\nname\n", so the old name can be substituted exactly.
    dumped = pickle.dumps(model, protocol=2)
    current = b"linear_product_model.regressor\nLinearProductRegressor\n"
    assert current in dumped
    old = dumped.replace(current, b"linear_product_model.regressor\nLinearProductRegressorBCD\n")
    np.testing.assert_array_equal(pickle.loads(old).predict(data), model.predict(data))


def test_levels_without_events_warn_unless_l2_is_set():
    data, groups, w, parts = poisson_data()
    data.response[parts.state[:, 3] == 1] = 0.
    with pytest.warns(UserWarning, match=r"Levels without events .*state\[3\]"):
        fit_log(data, groups, weights=w, solver_options={"max_iter": 5})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = fit_log(data, groups, weights=w, solver_options={"l2": 1e-8})
    assert np.isfinite(model.predict(data)).all()
