"""Model forms: the registry, their formulas, and the single block-evaluation entry point."""
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantbullet.linear_product_model import (InteractionCoef, LinearProductModelToolkit, LinearProductRegressor,
                                              ModelForm, resolve_form)
from quantbullet.linear_product_model.datacontainer import ProductModelDataContainer
from quantbullet.linear_product_model.forms import IdentityMSE, IdentityPoisson, LogPoisson, form_of


class FrozenCurve:
    """A fixed block, as a pre-fitted submodel would be."""

    def __init__(self, coef):
        self.coef = np.asarray(coef, dtype=float)

    def predict(self, X):
        return X @ self.coef


def fitted(loss, submodel=False):
    rng = np.random.default_rng(3)
    n = 4000
    age = rng.uniform(1., 120., n)
    purpose = rng.integers(0, 3, n)
    state = rng.integers(0, 4, n)
    rate = .05 * (.6 + .8 * np.minimum(age, 36.) / 36.) * np.array([.8, 1.2, 1.])[purpose] * (1. + .1 * state)
    matrices = {"purpose": np.eye(3)[purpose], "state": np.eye(4)[state],
                "age": np.column_stack((np.ones(n), np.minimum(age, 36.) / 36., np.maximum(age - 36., 0.) / 84.))}
    groups = {name: [f"{name}_{i}" for i in range(X.shape[1])] for name, X in matrices.items()}
    expanded = pd.DataFrame(np.column_stack(list(matrices.values())), columns=sum(groups.values(), []))
    orig = pd.DataFrame({"age": age, "purpose": pd.Categorical(np.array(["C", "N", "P"])[purpose]),
                         "state": pd.Categorical(state)})
    container = ProductModelDataContainer(orig, expanded, response=rng.poisson(rate).astype(float),
                                          feature_groups=groups)
    submodels = {"state": FrozenCurve([1., 1.1, 1.2, 1.3])} if submodel else None
    model = LinearProductRegressor().fit(container, groups, interactions={"age": "purpose"}, submodels=submodels,
                                            loss=loss, n_iterations=5, verbose=0)
    return model, container


def test_registry_resolves_supported_forms_and_rejects_others():
    assert isinstance(resolve_form("identity", "mse"), IdentityMSE)
    assert isinstance(resolve_form("identity", "poisson"), IdentityPoisson)
    assert isinstance(resolve_form("log", "poisson"), LogPoisson)
    assert resolve_form("identity", "poisson").name == "identity/poisson"
    assert [resolve_form(*key).default_solver for key in (("identity", "mse"), ("log", "poisson"))] == ["bcd", "newton"]
    for link, loss in (("log", "mse"), ("identity", "huber")):
        with pytest.raises(ValueError, match=f"Unsupported model form {link}/{loss}; "
                                             "supported: identity/mse, identity/poisson, log/poisson"):
            resolve_form(link, loss)


def test_log_poisson_formulas():
    log, identity = resolve_form("log", "poisson"), resolve_form("identity", "poisson")
    y, w, m = np.array([0., 1., 3.]), np.array([1., 2., .5]), np.array([.2, .4, .3])
    eta = np.array([-1., 0., 1.])
    X, coef = np.array([[1., 2.], [1., 0.], [1., -1.]]), np.array([.1, .3])
    np.testing.assert_allclose(log.factor(X, coef), np.exp(X @ coef))
    np.testing.assert_allclose(log.factor(X, log.rescale(coef, 2., np.array([1., 0.]))), np.exp(X @ coef) / 2.)
    np.testing.assert_array_equal(log.loss_terms(y, np.exp(eta)), identity.loss_terms(y, np.exp(eta)))
    numer, denom = log.implied_actual_terms(y, m, np.exp(eta), w)
    np.testing.assert_allclose(numer, w * y)
    np.testing.assert_allclose(denom, w * m)
    mu, deviance, gradient, hessian = log.newton_terms(eta, y, w)
    np.testing.assert_allclose(mu, np.exp(eta))
    np.testing.assert_allclose(gradient, w * (y - mu))
    np.testing.assert_allclose(hessian, w * mu)
    expected = w * 2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1.) / mu), 0.) - (y - mu))
    np.testing.assert_allclose(deviance, expected)


def test_form_formulas():
    y, y_hat, w = np.array([0., 1., 2.]), np.array([.5, 1.5, 1.]), np.array([1., 2., 3.])
    m, p = np.array([.2, .4, .3]), np.array([1., 2., 0.])
    mse, poisson = resolve_form("identity", "mse"), resolve_form("identity", "poisson")
    assert mse.loss(y, y_hat, w) == pytest.approx(np.average((y - y_hat) ** 2, weights=w))
    y_safe = np.maximum(y, 1e-10)
    assert poisson.loss(y, y_hat) == pytest.approx(np.mean(2 * (y_safe * np.log(y_safe / y_hat) - (y - y_hat))))

    numer, denom = mse.implied_actual_terms(y, m, p, w)
    np.testing.assert_allclose(numer, w * m * y)
    np.testing.assert_allclose(denom, w * m ** 2)
    numer, denom = poisson.implied_actual_terms(y, m, p, w)
    assert np.isfinite(numer).all() and np.isfinite(denom).all()   # a zero factor is floored
    np.testing.assert_allclose(numer, w * y / np.maximum(p, 1e-10))
    np.testing.assert_allclose(denom, w * m / np.maximum(p, 1e-10))

    X, coef = np.array([[1., 2.], [3., 4.]], dtype=np.float32), np.array([.5, -1.])
    np.testing.assert_array_equal(poisson.factor(X, coef), np.dot(X, coef))


def test_mse_block_step_variants():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(50), rng.uniform(1., 2., 50)])
    fixed, y, w = rng.uniform(.5, 1.5, 50), rng.normal(2., .1, 50), np.ones(50)
    form = resolve_form("identity", "mse")
    scaled = form.block_step(X, y, 2., fixed, None, None)
    np.testing.assert_allclose(scaled, np.linalg.lstsq(2. * X * fixed[:, None], y, rcond=None)[0], rtol=1e-7)
    np.testing.assert_allclose(form.block_step(X, y, 2., fixed, None, None, use_svd=True), scaled, rtol=1e-7)
    qr = form.block_step(X, y, 2., fixed, None, None, qr=np.linalg.qr(X))
    np.testing.assert_allclose(qr, np.linalg.lstsq(X, y / 2. / fixed, rcond=None)[0], rtol=1e-10)
    for option in ({"qr": np.linalg.qr(X)}, {"use_svd": True}):
        with pytest.raises(NotImplementedError, match="Weighted least squares"):
            form.block_step(X, y, 2., fixed, w, None, **option)
    with pytest.raises(ValueError, match="no grouped one-hot step"):
        form.block_step(X, y, 2., fixed, None, None, codes=np.zeros(50, dtype=int))
    assert form.step_options == ("qr", "use_svd") and not form.grouped_one_hot
    assert resolve_form("identity", "poisson").grouped_one_hot


def test_poisson_forms_need_a_non_negative_response():
    for key in (("identity", "poisson"), ("log", "poisson")):
        form = resolve_form(*key)
        form.check_response(np.array([0., .25, 1., 3.]))          # zeros, fractions and counts
        with pytest.raises(ValueError, match="y >= 0; found 1 negative and 1 NaN values"):
            form.check_response(np.array([0., -.1, np.nan]))
    resolve_form("identity", "mse").check_response(np.array([-1., 2.]))

    model, container = fitted("poisson")
    before = model.predict(container)
    container.response[0] = -1.
    for block_link in ("identity", "log"):
        with pytest.raises(ValueError, match="Poisson loss needs a response y >= 0; found 1 negative"):
            model.fit(container, model.feature_groups_, interactions={"age": "purpose"}, loss="poisson",
                      block_link=block_link, verbose=0)
        np.testing.assert_array_equal(model.predict(container), before)


def test_form_of_falls_back_to_the_identity_form_of_loss():
    assert form_of(SimpleNamespace(loss_="poisson")) is resolve_form("identity", "poisson")
    assert form_of(SimpleNamespace()) is resolve_form("identity", "mse")
    assert isinstance(form_of(SimpleNamespace(form_=IdentityPoisson(), loss_="mse")), IdentityPoisson)


@pytest.mark.parametrize("loss", ["mse", "poisson"])
@pytest.mark.parametrize("submodel", [False, True])
def test_predictions_are_products_of_block_factors(loss, submodel):
    model, container = fitted(loss, submodel)
    assert model.form_ is resolve_form("identity", loss)
    factors = {}
    for group in model.feature_groups_:
        coef = model.coef_[group]
        cat_series = container.orig[coef.by] if isinstance(coef, InteractionCoef) else None
        factors[group] = model.block_factor(group, container.get_expanded_array_for_feature_group(group),
                                            cat_series=cat_series)
        np.testing.assert_array_equal(model.single_feature_group_predict(group, container), factors[group])
    expected = model.global_scalar_ * np.prod(list(factors.values()), axis=0)
    np.testing.assert_allclose(model.predict(container), expected, rtol=1e-12)
    if submodel:
        X_state = container.get_expanded_array_for_feature_group("state")
        np.testing.assert_array_equal(factors["state"], model.submodels_["state"].predict(X_state))

    restored = pickle.loads(pickle.dumps(model))
    assert restored.form_ is model.form_
    np.testing.assert_array_equal(restored.predict(container), model.predict(container))


def test_fit_rejects_unsupported_loss_and_keeps_the_fitted_model():
    model, container = fitted("poisson")
    before = model.predict(container)
    with pytest.raises(ValueError, match="Unsupported model form identity/huber"):
        model.fit(container, model.feature_groups_, interactions={"age": "purpose"}, loss="huber", verbose=0)
    assert model.loss_ == "poisson"
    np.testing.assert_array_equal(model.predict(container), before)


def test_toolkit_aggregation_takes_a_form_or_a_loss_name():
    from quantbullet.model.feature import DataType, Feature, FeatureRole, FeatureSpec
    toolkit = LinearProductModelToolkit(FeatureSpec([
        Feature("x", DataType.FLOAT, FeatureRole.MODEL_INPUT), Feature("y", DataType.FLOAT, FeatureRole.TARGET)]))
    raw = {"x": pd.DataFrame({"feature_value": [1, 1, 2], "y": [0., 1., 0.], "m": [.2, .4, .3],
                              "model_pred": [1., 2., 1.5], "w": [1., 3., 1.]})}
    by_name = toolkit._aggregate_implied_data(raw, {"x": "discrete"}, 100, loss="poisson")["x"]
    form = resolve_form("identity", "poisson")
    assert isinstance(form, ModelForm)
    by_form = toolkit._aggregate_implied_data(raw, {"x": "discrete"}, 100, loss=form)["x"]
    pd.testing.assert_frame_equal(by_form, by_name)
