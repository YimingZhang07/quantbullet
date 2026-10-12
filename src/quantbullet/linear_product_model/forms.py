"""Block forms of the linear product model.

A form is the pair (block link, loss). It owns every formula that differs between model variants --
how a block's coefficients become its multiplicative factor, the loss tracked during fitting, and the
FOC-consistent implied actual used by diagnostics -- and nothing about iteration. Models, solvers and
the toolkit ask for a form instead of branching on the loss name.

Only the combinations registered in ``resolve_form`` exist; any other one is an error.
"""
from abc import ABC, abstractmethod

import numpy as np


class ModelForm(ABC):
    """Block link and loss of a linear product model. Forms are stateless."""

    link: str
    loss_name: str
    loss_label: str

    @property
    def name(self) -> str:
        return f"{self.link}/{self.loss_name}"

    @abstractmethod
    def factor(self, X, coef):
        """Multiplicative factor of a block with design ``X`` and coefficient vector ``coef``."""

    @abstractmethod
    def loss(self, y, y_hat, weights=None):
        """Mean (optionally weighted) loss, the value recorded in ``loss_history_``."""

    @abstractmethod
    def implied_actual_terms(self, y, m, p, w):
        """Per-row ``(numer, denom)``; a bin's implied actual is ``sum(numer) / sum(denom)``.

        ``m`` is the prediction without this block (global scalar included), ``p`` is this block's
        factor and ``w`` the sample weight. The ratio is the bin average of the block's IRLS working
        response under its working weights, so it follows from the link and the loss.
        """

    def __repr__(self):
        return f"{type(self).__name__}()"


class IdentityForm(ModelForm):
    """Block factor ``X @ coef``: each block is linear on the multiplier scale."""

    link = "identity"

    def factor(self, X, coef):
        return np.dot(X, coef)


class IdentityMSE(IdentityForm):
    loss_name = "mse"
    loss_label = "MSE Loss"

    def loss(self, y, y_hat, weights=None):
        per_obs = (y - y_hat) ** 2
        if weights is not None:
            return np.average(per_obs, weights=weights)
        return np.mean(per_obs)

    def implied_actual_terms(self, y, m, p, w):
        """MSE normal equations: ``sum(w·m·y) / sum(w·m²)``."""
        return w * m * y, w * m ** 2


class IdentityPoisson(IdentityForm):
    """Poisson deviance; factors are not bounded, so the loss floors predictions at 1e-10."""

    loss_name = "poisson"
    loss_label = "Poisson deviance"

    def loss(self, y, y_hat, weights=None):
        y_safe = np.maximum(y, 1e-10)
        yhat_safe = np.maximum(y_hat, 1e-10)
        per_obs = 2 * (y_safe * np.log(y_safe / yhat_safe) - (y - y_hat))
        if weights is not None:
            return np.average(per_obs, weights=weights)
        return np.mean(per_obs)

    def implied_actual_terms(self, y, m, p, w):
        """Poisson deviance FOC for an identity block: ``sum(w·y/p) / sum(w·m/p)``."""
        p_safe = np.maximum(p, 1e-10)
        return w * y / p_safe, w * m / p_safe


_FORMS = {(form.link, form.loss_name): form for form in (IdentityMSE(), IdentityPoisson())}


def resolve_form(link: str, loss: str) -> ModelForm:
    """The registered form for ``(link, loss)``."""
    try:
        return _FORMS[(link, loss)]
    except KeyError:
        supported = ", ".join(f"{link}/{loss}" for link, loss in _FORMS)
        raise ValueError(f"Unsupported model form {link}/{loss}; supported: {supported}") from None


def form_of(model) -> ModelForm:
    """``model.form_`` when present; otherwise the identity form of ``model.loss_`` (default ``'mse'``).

    The fallback covers objects that only carry ``loss_``, such as models outside the regressor family.
    """
    form = getattr(model, "form_", None)
    return form if form is not None else resolve_form("identity", getattr(model, "loss_", "mse"))
