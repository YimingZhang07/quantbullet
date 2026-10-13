"""Block forms of the linear product model.

A form is the pair (block link, loss). It owns every formula that differs between model variants --
how a block's coefficients become its multiplicative factor, the loss tracked during fitting, and the
FOC-consistent implied actual used by diagnostics -- and nothing about iteration. Models, solvers and
the toolkit ask for a form instead of branching on the loss name.

Only the combinations registered in ``resolve_form`` exist; any other one is an error. Each form names
the solver that fits it (``default_solver``), and solver capabilities are declared by the forms that
have them: ``block_step`` (with ``grouped_one_hot`` and ``step_options``) is what ``solvers.BCDSolver``
needs; ``newton_terms`` and ``rescale`` are what ``solvers.NewtonSolver`` needs.
"""
from abc import ABC, abstractmethod

import numpy as np
from scipy.special import xlogy

from ._acceleration import ols_normal_equation, ols_normal_equation_scaled


class ModelForm(ABC):
    """Block link and loss of a linear product model. Forms are stateless."""

    link: str
    loss_name: str
    loss_label: str
    default_solver: str

    @property
    def name(self) -> str:
        return f"{self.link}/{self.loss_name}"

    @abstractmethod
    def factor(self, X, coef):
        """Multiplicative factor of a block with design ``X`` and coefficient vector ``coef``."""

    @abstractmethod
    def loss_terms(self, y, y_hat):
        """Per-row loss; ``loss`` is its (optionally weighted) mean."""

    def check_response(self, y):
        """Raise ``ValueError`` if the loss is undefined for this response; any ``y`` is accepted by default."""

    def loss(self, y, y_hat, weights=None):
        """Mean (optionally weighted) loss, the value recorded in ``loss_history_``."""
        per_obs = self.loss_terms(y, y_hat)
        if weights is not None:
            return np.average(per_obs, weights=weights)
        return np.mean(per_obs)

    @abstractmethod
    def implied_actual_terms(self, y, m, p, w):
        """Per-row ``(numer, denom)``; a bin's implied actual is ``sum(numer) / sum(denom)``.

        ``m`` is the prediction without this block (global scalar included), ``p`` is this block's
        factor and ``w`` the sample weight. The ratio is the bin average of the block's IRLS working
        response under its working weights, so it follows from the link and the loss.
        """

    def __repr__(self):
        return f"{type(self).__name__}()"


# ----------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------

class _MSELoss:
    loss_name = "mse"
    loss_label = "MSE Loss"

    def loss_terms(self, y, y_hat):
        return (y - y_hat) ** 2


class _PoissonLoss:
    """Poisson deviance per row, with predictions floored at 1e-10 so it is defined for any factor."""

    loss_name = "poisson"
    loss_label = "Poisson deviance"

    def loss_terms(self, y, y_hat):
        y_safe = np.maximum(y, 1e-10)
        yhat_safe = np.maximum(y_hat, 1e-10)
        return 2 * (y_safe * np.log(y_safe / yhat_safe) - (y - y_hat))

    def check_response(self, y):
        """Poisson deviance needs ``y >= 0``: counts, 0/1 events, fractions or amounts, but not negatives or NaN."""
        y = np.asarray(y, dtype=float)
        if not np.all(y >= 0):
            raise ValueError(f"Poisson loss needs a response y >= 0; found {int(np.sum(y < 0))} negative and "
                             f"{int(np.sum(np.isnan(y)))} NaN values")


# ----------------------------------------------------------------------
# Identity link: blocks fitted one at a time by BCD
# ----------------------------------------------------------------------

class IdentityForm(ModelForm):
    """Block factor ``X @ coef``: each block is linear on the multiplier scale.

    Identity forms are fitted block by block: ``block_step`` returns a block's new coefficients
    with the other blocks fixed, where the full prediction is ``scalar · fixed · (X @ coef)``.
    """

    link = "identity"
    default_solver = "bcd"
    grouped_one_hot = False   # whether block_step takes the category codes of a full one-hot block
    step_options = ()         # optional block_step solve variants this form accepts

    def factor(self, X, coef):
        return np.dot(X, coef)

    @abstractmethod
    def block_step(self, X, y, scalar, fixed, w, coef, codes=None):
        """New coefficients of one block given the global ``scalar`` and the other blocks' product ``fixed``."""


class IdentityMSE(_MSELoss, IdentityForm):
    step_options = ("qr", "use_svd")

    def implied_actual_terms(self, y, m, p, w):
        """MSE normal equations: ``sum(w·m·y) / sum(w·m²)``."""
        return w * m * y, w * m ** 2

    def block_step(self, X, y, scalar, fixed, w, coef, codes=None, qr=None, use_svd=False):
        """Closed-form least squares for the block (``coef`` is not needed).

        ``qr`` is a cached ``np.linalg.qr(X)``: it solves ``X @ coef ≈ y / scalar / fixed`` without
        weights. ``use_svd`` solves the scaled problem with ``lstsq``. Neither supports weights.
        """
        if codes is not None:
            raise ValueError("IdentityMSE has no grouped one-hot step")
        if qr is not None:
            if w is not None:
                raise NotImplementedError("Weighted least squares with cached QR decomposition is not implemented yet.")
            Q, R = qr
            return np.linalg.solve(R, Q.T @ (y / scalar / fixed))
        if use_svd:
            if w is not None:
                raise NotImplementedError("Weighted least squares with SVD is not implemented yet.")
            return np.linalg.lstsq(scalar * X * fixed[:, None], y, rcond=None)[0]
        return ols_normal_equation_scaled(X, y, scalar * fixed, weights=w)


class IdentityPoisson(_PoissonLoss, IdentityForm):
    """Identity blocks are not bounded below; the loss floors predictions at 1e-10."""

    grouped_one_hot = True

    def implied_actual_terms(self, y, m, p, w):
        """Poisson deviance FOC for an identity block: ``sum(w·y/p) / sum(w·m/p)``."""
        p_safe = np.maximum(p, 1e-10)
        return w * y / p_safe, w * m / p_safe

    def block_step(self, X, y, scalar, fixed, w, coef, codes=None):
        """One identity-link IRLS step: WLS of ``y / m`` on ``X`` with weights ``w · m / p``.

        ``m = scalar · fixed`` and ``p = X @ coef``, both floored at 1e-10. With ``codes`` of a full
        one-hot block, the same normal equations are accumulated by category.
        """
        m = np.maximum(scalar * fixed, 1e-10)
        if codes is not None:
            p = np.maximum(coef[codes], 1e-10)
            w_over_p = (w if w is not None else 1.) / p
            numerator = np.bincount(codes, weights=w_over_p * np.asarray(y), minlength=len(coef))
            denominator = np.bincount(codes, weights=w_over_p * m, minlength=len(coef))
            return numerator / (denominator + 1e-8)
        p = np.maximum(X @ coef, 1e-10)
        w = w if w is not None else np.ones(len(y))
        return ols_normal_equation(X, y / m, weights=w * m / p)


# ----------------------------------------------------------------------
# Log link: all blocks fitted jointly by Newton
# ----------------------------------------------------------------------

class LogForm(ModelForm):
    """Block factor ``exp(X @ coef)``: blocks add on the log scale, so the model is a log-link GLM.

    Factors are always positive. A block is rescaled to mean 1 by shifting its coefficients along
    a constant direction ``c`` with ``X @ c = 1`` (a bias column, or all levels of a full one-hot block).
    """

    link = "log"
    default_solver = "newton"

    def factor(self, X, coef):
        return np.exp(np.dot(X, coef))

    def rescale(self, coef, mean, constant):
        """Coefficients whose factor is this one divided by ``mean``."""
        return coef - np.log(mean) * constant

    @abstractmethod
    def newton_terms(self, eta, y, w):
        """Per-row ``(mu, deviance, gradient weight, Hessian weight)`` at linear predictor ``eta``.

        For design ``Z`` the gradient of the log-likelihood is ``gradient_weight @ Z`` and its
        negative Hessian ``Z.T @ diag(hessian_weight) @ Z``; the deviance rows sum to the objective.
        """


class LogPoisson(_PoissonLoss, LogForm):
    """Weighted Poisson likelihood with a log link."""

    def implied_actual_terms(self, y, m, p, w):
        """Poisson deviance FOC for a log block: ``sum(w·y) / sum(w·m)``."""
        return w * y, w * m

    def newton_terms(self, eta, y, w):
        mu = np.exp(eta)
        return mu, w * 2 * (xlogy(y, y) - y * eta - y + mu), w * (y - mu), w * mu


_FORMS = {(form.link, form.loss_name): form for form in (IdentityMSE(), IdentityPoisson(), LogPoisson())}


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
