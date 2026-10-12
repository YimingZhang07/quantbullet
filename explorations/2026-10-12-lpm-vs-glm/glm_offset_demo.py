"""How statsmodels GLM offset works for a Binomial (logit) model.

Run from the repository root:
    .venv/Scripts/python.exe explorations/2026-10-12-lpm-vs-glm/glm_offset_demo.py
"""
import warnings

import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(0)
n = 50_000
x1 = rng.normal(size=n)
x2 = 0.5 * x1 + rng.normal(size=n)  # correlated with x1
y = rng.binomial(1, 1 / (1 + np.exp(-(-2.0 + 0.8 * x1 + 0.5 * x2))))
binomial = sm.families.Binomial()

# 1) Freeze the "old" model logit(p) = -2 + 0.8*x1 and fit only the new piece.
old_eta = -2.0 + 0.8 * x1
X_new = sm.add_constant(x2)
res = sm.GLM(y, X_new, family=binomial, offset=old_eta).fit()
print("offset fit           :", res.params.round(4))

# 2) The same fit by hand: IRLS with the offset in eta and out of the working response.
beta = np.zeros(2)
for _ in range(25):
    eta = X_new @ beta + old_eta
    mu = 1 / (1 + np.exp(-eta))
    w = mu * (1 - mu)
    z = eta + (y - mu) / w - old_eta
    beta = np.linalg.solve(X_new.T @ (w[:, None] * X_new), X_new.T @ (w * z))
print("manual IRLS          :", beta.round(4))

# 3) Joint fit for reference.
full = sm.GLM(y, sm.add_constant(np.column_stack([x1, x2])), family=binomial).fit()
print("joint fit            :", full.params.round(4))

# 4) A biased frozen part biases the new coefficient: the x1-only fit absorbs some of x2.
X1 = sm.add_constant(x1)
base = sm.GLM(y, X1, family=binomial).fit()
res2 = sm.GLM(y, X_new, family=binomial, offset=X1 @ base.params).fit()
print("base fit on x1 only  :", base.params.round(4))
print("offset on that base  :", res2.params.round(4))

# 5) predict() with new exog silently uses offset=0 unless the offset is passed.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    print("predict, no offset   :", res.predict(X_new[:3]).round(4), f"({len(caught)} warnings)")
print("predict, with offset :", res.predict(X_new[:3], offset=old_eta[:3]).round(4))
print("in-sample fitted     :", res.fittedvalues[:3].round(4))

# 6) exposure means log(exposure) in the offset, so it requires the log link.
try:
    sm.GLM(y, X_new, family=binomial, exposure=np.ones(n)).fit()
except ValueError as error:
    print("exposure with logit  :", error)
