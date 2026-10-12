"""Fit time and holdout log loss: statsmodels / sklearn GLMs vs LPM BCD, plus an offset-only fit.

Run from the repository root (argument = training rows, default 1,000,000):
    .venv/Scripts/python.exe explorations/2026-10-12-lpm-vs-glm/bench_speed.py 1000000
"""
import sys, time, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from quantbullet.linear_product_model import LinearProductRegressorBCD
from quantbullet.linear_product_model.datacontainer import ProductModelDataContainer

warnings.filterwarnings("ignore")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
rng = np.random.default_rng(0)
N = int(n * 1.25)                      # 80/20 train/test

KNOTS = np.linspace(0, 1, 8)            # 7 ramps per continuous driver
def ramps(x):                           # flat-ramp basis, flat outside [0,1]
    lo, hi = KNOTS[:-1], KNOTS[1:]
    return np.clip(x[:, None] - lo, 0, hi - lo) / (hi - lo)

xs = [rng.uniform(-0.1, 1.1, N) for _ in range(4)]
c10 = rng.integers(0, 10, N); c50 = rng.integers(0, 50, N)

# true factor curves (multiplier scale), each roughly mean 1
curves = [lambda x: 0.4 + 1.6 / (1 + np.exp(-10 * (x - 0.5))),        # S-curve (incentive-like)
          lambda x: 0.3 + 1.4 * np.minimum(np.clip(x, 0, 1) / 0.3, 1),  # seasoning ramp
          lambda x: 1.5 - np.clip(x, 0, 1),                              # burnout-like decline
          lambda x: 0.8 + 0.4 * np.sin(6 * np.clip(x, 0, 1))]
lvl10 = rng.uniform(0.6, 1.4, 10); lvl50 = rng.uniform(0.8, 1.2, 50)
f = np.prod([c(x) for c, x in zip(curves, xs)], axis=0) * lvl10[c10] * lvl50[c50]  # product-form truth
p = np.clip(0.01 * f, 1e-6, 0.5)
y = rng.binomial(1, p).astype(float)
tr, te = slice(0, n), slice(n, N)

B = [ramps(x) for x in xs]
H10 = np.eye(10)[c10]; H50 = np.eye(50)[c50]

def ll(ph, yy):
    ph = np.clip(ph, 1e-9, 1 - 1e-9)
    return -np.mean(yy * np.log(ph) + (1 - yy) * np.log(1 - ph))

res = []
def record(name, secs, ph, extra=""):
    res.append((name, secs, ll(ph, y[te]), extra))

# ---------- GLM design: one intercept, ramps, one-hot with first level dropped
Xg = np.column_stack([np.ones(N)] + B + [H10[:, 1:], H50[:, 1:]])
print(f"n_train={n:,}  p_glm={Xg.shape[1]}  event_rate={y[tr].mean():.4f}", flush=True)

t = time.perf_counter()
r = sm.GLM(y[tr], Xg[tr], family=sm.families.Poisson()).fit()
record("statsmodels GLM Poisson/log (IRLS)", time.perf_counter() - t, np.exp(Xg[te] @ r.params), f"iters={r.fit_history['iteration']}")

t = time.perf_counter()
r = sm.GLM(y[tr], Xg[tr], family=sm.families.Binomial()).fit()
record("statsmodels GLM Binomial/logit (IRLS)", time.perf_counter() - t, 1 / (1 + np.exp(-Xg[te] @ r.params)), f"iters={r.fit_history['iteration']}")

t = time.perf_counter()
m = PoissonRegressor(alpha=0, solver="newton-cholesky", fit_intercept=False, max_iter=200).fit(Xg[tr], y[tr])
record("sklearn PoissonRegressor newton-cholesky", time.perf_counter() - t, np.exp(Xg[te] @ m.coef_), f"iters={m.n_iter_}")

t = time.perf_counter()
m = LogisticRegression(C=1e10, solver="newton-cholesky", fit_intercept=False, max_iter=200).fit(Xg[tr], y[tr])
record("sklearn LogisticRegression newton-cholesky", time.perf_counter() - t, m.predict_proba(Xg[te])[:, 1], f"iters={m.n_iter_[0]}")

t = time.perf_counter()
m = LogisticRegression(C=1e10, solver="lbfgs", fit_intercept=False, max_iter=2000).fit(Xg[tr], y[tr])
record("sklearn LogisticRegression lbfgs", time.perf_counter() - t, m.predict_proba(Xg[te])[:, 1], f"iters={m.n_iter_[0]}")

# ---------- LPM: product of linear blocks, each continuous block has its own constant
mats = {f"x{i}": np.column_stack([np.ones(N), b]) for i, b in enumerate(B)}
mats["c10"] = H10; mats["c50"] = H50
groups = {k: [f"{k}_{j}" for j in range(v.shape[1])] for k, v in mats.items()}
def container(sl):
    exp = pd.DataFrame(np.column_stack([v[sl] for v in mats.values()]), columns=sum(groups.values(), []))
    orig = pd.DataFrame({k: np.zeros(exp.shape[0]) for k in groups})
    return ProductModelDataContainer(orig, exp, response=y[sl], feature_groups=groups)
Ctr, Cte = container(tr), container(te)
for loss in ("poisson", "mse"):
    t = time.perf_counter()
    lpm = LinearProductRegressorBCD().fit(Ctr, groups, loss=loss, verbose=0, n_iterations=100, ftol=1e-7)
    secs = time.perf_counter() - t
    record(f"LPM BCD loss={loss}", secs, lpm.predict(Cte),
           f"sweeps={len(lpm.loss_history_)} stop={lpm.convergence_info_['reason']} setup={lpm.fit_timing_['setup_seconds']:.2f}s")

# ---------- offset fit: freeze everything but block x0 (7 ramps + intercept)
off = np.log(np.clip(lpm.leave_out_feature_group_predict("x0", Ctr), 1e-12, None))
X0 = np.column_stack([np.ones(n), B[0][tr]])
t = time.perf_counter()
r0 = sm.GLM(y[tr], X0, family=sm.families.Poisson(), offset=off).fit()
t_sm = time.perf_counter() - t
t = time.perf_counter()
e = np.exp(off)
m0 = PoissonRegressor(alpha=0, solver="newton-cholesky", fit_intercept=False).fit(X0, y[tr] / e, sample_weight=e)
t_sk = time.perf_counter() - t

print(f"\n{'method':45s} {'fit s':>8s} {'test logloss':>13s}  notes")
best = min(r[2] for r in res)
for name, secs, l, extra in res:
    print(f"{name:45s} {secs:8.2f} {l:13.6f}  (+{(l-best)*1e6:5.1f}e-6) {extra}")
print(f"\noffset fit, 8 params, n={n:,}: statsmodels {t_sm:.2f}s | sklearn Poisson weight trick {t_sk:.2f}s "
      f"| max |coef diff| = {np.max(np.abs(r0.params - m0.coef_)):.2e}")
