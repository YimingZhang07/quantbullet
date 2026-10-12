"""LPM BCD vs sklearn Newton GLMs when continuous drivers are correlated (rho = 0, 0.6, 0.9).

Run from the repository root (argument = training rows):
    .venv/Scripts/python.exe explorations/2026-10-12-lpm-vs-glm/bench_correlation.py 1000000
"""
import sys, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from quantbullet.linear_product_model import LinearProductRegressorBCD
from quantbullet.linear_product_model.datacontainer import ProductModelDataContainer

warnings.filterwarnings("ignore")
n = int(sys.argv[1])
KNOTS = np.linspace(0, 1, 8)
curves = [lambda x: 0.4 + 1.6 / (1 + np.exp(-10 * (x - 0.5))),
          lambda x: 0.3 + 1.4 * np.minimum(np.clip(x, 0, 1) / 0.3, 1),
          lambda x: 1.5 - np.clip(x, 0, 1),
          lambda x: 0.8 + 0.4 * np.sin(6 * np.clip(x, 0, 1))]


def ramps(x):
    lo, hi = KNOTS[:-1], KNOTS[1:]
    return np.clip(x[:, None] - lo, 0, hi - lo) / (hi - lo)


def ll(ph, yy):
    ph = np.clip(ph, 1e-9, 1 - 1e-9)
    return -np.mean(yy * np.log(ph) + (1 - yy) * np.log(1 - ph))


for rho in (0.0, 0.6, 0.9):
    rng = np.random.default_rng(0)
    N = int(n * 1.25)
    z = rng.normal(size=N)
    xs = [norm.cdf(np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.normal(size=N)) * 1.2 - 0.1 for _ in range(4)]
    c10 = rng.integers(0, 10, N); c50 = rng.integers(0, 50, N)
    lvl10 = rng.uniform(0.6, 1.4, 10); lvl50 = rng.uniform(0.8, 1.2, 50)
    f = np.prod([c(x) for c, x in zip(curves, xs)], axis=0) * lvl10[c10] * lvl50[c50]
    y = rng.binomial(1, np.clip(0.01 * f, 1e-6, 0.5)).astype(float)
    tr, te = slice(0, n), slice(n, N)
    B = [ramps(x) for x in xs]
    H10, H50 = np.eye(10)[c10], np.eye(50)[c50]

    Xg = np.column_stack([np.ones(N)] + B + [H10[:, 1:], H50[:, 1:]])
    rows = []
    t = time.perf_counter()
    m = PoissonRegressor(alpha=0, solver="newton-cholesky", fit_intercept=False, max_iter=200).fit(Xg[tr], y[tr])
    rows.append(("sklearn Poisson newton-cholesky", time.perf_counter() - t, ll(np.exp(Xg[te] @ m.coef_), y[te]), f"iters={m.n_iter_}"))
    t = time.perf_counter()
    m = LogisticRegression(C=1e10, solver="newton-cholesky", fit_intercept=False, max_iter=200).fit(Xg[tr], y[tr])
    rows.append(("sklearn Logistic newton-cholesky", time.perf_counter() - t, ll(m.predict_proba(Xg[te])[:, 1], y[te]), f"iters={m.n_iter_[0]}"))
    del Xg

    mats = {f"x{i}": np.column_stack([np.ones(N), b]) for i, b in enumerate(B)}
    mats["c10"], mats["c50"] = H10, H50
    groups = {k: [f"{k}_{j}" for j in range(v.shape[1])] for k, v in mats.items()}

    def container(sl):
        exp = pd.DataFrame(np.column_stack([v[sl] for v in mats.values()]), columns=sum(groups.values(), []))
        return ProductModelDataContainer(pd.DataFrame({k: np.zeros(exp.shape[0]) for k in groups}), exp,
                                         response=y[sl], feature_groups=groups)

    Ctr, Cte = container(tr), container(te)
    for ftol in (1e-5, 1e-9):
        t = time.perf_counter()
        lpm = LinearProductRegressorBCD().fit(Ctr, groups, loss="poisson", verbose=0,
                                              n_iterations=300, ftol=ftol, early_stopping_rounds=None)
        secs = time.perf_counter() - t
        h = np.array(lpm.loss_history_)
        k = int(np.argmax(h <= h.min() * (1 + 1e-6))) + 1
        rows.append((f"LPM BCD poisson ftol={ftol:g}", secs, ll(lpm.predict(Cte), y[te]),
                     f"sweeps={len(h)} ({lpm.convergence_info_['reason']}), within 1e-6 of best at sweep {k}"))

    best = min(r[2] for r in rows)
    print(f"\nrho={rho}  n_train={n:,}  corr(x0,x1)={np.corrcoef(xs[0][tr], xs[1][tr])[0, 1]:.2f}")
    for name, secs, l, extra in rows:
        print(f"  {name:36s} {secs:7.2f}s  logloss={l:.6f} (+{(l - best) * 1e6:5.1f}e-6)  {extra}")
