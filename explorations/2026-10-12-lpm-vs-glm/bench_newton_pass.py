"""Cost of one Newton pass per 500k rows on turnover-shaped arrays: dense vs one-hot-structured Hessian.

Turnover's log design has 156 columns: 82 dense (intercept, ramps, age x purpose) and 74 one-hot
(levels 3, 3, 5, 3, 12, 54 without references). Synthetic arrays understate per-call overhead, so the
eta/gradient choice was settled on 3M real turnover rows inside the solver (median seconds per Newton pass:
whole-chunk upcast for each product 4.59, 4096-row blocks 4.66, one upcast per chunk reused 3.71).
Run from the repository root:
    .venv/Scripts/python.exe explorations/2026-10-12-lpm-vs-glm/bench_newton_pass.py
"""
import time

import numpy as np
from scipy import sparse

rng = np.random.default_rng(0)
m, nd, levels = 500_000, 82, [3, 3, 5, 3, 12, 54]
codes = [rng.integers(0, L, m).astype(np.int32) for L in levels]
D = rng.normal(size=(m, nd)).astype(np.float32)
D[:, 0] = 1
onehot = np.hstack([np.eye(L, dtype=np.float32)[c][:, 1:] for L, c in zip(levels, codes)])
Z = np.hstack([D, ((onehot - onehot.mean(0)) / onehot.std(0)).astype(np.float32)])   # current: centered one-hot
beta = rng.normal(size=Z.shape[1]) * .01
y, w = (rng.random(m) < .01).astype(float), rng.uniform(.5, 2., m)
h, r = rng.uniform(.001, .01, m), rng.normal(size=m)


def seconds(f, repeat=3):
    f()
    started = time.perf_counter()
    for _ in range(repeat):
        f()
    return (time.perf_counter() - started) / repeat


def dense():
    Zs = Z * np.sqrt(h).astype(np.float32)[:, None]
    return Zs.T @ Zs, r @ Z, Z @ beta


def structured_gemm():
    """Dense block by GEMM, one-hot x dense by GEMM on a rebuilt 0/1 matrix, one-hot x one-hot by bincount."""
    Ds = D * np.sqrt(h).astype(np.float32)[:, None]
    C = np.hstack([np.eye(L, dtype=np.float32)[c][:, 1:] for L, c in zip(levels, codes)])
    blocks = [Ds.T @ Ds, C.T @ (D * h.astype(np.float32)[:, None])]
    for i in range(len(levels)):
        blocks.append(np.bincount(codes[i], h, levels[i]))
        for j in range(i + 1, len(levels)):
            blocks.append(np.bincount(codes[i] * levels[j] + codes[j], h, levels[i] * levels[j]))
    return blocks, r @ D, D @ beta[:nd]


def structured_sparse():
    """As above, with one-hot x dense from sparse category-by-row matrices."""
    Ds = D * np.sqrt(h).astype(np.float32)[:, None]
    blocks = [Ds.T @ Ds]
    for L, c in zip(levels, codes):
        blocks.append(sparse.csr_matrix((h, (c, np.arange(m))), shape=(L, m)) @ D)
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            blocks.append(np.bincount(codes[i] * levels[j] + codes[j], h, levels[i] * levels[j]))
    return blocks, r @ D, D @ beta[:nd]


def eta_gradient_upcast():
    eta = Z @ beta                       # NumPy upcasts the whole float32 chunk, once per product
    return eta, (w * (y - np.exp(eta))) @ Z


def eta_gradient_reused():
    """What NewtonSolver does: convert the chunk once and use it for both products."""
    Z64 = Z.astype(np.float64)
    eta = Z64 @ beta
    return eta, (w * (y - np.exp(eta))) @ Z64


def eta_gradient_blocked(block=4096):
    """Same float64 results, converting cache-sized row blocks once for both products."""
    eta, gradient = np.empty(m), np.zeros(Z.shape[1])
    for start in range(0, m, block):
        Zb = Z[start:start + block].astype(np.float64)
        eta[start:start + block] = e = Zb @ beta
        gradient += (w[start:start + block] * (y[start:start + block] - np.exp(e))) @ Zb
    return eta, gradient


for name, f in [("dense Hessian, 156 columns (solver)", dense),
                ("structured: one-hot cross terms by GEMM", structured_gemm),
                ("structured: one-hot cross terms by sparse", structured_sparse),
                ("eta + gradient, upcast per product", eta_gradient_upcast),
                ("eta + gradient, one upcast reused (solver)", eta_gradient_reused),
                ("eta + gradient, 4096-row blocks", eta_gradient_blocked)]:
    print(f"{name:46s} {seconds(f) * 1000:7.1f} ms per {m:,} rows")
print(f"float32 design cache for 17M rows: {17e6 * Z.shape[1] * 4 / 1e9:.1f} GB dense, "
      f"{17e6 * nd * 4 / 1e9:.1f} GB with one-hot kept as codes")
