import unittest

import numpy as np
from numba import njit
import numexpr as ne
import time

# ========================================
# 1. 原始 Python 循环乘法
# ========================================
def product_loop(dct):
    """
    Python 循环版本，逐个向量相乘
    """
    it = iter(dct.values())
    try:
        result = next(it).astype(np.float32, copy=True)
    except StopIteration:
        raise ValueError("dict 为空")

    for arr in it:
        result *= arr
    return result


# ========================================
# 2. NumPy in-place 相乘
# ========================================
def product_numpy(dct):
    """
    NumPy in-place 版本，使用 np.multiply(out=...)
    """
    it = iter(dct.values())
    try:
        result = next(it).astype(np.float32, copy=True)
    except StopIteration:
        raise ValueError("dict 为空")

    for arr in it:
        np.multiply(result, arr, out=result)
    return result


# ========================================
# 3. Numba JIT 加速
# ========================================
@njit(parallel=True)
def product_numba_inner(arrays_2d):
    n_blocks, n = arrays_2d.shape
    res = np.ones(n, dtype=np.float32)
    for j in range(n_blocks):
        res *= arrays_2d[j]
    return res

def product_numba(dct):
    arrays = np.stack(list(dct.values()), axis=0)  # (n_blocks, n_samples)
    return product_numba_inner(arrays)


# ========================================
# 4. NumExpr 多线程 SIMD
# ========================================
def product_numexpr(dct):
    """
    NumExpr 版本，多线程 + SIMD
    """
    keys = list(dct.keys())
    local_dict = {f"x{i}": dct[k] for i, k in enumerate(keys)}
    expr = "*".join([f"x{i}" for i in range(len(keys))])
    return ne.evaluate(expr, local_dict=local_dict)


# ========================================
# Benchmark 测试
# ========================================
class TestSpeed(unittest.TestCase):
    def setUp(self):
        n_obs = 1_400_000   # 样本数
        n_blocks = 18       # 向量数量
        np.random.seed(42)

        # 自动生成 n_blocks 个 key → 向量
        self.data = {
            f'block_{i}': np.random.rand(n_obs).astype(np.float32)
            for i in range(n_blocks)
        }

    def test_product_methods(self):
        # 预热 Numba，避免编译时间干扰
        product_numba(self.data)

        def bench(func, data, repeat=5):
            times = []
            result = None
            for _ in range(repeat):
                start = time.perf_counter()
                result = func(data)
                times.append(time.perf_counter() - start)
            return result, np.mean(times), np.std(times)

        # Python 循环版本
        res_loop, time_loop, std_loop = bench(product_loop, self.data)
        print(f"Python 循环版本耗时:   {time_loop:.6f} ± {std_loop:.6f} 秒")

        # NumPy in-place 版本
        res_numpy, time_numpy, std_numpy = bench(product_numpy, self.data)
        print(f"NumPy in-place 版本耗时: {time_numpy:.6f} ± {std_numpy:.6f} 秒")

        # Numba JIT 版本
        res_numba, time_numba, std_numba = bench(product_numba, self.data)
        print(f"Numba JIT 版本耗时:     {time_numba:.6f} ± {std_numba:.6f} 秒")

        # NumExpr 版本
        res_numexpr, time_numexpr, std_numexpr = bench(product_numexpr, self.data)
        print(f"NumExpr 版本耗时:       {time_numexpr:.6f} ± {std_numexpr:.6f} 秒")

        # 验证结果一致性
        np.testing.assert_allclose(res_loop, res_numpy, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(res_loop, res_numba, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(res_loop, res_numexpr, rtol=1e-5, atol=1e-8)


def lstsq_svd(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]

def lstsq_normal_eq(X, y, ridge=1e-8):
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), Xty)

def lstsq_qr(X, y):
    Q, R = np.linalg.qr(X, mode='reduced')
    return np.linalg.solve(R, Q.T @ y)

def bench(func, X, y, repeat=5):
    # 预热
    func(X, y)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func(X, y)
        times.append(time.perf_counter() - t0)
    return np.mean(times), np.std(times)

class TestLstsqSpeed(unittest.TestCase):
    def setUp(self):
        n_obs = 1_400_000   # 样本数
        n_feat = 10         # 特征数
        np.random.seed(42)
        self.X = np.random.rand(n_obs, n_feat).astype(np.float32)
        self.y = np.random.rand(n_obs).astype(np.float32)

    def test_methods(self):
        methods = {
            "SVD (np.linalg.lstsq)": lstsq_svd,
            "Normal Equation": lstsq_normal_eq,
            "QR Decomposition": lstsq_qr,
        }

        results = {}
        for name, func in methods.items():
            mean, std = bench(func, self.X, self.y, repeat=3)
            coef = func(self.X, self.y)
            results[name] = (mean, std, coef)
            print(f"{name} 耗时: {mean:.6f} ± {std:.6f} 秒")

        # 校验结果一致性
        ref = results["SVD (np.linalg.lstsq)"][2]
        for name, (_, _, coef) in results.items():
            np.testing.assert_allclose(coef, ref, rtol=1e-4, atol=1e-4,
                                       err_msg=f"{name} 与 SVD 结果不一致")

def cpu_step(X, y, f, ridge=1e-6):
    """CPU 正规方程求解"""
    mX = X * f[:, None]
    XtX = mX.T @ mX + ridge * np.eye(mX.shape[1], dtype=mX.dtype)
    Xty = mX.T @ y
    beta = np.linalg.solve(XtX, Xty)
    return beta

def gpu_step(X, y, f, ridge=1e-6):
    """GPU 正规方程求解 (CuPy)"""
    mX = X * f[:, None]
    XtX = mX.T @ mX + ridge * cp.eye(mX.shape[1], dtype=mX.dtype)
    Xty = mX.T @ y
    beta = cp.linalg.solve(XtX, Xty)
    return beta

class TestGPUSpeed(unittest.TestCase):

    def setUp(self):
        n, p = 1_400_000, 10
        np.random.seed(42)
        self.X = np.random.randn(n, p).astype(np.float32)
        self.y = np.random.randn(n).astype(np.float32)
        self.f = np.random.rand(n).astype(np.float32)

        # GPU 数据
        self.X_gpu = cp.asarray(self.X)
        self.y_gpu = cp.asarray(self.y)
        self.f_gpu = cp.asarray(self.f)

    def bench(self, func, *args, repeat=3):
        # 预热
        func(*args)
        if isinstance(args[0], cp.ndarray):
            cp.cuda.Stream.null.synchronize()

        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            res = func(*args)
            if isinstance(args[0], cp.ndarray):
                cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        return res, np.mean(times), np.std(times)

    def test_cpu_vs_gpu(self):
        beta_cpu, t_cpu, std_cpu = self.bench(cpu_step, self.X, self.y, self.f)
        beta_gpu, t_gpu, std_gpu = self.bench(gpu_step, self.X_gpu, self.y_gpu, self.f_gpu)

        beta_gpu_host = cp.asnumpy(beta_gpu)

        print(f"\nCPU 耗时: {t_cpu:.4f} ± {std_cpu:.4f} 秒")
        print(f"GPU 耗时: {t_gpu:.4f} ± {std_gpu:.4f} 秒")
        print("参数差异 (L2 norm):", np.linalg.norm(beta_cpu - beta_gpu_host))

        np.testing.assert_allclose(beta_cpu, beta_gpu_host, rtol=1e-5, atol=1e-5)