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