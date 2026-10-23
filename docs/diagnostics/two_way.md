# Two-way diagnostics

`TwoWayDiagnostic` 在两个变量 x、z 上检查一个模型的预测。它只需要五列：x、z、y、pred（模型预测）和可选的 weight，不需要模型对象，
所以 LPM、GBM、GLM 的预测都能用。两个方法回答两个不同的问题：

| 方法 | 回答 | 结果 |
|---|---|---|
| `diag.ae(x_bins=..., z_bins=...)` | 在 x × z 的每个格子里，实际比模型多还是少（原始比值） | `TwoWayAE` |
| `diag.correction(x_knots=..., z_knots=..., ...)` | 模型的预测还需要乘上一个怎样的二维修正面 D(x, z)；设了 reference 时，就是可分离的主效应漏掉了多少交互 | `TwoWayCorrection` |

两个结果都有 `.plot()`：上方一张热力图，下方每个 x bin（或节点）一个切片 panel。

可以直接运行的用法示例（假数据，每个用法一个 test，输出写到 `examples/diagnostics/output/`）见
[examples/diagnostics/test_two_way_examples.py](../../examples/diagnostics/test_two_way_examples.py)。

## 输入

```python
from quantbullet.diagnostics import TwoWayDiagnostic

diag = TwoWayDiagnostic(
    df,                     # pandas 或 eager Polars，不会被修改
    x="balance",            # 热力图的行；默认每个 x bin / 节点一个切片 panel
    z="incentive",          # 热力图的列；切片的横轴
    y="prepaid",            # 实际值
    pred="model_smm",       # 模型预测
    weight="upb",           # 可选；不给时等权
    x_scale="log",          # "linear" | "log"：变量本身的尺度，两个方法共用
    labels={"balance": "current balance"},
)
diag.n_obs, diag.excluded
```

- 只取这五列并转成 Float64，之后两个方法都复用，所以在大数据上换 bins 或 knots 反复看时不会重新读数。
- 五列中任一为 null 或非有限值的行剔除，计入 `excluded["missing"]`。log 轴上 ≤ 0 的值剔除，计入 `nonpositive_x` / `nonpositive_z`。
- weight 为负时报错。
- y 和 pred 的符号由各方法自己检查：A/E 和 MSE 不要求它们非负，Poisson 要求。
- 要看子集（例如某个时期），就用过滤后的 df 再建一个实例。

## A/E：`diag.ae()`

```python
from quantbullet.plot import BinSpec

ae = diag.ae(x_bins=BinSpec.edges([0, 1e5, 2.5e5, 5e5, 2e6]), z_bins=BinSpec.step(1.0))
ae.table          # 长表，每个 (x bin, z bin) 一行
ae.pivot("ae")    # 宽表：x bins 为行，z bins 为列；可取 table 的任一数值列
ae.plot(min_rows=500)
```

- 分箱规则与 binned means 相同（`BinSpec` 的 edges / quantile / step / round）：区间右闭，最低边界包含在内；
  落在 edges 之外的行剔除，计入 `excluded["outside_bins"]`。
- 每个出现过的 x bin 都补全所有 z bin，空格子的 `rows` 为 0、`ae` 为 NaN。

`table` 的列：

| 列 | 含义 |
|---|---|
| `x_bin`、`z_bin` | 区间（`pd.Interval`），或 `BinSpec.round` 的取值 |
| `rows`、`weight_sum` | 行数、权重和 |
| `actual`、`expected` | Σw·y、Σw·pred |
| `ae` | actual / expected；expected ≤ 0 时为 NaN |
| `ae_se` | √Σw²(y − pred)² / expected |
| `actual_mean`、`pred_mean` | 加权均值 |

`ae_se` 不假设 y 的分布，但把行当作相互独立：同一笔贷款的多行之间有相关，实际不确定性会更大一些。

**读法。** A/E 是原始比值，混着主效应的残差和交互两部分。例如某一行（一个 x bin）整体偏高，可能只是 x 的主效应没拟合好。
要判断交互，看同一行的 A/E 是否随 z 倾斜，并且倾斜方向随 x 有规律地变化；或者直接用下面的 D̂。

## 修正面 D̂：`diag.correction()`

```python
fit = diag.correction(
    x_knots=(5e4, 1.5e5, 2.5e5, 4.5e5, 8e5), z_knots=(-5, -4, -3, -2, -1, -.5),
    x_reference=2.5e5, z_reference=-3,   # 每个轴可选，必须是 knots 之一
    loss="poisson",                      # 或 "mse"
    ridge=0.01,
)
fit.table("nodes")    # 也可取 se / lower / upper / z_score / rows / weight / actual / expected / information
fit.deviance          # baseline（D ≡ 1）/ fitted / change
fit.predict(df)       # 每行的 D
fit.plot(overlays={"2022+": fit_recent})
```

### 模型

μ = pred × D(x, z)。D 是两个轴 tent basis（一次 B-spline）的张量积：在节点之间对各轴的 scale 坐标做双线性插值（log 轴就在 log(x)
上插值），在两端节点之外取端点值。超出范围的行数计入 `clamped`。D 对节点值是线性的，所以每个节点值直接读作乘数。

pred 在拟合中固定不动，作为 offset。

### reference 的三种情况

| reference | D 表示什么 | 节点怎么读 |
|---|---|---|
| x 和 z 都给 | 只剩交互：reference 所在的行和列固定为 1，D 不能再表示 x 或 z 单独的效应 | 双重比值，见下 |
| 都不给 | 相对 pred 的平滑二维 A/E 面，包括主效应的偏差 | 该点上实际 / 模型的比值。如果 pred 不含这两个主效应，它就是完整二维面的 benchmark |
| 只给一个 | 只固定那一条线 | 相对那条线的比值 |

两个 reference 都给时，节点值近似为

$$D(x,z) \approx \frac{\text{A/E}(x,z)\,/\,\text{A/E}(x,z^*)}{\text{A/E}(x^*,z)\,/\,\text{A/E}(x^*,z^*)}$$

例如 D(50K, −4) = 1.2 的意思是：对 50K 的贷款，−4 处相对 −3 处，实际比模型多出约 20%，并且已经扣掉了 reference 行
（例如 250K）在同一位置的比值。reference 应选在数据最密的位置，这样主效应的解释最稳。

### 拟合

- **目标函数**：Poisson 为 Σw(μ − y log μ)，要求 y ≥ 0、pred ≥ 0（pred = 0 且 y = 0 的行剔除，计入 `excluded["zero_pred"]`）；
  MSE 为 ½Σw(y − μ)²。
- **罚项**：½ · ridge · Ī · ‖d − 1‖²，只作用于自由节点。Ī（`penalty_scale`）是 D = 1 时自由节点 Fisher 信息的均值，
  所以 ridge 无量纲：信息量为 I 的节点大约被拉向 1 的比例为 ridge·Ī / (ridge·Ī + I)。默认的 0.01 只明显影响数据很少的节点。
- **求解**：Fisher scoring，加 step-halving。Poisson 下节点保持为正；某个区域没有事件时，那里的节点会贴近 0。
  法方程按每行所在格子的四个角累加，不构造按行展开的设计矩阵。在 1700 万行上一次拟合约 20 秒。
- **SE**：penalized sandwich，把行当作相互独立，不按贷款聚类。

### 注意事项

- **要把 D 读成交互，pred 必须已经包含 x 和 z 的主效应。** 漏掉的主效应会被扭曲地涂到整张面上。
- **改善是样本内的。** 20 个左右的节点在样本内总会改善拟合，`deviance["change"]` 只说明方向和量级，不能当作检验。
- **子集比较。** 浅端、深端等局部的偏差可能来自时代或其他构成差异。可以对子集各拟合一次，用 `overlays` 叠加对比。
- **节点之外取端点值。** knots 应覆盖数据的主要范围，两端可以与模型输入的 clip 一致。

## 画图

```python
ae.plot(slices="x", min_rows=0, ci=.95, n_cols=4, panel_size=(3.2, 2.6), heatmap_height=3.6, theme=..., style=...)
fit.plot(overlays=None, slices="x", ci=.95, label="fit", ...)
```

- **热力图**：按 log 值着色，以 1 为中心的对称 diverging 色阶（蓝 < 1，红 > 1）。每格标注"值 ± CI 半宽"；D̂ 的 reference 格子标"1 · ref"；
  A/E 中行数少于 `min_rows` 的格子留空。
- **A/E 切片**：每个 x bin 一个 panel，画 A/E 对 z 的曲线（带 CI 误差棒）、1 的参考线和行数柱（右轴）。所有 panel 共用刻度。
  log 轴上各 bin 放在 log 中点，刻度标原始单位。
- **D̂ 切片**：每个 x 节点一个 panel（reference 行全为 1，不画），画节点值和 CI，节点间的连线就是该 x 上 D 的精确取值。
  `overlays` 把其他拟合（同样的 x、z 列和 scale，knots 可以不同）沿同一条线叠加。
- `slices="z"` 时两个维度互换。字号和文字颜色跟随 `theme`（例如 `PRINT_THEME`），色阶和线宽在 `TwoWayStyle` 里。

## 结构

| 模块 | 职责 |
|---|---|
| `diagnostics/two_way/diagnostic.py` | `TwoWayDiagnostic`：取数、剔除、校验，两个方法的入口 |
| `diagnostics/two_way/ae.py` | `TwoWayAE`：用 `summarize_binned_means` 做二维加权汇总 |
| `diagnostics/two_way/correction.py` | `TwoWayCorrection`：二维 tent 网格、Fisher scoring、SE |
| `plot/two_way.py` | `draw_two_way_ae`、`draw_two_way_correction`、`TwoWayStyle` |

构造 `TwoWayDiagnostic` 和拟合 `correction` 都不加载 matplotlib。`.ae()` 用 binned means 的汇总，那个包会带上 matplotlib；
`.plot()` 才加载 `plot/two_way.py`。
来历和一次完整的应用（turnover 的 balance × incentive）见 `explorations/2026-10-05-balance-incentive-2d/report.md`。
