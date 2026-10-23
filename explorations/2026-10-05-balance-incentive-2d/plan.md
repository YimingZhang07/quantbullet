# Plan：CrossTerm 工具（在 offset 上拟合二维交互修正面）

日期：2026-10-06 · 状态：计划，未实施 · 背景与证据见 [report.md](report.md)

这份计划写于 lpm-terms（[design.md](../2026-10-20-lpm-terms/design.md)）之前，原文保留在"原计划"一节。实施前先按下面的
"与现状的关系"重新评估。

## 与现状的关系（2026-10-22 补记）

- **模型侧的二维 term 已经有了。** lpm-terms 的 P2 实现了 `TensorTerm`：tent × tent 的节点表就是节点上的双线性插值，
  reference 所在的行和列为 1，删除规则由 `drop` 显式写出，identity 和 log 都能拟合。所以"把交互面放进模型"不再需要本计划。
- **默认 form 变了。** turnover 现在默认用 log-link Poisson（Newton）。本计划选的 identity link 是在乘数尺度上做双线性插值，
  log 下的 `TensorTerm` 是在 log 尺度上做双线性插值，两者是不同的函数空间。
- **本计划仍有价值的部分**是与模型无关的诊断：只需要 (y, offset, weight)，给出 SE（可按 loan 聚类）、每个节点的支持度、
  在 holdout 和子集上的 `evaluate`，以及热力图加切片图。这正是 lpm-terms 待决问题 20（张量积的二维诊断）。
- **实施前要决定：**
  1. 做成独立模块，还是在 LPM 里实现为"冻结其余 term，只拟合一个 `TensorTerm`"；
  2. link 是否改成与模型一致（log 为默认，identity 可选）。

## 原计划（2026-10-06）


### Context

这次 turnover 诊断里，我们在 scratch 里做了一个离线分析：固定现有模型的预测作为 offset，在 balance × incentive 的节点网格上拟合一个双线性修正面 D̂（锚定一行一列为 1）。它清楚地显示了交互项存在、在哪里、多大，也被用来检验参数化形式（C2、tilt）是否表示得了它的形状。

现在要把这一步做成 repo 里一个通用、与具体模型无关的工具：输入 (y, offset, weight) 和两个连续变量，拟合交互面、给出不确定性和支持度、在新数据上评估、画出热力图和切片图。本期只做 CrossTerm 本身。

**已确认的范围：**
- loss：Poisson 与 MSE，identity link（D 对节点线性，与 product model 的线性 block 一致）。
- 轴：连续 × 连续，每个轴可选 linear / log 插值尺度。
- 约束：anchored（思路 A，主效应在 offset 里）与 full surface（思路 B，offset 不含两个主效应，作为 benchmark）。

**本期不做：** `ParametricCross`、`screen_cross_terms`、categorical 轴、log link、centered 约束、自动选 λ、fit + draw 合一的 `plot_cross_term`、接入 `report_turnover`。

### 模块与文件

沿用 binned_means 的分层：计算与绘图分离，计算部分不引入 plotting 依赖；对外接口走 lazy exports。

| 文件 | 内容 |
|---|---|
| `src/quantbullet/model/cross_term/__init__.py` | 导出 `CrossTermAxis`、`CrossTermSurface`、`fit_cross_term` |
| `src/quantbullet/model/cross_term/axis.py` | `CrossTermAxis`、单轴插值（cell + frac）、`from_quantiles` |
| `src/quantbullet/model/cross_term/fit.py` | `fit_cross_term`：数据检查、角点累加、penalty、Fisher scoring、sandwich SE |
| `src/quantbullet/model/cross_term/surface.py` | `CrossTermSurface` 结果对象：`predict`、`evaluate`、`table` |
| `src/quantbullet/model/__init__.py` | `_lazy_by_module` 加 `".cross_term"` 的三个名字 |
| `src/quantbullet/plot/cross_term.py` | `draw_cross_term`、`CrossTermPlot`、`CrossTermStyle`、`DEFAULT_CROSS_TERM_STYLE` |
| `src/quantbullet/plot/__init__.py` | `submod_attrs` 加 `"cross_term"` |
| `docs/model/cross_term.md` | 中文说明（见"文档"） |
| `tests/model/test_cross_term.py`、`tests/plot/test_cross_term.py` | 测试 |

复用：`PlotTheme` / `MINIMAL_THEME` / `PRINT_THEME`（[theme.py](../../src/quantbullet/plot/theme.py)，曲线颜色取 `theme.palette`），`compact_number`（[formatter.py](../../src/quantbullet/plot/formatter.py)，balance 类刻度），frozen dataclass + `__post_init__` 校验的写法（参考 [binning.py](../../src/quantbullet/plot/binned_means/binning.py) 的 `BinSpec`）。repo 里没有现成的二维 tensor / bilinear 基函数（`FlatRampTransformer` 是一维 ramp，语义不同），所以新写。

### 接口

```python
@dataclass(frozen=True)
class CrossTermAxis:
    column: str
    knots: tuple[float, ...]                 # 有限、严格递增、至少 2 个；log 时须 > 0
    scale: Literal["linear", "log"] = "linear"
    reference: float | None = None           # anchored 用，必须是 knots 之一；None 时 fit 取 weight 支持度最大的节点
    label: str | None = None                 # 图上的轴名，默认 column

    @classmethod
    def from_quantiles(cls, df, column, n_knots=5, *, weight=None, trim=0.01, **kwargs) -> CrossTermAxis: ...


def fit_cross_term(
    df, *,                                   # pandas 或 eager Polars
    row_axis: CrossTermAxis, col_axis: CrossTermAxis,
    y: str, offset: str, weight: str | None = None, cluster: str | None = None,
    loss: Literal["poisson", "mse"] = "poisson",
    constraint: Literal["anchored", "full"] = "anchored",
    ridge: float = 0.01, roughness: float = 0.0,
    init: np.ndarray | None = None,          # full：起点与 ridge 目标，形状 (n_row, n_col)，默认全 1
    max_iter: int = 50, tol: float = 1e-10,
) -> CrossTermSurface: ...


@dataclass(frozen=True)
class CrossTermSurface:
    row_axis: CrossTermAxis; col_axis: CrossTermAxis      # reference 已解析
    loss: str; constraint: str; ridge: float; roughness: float
    nodes: np.ndarray; se: np.ndarray; free: np.ndarray   # (n_row, n_col)；固定节点 se = 0
    support: dict[str, np.ndarray]   # rows / weight / actual(Σwyz) / expected(Σwmz) / information，均按插值权重分摊
    scale: float                     # Σ w m D / Σ w m
    deviance_baseline: float; deviance_fitted: float      # D ≡ 1 vs 拟合后
    n_obs: int; excluded: dict[str, int]; clamped: dict[str, int]
    iterations: int; converged: bool
    columns: dict[str, str | None]   # y / offset / weight，供 evaluate 使用

    def predict(self, df) -> np.ndarray: ...              # 每行的 D
    def evaluate(self, df) -> dict: ...                   # 任意数据（holdout、子集）上的 baseline / fitted / change
    def table(self, value="nodes", ci=.95) -> pd.DataFrame: ...
    # value 可取 nodes / se / lower / upper / z / 任一 support 键；index = row knots，columns = col knots


def draw_cross_term(
    surface, *, overlays: Mapping[str, CrossTermSurface] | None = None,  # 例如 {"2022+": surface_recent}
    slices: Literal["rows", "cols"] = "rows", ci: float = .95, label: str = "fit",
    n_cols: int = 4, panel_size=(3.2, 2.6), heatmap_height: float = 3.6, title: str | None = None,
    theme: PlotTheme = MINIMAL_THEME, style: CrossTermStyle = DEFAULT_CROSS_TERM_STYLE,
) -> CrossTermPlot: ...   # CrossTermPlot(fig, heatmap_ax, slice_axes, surface)
```

### 数学与求解（fit.py）

- **插值。** 每个轴先做变换 t = g(v)（linear 或 log），clamp 到 [t₀, t_K]（区间外 flat，计数记入 `clamped`），得到 cell 和 frac。每行只存两轴的 (cell, frac)，约 24 bytes/行。四个角点权重 (1−u)(1−v)、u(1−v)、(1−u)v、uv 在每次累加时现算。D_i = Σ 角点权重 × 节点值。
- **两种约束。**
  - anchored：reference 行、列上的节点固定为 1，其余自由。双线性插值保证整条参考线上 D = 1。
  - full：全部节点自由，从 `init` 出发。
- **目标函数**（μ = m·D）：
  - Poisson：Σ w(μ − y log μ)；MSE：½ Σ w(y − μ)²；
  - 罚项：½ · Ī · [ridge · ‖d_free − target‖² + roughness · ‖R d‖²]。target 在 anchored 时为 1，在 full 时为 `init`。
  - Ī 是起点处自由节点 Fisher 信息对角线的均值，所以 ridge / roughness 无量纲、对两种 loss 通用：节点 k 的收缩约为 ridge·Ī / (ridge·Ī + I_k)。
  - R 是沿两个方向的二阶差分，按节点间距缩放（节点等距时就是普通二阶差分）；固定节点进入线性项。
- **求解。** Fisher scoring，两种 loss 共用一个循环：
  - Hessian 行权重：Poisson 为 w·m/D，MSE 为 w·m²；
  - 梯度行值：Poisson 为 w(mD − y)/D，MSE 为 w·m(mD − y)。
  - 法方程用 `np.bincount` 累加（Hessian 16 次，梯度 4 次），不建 dense 设计矩阵。
  - step-halving 保证目标下降；Poisson 下自由节点保持 > 0。MSE 是二次的，一步即精确解。
  - 未收敛时 warning，结果里记 `converged=False`。
- **SE。** penalized sandwich：bread = (H + P)⁻¹，meat = Σ_g s_g s_gᵀ。给了 `cluster` 时，先按 cluster 把 score 聚合（bincount 到 (cluster, node)）再做外积。
- **数据检查。**
  - 用到的列有 null 或非有限值的行剔除，按原因计入 `excluded`。
  - 报错：weight < 0；Poisson 下 y < 0 或 offset < 0；Poisson 下 offset = 0 且 y > 0。Poisson 下 offset = 0 且 y = 0 的行剔除计数。
  - 报错：reference 不在 knots 上；knots 不递增；log 轴有非正 knot；某自由节点 information = 0 且 ridge = 0（报出节点位置）。
- **deviance。** Poisson 为加权平均 deviance（与 `fit_turnover.prediction_metrics` 的定义一致），MSE 为加权 MSE；baseline 即 D ≡ 1。

### 绘图（plot/cross_term.py）

- **上方热力图**：rows = row_axis 的节点，columns = col_axis 的节点（低值在下）。
  - 颜色按 log D 取对称的 diverging 色阶，中心在 1（anchored）或 `scale`（full）。
  - 每格标注"值 ± CI 半宽"；anchor 格用 muted 文字"1 / ref"；colorbar 标"multiplier"。
- **下方切片图**：`slices="rows"` 时每个 row 节点一个 panel，横轴是 col_axis；anchored 模式下省略全为 1 的 reference 行。`"cols"` 时两轴互换。
  - 主 surface 画节点 + CI 误差棒；`overlays` 中的每个 surface 在同一 row 值上沿 col 方向连续求值后叠加，可以和主 surface 的节点不同，但两轴的 column 必须一致，否则 `ValueError`。
  - y 轴共享，y = 1 画水平参考线，reference col 处画竖线；全图一个 figure legend。
- 字号、颜色和 spine 跟随 `theme`，曲线颜色取 `theme.palette`。`CrossTermStyle` 放色阶端点、线宽、marker、capsize、是否标注。

### 测试（只用合成数据）

`tests/model/test_cross_term.py`：
1. **轴。** 角点权重和为 1；节点上的值精确落在对应节点；区间外 clamp 并计数；log 插值与手算一致；各类非法输入报错。
2. **anchored + Poisson 复原。** 约 20 万行，m ~ U(0.002, 0.02)，用带锚点的已知 D_true 生成 y：拟合节点与真值的差在 3 SE 内；anchor 恰为 1；参考线上 `predict` = 1。
3. **零交互。** D_true ≡ 1 时节点 ≈ 1，|z| 基本 < 3，deviance 改善极小。
4. **MSE。** 能复原；结果等于直接解加罚 WLS 的闭式解。
5. **full。** 能复原带整体水平（如 1.3）的 surface；`init` 作为 ridge 目标生效；`scale` 正确。
6. **罚项。** ridge 极大时节点 → target；roughness 极大时沿两轴的二阶差分 → 0。
7. **Poisson 正值。** 节点被推向 0 时仍 > 0，`converged` 正确。
8. **cluster SE。** cluster = 行号时等于无 cluster 的 SE；同一 cluster 内复制行时 SE 变大。
9. **数据检查。** null 行剔除计数；负 weight / 负 offset、Poisson 下 offset = 0 且 y > 0 报错。
10. **其他。** pandas 与 Polars 输入结果一致；在拟合数据上 `evaluate()` 等于结果里的 deviance；`table()` 的标签和形状正确。

`tests/plot/test_cross_term.py`：
- 热力图存在；切片 panel 数：anchored 为 n_row − 1，full 为 n_row，`slices="cols"` 时互换。
- anchor 格有标注；overlays 增加曲线，全图只有一个 legend；用 `PRINT_THEME` 时刻度字号生效；overlay 的 column 不一致时报错。

### 文档：`docs/model/cross_term.md`（中文）

- 它是什么：offset 上的锚定交互面拟合，以及与 GLM offset、tensor-product spline、corner-point 约束、score test 这些标准名词的对应关系。
- anchored 与 full 的区别和适用场景（按这次对话里的解释写）。
- 数学：插值、目标函数、ridge / roughness 的无量纲含义。
- 用通用列名的 API 示例。
- 怎么读图：双重比值、CI、支持度。
- 注意事项：主效应必须在 offset 里；样本内改善偏乐观，要用 `evaluate` 看 holdout；按时间子集比较；区间外 clamp。
- 不写任何本地路径或数据内容。

### 验证

1. 只跑本次涉及的测试：

   ```bash
   .\.venv\Scripts\python.exe -m pytest tests/model/test_cross_term.py tests/plot/test_cross_term.py
   ```

2. **对照 scratch 结果。** 在 scratchpad 写一个脚本（不进 repo），在 turnover 数据上用与 `dhat_vs_c2.py` 相同的节点和锚点（B* = 250K，I* = −3），ridge 换算成与绝对 λ = 10 等价（ridge = 2λ / Ī）：
   - 全量与 2022+ 两个拟合的节点，与 `dhat_vs_c2.json` 中的值差 ≤ 1e-3；
   - deviance 变化分别约为 −0.018% 和 −0.027%；
   - 加上 `cluster="loan_identifier"` 后 SE 应不小于原 sandwich SE；
   - 记录 11.7M 行上单次拟合的耗时。
3. 用 `draw_cross_term(surface_all, overlays={"2022+": surface_recent})` 出图，和之前的图对照检查。
4. 检查 diff：没有个人路径、没有数据文件、没有无关改动。**不提交**，留给你审阅。
