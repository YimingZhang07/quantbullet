# Design：LPM 的 Term 抽象（tent basis、reference、cross term 与参数化 term）

日期：2026-10-20 · 状态：设计草案；P0（第 9 节）、P1 与 P2 已实施，P3 起未实施

## 目标与非目标

目标：

- 把 term（一个 feature group 怎样变成乘积里的一个 factor）做成模型里的一等对象，由它自己负责 basis、参数、初值、约束和 level。
  模型写成 `mu = global_scalar × prod_k f_k(x; theta_k)`，`f_k` 可以对 `theta_k` 线性，也可以是任意参数形式。
- 支持 tent basis（一次 B-spline）：系数就是曲线在 knot 上的值。
- 协议能容纳参数以非线性方式进入的 term（logistic、指数平台等）。这些参数可以初始化、可以固定，并和其他 term 联合优化。
- frozen submodel 和 interaction 收进 term，不再是 `block_factor` 和 solver 里的特例分支。
- reference 由每个 term 自己声明，可以是一个或多个；求解时的 pivot 和 reference 分开（3.2 节）。
- 支持 cross term：分类 × 连续（现有的 interaction）和连续 × 连续的张量积。删除规则必须显式写出，模型按笔记 4.2.3 检查（3.2 节）。
- 默认路径行为不变：现有 `fit(feature_groups=...)` 调用、数值结果、已保存的 pickle 都不受影响。

非目标：

- 不改 form 的集合（`identity/mse`、`identity/poisson`、`log/poisson`），不加 log/MSE。
- 不改 `LinearProductClassifier*` 和 `LinearProductRegressorScipy`。
- 第一版不做自由 knot（knot 位置作为参数）。

## 1. 现状

模型只认识"设计矩阵 `X_k` × 系数向量 `beta_k`"。term 本身的定义（FlatRamp 的 knots、bias、OneHot 的 level）放在 toolkit 的
`preprocess_config` 里，模型看到的只是展开后的列。对系数线性的假设分散在下面这些地方（行号为 2026-10-20 的版本）：

| 假设 | 位置 |
|---|---|
| factor 为 `X @ coef` 或 `exp(X @ coef)` | `forms.py` 114、194 |
| identity 路径里直接写 `X @ coef` 求 block 预测 | `solvers.py` 334（interaction 曲线）、353（`by` block）、373（普通 block） |
| `block_step` 为闭式 WLS 或一步 IRLS | `forms.py` 130–176 |
| 归一化靠系数除以均值，要求 `f(c·beta) = c·f(beta)` | `solvers.py` 337、356、376 |
| 初值按列均值缩放 | `utils.py` 5（`init_betas_by_response_mean`）、`base.py` 204 |
| Newton 的可识别性靠恒为 1 的单列或完整 one-hot | `solvers.py` 525（`_constant_column`）、553–577 |
| Newton 的 design 就是 `X` 的列，可缓存为 float32 | `solvers.py` 545–656 |
| `coef_` 为 ndarray，flatten 按 `len(feature_groups_[k])` 切分 | `base.py` 119–146、164–170 |
| term 定义在 toolkit，模型只见展开列 | `toolkit.py` 123–156 |
| 画单变量曲线要 transformer + 临时 container | `toolkit.py` 193–218 |
| 报告按 transformer 类型分支，遇到未知类型报错 | `toolkit.py` 744–764 |
| frozen submodel 和 `InteractionCoef` 是特例分支 | `base.py` 349–364；`solvers.py` 中 `interactions_` / `submodels_` 检查约 20 处 |

外部使用者：`projects/freddie_prepayment/fit_turnover.py`、`tests/linear_product_model/`（regressor、interaction、categorical_poisson、
solvers、forms、toolkit、component_manager 等）、`tests/projects/test_freddie_turnover.py`，以及已保存的 `turnover_model.pkl`。

## 2. 分层

| 层 | 负责什么 | 现状 |
|---|---|---|
| **Term** | 原始列到输出值的映射、参数、Jacobian、level、惩罚 | 新增。今天散在 transformer、`feature_groups_`、`submodels_`、`interactions_` 里 |
| **Form**（link + loss） | link 尺度和 factor 尺度之间的换算、loss、working weights、implied actual、归一化选哪种操作 | 已有 `ModelForm`，增加少量方法 |
| **Solver** | 迭代 | 已有 `BCDSolver`、`NewtonSolver`，改为通过 term 取值和 Jacobian |

上一轮 form / solver 分层里设想的第 4 期 Block 类，由 Term 取代。

### 2.1 输出尺度

每个 term 声明自己的 `output`：

- `"link"`：`value` 返回 link 尺度上的贡献 `g`，factor 为 `link⁻¹(g)`。今天所有 block 都是这一类：identity 下 `f = X @ beta`，log 下 `f = exp(X @ beta)`。
- `"factor"`：`value` 直接返回乘数 `f`。参数化形状（从 piecewise 曲线拟合出来的 logistic 等）和 frozen submodel 属于这一类。

solver 一律在 link 尺度上工作，由 form 统一换算：`g = link(f)`，`J_g = link'(f) · J_f`。identity link 下两种尺度相同；
log link 下 `g = log f`，`J_g = J_f / f`，要求 `f > 0`。

这样不用把 link 下放到每个 term，form 的注册表也保持不变，同时能表达"log-link 模型里放一个乘数尺度的 tent"这类组合
（它对 Newton 来说是非线性 term）。

## 3. Term 协议

```python
@dataclass
class Param:
    name: str
    value: float                                  # initial value; the estimate after fit
    fixed: bool = False
    bounds: tuple[float, float] = (-np.inf, np.inf)   # enforced by reparametrization, see 6.5


class Term(ABC):
    """One factor of the product: maps raw columns to a link-scale contribution or to the factor itself."""

    name: str
    inputs: tuple[str, ...]                       # raw columns of the container this term reads
    output: Literal["link", "factor"]
    linear: bool                                  # value = basis @ theta: constant Jacobian, closed-form block step

    def setup(self, frame) -> None: ...           # fit-time state: levels, default knots, input checks
    def prepare(self, frame) -> TermData: ...     # per-row inputs: basis matrix (linear) or raw columns
    def params(self) -> list[Param]: ...
    def value(self, data, theta) -> np.ndarray: ...       # (n,) on the term's output scale
    def jacobian(self, data, theta) -> np.ndarray: ...    # (n, p) d value / d theta; default: finite differences

    # Level: the degree of freedom redundant with global_scalar_ (sections 3.1 and 3.2)
    def shift(self, theta, t) -> np.ndarray | None: ...   # theta whose value is value + t; None if not possible
    def scale(self, theta, s) -> np.ndarray | None: ...   # theta whose value is s · value; None if not possible
    def pivot(self) -> int | None: ...                    # theta coordinate held fixed in joint fits; the reference when declared

    def penalty(self) -> np.ndarray | None: ...           # (p, p) quadratic penalty, e.g. tent second differences
    def describe(self) -> dict: ...                       # kind and config for reports (knots, parameter values)
```

- `theta` 是该 term 全部参数（含 fixed）按 `params()` 顺序排成的向量，所以 `coef_` 仍然是 `{name: ndarray}`，
  `coef_dict` 仍然是 `{name: {param_name: value}}`。线性 term 的参数名就是 basis 列名，和今天一致。
- 线性 term（`linear=True`）的 `jacobian` 就是 `prepare` 产出的 basis 矩阵，solver 可以照旧缓存、走闭式解。
- `jacobian` 的默认实现是中心差分，参数化形状以后再逐个补解析式。

### 3.1 Level 与可识别性

`global_scalar_` 和每个 term 的整体水平是冗余的。今天的做法有两处：BCD 每步把 block 归一到均值 1，均值并入 `global_scalar_`；
Newton 去掉常数方向，只留一个全局 intercept。推广到 term 后，规则是：

**归一化**：把 factor 变成原来的 `1/mean`。form 根据 link 和 term 的输出尺度决定调用哪个操作：

| term 输出 | identity link | log link |
|---|---|---|
| `"link"` | `scale(theta, 1/mean)` | `shift(theta, -log(mean))` |
| `"factor"` | `scale(theta, 1/mean)` | `scale(theta, 1/mean)` |

对应操作返回 `None` 时，这个 term 不归一化，它的均值留在 term 里。这和今天 Newton 对"没有常数列的数值 block"的处理相同。

**联合求解时的可识别性**：`pivot()` 给出一个参数坐标，联合求解时固定住它，以消除和全局水平的冗余；归一化只在求解后进行。
pivot 和用户声明的 reference 是两回事，见 3.2 节。

| term | `shift` | `scale` | `pivot` |
|---|---|---|---|
| 线性，有常数方向 `c`（`X @ c = 1`） | `theta + t·c` | `s·theta` | 一个 `c_j ≠ 0` 的列：bias 列、tent 的某个 knot、one-hot 的某个 level |
| 线性，无常数方向 | `None` | `s·theta` | `None` |
| 参数化，有幅度参数 `a`（`f = a·h`） | `None` | `a → s·a` | `a` |
| 参数化，有偏移参数 `b`（`g = b + h`） | `b → b + t` | `None` | `b` |
| frozen | `None` | `None` | `None` |

`LogForm.rescale(coef, mean, constant)` 已经接受方向向量，所以线性 term 的 `shift` 就是现有公式。

fit 开始时校验声明：线性 term 的 `c` 用数据检查 `X @ c ≈ 1`。如果 Jacobian 中某个方向近似常数却没有声明，就发出警告，
因为它会让 Hessian 奇异。

### 3.2 Reference 与 pivot

两个概念要分开：

| | pivot | reference |
|---|---|---|
| 是什么 | 联合求解时固定的一个参数坐标，用来消除 term 和全局水平的冗余 | 建模选择：term 在这里的乘数为 1 |
| 谁来定 | 用户声明了 reference 就用它；没有声明时由 solver 自动选 | 用户在 term 上声明，一个或多个 |
| 影响 | `l2 = 0` 时不影响拟合、预测和最终的 `coef_`，只是求解时的参数化；`l2 > 0` 时只有很小的影响 | 第一个只挪动水平（gauge）；之后每多一个，就多一个约束 |
| 出现在哪里 | 只有 Newton（log link）需要；BCD 逐块求解，没有 pivot | 两种 link |

**pivot 怎么选。**

1. term 声明了 reference，且 reference 落在单个系数上（tent 的某个 knot、分类的某个 level，或下面合并出来的那一列）：
   pivot 就是它，Newton 求解时这一列不进 design，没有任何自动选择。实现上求解后的换算沿用原路径（先 mean-1，再按 9.2 节换算到
   reference），结果和"求解后直接把水平留在 `global_scalar_`"相同，reference 处的系数正好是 1（identity）或 0（log）。
2. reference 是两个 knot 之间的 x 值（只允许单个 reference）：pivot 按 9.3 节自动选，求解后用 9.2 节的 gauge 换算到 reference。
   结果和第 1 种相同。
3. 没有声明 reference：pivot 按 9.3 节自动选（bias 列，或权重最大的列），求解后做 mean-1 归一化。

每个 group 的 pivot 记进 `convergence_info_['pivots']`：`{group: {'column': 列名, 'from': 'reference' 或 'auto'}}`。
identity link 下 BCD 没有 pivot，reference 一律用 9.2 节的 gauge 换算。

只有当常数方向是精确的，换 pivot 才是精确的重参数化。实现 P1(c) 时发现 float32 的 tent 行和只精确到约 6e-8，换 pivot 会让预测
变化约 1e-8，沿全 1 方向平移系数也是。所以 `TentTransformer` 在输出 dtype 里直接构造每行的两个权重：较大的那个落在 [0.5, 1]，
另一个取 1 减它，按 Sterbenz 引理是精确的，于是 float32 下每行也精确和为 1。此后不同 pivot 的拟合一致到约 1e-15。

**多个 reference：合并列，再做 gauge。**

- 多个 reference 必须都落在 knot 或 level 上。把这些列相加，合并成一列，它们共用一个系数，表示"这些点上的乘数相同"。
  这是线性的重参数化，BCD 和 Newton 都不用改，basis 少 r − 1 列。
- 合并出来的系数再按上面第 1 点处理：log 下作为 pivot 固定为 0；identity 下用 gauge 设为 1。
- log 下这和"删掉这些列"完全等价；identity 下删列会把乘数固定为 0（9.2 节），合并才对。
- 合并后每行仍然和为 1，常数方向不变。
- `coef_` 保持 solver 的参数化，即合并后的列（例如 `c_orig_fico_fit_at_740+780`）；`model.term(name).table()` 展开成完整的
  knot 或 level 表，合并的点都是 reference 值，`describe()` 列出全部 reference。
- 作为 `ByTerm` 的 `by` 的 term 只能有一个 reference：`by` 的每个 level 都要有自己的列，才能接收各条曲线的水平。
- 第一个 reference 不改变模型，之后每一个都是约束（笔记第 4 节："多删的每一个都是约束"）。

**cross term 的删除规则。**

cross term 由几个变量的 basis 逐列相乘得到：分类 × 连续是 `ByTerm`（现在的 `interactions`），连续 × 连续是张量积 `TensorTerm`。
每个变量用它自己 term 上声明的 reference。这里的"删除"统一指"固定为 reference 值"，实现同上：log 下删列，identity 下合并加 gauge。
固定哪些格子，由规则决定（下表举例时 x 的 reference 为 a，y 的为 p）：

| 规则 | 固定的格子 | 例 |
|---|---|---|
| 变量子集 `S` | `S` 中任一变量取 reference 的格子 | `S = {x}`：a 那一行 |
| `"any"` | 任一变量取 reference，即 `S` 为全部变量 | a 那一行 + p 那一列 |
| `"all"` | 所有变量同时取 reference | 只有 (a, p) 一格 |

**没有默认，只有检查（2026-10-21 决定）。** 每个 cross term 都必须显式写出 `drop="any" | "all" | (变量, ...)`，没写就报错。
模型按笔记 4.2.3 推出唯一正确的规则，只用来检查：变量 v 必须在 `S` 里，当且仅当 cross term 去掉 v 后剩下的那一项已经在模型里，
`global_scalar_` 算作空项。`S` 为空、但 cross term 仍和 `global_scalar_` 重复时（模型里只有截距和这个 cross term），正确的写法是
`"all"`（笔记 4.2.5）。

| 模型 | 唯一正确的写法 |
|---|---|
| `1 + x + y + x×y` | `drop="any"`（即 `S = {x, y}`） |
| `1 + x + x×y` | `drop=("y",)` |
| `1 + c + c×age`（现在的 `interactions={"age": "purpose"}`） | `drop=("age",)`：每个 purpose level 保留一条完整的 age 曲线，正是今天 Newton 的做法 |
| `1 + x×y` | `drop="all"` |

检查时，frozen term 算作不在模型里：它的块是固定的，吸收不了被删掉的部分。

**写错了就报错**，在 fit 开始、模型状态重置之前。报错信息说明错在哪里，并给出正确的写法：

实现上按笔记 4.2.1 的块来检查（P2）：全局 scalar 提供常数块，每个主效应提供只随自己变化的块，每个 cross term 按自己的 `drop`
提供它保留的块（规则 `S` 保留包含 `S` 的块，`"all"` 保留常数块以外的块）。某个块被两处提供就是共线，完整的块没有任何地方提供
就是约束。报错会写出是哪一块、由谁提供，例如 "it keeps the part that varies only with purpose, which CategoricalTerm('purpose')
also provides, so the model is collinear"。正确写法在所有变量子集和 `"all"` 里找；都不行时（需要删单个格子之外的东西）也会说明。

- 少删了（留下了其他 term 已经提供的块）：模型共线。
- 多删了（删掉了没有其他 term 提供的块）：模型多了约束。
- `S` 里的变量没有在自己的 term 上声明 reference。

`ByTerm` 只支持 `drop=(内层 term,)`，因为 solver 为 `by` 的每个 level 拟合一条完整的曲线；其他结构用 `TensorTerm`
（分类 × 连续也可以）。`TensorTerm` 的各个 term 只能是 `TentTerm` 或 `CategoricalTerm`，每个只有一个 reference，并且必须落在
knot 或 level 上；被固定的格子并成一列 `{name}:reference`，拟合后换算到 reference 值，log 下 Newton 也用这一列当 pivot。

最后一条有一个例外：`ByTerm` 的内层变量（例如 `ByTerm(tent_age, "purpose", drop=("age",))` 里的 age）可以不声明 reference。
这时它和主效应一样，pivot 自动选，求解后每个 level 的曲线归一到均值 1、水平转入 `by`，这就是今天的做法。`TensorTerm` 没有这个例外：
张量积没有像 mean-1 这样自然的口径，不声明 reference，系数就只能由自动选的 pivot 决定，而这会体现在系数上。

`feature_groups` 路径的 `interactions={"age": "purpose"}` 是旧接口，它本身就定义了 `drop=("age",)`：兼容层构造 `ByTerm` 时显式写出这个规则，
再照常检查。旧接口要求 `by` 在 `feature_groups` 里、`age` 没有单独的主效应，所以这条规则总能通过；age 没有声明 reference，
走上面的例外，现有调用不受影响。

**两点提醒。**

- 笔记的块分析在 log link 下精确成立，因为对数尺度上是加法。identity link 下乘积不是加法：推出的规则在节点上去掉的自由度仍然正确，
  但节点之间两种 link 的曲面本来就不同。
- 笔记 4.4：cross term 里某个变量用了主效应没有的 knot 时，另一个变量的 reference 位置会影响拟合。`check()` 检查 cross term 的
  knots 是否为同名主效应 knots 的子集，不是就警告（用同一个 term 对象时自然满足）。
- tent × tent 的格子是两个 float32 的乘积，float32 下行和只在舍入误差内等于 1；需要精确的重参数化时用 `dtype=np.float64` 准备数据。
- identity link 下 BCD 迭代时，cross term 的 reference 列可能经过负值：只要求拟合结果和最优一步的 reference 因子为正，history 中
  因子为负的步照样换算（预测不变），为 0 的步保持 mean-1 口径。

## 4. Term 类型

| Term | output | linear | theta | level | 取代 |
|---|---|---|---|---|---|
| `ColumnsTerm(columns)` | link | 是 | 原 `coef_[group]` | 按数据检测常数方向 | `fit(feature_groups=...)` 的兼容层：读 container 的展开列 |
| `BasisTerm(transformer)` | link | 是 | basis 系数 | transformer 声明，或按数据检测 | toolkit 的 `preprocess_config` |
| `TentTerm(column, knots, reference=None, penalty=None, output="link")` | link 或 factor | link 时是 | 每个 knot 上的值 | `c = 1` | 新增，见第 5 节 |
| `CategoricalTerm(column, levels=None, reference=None)` | link | 是 | 每个 level 的值 | `c = 1` | `OneHotEncoder` + one-hot 检测 |
| `ParametricTerm(column, shape, params)` | factor | 否 | 形状参数 | 幅度或偏移参数 | `parametric_model` 的类（`ComponentManager` 的用法） |
| `FrozenTerm(column, model)` | factor | — | 无 | 无 | `submodels_` |
| `ByTerm(term, by, drop)` | 同内层 | 同内层 | 每个 level 一份内层 theta | 每个 level 的水平转入 `by` term | `interactions_` + `InteractionCoef` |
| `TensorTerm(*terms, drop)` | link | 是 | 节点表（各变量列数的乘积） | 由 `drop` 固定的格子 | 新增：连续 × 连续交互（笔记第 3 节） |

- `column` 是 term 读的原始列（`TensorTerm` 读各个 term 的列，见 `term.inputs`）。单列 term 的名字就是输入列名，`ByTerm` 用内层
  term 的名字，和旧的 `interactions` 一致；`TensorTerm` 默认叫 `a×b`，可以用 `name=` 改。spec 准备的数据不再要求 group 名是原始列
  （`ProductModelDataContainer(groups_in_orig=False)`），toolkit 按 term 的输入列取值。`by` term 的名字仍须是它的列，interaction
  按 label 取行时要用。
- `reference`（3.2 节）：`None`、一个值（tent 可以是两个 knot 之间的 x 值），或多个 knot / level（合并成一列）。
- `drop`（3.2 节）：cross term 必填，没有默认值；模型按笔记 4.2.3 检查，写错就报错。里面写的是 term 名。
- `ByTerm` 的 `by` 和 `TensorTerm` 的各个变量都是 term 对象（`by` 是 `CategoricalTerm`），见 7.1 节。
- `ParametricTerm` 包住现有 `ParametricModel` 子类：它们已有 `func_with_args` 和 `get_param_names`，缺的只是 Jacobian（先用差分）。
- `FrozenTerm` 等价于所有参数都 fixed 的 `ParametricTerm`，单独列出来是为了兼容 `submodels_`。
- `ByTerm` 的 theta 继续存成 `InteractionCoef`，toolkit 不用改；solver 看到的是各 level 的 theta 拼成的向量。
- `ByTerm` 是 cross term 在"分类 × 任意 term"时的形式，每个 level 一条曲线。`TensorTerm` 的列是各变量 basis 的逐列乘积
  （tent × tent 时就是节点表上的双线性插值，笔记 4.5），报告按节点表展示，reference 格子填 reference 值。

## 5. Tent basis

定义见 notes-factory 仓库的 `docs/stats/tent-basis-linear-spline/tent-basis-construction.md`。knots `k_1 < … < k_K` 各对应一列；
每行非负、和为 1；`f(k_j) = theta_j`；相邻 knot 之间线性插值，两端外是平的。

### 5.1 与 FlatRamp 的关系

`FlatRampTransformer(knots=k, include_bias=True)` 的列是 `1, min(x, k_1), clip(x, k_1, k_2), …, max(x, k_K)`，两端是**线性外推**；
tent 两端是平的。在区间 `[lo, hi]` 上，tent knots `(lo, k_1, …, k_K, hi)` 和 FlatRamp(+bias) 张成同一个列空间（都是 K+2 维），
只有外推不同。

turnover 在建模前把所有数值输入截断到 `CLIP`（`fit_turnover.py` 的 `to_model_data`），所以在 turnover 上，取 `lo, hi` 为截断边界的
tent 和现有 FlatRamp 是**同一个模型**，只是换了参数化。P0 的等价验收就利用这一点。

### 5.2 系数的含义与归一化

- identity link：`theta_j` 是 knot `k_j` 上的乘数；mean-1 归一化后是相对平均水平的乘数。
- log link：`exp(theta_j)` 是 knot 上的乘数，knot 之间按乘数的几何插值。
- 用 reference 代替 mean-1：让 reference knot 上的系数为 1（identity）或 0（log），系数就是"相对 reference knot 的乘数"，
  和分类变量 reference level 的口径一致。单个 reference 只在 term 和 `global_scalar_` 之间挪动水平，不改变拟合；多个 reference
  从第二个起是约束。identity 下不能靠删列实现（9.2 节），一律合并列再做 gauge（3.2 节）。
- 在乘积模型里，`global_scalar_` 起截距的作用。联合求解（Newton，log 尺度）时，每个有常数方向的主效应都要固定一个 pivot 来消除共线
  （笔记 4.2.2）。声明了 reference 时 pivot 就是 reference，否则自动选，这只是求解时的参数化（3.2 节）。
  `ByTerm(tent, by, drop=("x",))` 在 `by` term 存在时删 `x` 的 reference、保留 `by` 的全部 level，正是今天 Newton 的
  "非常数列 × 完整 one-hot"，也是笔记 4.2.3 下唯一正确的写法。

### 5.3 惩罚与约束

- 二阶差分惩罚 `lambda · ||D_2 theta||²`：可以多放 knots，再用惩罚控制平滑度，少一些手挑 knot 的工作。
  `D_2 · 1 = 0`，所以惩罚不受常数方向和 reference 选择的影响。
  - Newton：现在的 `l2` 是对角的，作用在标准化后的系数上。要推广成矩阵 `S⁻¹ P S⁻¹`，其中 `S` 为保留列的 std。
  - BCD：`ols_normal_equation_scaled` 的 `ridge` 要推广成矩阵。identity 下归一化会把惩罚的尺度改变 `1/mean²`，需要在文档里写明。
- 单调约束：`theta = theta_1 + cumsum(softplus(delta))`。这是非线性参数化，属于 P3/P4。

## 6. Solver 的推广

### 6.1 统一的视角

记 link 尺度的线性预测为 `eta = log(global_scalar) + sum_k g_k`（log link），或 `mu = global_scalar · prod_k f_k`（identity link）。
两个 solver 都是 Gauss-Newton / Fisher scoring 的特例，只是线性 term 的 Jacobian 恒定，有闭式解：

| | Jacobian | 每一步 |
|---|---|---|
| Newton（log/Poisson） | `Z = [J_1, …, J_K]`，`J_k = ∂g_k/∂theta_k`；线性 term 为 `X_k` | `(Zᵀ diag(w·mu) Z + P) Δ = Zᵀ(w(y - mu)) - P·theta`，权重即 `newton_terms` 现在返回的 `hess_w` |
| BCD（identity） | `J_k = ∂f_k/∂theta_k`；线性 term 为 `X_k` | 对一个 block 解加权最小二乘：working response `z`、weights `W` 由 form 给出 |

对 BCD，两种 identity form 的 working response 和 weights：

| form | `z` | `W` | 现在的实现 |
|---|---|---|---|
| `IdentityMSE` | `y / (scalar · fixed)` | `w · (scalar · fixed)²` | `ols_normal_equation_scaled` |
| `IdentityPoisson` | `y / m` | `w · m / p` | `ols_normal_equation(X, y/m, weights=w·m/p)` |

线性 term 的一步 WLS 就是今天的 `block_step`；非线性 term 用同样的 `z`、`W` 做一步阻尼 GN：
`(Jᵀ W J + lambda·diag(Jᵀ W J) + P) Δ = Jᵀ W (z - f) - P·theta`。

### 6.2 NewtonSolver（log forms）

- `_LogDesign` 的 part 来自 term：
  - 线性、link 输出的 term 走现在的路径：basis 列，按 `pivot()` 删列（3.2 节；没有声明 reference 时即 9.3 节的自动选择），
    标准化，可缓存 float32。cross term 按 `drop` 固定的格子同样不进 design。
  - 非线性 term（以及 log link 下的 factor 输出 term）成为"动态 part"：每一遍在分块循环里重算 `J_g`，用 float64，
    不缓存；`pivot()` 给出的参数固定。
  - frozen term 进 offset（现状）。
- 步长：全是线性 term 时问题是凸的，保留现在的 Newton + step halving，结果不变。有动态 part 时改用 Levenberg-Marquardt 阻尼
  （`lambda` 成功时减小、失败时增大），Hessian 换成 Fisher 信息，所以收敛速度变成线性的，这需要写明。
- `_block_coefficients`：线性 part 的反标准化照旧，常数方向从"单列"推广为向量 `c`（`coef += intercept · c`）；
  动态 part 直接取 theta。归一化都按 3.1 节的表。
- `_factor_means`（history 用）：动态 part 在分块循环里调用 `term.value`。

### 6.3 BCDSolver（identity forms）

- 线性 term 保留现在的 `form.block_step` 和全部快速路径（QR 缓存、SVD、one-hot 分组求和）。
- form 新增 `working(y, scalar, fixed, f, w) -> (z, W)`，非线性 term 每个 sweep 做一步（或几步）阻尼 GN；只有 block loss
  下降才接受这一步，否则增大阻尼，保持 BCD 的单调性。
- 归一化改为 3.1 节的 `form.rescale(term, theta, mean)`；`solvers.py` 334、353、373 处的 `X @ coef` 改为 `term.value`。
- `ByTerm` 按 level 做同样的事，level 转移规则不变。

### 6.4 identity 下的联合求解（可选）

`LinearProductRegressorScipy.jacobian`（`regressor.py` 232–261）已经用前缀/后缀乘积算整个乘积模型的联合 Jacobian。
推广到 term：`∂mu/∂theta_k = global_scalar · prod_{j≠k} f_j · J_{f,k}`。如果非线性 term 在 BCD 里收敛慢，可以基于它加一个联合 GN solver。
在出现需要之前不做。

### 6.5 参数约束与固定

solver 在无约束坐标 `u` 上工作，`theta = T(u)`，`J_u = J_theta · T'(u)`：

- `(0, ∞)`：`theta = exp(u)`；`(a, b)`：`theta = a + (b - a) · sigmoid(u)`。
- fixed 参数和 `pivot()` 的参数从 `u` 里去掉。

这些逻辑放在 `Param` 列表上统一处理，各个 term 不用自己实现。

### 6.6 初值

- `ColumnsTerm` / `BasisTerm`：保持今天的 `init_betas_by_response_mean`，P1 要求数值不变。
- 新的线性 term（`TentTerm`、`CategoricalTerm`）：identity link 下取常数方向 `c`（factor 恒为 1；tent 就是全 1），log link 下取 0。
  Newton 本来就从 0 起步，只有 BCD 受影响。
- 参数化 term：用 `Param.value`。推荐的流程是先用 tent / FlatRamp 拟合出形状，用 `curve_fit` 把参数化形状拟合到这条曲线上得到初值，
  再联合微调。这就是 `ComponentManager` 现在的用法，只是最后一步从"冻结"变成"联合优化"。

### 6.7 性能

- 线性 term 的 design 和缓存不变，turnover 的内存占用不变。
- 动态 part 每一遍多算 `n × p_k` 的 Jacobian，`p_k` 通常只有 3–5，相对 design 可以忽略。用差分时每一遍要多做 `p_k + 1` 次 `value`，v1 可以接受。
- `TensorTerm` 的列数是各变量列数的乘积，design 的内存按比例增加（例如 6 × 6 个 knot 在 turnover 全量上约多 2.4 GB 的 float32）。
  tent 的张量积每行最多 `2^m` 个非零，以后可以考虑稀疏存储。

## 7. 用户接口、模型与 container

### 7.1 用户接口

P1 起的新接口。旧的 `fit(container, feature_groups=..., interactions=..., references=...)` 继续可用，内部转成 term（7.2 节）。

以 turnover 为例：

```python
from quantbullet.linear_product_model import LinearProductRegressor, ModelSpec
from quantbullet.linear_product_model.terms import TentTerm, CategoricalTerm, ByTerm, TensorTerm

# 1) 每个 term 独立声明：读哪一列、用什么 basis、reference 在哪里
age     = TentTerm("c_age_fit", knots=(1, 3, 6, 12, 18, 24, 36, 60, 84, 108, 120))
purpose = CategoricalTerm("f_purpose", reference="P")
state   = CategoricalTerm("f_state")                                   # 不给 reference：mean-1
ltv     = TentTerm("c_updated_ltv_fit", knots=(5, 20, 40, 60, 80, 95, 100), reference=80)
hpi     = TentTerm("c_hpi_ratio_fit", knots=(.8, .95, 1, 1.1, 1.25, 1.5, 1.75, 2), reference=1)
fico    = TentTerm("c_orig_fico_fit", knots=(620, 660, 700, 740, 780, 800),
                   reference=[740, 780])                               # 两个 reference：第二个是约束

# 2) 模型就是 term 的列表；cross term 引用上面的 term 对象，drop 必须显式写出
spec = ModelSpec([
    ByTerm(age, by=purpose, drop=("c_age_fit",)),
    purpose, state, ltv, hpi, fico,
    TensorTerm(ltv, hpi, drop="any"),
])

# 3) 先检查结构，再准备数据，最后拟合
spec.check()                    # 不需要数据：名字不重复、drop 写对、cross term 的 knots 是主效应 knots 的子集
data = spec.prepare(frame, response="y_full_prepay", weights="weight", dtype=np.float32)   # 学 level、建 basis，只做一次
model = LinearProductRegressor().fit(data, loss="poisson", block_link="log")
model.predict(new_frame)        # 模型带着 spec，直接接收原始列
```

`drop` 写错时，`spec.check()` 在读任何数据之前就报错，并给出正确的写法：

```text
TensorTerm(c_updated_ltv_fit × c_hpi_ratio_fit) has drop=("c_hpi_ratio_fit",), which leaves a duplicate block:
the part that varies only with c_updated_ltv_fit is also provided by TentTerm(c_updated_ltv_fit).
The correct rule for this model is drop="any".
```

拟合之后这样读结果：

```python
model.term("c_updated_ltv_fit").curve([60, 80, 95])      # 乘数，80 处为 1
model.term("c_updated_ltv_fit").table()                  # tent 为每个 knot 上的乘数，分类 term 为每个 level
model.term("c_updated_ltv_fit×c_hpi_ratio_fit").table()  # 节点表，reference 所在的行和列都为 1
model.term("c_age_fit").table()                          # ByTerm：每个 purpose level 一条 age 曲线
model.global_scalar_                                     # reference state 上的速度
model.coef_                                              # 兼容：{name: theta}
```

P3 起加上参数化 term：

```python
incentive = ParametricTerm("c_incentive_fit", shape=SigmoidModel,
                           params=[Param("k", 3., bounds=(0, np.inf)), Param("x0", -1.)])   # 参数名来自 shape
sato      = FrozenTerm("c_sato_fit", fitted_submodel)
```

**设计选择。**

1. **term 是声明式的值。** 构造时不碰数据，参数原样保存（sklearn 约定）。拟合结果放在模型上，由 `model.term(name)` 读取，
   spec 本身不会被 `fit` 改动，所以同一个 spec 可以分别用 identity 和 log 拟合再对比。
2. **`ModelSpec` 是单独的对象。** 它承担两件事：
   - 不需要数据就能做结构检查（drop 规则、名字、knots 是否为子集），这类检查本来就依赖完整的 term 集合（3.2 节）；
   - `prepare` 一次、多次拟合时复用，相当于 turnover 里现在 `make_container` 的角色。
   需要数据的检查（tent 的 knot 附近有没有数据、`ColumnsTerm` 的常数方向）在 `prepare` 时做。
3. **cross term 引用 term 对象，不重新写 knots 和 reference。** 主效应和交互项天然用同一套 knots 和 reference，即笔记 4.4 的建议。
   确实要用不同的 knots 时就传一个新的 term，由检查判断它是不是子集。
4. **`drop` 里写 term 名。** 名字的默认值见第 4 节。
5. **reference 写原始值。** 数值写 x，分类写 level 标签，多个就写成 list。
6. **输出尺度、惩罚也是 term 的参数**，例如 `TentTerm(..., output="factor")`、`penalty=1e-3`（P4）。

不先做公式字符串（类似 R 的 `age*purpose`）：knots、reference、drop 写进字符串后很快就难读也难检查。以后需要时，可以在对象 API 上再加一层。

### 7.2 模型与 container

- 模型以 `terms_`（有序 `{name: Term}`）为准；`feature_groups_`、`submodels_`、`interactions_` 变成派生视图，为兼容保留。
- `coef_` 仍为 `{name: theta}`（`ByTerm` 为 `InteractionCoef`）。`block_factor` 改为按 term 求值：`term.value`，再按输出尺度经 form 换算。
- `spec.prepare(frame, ...)` 返回带着 spec 的数据对象，内部是 `ProductModelDataContainer` 加上每个 term 的 `term_data` 缓存：
  `ColumnsTerm` 读现有的展开数组，其他 term 调用 `term.prepare(orig)`，按 `dtype` 转成 float32。不再需要 `expanded_df` 的 term 只用原始 frame。
- `fit(data, ...)`：`data` 是 `spec.prepare` 的结果，spec 从它身上取；或者是旧的 container 加 `feature_groups=`，
  兼容层构造 `ColumnsTerm` / `FrozenTerm` / `ByTerm(drop=("x",))`，数值结果和今天完全一样。两种入口二选一。
  拟合时模型保存 spec 的副本，`model.term(name)` 返回带拟合系数的视图（7.1 节）。
- `predict` 接受原始 DataFrame（用保存的 spec 来 `prepare`）、`spec.prepare` 的结果或旧的 container；展开列的 DataFrame 照旧可用。
- pickle：`LinearProductRegressor.__setstate__` 在没有 `terms_` 时，从 `feature_groups_`、`submodels_`、`interactions_` 重建。

## 8. 下游改动

- **toolkit**：单变量曲线改为 `term.prepare` + `term.value`，不再需要临时 container。implied actual 只用 `m`、`p` 和
  `form.implied_actual_terms`，不改。特征配置表改用 `term.describe()`，去掉 `isinstance` 分支。toolkit 只接收
  `ModelSpec`，取代 `FeatureSpec` + `preprocess_config`（待决问题 19，已定）。
- **`MortgageDiagnostics`**：只读预测，不改。
- **`fit_turnover`**：P1 继续走 `feature_groups` 路径。数值特征改用 tent 是独立的建模决定，需要单独的报告对比。
- **`ComponentManager`**：`ParametricModel` 子类成为 `ParametricTerm` 的形状；序列化 term 配置留到 P3 之后再考虑。

## 9. P0 的接口

P0 不引入 Term，只在现有 `feature_groups` 路径上加三样东西：tent basis、按 reference 归一化、Newton 识别常数方向。

tent 和 references 同时支持 identity 和 log 两种 link（2026-10-21 决定）。log 是 turnover 的默认：凸、预测恒为正、
implied actual 是闭式的 A/E；identity 保留"系数就是乘数、knot 之间乘数线性插值"，也便于和现有 identity 模型对比。
同时支持的额外代价只有 identity 的换算分支和 `f_ref > 0` 这一条检查。

### 9.1 TentTransformer

放在 `preprocessing/transformers.py`，和 `FlatRampTransformer` 并列，并加入 `preprocessing/__init__.py` 的懒加载列表。

继续用 Transformer 命名：

- P0 必须走这个接口：toolkit 的 `preprocess_config`、`get_single_feature_pred_given_values` 和 `fit_turnover.make_container`
  都按 sklearn 的 `fit` / `transform` / `get_feature_names_out` 调用它。
- 后缀标明层次：`*Transformer` 只把 x 变成列，不带建模语义；P1 的 `TentTerm` 在外面加 reference、penalty 和输出尺度。
- 类名按 basis 函数的形状取，和 `FlatRampTransformer` 对称。

```python
class TentTransformer(BaseEstimator, TransformerMixin):
    """Tent basis (degree-1 B-spline): one column per knot, rows sum to 1, coefficients are the curve's values at the knots."""

    def __init__(self, knots): ...
    def fit(self, X, y=None): ...
    def transform(self, X): ...
    def get_feature_names_out(self, input_features=None): ...
```

| 项 | 规则 |
|---|---|
| `knots` | 必填，严格递增，至少 2 个 |
| 端外 | 两端是平的：`x ≤ k_1` 时 `B_1 = 1`，`x ≥ k_K` 时 `B_K = 1`。这和 sklearn `SplineTransformer` 的默认 `extrapolation="constant"` 相同 |
| 输入 | 单列：`Series`、单列 `DataFrame` 或 `(n, 1)` 数组，同 FlatRamp |
| 输出 | `(n, K)` ndarray，默认 float64；`transform(X, dtype=np.float32)` 直接输出 float32，且每行精确和为 1（3.2 节） |
| 列名 | `{feature}_at_{knot}`，knot 用 `np.format_float_positional`（最短的精确写法）格式化，例如 `c_age_fit_at_24`、`c_orig_balance_real_fit_at_125000`。不用 FlatRamp 的 `numeric_to_string`：它对 ≥ 1000 的值用一位小数的科学计数法，125000 和 120000 会得到同一个列名 |
| 校验 | 参数原样保存，在 `fit` 里检查 `knots`（sklearn 约定，`clone` 可用；FlatRamp 在构造时校验，这里不照搬）。某一列在训练数据上全为 0（knot 附近没有数据）时报错，因为这一列的系数不可识别 |

刻意不提供的参数：

- `extrapolation`：只做平的外推（2026-10-21 决定）。以后需要线性外推时再加这个参数，默认值保持 `"constant"`，不影响现有调用。
- `include_bias`：每行和为 1，常数方向已经在 basis 里。
- `drop` / `reference`：删列在 identity link 下是约束（9.2 节），而 transformer 不知道 link，所以 reference 不放在 basis 层。
  sklearn `SplineTransformer` 的 `include_bias=False` 就是这种删法。
- `get_bins_and_labels`：FlatRamp 的这个方法没有调用方。

不直接用 sklearn 的 `SplineTransformer(degree=1, extrapolation="constant")`。它在数学上就是 tent（笔记的例子上与逐列 `np.interp`
构造的 tent 完全一致，sklearn 1.9.1），但列名是 `x_sp_0`，而 tent 的价值就在于系数可读。测试里用它做对照。

toolkit 的特征配置表（`toolkit.py` 744–764）增加 `TentTransformer` 分支，显示 knots。

`FlatRampTransformer` 保持现状，不做任何修改。

### 9.2 references

**为什么不删列。** 笔记里"删掉一列 = 把该 knot 上的系数固定为 0"是加法模型的规则。放到乘积模型里，结果取决于 link：

| link | 删掉 knot 24 的列 | 含义 |
|---|---|---|
| log | `theta_24 = 0`，乘数为 1 | 正确的 reference |
| identity | `theta_24 = 0`，乘数为 0 | 约束：age = 24 处的预测被强制为 0 |

**reference 是 gauge。** 只要 group 有常数方向，`f'(x) = f(x) / f(x_ref)`、`global_scalar' = global_scalar · f(x_ref)` 不改变预测，
也不改变拟合：BCD 每一步只依赖其他 block 与 scalar 的乘积，Newton 的 design 与归一化无关。所以 reference 不进 solver，
在 solver 结束后统一换算一次。

**接口。** 用户给"特征 → reference 值"的映射。P0 的模型只认识 basis 列，所以由 toolkit 先把值转成 basis 行：

```python
REFERENCES = {"c_age_fit": 24, "f_purpose": "P"}           # feature -> x value or level label
model.fit(container, feature_groups=..., interactions=...,
          references=toolkit.reference_rows(REFERENCES), ...)
```

- `toolkit.reference_rows(values) -> dict[str, np.ndarray]`：对每个特征，用它自己的 transformer transform 一行数据，得到长度等于
  该 group 列数的 basis 行；稀疏结果转成 dense；没有 transformer 的特征，行就是 `[value]`。对 tent、FlatRamp、OneHot 都适用。
  特征不在 toolkit 里时报错；level 不存在时由 transformer 报错（OneHot 的 `handle_unknown="error"`）。
- `fit(references=None)`：`{group: basis_row}`。不传时行为和数值都不变。
- P1(b) 起 toolkit 只接收 `ModelSpec`，这个函数移到拟合后的 spec 上：`ModelSpec.reference_rows`（例如 `model.spec_.reference_rows`）。

**语义。**

- 数值特征的 reference 是 x 值，可以不在 knot 上。取在 tent 的 knot 上时，该 knot 的系数正好是 1（identity）或 0（log），最好读。
- 分类特征的 reference 是 level 标签，该 level 的系数为 1 或 0。
- 没有指定 reference 的 group 照旧做 mean-1 归一化。`global_scalar_` 的含义是：指定了的特征取 reference 值、其余特征取平均水平时的
  预测。全部特征都指定时，它就是 reference state 上的预测。

**换算。** 记 `f_ref = form.factor(row, theta)`；interaction group 的每个 level `l` 记 `f_l = form.factor(row, theta_l)`。
两种 link 都通过 `form.rescale` 换算（`IdentityForm.rescale` 为此新增，即 `theta / mean`）：

| | identity | log |
|---|---|---|
| 普通 group | `theta / f_ref`，`global_scalar_ × f_ref` | `theta - log(f_ref) · c`，`global_scalar_ × f_ref` |
| interaction group | 曲线 `theta_l / f_l`；`by` group 中 level `l` 那一列的系数 `× f_l` | 曲线 `theta_l - log(f_l) · c`；`by` 列的系数 `+ log(f_l)` |

`c` 为 9.3 节检测到的常数方向。interaction group 的水平转入 `by` group，`global_scalar_` 不变。

- **顺序**：先换算 interaction group，再换算其他 group（含 `by` group）。这样 `{"c_age_fit": 24, "f_purpose": "P"}` 里 purpose 的
  reference 作用在已经接收了各 level 水平的 `by` 系数上，最后 `global_scalar_` 就是 age 24、purpose P 时的预测水平。
- **位置**：`LinearProductRegressor.fit` 在 `solver.run` 之后调用 `_apply_references`，换算后重新计算 `block_means_`
  （`_compute_block_means` 现在在 solver 里调用，换算后要再调一次，或者把它移到 `fit`）。
  label 到 `by` 列的对应关系用 BCD 的 `_interaction_level_columns` 的同一套规则从数据求出。
- **history**：`coef_history_`、`interaction_params_history_`、`global_scalar_history_`、`best_params_`、`best_interaction_params_`
  逐步用同样的规则换算（每一步按该步自己的 theta 求 `f_ref`），保持 `global_scalar_ == global_scalar_history_[best_iteration_]`。
  `global_scalar_step_history_` 记录的是 sweep 内部的过程，没有对应的系数，保持拟合时的 mean-1 口径。
- **记录**：`references_ = {group: row}`；`fit_metadata_` 记下指定了 reference 的 group。

**报错。** 能提前检查的都在 `fit` 开始、模型状态重置之前检查：

| 情况 | 时机 |
|---|---|
| key 不在 `feature_groups` 里 | fit 开始 |
| group 是 frozen submodel（在 `submodels_` 里） | fit 开始 |
| basis 行的长度不等于 group 的列数，或含 NaN | fit 开始 |
| log link 下 group 没有常数方向（例如不带 bias 的 FlatRamp） | fit 开始，按 9.3 节检测 |
| interaction group 的 `by` block 不是可训练的完整 one-hot，无法接收各 level 的水平 | fit 开始 |
| identity link 下 `f_ref ≤ 0`，或某个 level 的 `f_l ≤ 0` | 拟合结束后 |

identity link 不需要常数方向：`theta / f_ref` 对任何线性 block 都成立。

**P1。** reference 改由 term 声明（3.2 节）：`TentTerm(..., reference=24)` 或多个 knot、`CategoricalTerm(..., reference="P")`。
落在单个系数上时，Newton 直接用它当 pivot；其余情况调用本节同一个换算函数。`fit(references=...)`（按 basis 行）继续作为
`feature_groups` 路径的接口。

### 9.3 Newton：常数方向

这里选出的是 3.2 节的 pivot，不是用户的 reference：它只是求解时固定的一列，`l2 = 0` 时不影响任何结果。P0 里用户的 reference
一律走 9.2 节的 gauge；P1 起，用户声明的 reference 落在单个系数上时直接当作 pivot，不再自动选。

- `_constant_column(X, chunk_rows)` 推广为 `_constant_direction(X, w, chunk_rows) -> (c, pivot)`，按顺序检查：
  1. 有一列恒为 1：`c` 为这一列的单位向量，pivot 为这一列（现状，FlatRamp + bias）。
  2. 否则每行和为 1：`c` 为全 1，pivot 为加权列和最大的列（同分类 block 选 reference level 的规则）。
     float32 的 design 下行和有舍入误差，容差按 float32 定（例如 1e-6）。
  3. 否则返回 `(None, None)`。

  和现在一样逐块检查，不生成 n × p 的临时数组。完整 one-hot 的 block 先走分类路径，不受影响。
- `_Part.constant` 从列下标改为方向向量 `c`；`columns` 为去掉 pivot 列后的列，numeric 和 interaction part 相同。
- `_block_coefficients`：`coef[part.constant] += intercept` 改为 `coef += intercept * c`；`form.rescale(coef, mean, c)`
  本来就接受方向向量。
- `_factor_means` 的筛选条件（有常数方向的非分类 part）不变；分类 part 不变。
- 9.2 节 log link 下的检查和换算复用 `_constant_direction`。BCD 不需要它。

### 9.4 P0 测试

- `TentTransformer`：
  - 每行和为 1；`f(k_j) = theta_j`；端外是平的；列名；
  - 与 `SplineTransformer(degree=1, knots=..., extrapolation="constant")` 一致；
  - knots 不合法、某列没有数据时报错；`clone` 可用。
- 等价：数据截断到 `[lo, hi]` 时，tent knots `(lo, *k, hi)` 与 `FlatRamp(k, include_bias=True)`：
  - log/Newton（凸，`l2 = 0`）：预测一致到 1e-10。测试用 float64 Hessian；默认的 float32 Hessian 在相对误差约 1e-8 处就停下；
  - identity/BCD：两边初值不同，比较收敛后的 loss 和预测，容差按 `ftol` 定；
  - 含 interaction（`c_age × f_purpose` 的形式）时同样成立。
- Newton + tent：design 列数和 FlatRamp 相同（去掉了一列），block 均值为 1；`_constant_direction` 的单元测试覆盖 bias、
  float32 的 tent 和没有常数方向的 block。
- references：
  - 三种 form 下换算前后的预测一致，含 interaction；
  - tent 在 reference knot 上的系数为 1（identity）或 0（log），分类变量的 reference level 同理；
  - 全部特征都指定时，`global_scalar_` 等于 reference state 上的预测；
  - `global_scalar_ == global_scalar_history_[best_iteration_]`；
  - 9.2 节报错表里的每一种情况；
  - `toolkit.reference_rows` 对 tent、FlatRamp、OneHot 返回正确的行。
- 现有测试通过；不传 `references` 时数值不变；FlatRamp 路径数值不变。

## 10. 分期与验收

| 期 | 内容 | 验收 |
|---|---|---|
| P0 | 第 9 节：`TentTransformer`；`fit(references=...)` 与 `toolkit.reference_rows`；Newton 识别常数方向（含 interaction）；toolkit 报告的 tent 分支 | 9.4 节 |
| P1 | Term 协议与线性 term（`ColumnsTerm`、`BasisTerm`、`TentTerm`、`CategoricalTerm`、`FrozenTerm`、`ByTerm`）；`ModelSpec`（`check`、`prepare`）与 `model.term(name)`（7.1 节）；spec 编译成 solver 现有的 block（feature groups、interactions、frozen submodels、references），solver 不动；toolkit 只接收 `ModelSpec`，turnover 随之迁移；3.2 节的 reference（单个、多个合并加 gauge）、Newton 用 reference 当 pivot、cross term 的 `drop` 检查（先用于 `ByTerm`） | spec 路径和 container 路径的 `coef_`、预测、`loss_history_` 完全一致；turnover `--smoke-rows 100000` 的 metrics 一致；现有 `turnover_model.pkl` 可加载，预测一致。另见下方 P1 测试 |
| P2 | `TensorTerm`（连续 × 连续）；`drop` 检查推广到多个连续变量；knots 子集检查（笔记 4.4） | 见下方 P2 测试 |
| P3 | `Param` 与重参数化；`ParametricTerm`（差分 Jacobian）；BCD 的阻尼 GN 步；Newton 的动态 part + LM；`pivot()` 与归一化；从曲线拟合初值 | 见下方 P3 测试 |
| P4 | tent 的二阶差分惩罚与单调重参数化；identity 联合 GN solver；自由 knot | 按需，各自测试 |

P1 分三个提交：

- (a) term 类、`ModelSpec`（`check`、`prepare`、`transform`）、`fit(spec.prepare(...))` 与 `predict(原始 frame)`、`ByTerm` 的 `drop` 检查、
  单个 reference（编译成 P0 的 `fit(references=...)`）。已实施，测试在 `tests/linear_product_model/test_terms.py`。
- (b) toolkit 只接收 `ModelSpec`、`model.term(name)` 的读取接口（`coef`、`curve`、`table`）、turnover 迁移到 spec。已实施：
  toolkit 只用 term 的名字和类型（`CategoricalTerm` 为分类，其余为数值），报告用 `term.describe()`；`reference_rows` 移到
  `ModelSpec`；turnover 用 `BasisTerm(FlatRamp)` 和 `CategoricalTerm`，拟合与迁移前完全相同，`spec_` 之前保存的 model 需要重新 fit 才能出报告。
- (c) 多个 reference（合并列加 gauge）、Newton 用 reference 当 pivot。已实施：pivot 记进 `convergence_info_['pivots']`；
  `TentTransformer` 的行在 float32 下也精确和为 1；reference 的换算按 `feature_groups_` 的顺序进行，与传入顺序无关。

原计划里"solver 改为通过 term 求值"一步取消（2026-10-21）：P1 只有线性 term，spec 编译成 solver 现有的 block 就够了，数值结果按构造与
container 路径一致。让 solver 通过 term 求值、取 Jacobian 是 P3 参数化 term 才需要的，放到 P3。

P1 测试（reference 与 `drop`）：

- log 下声明了 reference 的 tent：pivot 就是 reference 列（`convergence_info_` 里能看到），reference 上的系数正好为 0，
  预测和 P0 的 `fit(references=...)` 一致。
- 多个 reference：log 下合并列的拟合和"直接删掉这些列"的模型一致；identity 下各 reference knot 上的乘数都是 1，
  拟合和手工构造的合并列 design 一致。
- `ByTerm` 不写 `drop` 报错；写错（多删、少删、`S` 里的变量没有 reference）报错，信息里给出正确写法；
  旧接口 `interactions={"age": "purpose"}` 构造出的 `drop=("age",)` 通过检查，结果和今天一致。

P2 测试：

- 3.2 节表里的四种模型结构：只有表中的写法通过检查，其他写法都按"共线"或"约束"报错。
- `1 + x + y + x×y`（log）的拟合和手工删掉 reference 行、列（以及主效应 reference）的 design 一致到 1e-10。
- cross term 的 knots 不是主效应 knots 的子集时发出警告。
- 已实施，测试在 `tests/linear_product_model/test_terms.py`，另外还验证了：张量积的 `table()` 在 reference 行、列上为 1（两种 link）；
  `TensorTerm(purpose, age, drop=("age",))` 和 `ByTerm(age, by=purpose, drop=("age",))` 的拟合一致；toolkit 的一维 implied actual
  跳过张量积；`leave_out × single` 等于完整预测。

P3 测试：

- 用已知的参数化乘积（logistic × tent × categorical）生成合成数据，两种 form 都能在容差内恢复参数。
- 能精确表示线性 term 的参数化 term（如 `g = a + b·x`）和对应线性 term 的拟合一致。
- 非线性 term 的 block 步不会让 loss 上升（6.3 节的接受规则）；Newton 带动态 part 能收敛。
- 所有参数 fixed 的 `ParametricTerm` 与 `FrozenTerm` 结果一致。

## 11. 风险

- P1 改动面大（模型状态、container、toolkit、solver）：用逐期、逐提交的等价检查控制。
- pickle 兼容：`__setstate__` 兜底，并用现有 `turnover_model.pkl` 做加载测试。
- 非凸：P3 的结果依赖初值。需要写清楚初值流程，并在 `convergence_info_` 里按 term 记录收敛情况。
- 可识别性错误不会自己暴露：靠 3.1 节的声明校验和警告，以及 3.2 节的 `drop` 检查。
- `drop` 检查依赖模型里的 term 集合（frozen term 算作不在模型里）：检查必须在所有 term 都确定之后、fit 开始时做，不能在构造 term 时做。
- 范围膨胀：P3 等到 turnover 里出现具体需要联合拟合的参数形状再做，P4 各项各自按需。

## 12. 待决问题

1. **term 的输出尺度**：只允许 link 尺度，还是 link 和 factor 两种（2.1 节）。建议两种：换算集中在 form，代价小，
   而且能表达乘数尺度的 tent 和参数化乘数形状。
2. **归一化口径**：已定，mean-1 为默认，`references` 按 gauge 换算（9.2 节）。剩下的问题是报告怎样展示指定了 reference 的 group
   （它们的 `block_means_ ≠ 1`），以及 implied actual 图是否标出 reference 点。
3. **term 定义放在哪里**：已有设想，由 `ModelSpec` 持有 term，`prepare` 后交给模型，模型保存一份副本（7.1 节）。
   toolkit 的角色见第 19 条。
4. **`feature_groups=` 路径**：长期作为兼容层保留，还是 turnover 迁移后废弃。
5. **`ByTerm` 的 theta 存储**：继续用 `InteractionCoef`，还是统一成向量加结构信息。
6. **identity 下的非线性求解**：在 BCD 里做阻尼 GN 步（建议先这样），还是新增联合 GN solver（6.4 节）。
7. **Jacobian**：P3 先用差分，库里的形状逐个补解析式，还是一开始就要求解析式。
8. **P0 的常数方向**：已定，P0 按数据检测两种情况（9.3 节）；P1 起改为由 term 声明、按数据校验。
9. **turnover 是否改用 tent、设哪些 reference**：这是独立的建模决定。在 P0 的等价验证之后，单独做一次报告对比再决定。
10. **`extrapolation="linear"`**：已定，不做，tent 只做平的外推（9.1 节）。
11. **类名**：已定，用 `TentTransformer`；`FlatRampTransformer` 保持现状。
12. **pivot 与 reference**：已定，两者分开；声明了 reference 时 Newton 用它当 pivot，多个 reference 合并列再做 gauge（3.2 节）。
13. **cross term 的删除规则**：已定，没有默认值，必须显式写 `drop`，按笔记 4.2.3 检查，写错就报错（3.2 节）。
14. **有意的约束**：按 13，多删（引入约束）也会报错。如果以后确实需要有意的约束（例如某个 level 上不随 x 变化），
    是加一个显式开关，还是用多个 reference 表达。
15. **不在 knot 上的多个 reference**：v1 要求多个 reference 都落在 knot 或 level 上；`f(x_a) = f(x_b)` 这类一般线性约束是否需要支持。
16. **`ByTerm` 与 `TensorTerm`**：`ByTerm` 是否最终并入 `TensorTerm`（分类 × 连续也存成节点表），还是为了 `InteractionCoef` 的兼容一直分开。
17. **类名风格**：已定，用 `TentTerm`、`CategoricalTerm` 这样的全名（2026-10-21）。
18. **`ModelSpec` 的序列化**（`to_dict` / `from_dict`）：已定，以后再做（2026-10-21）。等 turnover 真要从 TOML 读 spec 时再加，
    届时和 `ComponentManager` 的序列化（第 8 节）一起考虑。
19. **toolkit 的角色**：已定，toolkit 只接收 `ModelSpec`，取代现在的 `FeatureSpec` + `preprocess_config`，报告用 `term.describe()`；
    turnover 在 P1(b) 随之迁移（2026-10-21）。
20. **张量积的诊断图**：一维 implied actual 跳过张量积。是否加二维的 implied actual（例如节点表上的 A/E 热图），放在哪一期。
