# Design：one-hot、tent 与 cross 的加法设计矩阵

`Design` 把一个数据框写成加法设计矩阵 Z，用于 `eta = Z @ beta`，可以交给 GLM、log link 或任何线性模型。
矩阵总是按写法生成；不满秩、或者给模型加了约束的地方，会记成 violation 并给出一条警告（见[检查与 violation](#检查与-violation)）。

- 分类列用 one-hot，数值列用 tent basis。
- 可以加任意阶、任意组合的 cross（cat×cont、cont×cont、cat×cont×cont……）。
- 它和乘积模型的 `ModelSpec` 是两个独立的工具：ModelSpec 把 term 编成 block，由 solver 逐块优化；Design 只生成矩阵。

可以直接运行的用法示例在 [examples/preprocessing/test_design_examples.py](../../examples/preprocessing/test_design_examples.py)：
用假的 mortgage 数据（`quantbullet.utils.data.generate_fake_loan_months`），每个用法一个 test。每个 example 把原始列和它生成的矩阵的前几行
写到 `examples/preprocessing/output/design/design_examples.xlsx` 里自己的 sheet。

[examples/preprocessing/test_design_reconciliation_examples.py](../../examples/preprocessing/test_design_reconciliation_examples.py)
把 `Design` 和另一种实现（每个 spec 是 (变量, knots 或 levels, 1-based reference) 的序列，factor 逐格相乘）逐格对账：
tent 的值只差舍入误差（最后一位），drop 规则一致。那种实现没有 intercept、cross 不带主效应、只 cross 一个 level、一个 factor 有两个
reference；`Design` 都能照写（后两种用 reference 列表），并把它们加的约束记成 violation，所以每个 spec 都逐格对上。

所有选择都要显式写出，没有默认值：

| 参数 | 写在哪里 |
|---|---|
| `reference=` | 每个编码 |
| `drop=` | 每个 cross |
| `intercept=` | Design |

```python
from quantbullet.preprocessing import Cross, Design, OneHot, Tent

purpose   = OneHot("purpose", reference="P")
fico      = Tent("fico", (620, 680, 740, 800), reference=680)
age       = Tent("age", (0, 12, 24, 60, 120), reference=24)
age_cross = Tent("age", (0, 24, 60, 120), reference=24)          # cross 用更粗的 knots
incentive = Tent("incentive", (-3, -1, 0, 1, 3), reference=0)

design = Design([
    purpose, fico, age, incentive,
    Cross(age_cross, incentive, drop=["age", "incentive"]),
    Cross(purpose, age_cross, incentive, drop=["purpose", ("age", "incentive")]),
], intercept=True)
Z = design.fit_transform(df)          # 带列名，index 同 df；这个设计满秩，没有 violation
Z_new = design.transform(new_df)      # 新数据，沿用学到的 levels
design.groups()                       # {term 名: 列名}，有 intercept 时排在最前
```

## 编码与 reference

| 编码 | 列 | 系数的含义 | reference |
|---|---|---|---|
| `Tent(column, knots, *, reference)` | 每个 knot 一列，`{column}_at_{knot}` | 曲线在该 knot 上的值；knot 之间线性，两端之外取端点值 | knots 之一、它们的列表，或 `None` |
| `OneHot(column, *, reference, levels=None)` | 每个 level 一列，`{column}_{level}` | 该 level 的值 | levels 之一、它们的列表，或 `None` |

- reference 就是 Design 删掉的那一列：`reference=24` 删 `age_at_24`，`reference="P"` 删 `purpose_P`。在 log link 下，这个点的乘数是 1。
- `reference=None` 表示保留全部列，常数由这个编码承担（见下一节）。
- reference 也可以是一个列表，每个值删掉一列。系数相对**第一个** reference 读；后面的每一个都是约束，记一条 `constraint` violation：
  - `Tent("upb", knots, reference=[250_000, 400_000])`：曲线在 400K 也是 0；
  - `OneHot("purpose", reference=["C", "P"])`：P 和 C 取同一个值。放在 cross 里，就是只 cross 一个 level（这里只有 N 有自己的格）。
  主效应仍写 `reference="C"` 时，两处的第一个 reference 相同，所以不算"reference 不同"。
- 不写 `reference` 会报 `TypeError`；写了但不在 knots 或 levels 里、列表里有重复、或列表包含全部 knots 或 levels，也会报错。
- `OneHot` 的 levels 默认取 `fit` 时数据里排好序的 levels。缺失值、以及 `transform` 时出现的新 level 都会报错。
- `fit` 时 tent 的每个 knot 附近都必须有数据（见 `TentTransformer`）。

## 常数只出现一次：`intercept=`

tent 和 one-hot 的每一行加起来都等于 1，所以一个编码的全部列相加就是常数列。常数在 Z 里只能由一个 term 承担：

| 承担常数的 | 写法 | 其余编码 |
|---|---|---|
| intercept | `intercept=True`，Z 的第一列是 `intercept` | 都要写 reference |
| 某一个主效应 | `intercept=False`，这个编码写 `reference=None`，保留全部列 | 都要写 reference |
| 一个 cross | `intercept=False`，它的 drop 保留常数部分 | 都要写 reference |

没有任何 term 承担常数，或者有不止一个 term 承担，都会记一条 violation，并列出是哪些 term：
- 不止一个：设计共线；
- 一个都没有：等于加了一个约束，所有 term 都在 reference 上时 eta 为 0。用 offset 承担水平的模型就是这样，这时可以有意这样写。

只有一个 feature 时：

```python
Design([Tent("age", (0, 12, 24, 60, 120), reference=None)], intercept=False)
# 5 列，满秩；log link 下系数就是各 knot 上的 log 值
```

它和 `intercept=True` 加 `reference=24` 是同一个模型，只是参数化不同：那种写法的系数是相对 age=24 的差，这种写法的系数是各 knot 上的值。

## Cross 和 `drop`

`Cross(*encodings, drop=...)` 把几个编码的 basis 相乘，每个 knot 或 level 的组合（格）占一列。列名形如
`age_at_60_x_incentive_at_-1`，term 名是 `age_x_incentive`。

`drop` 是一个条件列表：
- 每个条件是一个列名，或一个列名的 tuple，意思是"这些列同时在 reference 上"（reference 是列表时，在其中任何一个上）；
- 满足任何一个条件的格都会被删掉；
- 条件里写到的列必须有 reference。

例如 age 的 reference 是 24、incentive 的是 0，两个主效应都在，写 `drop=["age", "incentive"]`：

```
                 incentive
age        -3    -1     0     1     3
  0         ·     ·     ✗     ·     ·
 12         ·     ·     ✗     ·     ·
 24         ✗     ✗     ✗     ✗     ✗
 60         ·     ·     ✗     ·     ·
120         ·     ·     ✗     ·     ·
```

- age=24 这一行要删：如果 cross 在这一行可以变化，它就是"age 在 24 时 incentive 的曲线"，而 incentive 主效应已经是这条曲线。
- incentive=0 这一列要删：同理，age 主效应已经提供了它。

**怎么写 drop。** 设计里已有的、只涉及 cross 部分列的每一个效应，都对应一条条件：cross 里不属于这个效应的列，同时在
reference 上。这些效应包括：
- 常数（intercept，或承担常数的那个 term）；
- 各主效应；
- 更低阶的 cross。

| 设计里已有的 | `age_x_incentive` 的 drop |
|---|---|
| 常数 + age + incentive | `["age", "incentive"]`，删 9 格 |
| 只有常数 | `[("age", "incentive")]`，只删 (24, 0) 一格 |
| 常数 + age | `["incentive"]`，删 incentive=0 这一列 |
| 什么都没有（`intercept=False`，只有这个 cross） | `[]`，完整的张量积；这时编码可以写 `reference=None` |

主效应也服从同一条规则，可以看成只有一列的 cross：常数由别的 term 承担时，常数给出的条件是"age 在 reference 上"，也就是删掉 reference 那一列。

三阶也一样。例如 `purpose_x_age_x_incentive`，设计里已有常数、三个主效应和 `age_x_incentive`：

| 已有的效应 | 条件 |
|---|---|
| 常数 | `("purpose", "age", "incentive")` |
| purpose | `("age", "incentive")` |
| age | `("purpose", "incentive")` |
| incentive | `("purpose", "age")` |
| `age_x_incentive` | `"purpose"` |

五条可以全部照写，也可以只写 `["purpose", ("age", "incentive")]`，因为其余三条已被 `"purpose"` 覆盖。两种写法删掉的格相同，都能通过检查。

再如三个主效应加三阶交互、不要两两交互：`drop=[("age", "incentive"), ("age", "ltv"), ("incentive", "ltv")]`。

**检查。** `Design.check()` 不需要数据，`fit` 会调用它。`drop` 删掉的应当恰好是其他 term 已经提供的部分：
- 多留了，设计共线；
- 多删了，等于多加了一个约束。

不符合时记一条 violation，消息会逐条列出上面那张表，并给出这个设计的正确写法：

```
Cross('age_x_incentive') has drop=['age']: it keeps the part that varies only with age, also provided by Tent('age'), so the design is collinear.
Each effect on its columns gives one condition, the cross's other columns all at their references:
  the intercept: ('age', 'incentive')
  Tent('age'): 'incentive'
  Tent('incentive'): 'age'
so for this design drop=['age', 'incentive']; conditions that others imply may be left out
```

`drop` 必须是 list。传 tuple 会报错，因为 `("age", "incentive")` 既可以读成两个条件，也可以读成一个条件。

## Cross 自己的 knots 和 reference

cross 接收编码对象，有两种用法：
- 传主效应的**同一个对象**，就共用它的 knots、levels 和 reference，`fit` 只做一次；
- 传**另一个对象**，就用它自己的设置。

`drop` 里写的是列名，和用的是哪个对象无关。

| 情形 | 结果 |
|---|---|
| cross 的 knots 是主效应 knots 的子集（更粗） | 允许。交互项更平滑，模型随之改变 |
| cross 的 knots 有主效应没有的 knot | violation（`knots`）。那个拐点只会在其他变量的 reference 那一行被压掉，所以拟合会随 cross 的 reference 放在哪里而变化 |
| 同一列在不同 term 里的 reference 不同 | violation（`references`）。张成的模型相同，但主效应的读法变了：incentive 主效应变成"age 在 cross 的 reference 上时的 incentive 效应" |
| 同一列在一处是 Tent、另一处是 OneHot | violation（`encoding`） |

如果 cross 依赖一个更低阶的 cross 提供某一部分，规则相同：它的 knots 必须是那个低阶 cross 的 knots 的子集。

## 检查与 violation

`Design` 只对生成不了矩阵、或者矩阵有歧义的写法报错：knots 不递增、reference 不在 knots 或 levels 里、数据有缺失或未知的 level、
`drop` 不是 list 或写了 cross 没有的列（或没有 reference 的列）、两个 term 同名或叫 `intercept`。

其余的问题都是建模选择，矩阵照样按写法生成，由用户负责。`check()` 返回一个 `Violation` 列表，`fit` 把它存在 `design.violations_`，
有 violation 时给出**一条** `DesignWarning`，列出每条的种类和位置；完整的消息在 `violations_` 里：

| `kind` | 含义 |
|---|---|
| `collinear` | 有列重复了别的列已经张成的部分：常数有多个承担者、cross 多留了、cross 什么都没加 |
| `constraint` | 删掉了没有别的 term 提供的部分：没有常数、cross 多删了、reference 列表里第一个之后的值 |
| `hierarchy` | cross 没有合适的 drop：别的 term 提供了某一部分，却没有提供它下面的部分 |
| `knots` | cross 的 knots 不是它所依赖的 term 的 knots 的子集 |
| `references` | 同一列在不同 term 里的第一个 reference 不同（模型相同，读法不同） |
| `encoding` | 同一列在一处是 Tent、另一处是 OneHot |
| `levels` | 同一列在不同 term 里的 levels 不同 |
| `no data` | `fit_transform` 时有的列没有数据，系数无法识别 |

想要严格检查（例如测试或生产流程里）时，把这类警告变成错误：

```python
import warnings
from quantbullet.preprocessing import DesignWarning

warnings.simplefilter("error", DesignWarning)
```

## 其他行为

- `fit` 在 term 的深拷贝上学习，传进去的对象不会改变。拟合后的 term 在 `design.terms_` 里。
- `transform` 的 `dtype` 默认是 `float64`，也可以传 `np.float32`。
- `fit_transform` 发现训练数据里没有数据的列时（通常是 cross 的空格）记一条 `no data` violation，因为这些列的系数无法识别。
  `transform` 不做这个检查，因为新数据里出现空格是正常的。

## 代码

| 文件 | 内容 |
|---|---|
| `preprocessing/design.py` | `Tent`、`OneHot`、`Cross`、`Design`、`Violation`、`DesignWarning` |
| `tests/preprocessing/test_design.py` | 满秩与张成、常数只出现一次、drop 的检查与 violation、cross 的 knots 和 reference |
| `examples/preprocessing/test_design_examples.py` | 用法示例：分类列、连续列、cat × cat 和 cont × cont 的 cross |
| `examples/preprocessing/test_design_reconciliation_examples.py` | 和另一种实现逐格对账 |
