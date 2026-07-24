# GAM Adapter Plan

## Goal

让 fitted GAM transition models 可以接进 framework：coefficient dump / partial-dependence payload -> `GAMReplayModel`-compatible runtime object -> callable edge value -> transition model。

核心约束（用户明确要求）：**roll-rate 的 coefficient 格式只是第一个/测试用格式**。未来会有其他 GAM coefficient 的组织方式进来。所以格式解析必须是 adapter 层的事，framework 标准化的是：

```text
GAMReplayModel-compatible runtime object
+ callable 接口 FeatureContext -> float
```

新格式进来只加 parser / converter，`GAMReplayModel` 求值器、transition assembly、simulator 尽量不动。

## Existing Quantbullet GAM Replay

`src/quantbullet/model/gam_replay.py` 已经有 `GAMReplayModel`，它能从 `GAMTermData` partial dependence payload 重放 GAM prediction：

- `SplineTermData`: 1D spline curve replay
- `SplineByGroupTermData`: smooth by categorical group
- `TensorTermData`: 2D tensor interpolation
- `FactorTermData`: categorical lookup
- `predict(X: pd.DataFrame) -> np.ndarray`
- `decompose(X)` for debugging / contribution inspection

这比另起一个 `AdditiveGAMModel` 更完整，也更符合 repo 现有设计。下一阶段应该尽量复用并增强 `GAMReplayModel`，而不是在 `loan_simulation` 里维护第二套 GAM evaluator。

两个已知差异要在 spike / tie-out 时处理：

- **插值方法**：`GAMReplayModel` 的 spline replay 用 PCHIP（`make_monotone_predictor_pchip`），roll-rate `Smooth1D.eval` 用线性插值。grid 点上相等，点之间有小的系统性差异；tie-out 要么用容差吸收，要么给 replay 增加 linear interpolation 选项。
- **依赖链**：`gam_replay` -> `model.gam.__init__` -> `wrapper.py` -> `pygam`，而 pygam 不在 `pyproject.toml` 依赖里。parser 必须在函数内 lazy import，避免 `import quantbullet.loan_simulation` 硬依赖 pygam。

## Roll-Rate Coefficient Format（第一个 adapter 的输入）

每个 `from_status` 一个 TSV 文件（`fromC.txt` 等），六列：

```text
model    var_name1    var_val1    var_name2    var_val2    value
```

- `model` 列是 `to_status`，一个文件里放该 from_status 的所有 edge models。
- 行类型：
  - intercept: `var_name1 = intercept`
  - categorical level: `purpose | Business | -> coef`（treatment coding，reference level 不出现在表里）
  - smooth grid point: `opti | 0.001 | v_opti | -> value`。spline 已被 R 侧预求值成稠密 `(x, f(x))` grid，引擎只做 clamp + 线性插值，不需要 mgcv basis functions。
  - `var_name2` 位置的 `v_*` 是 validity / multiplier feature：smooth 贡献 = `interp(x) * multiplier`，数据缺失时 multiplier=0 把整个 term 归零。（注意：这个乘法语义是从 roll-rate 代码注释和调用形态推断的，`SmoothByNum.eval` 的实现体未逐行核对；Step 4 tie-out 时对照验证，不一致则修正 term 定义。）

关于 multiplier feature 的语义边界：

- 它是 **prediction-time 的 term contribution 乘子**，等价于 mgcv 里 numeric `by=` 的 varying-coefficient smooth（roll-rate 内部对应 `SmoothByNum`）。`v_*` 的 0/1 validity flag 只是它的特例，连续缩放同样支持。
- 它**不是 training sample weight**（那是 fit 时的 observation 权重）。
- 它也**不是 roll-rate 的 dials**：dials 是 probability-space 的 post-model 调整，作用在 transition probabilities 上，仍然 deferred。
- 落地到 `model.gam` 时字段命名（沿用现有 `by_feature` 约定 vs `multiplier_feature`）由 Step 1 spike 结合现有 schema 决定；`SplineByGroupTermData` 已经把 `by_feature` 用于 factor-by 情形。

roll-rate 这个具体格式里，一个 edge 的 scalar 是 logit：

```text
z = intercept + Σ categorical lookups + Σ spline interpolations (× optional flag)
```

因此 roll-rate adapter 会把 replay model 的 output 解释为 logit，并喂给 `SoftmaxTransitionModel`。其他 GAM 格式如果输出 probability 或 multiplier，由 caller 选择不同 transition model / wrapper 来解释。

## Layered Design

```text
Layer 1: format parsers / converters（每种来源一个）
  parse_rollrate_coefficients(path) -> dict[to_status, GAMReplayModel-compatible object]
  future: parse_xxx_coefficients(...) -> 相同输出

Layer 2: GAMReplayModel-compatible runtime evaluator
  predict(one_row_dataframe) -> scalar
  decompose(one_row_dataframe) -> optional debug details

Layer 3: bridge（不新增类）
  caller 自己声明 scalar 语义，把模型包成 callable：
  lambda ctx: model.predict(build_feature_frame(ctx))[0]
  然后喂 CompositeTransitionModel / SoftmaxTransitionModel / ProbabilitySoftmaxTransitionModel
```

Layer 2 不依赖 `FeatureContext` / simulator，可以独立测试，也可复用到 simulation 之外。

## Output Semantics

GAM replay object 只返回一个 scalar。它不知道这个 scalar 是 logit、probability、hazard 还是 multiplier。caller 必须显式选择解释方式：

- scalar 是最终 edge probability，且 row-compatible：放进 `CompositeTransitionModel`。
- scalar 是 independently trained binary probability：放进 `ProbabilitySoftmaxTransitionModel`。
- scalar 是 logit / score：放进 `SoftmaxTransitionModel`。
- scalar 是 multiplier：在 wrapper callable 里和 base curve 相乘，再选择 `CompositeTransitionModel` 或 `ProbabilitySoftmaxTransitionModel`。

这个设计避免为 probability-GAM / logit-GAM / multiplier-GAM 重复定义三套 term classes。

## V1 Decisions

1. **优先复用 `GAMReplayModel`**。不新增 `AdditiveGAMModel` / `CategoricalGAMTerm` / `SplineGridGAMTerm` 等平行 evaluator。
2. **roll-rate parser 的目标是产出 `GAMReplayModel` 或 replay-compatible object**，而不是产出 loan-simulation 专用 term classes。
3. **如果 roll-rate `v_*` multiplier smooth 无法用现有 `GAMTermData` 表达，优先增强 `model.gam` 的 replay capability**（例如新增 numeric-by / multiplier spline term 或 replay adapter），不要在 `loan_simulation` 里偷偷另开一套。
4. **Categorical 未知 level 贡献 0**。这是 treatment coding 的语义，也是 `GAMReplayModel` 当前 `FactorTermData` lookup 的行为。Known limitation：dump 里没有完整 level 集合，拼写错误 level 和 reference level 无法区分，靠 tie-out 和上游数据质量把关。
5. **缺失 required feature -> 报错**。这和 `GAMReplayModel.decompose(...)` 当前行为一致。missingness 应该由 feature builder 显式编码，例如给 `v_* = 0`。
6. **Parser 遇到不认识的行结构 -> fail fast**，报错里带行内容。不假装覆盖所有 roll-rate term 变体；tie-out 撞到再扩展。
7. **不做 static/dynamic term 缓存**。先用 existing replay evaluator 每期求值，性能问题以后统一处理。已知代价：每 edge 每期构造 one-row DataFrame 调 `predict(...)`，比 dict lookup 重不少；tie-out 规模没问题，大 portfolio simulation 时再统一优化（batch evaluation / static logit cache）。
8. **不新增 bridge 类**。`lambda ctx: replay.predict(build_feature_frame(ctx))[0]` 一行就够，文档写清 pattern 即可。

## Proposed API

```python
import pandas as pd

from quantbullet.model.gam_replay import GAMReplayModel
from quantbullet.loan_simulation import parse_rollrate_coefficients

# Layer 1: roll-rate parser
edge_models = parse_rollrate_coefficients("fromC.txt")
# {"D1M": GAMReplayModel(...), "PIF": GAMReplayModel(...)}

# Layer 3: bridge 到 simulator
def build_features(context):
    return pd.DataFrame([{
        "purpose": context.loan.metadata["purpose"],
        "opti": context.loan.metadata["opti"],
        "v_opti": 1.0,
    }])

transition_model = SoftmaxTransitionModel(
    logits={
        "C": {
            to_status: (lambda ctx, m=m: float(m.predict(build_features(ctx))[0]))
            for to_status, m in edge_models.items()
        },
        ...
    },
    status_config=status_config,
)
```

如果某个未来 GAM 格式直接输出 edge probability，同一个 replay-compatible object 可以这样接：

```python
transition_model = ProbabilitySoftmaxTransitionModel(
    probabilities={
        "C": {
            to_status: (lambda ctx, m=m: float(m.predict(build_features(ctx))[0]))
            for to_status, m in edge_models.items()
        },
    },
    status_config=status_config,
)
```

## Implementation Steps

### Step 1: roll-rate parser spike

- 目标：证明 roll-rate TSV 是否能自然转换为现有 `GAMReplayModel` / `GAMTermData`。
- 先不要写大量生产 parser；先在 notebook/script 或小测试里构造最小 `GAMTermData` payload：intercept + factor + spline。
- 确认 `GAMReplayModel.predict(...)` 能复现手算 scalar。
- 量化 PCHIP vs 线性插值在稠密 grid（roll-rate dump 约 100-200 点）上的差异量级，决定 tie-out 容差还是加 linear 选项。
- 如果 `v_*` multiplier smooth 不能自然表达，决定是扩展 `model.gam`（含字段命名：`by_feature` 约定 vs `multiplier_feature`）还是写一个 replay-compatible adapter。

### Step 2: parser implementation

- `parse_rollrate_coefficients(path) -> dict[str, GAMReplayModel]`
- `gam_replay` 相关 import 放在函数内（lazy import），`import quantbullet.loan_simulation` 不得硬依赖 pygam
- 加入 `loan_simulation/__init__.py` 导出
- 用 tests 里的小型 synthetic fixture 文件（不依赖 roll-rate repo）
- 测试：
  - intercept / categorical / smooth grid 各自解析正确
  - `v_*` flag 的表达按 Step 1 spike 的结论落地并测试
  - 未知行结构 fail fast

### Step 3: integration test

- parser 产出的 replay models 包成 logits，进 `SoftmaxTransitionModel`，通过 `LoanSimulator` 跑通
- 验收标准：不改动 `model_transition.py` / `simulator.py` / `cashflow.py` 任何代码

### Step 4: tie-out script（docs，不进 package）

- `docs/loan_simulation/gam_tieout.py`
- 读真实 `input/coef/GENERIC_v4/fromC.txt`
- 同一批 synthetic loan feature dicts：
  - 我们的 replay model prediction vs roll-rate 的 `calc(link, loan)` 比 scalar/logit
  - 我们的 `SoftmaxTransitionModel` row vs roll-rate 的 `_softmax_transition` 比概率
- roll-rate repo 位置用 `--roll-rate-root` / `ROLL_RATE_MODEL_ROOT`，不 hardcode 本机路径
- 验证 `v_*` flag 的乘法语义假设与 roll-rate `SmoothByNum` 实际实现一致
- parser 撞到 GENERIC_v4 里未支持的 term 变体时，回到 Step 1/2 扩展 term 类型

## Deferred

- 其他 coefficient 格式的 parsers（设计已预留：新 parser -> replay-compatible object）
- 如果需要，增强 `quantbullet.model.gam` 的 term schema 以支持 multiplier (numeric-by) spline / roll-rate-specific flags
- smooth-by-factor / 组合 key lookup（tie-out 需要时再加）
- basis-coefficient spline term
- static/dynamic term 缓存等性能优化
- dials / overlays
