# Model Integration Plan

## Goal

下一阶段目标是让 loan simulation framework 可以接入真实模型，而不只是 `ConstantTransitionModel`。

核心问题不是 "怎么读 GAM 文件"，而是先定义清楚：

```text
loan + current_state + macro_features + path_features
-> feature values
-> model output
-> transition probability row
```

simulation engine 仍然只认识 `TransitionModel.predict(...)`。具体概率可以来自 constants、callables、sklearn-like models、GAM coefficients，或者它们的组合。

## Design Principles

各层职责不变：

- `LoanSimulator`: path loop、seed、macro lookup、path feature tracker、sampling。
- `CashflowEngine`: accounting。
- `TransitionModel`: 输出下一期状态概率。
- model integration layer: 把 features 转成 transition probabilities。

model engine 不进入 `CashflowEngine` / `LoanSimulator`。换模型不影响 accounting。

延续 framework 的 fail-fast 哲学：不做 silent fallback，配置缺失或概率非法就报错。

## Why Not Directly Copy Roll-Rate GAM Engine

`roll-rate-model` 的 model layer 是 production-oriented：coefficient dumps、variable registry、static/dynamic term cache、macro registry、C++ hot path、dials。一次性搬过来会让 framework 过重。先做 thin integration layer，把模型接入的 shape 定稳；GAM 作为后续 adapter。

## V1 Decisions

以下是已拍板的决定，不是 open questions：

1. **Row assembly 用 direct probability + residual stay**。每个 configured edge 输出一个概率，剩余概率给当前 status。softmax-over-logits assembly 不进 v1，作为 GAM adapter 步骤的配套工作（roll-rate 的 GAM 是 multinomial logistic，直接概率组合会和 roll-rate 不一致，两种 assembly 必须区分清楚）。
2. **Residual 固定给当前 status**。不做 `residual_status` 参数，未来有真实需求再加。
3. **Edge spec 是 `float | Callable[[FeatureContext], float]`**。数字就是常数概率，callable 每期求值。不做 `ConstantEdgeProbability` / `CallableEdgeProbability` wrapper 类。
4. **每个 non-terminal valid status 必须显式出现在 edges 配置里**。想表达 "纯 stay" 就显式给空 dict。缺失在构造时报错，和 `ConstantTransitionModel` 当初移除 stay fallback 的决定保持一致。
5. **Terminal statuses 不配置、自动 self-loop**。给 terminal status 配 edges 会在构造时报错。
6. **概率越界 / row sum > 1 在运行时报错**。错误信息包含 loan_id、period、from_status 和整行概率，方便定位。这是 per-path runtime error，模型在压力情形下可能触发；宁可 fail fast，也不 silent normalize。
7. **整行输出的模型不走 composite**。multinomial 模型（一个模型直接输出整行概率）直接 subclass `TransitionModel`。composite 只服务 per-edge 组装，不要把整行模型硬拆成 edges。
8. **dials / overlays、probability trace 不进 v1**，后续单独做。

## Proposed Design

### FeatureContext

data-only 的 frozen dataclass，把 `predict(...)` 的四个输入打包成一个参数，方便 edge callables 接收：

```python
@dataclass(frozen=True)
class FeatureContext:
    loan: Loan
    current_state: LoanState
    macro_features: Mapping[str, Any]
    path_features: Mapping[str, Any]
```

原则：

- 不做 getter 方法（`get_macro(...)` 之类没有信息量，直接 dict 访问）。
- 不做 derived features（`factor`、`incentive` 由模型自己算，遵循 "模型自己 derive 自己需要的 features"）。
- 以后加字段不破坏 callable 签名。

### Edge Specs

```python
EdgeSpec = float | Callable[[FeatureContext], float]
```

例子：

```python
def prepay_probability(context: FeatureContext) -> float:
    incentive = context.loan.annual_rate - context.macro_features["market_rate"]
    return _sigmoid(a + b * incentive)
```

乘法模型同样是一个 callable：内部把 base curve × seasoning × burnout × incentive multipliers 乘起来，返回一个概率。framework 不需要知道模型内部形式。

### CompositeTransitionModel

```python
CompositeTransitionModel(
    edges={
        "C": {
            "PIF": prepay_probability,   # callable model
            "D1M": 0.02,                 # constant
            "LIQ": 0.005,
        },
        "D1M": {
            "C": cure_probability,
            "D2M": 0.15,
            "LIQ": 0.02,
        },
        "D2M": {},                       # explicit pure-stay
    },
    status_config=status_config,
)
```

构造时校验（fail fast）：

- 所有 from/to statuses 必须属于 `status_config.valid_statuses`。
- 每个 non-terminal valid status 必须有 edges entry（空 dict 表示显式 pure-stay）。
- terminal statuses 不允许出现在 edges 里。
- to_status 不能等于 from_status（stay 是 residual，不显式配置）。
- constant edge 概率必须在 [0, 1]。

`predict(...)` 每期逻辑：

- terminal status 直接返回 `{status: 1.0}`。
- 逐个 edge 求值：常数直接用，callable 传入 `FeatureContext`。
- 校验每个概率在 [0, 1]、row sum <= 1，违反就报错并带 loan/period/probabilities 上下文。
- `stay = 1 - sum(edge probabilities)`，返回完整概率行。

### Full-Row Models（escape hatch）

一个模型直接输出整行概率时（multinomial logistic、xgboost multi-class 等），不要拆成 edges，直接实现 `TransitionModel`：

```python
class MyMultinomialTransitionModel(TransitionModel):
    def predict(self, loan, current_state, macro_features=None, path_features=None):
        features = self._build_features(loan, current_state, macro_features, path_features)
        return dict(zip(self.statuses, self.model.predict_proba(features)[0]))
```

## Model Type Compatibility

- **constant**: edge spec 直接写 float。
- **乘法模型**: callable 内部算乘积。
- **ML binary per-edge**: callable 包一层 `predict_proba`，返回单个概率。
- **ML multinomial full-row**: 直接 subclass `TransitionModel`。
- **GAM（roll-rate style logits）**: 后续 GAM adapter + softmax assembly，不进 v1。

## Implementation Steps

### Step 1: `model_transition.py`

新增 `src/quantbullet/loan_simulation/model_transition.py`，包含：

- `FeatureContext`
- `CompositeTransitionModel`

公开 API（加入 `loan_simulation/__init__.py` 导出）：

- `FeatureContext`
- `CompositeTransitionModel`

### Step 2: Unit Tests

新增 `tests/loan_simulation/test_model_transition.py`：

- constant + callable 混合 row 输出正确概率和 residual stay。
- callable 收到的 `FeatureContext` 内容正确（loan / state / macro / path features）。
- 缺 non-terminal status entry 时构造报错。
- terminal status 配置 edges 时构造报错。
- 未知 status 构造报错。
- edge 概率越界或 row sum > 1 时运行时报错，且错误信息含 loan/period。

### Step 3: Integration Test

加一个端到端 case（放 `test_model_transition.py`）：

- prepay edge 依赖 macro feature（例如 market_rate）。
- delinquency edge 依赖 path feature（例如 `ever_delinquent`）。
- 通过 `LoanSimulator` 跑通并产生 cashflow 输出。

验收标准：整个 Step 1-3 不改动 `CashflowEngine` / `LoanSimulator` 任何代码。

### Step 4: GAM Adapter（deferred）

等 Step 1-3 稳定后单独设计：

- GAM coefficient table 解析。
- softmax row assembly（multinomial logistic，stay 为 base category）。
- 和 roll-rate transition probabilities 的 tie-out。

## Deferred

- GAM coefficient parser + softmax assembly
- dials / overlays
- probability trace
- batch prediction / performance optimization
