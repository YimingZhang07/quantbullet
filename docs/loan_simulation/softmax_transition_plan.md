# Softmax Transition Plan

## Why This Exists

当前已经有两种 transition model：

- `ConstantTransitionModel`: 直接给完整 probability table。
- `CompositeTransitionModel`: 每个 outward edge 直接给 probability，stay 是 residual。

这两种都处理 **probabilities**。

`roll-rate-model` 的 GAM transition 不是直接输出 probability，而是输出每个 non-stay edge 的 logit / score，然后做 softmax：

```text
score(stay) = 0
score(edge_i) = z_i

P(edge_i) = exp(z_i) / (1 + sum(exp(z_j)))
P(stay)   = 1 / (1 + sum(exp(z_j)))
```

所以需要一个独立的 `SoftmaxTransitionModel`。它不是 `CompositeTransitionModel` 的小改动，而是另一种 row assembly math。

## Mental Model

这些 models 不是简单的线性继承关系，而是同一个 `TransitionModel` interface 下的不同 probability assembly strategy。

```text
TransitionModel
├── ConstantTransitionModel
│   └── full row probabilities are provided directly
├── CompositeTransitionModel
│   └── outward edge probabilities are provided directly; stay is residual
└── SoftmaxTransitionModel
    └── outward edge logits are provided; row probabilities come from softmax
```

关系可以这样理解：

- `ConstantTransitionModel`: 最适合 baseline / deterministic assumptions / tests。
- `CompositeTransitionModel`: 适合 event probability models，例如 prepay probability、default probability、multiplicative assumptions。
- `SoftmaxTransitionModel`: 适合 multinomial logit / GAM logits，尤其是 roll-rate style transition models。

它们都可以表达 constants，但语义不同：

- constant probabilities: 直接是最终概率。
- direct edge probabilities: 每个 edge model 已经输出最终 event probability。
- logits: 每个 edge model 输出相对 stay 的 score，最终概率要通过 softmax 归一化。

## Important Difference: Direct Probability vs Logit

Direct probability model:

```text
C -> PIF = 0.10
C -> D1M = 0.03
C -> LIQ = 0.01
C -> C   = 0.86
```

Softmax logit model:

```text
z_PIF = 1.2
z_D1M = -0.5
z_LIQ = -2.0

probabilities are derived from exp(z)
```

In softmax, increasing one edge's logit reduces other probabilities because all outcomes compete within the same row. In direct probability assembly, edges do not compete except through the residual stay constraint.

## Proposed V1 API

`SoftmaxTransitionModel` should look similar to `CompositeTransitionModel`, but edges are logit specs:

```python
LogitSpec = float | Callable[[FeatureContext], float]

SoftmaxTransitionModel(
    logits={
        "C": {
            "PIF": prepay_logit,
            "D1M": dq_logit,
            "LIQ": -4.0,
        },
        "D1M": {
            "C": cure_logit,
            "D2M": roll_logit,
            "LIQ": -2.5,
        },
    },
    status_config=status_config,
)
```

V1 rules:

- Every non-terminal valid status must appear in `logits`.
- Empty dict means pure stay.
- Terminal statuses cannot define logits and auto self-loop.
- Explicit stay logits are not allowed; stay is the base category with score 0.
- Unknown statuses fail at construction.
- Callable logits receive `FeatureContext`.
- Runtime logits must be finite numbers; non-finite logits raise errors that include `loan_id` / `period` / `from_status` / `to_status`，与 `CompositeTransitionModel` 的错误上下文一致。
- Probabilities returned by `predict(...)` must sum to 1.

## Independent Binary Probability Models

Another common input shape is not logits but independent binary probabilities:

```text
C -> PIF model returns 0.90
C -> DQ30 model returns 0.90
```

These probabilities cannot both be final mutually-exclusive transition probabilities. For this use case we add `ProbabilitySoftmaxTransitionModel`:

```python
ProbabilitySoftmaxTransitionModel(
    probabilities={
        "C": {
            "PIF": prepay_probability,  # returns probability in [0, 1)
            "D1M": dq_probability,
        }
    },
    status_config=status_config,
)
```

It converts each independent binary probability to odds and normalizes against stay odds of 1:

```text
odds_i = p_i / (1 - p_i)

P(edge_i) = odds_i / (1 + sum(odds_j))
P(stay)   = 1      / (1 + sum(odds_j))
```

Example:

```text
p_prepay = 0.90
p_dq30   = 0.90

odds_prepay = 9
odds_dq30   = 9
stay_odds   = 1

P(PIF) = 9 / 19 = 47.37%
P(DQ30) = 9 / 19 = 47.37%
P(C) = 1 / 19 = 5.26%
```

Probability specs must be `>= 0` and `< 1`; `1.0` creates infinite odds and should be represented with a direct deterministic model instead.

## Example

If current status is `C`:

```python
logits = {
    "C": {
        "PIF": 1.0,
        "D1M": -1.0,
    }
}
```

Then:

```text
denominator = 1 + exp(1.0) + exp(-1.0)

P(C -> C)   = 1 / denominator
P(C -> PIF) = exp(1.0) / denominator
P(C -> D1M) = exp(-1.0) / denominator
```

## Numerical Stability

`math.exp(z)` 在 `z > ~709` 时会直接抛 `OverflowError`，raw softmax 会被 badly scaled model 输出 crash 掉。实现必须先做 max-shift：

```text
m = max(0, max(logits))

P(edge_i) = exp(z_i - m) / (exp(-m) + sum(exp(z_j - m)))
P(stay)   = exp(-m)      / (exp(-m) + sum(exp(z_j - m)))
```

分子分母同乘 `exp(-m)`，概率完全不变，但所有指数都 <= 0，不会 overflow。极端 logit 只会产生接近 0/1 的概率，而不是让 simulation 中途崩掉。

## Where GAM Fits

The GAM adapter should not change simulator/cashflow code.

It should only create callable logits:

```python
def pif_logit(context: FeatureContext) -> float:
    return intercept + spline(age) + beta * incentive + ...
```

Then:

```python
SoftmaxTransitionModel(
    logits={"C": {"PIF": pif_logit, "D1M": dq_logit}},
    status_config=status_config,
)
```

This mirrors roll-rate's softmax logic without copying its full C++/registry system into the core framework.

## Implementation Notes

- 实现放在现有 `src/quantbullet/loan_simulation/model_transition.py`，复用 `FeatureContext`。
- 结构校验（valid statuses、non-terminal 全覆盖、terminal 不许配置、不许显式 stay）与 `CompositeTransitionModel` 完全相同：把 freeze helper 参数化（传入 constant leaf validator：probability in [0, 1] vs finite logit），不要复制粘贴。
- `LogitSpec = float | Callable[[FeatureContext], float]`。类型与 `EdgeSpec` 相同，但语义不同（logit vs probability），保留独立 alias。
- `ProbabilitySpec = float | Callable[[FeatureContext], float]`。类型与 `EdgeSpec` 相同，但语义是 independent binary event probability，保留独立 alias。
- 公开 API：`SoftmaxTransitionModel`、`LogitSpec`、`ProbabilitySoftmaxTransitionModel`、`ProbabilitySpec` 加入 `loan_simulation/__init__.py`。

## Out of Scope

- 同一个模型实例内混用 direct-probability 和 softmax assembly（例如 `C` 行 softmax、`D1M` 行 direct probability）。需要时写 custom `TransitionModel`。
- GAM coefficient parsing（后续单独一步）。
- dials / overlays、probability trace。

## Tests

Add `tests/loan_simulation/test_softmax_transition.py`:

- constant logits match hand-computed softmax probabilities.
- callable logits receive `FeatureContext`.
- terminal statuses self-loop.
- missing non-terminal rows fail.
- explicit stay logit fails.
- non-finite logit fails.
- probabilities sum to 1.
- simulator integration works with a callable logit depending on macro/path features.
- independent binary probabilities such as 90% / 90% normalize through odds as expected.

## Recommendation

Implement `SoftmaxTransitionModel` next, but do not implement GAM file parsing yet.

This gives us the correct mathematical assembly layer for roll-rate style models. The later GAM adapter can focus only on turning coefficients/features into logit callables.
