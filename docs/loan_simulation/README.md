# Loan Simulation Docs

这个目录按 development 阶段放设计说明、demo scripts 和 tie-out scripts。每个阶段先看 plan，再看对应的可运行脚本或验证脚本。

## Phase 1: Core Simulator

- `plan.md`: 第一阶段总览，定义 `Loan` / `LoanState` / cashflow engine / simulator / metrics 的边界。
- `portfolio_example.py`: 最小 portfolio demo，使用固定 transition table 跑 cashflows 和 metrics。
- `reconcile.py`: cashflow-level tie-out，用固定 assumptions 对比本 framework 和 `roll-rate-model`。

## Phase 2: Model-Backed Direct Probabilities

- `model_integration_plan.md`: 设计 `FeatureContext` 和 `CompositeTransitionModel`，让 callable edge models 输出 direct probabilities。
- `model_backed_example.py`: direct-probability model demo，展示 edge callable 如何读取 loan、state、macro 和 path features。

## Phase 3: Softmax Transition Assembly

- `softmax_transition_plan.md`: 设计 `SoftmaxTransitionModel` 和 `ProbabilitySoftmaxTransitionModel`，区分 logits、direct probabilities 和 independent binary probabilities。

## Phase 4: GAM Adapter

- `gam_adapter_plan.md`: 设计 roll-rate coefficient TSV 到 `GAMReplayModel` / callable logits 的 adapter。
- `gam_tieout.py`: transition-level tie-out，比较真实 coefficients 的 raw logits 和 softmax probabilities。

## Running Scripts

```powershell
.\.venv\Scripts\python.exe docs\loan_simulation\portfolio_example.py
.\.venv\Scripts\python.exe docs\loan_simulation\model_backed_example.py
.\.venv\Scripts\python.exe docs\loan_simulation\gam_tieout.py --roll-rate-root C:\path\to\roll-rate-model
.\.venv\Scripts\python.exe docs\loan_simulation\reconcile.py --roll-rate-root C:\path\to\roll-rate-model
```

`ROLL_RATE_MODEL_ROOT` can be used instead of `--roll-rate-root`. Generated `.xlsx` workbooks stay local and are ignored by `.gitignore`.
