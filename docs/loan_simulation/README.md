# Loan Simulation Docs

这个目录放 loan simulation framework 的设计说明、demo scripts 和 tie-out scripts。当前目录暂时保持平铺；先用这个 README 按用途分组，避免在重构仍在推进时频繁移动脚本路径。

## Recommended Reading Order

1. `plan.md`: Phase 1 framework 总览，记录状态语义、cashflow engine、simulator、metrics 和当前设计边界。
2. `portfolio_example.py`: 最小端到端 demo，使用固定 transition table 跑 portfolio cashflow 和 metrics。
3. `model_backed_example.py`: 模型接入 demo，展示 callable edge probability 如何通过 `FeatureContext` 读取 loan、state、macro 和 path features。
4. `softmax_transition_plan.md`: 当前 softmax transition API 的设计说明，解释 logit softmax 和 independent-probability odds softmax 的区别。
5. `gam_adapter_plan.md`: roll-rate / GAM coefficient adapter 的设计路线。

## Demo Scripts

- `portfolio_example.py`: 基础 demo。核心链路是 `ConstantTransitionModel + MatrixPaymentPolicy + CashflowEngine + LoanSimulator + PortfolioSimulator + compute_period_metrics`。
- `model_backed_example.py`: 模型接入 demo。核心链路是 `CompositeTransitionModel + callable edge models + macro/path features`。

Run from repo root:

```powershell
.\.venv\Scripts\python.exe docs\loan_simulation\portfolio_example.py
.\.venv\Scripts\python.exe docs\loan_simulation\model_backed_example.py
```

## Tie-Out Scripts

- `gam_tieout.py`: GAM coefficient / transition row tie-out。读取 `roll-rate-model` 的真实 coefficient TSV，比较本 framework 的 `parse_rollrate_coefficients + SoftmaxTransitionModel` 与 roll-rate Python reference 的 raw logits 和 softmax probabilities。这个脚本只验证 transition math，不跑 cashflow simulation。
- `reconcile.py`: cashflow / metrics tie-out。用 synthetic loans、固定 transition table、payment matrix、seed、severity 和 recovery timing 对比本 framework 与 `roll-rate-model` 的 cashflows/metrics。这个脚本刻意绕过 GAM model。

Run from repo root:

```powershell
.\.venv\Scripts\python.exe docs\loan_simulation\gam_tieout.py --roll-rate-root C:\path\to\roll-rate-model
.\.venv\Scripts\python.exe docs\loan_simulation\reconcile.py --roll-rate-root C:\path\to\roll-rate-model
```

`ROLL_RATE_MODEL_ROOT` can be used instead of `--roll-rate-root`.

## Design Notes

- `model_integration_plan.md`: `CompositeTransitionModel`、`FeatureContext` 和 direct-probability edge assembly 的设计记录。
- `softmax_transition_plan.md`: `SoftmaxTransitionModel` / `ProbabilitySoftmaxTransitionModel` 的设计记录。
- `gam_adapter_plan.md`: GAM replay / roll-rate coefficient adapter 的设计记录。
- `plan.md`: 总览和长期 backlog。

Some files still use "plan" wording because they were written while the framework was being built. Treat the source code and focused tests as the current behavior, and use these notes for rationale and next-step context.

## Generated Outputs

Scripts in this directory may write `.xlsx` workbooks next to themselves. Those generated files stay local and are intentionally ignored by `.gitignore`.
