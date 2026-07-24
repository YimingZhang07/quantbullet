# Loan Simulation Docs

这个目录放 loan simulation framework 的设计说明、demo scripts 和 reconciliation scripts。

## Files

- `plan.md`: Phase 1 framework 总览，记录当前模块结构、核心设计决定和后续方向。
- `model_integration_plan.md`: 下一阶段 model-backed transition 的设计计划，重点是 `CompositeTransitionModel`、`FeatureContext` 和未来 GAM/ML adapter。
- `portfolio_example.py`: 基础 demo，核心链路是 `ConstantTransitionModel + MatrixPaymentPolicy + CashflowEngine + Simulator + Metrics`。
- `model_backed_example.py`: 模型接入 demo，核心链路是 `CompositeTransitionModel + callable edge models + macro/path features`。
- `reconcile.py`: synthetic portfolio tie-out 脚本，用相同 assumptions 对比本 framework 和 `roll-rate-model` 的 cashflows/metrics。
- `.gitignore`: 忽略这些 scripts 生成的 `.xlsx` workbooks。

Generated Excel files stay local and are intentionally not committed.
