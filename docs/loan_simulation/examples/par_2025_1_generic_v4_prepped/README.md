# PAR_2025_1 + GENERIC_v4 Prepared Deal Benchmark

这个 example 用 roll-rate-model 已经 prepped 好的 real deal loan features，检查 `GENERIC_v4` coefficients 需要的 features 是否都能在 prepared loan dict 里找到。

## Current Scope

- `feature_inventory.py`: 比较真实 coefficient files 的 required features 和 `PAR_2025_1/loans_prepped.json` 的 available fields。
- `feature_builder.py`: run-level enrichment，补 `days_to_month_end` 和 `month_group`，feature semantics 留在这个 example 里。
- `runtime_feature_tieout.py`: multi-period feature update tie-out，对比本 example 的 runtime updater 和 roll-rate registry。
- `period1_transition.py`: period-1 transition benchmark，对比 quantbullet 和 roll-rate 的 logits / softmax probabilities。
- `cashflow_benchmark.py`: small Monte Carlo cashflow benchmark，对比 quantbullet 和 roll-rate 的 averaged portfolio cashflows。
- 不从 raw tape derive features；第一版从 roll-rate 的 prepared loan snapshot 出发。
- Cashflow benchmark 已按 roll-rate sampling order 和 CPR 口径对齐；剩余差异为 floating point noise。

当前 coverage：`GENERIC_v4` 的 23 个 required features 全部覆盖，missing count = 0。

当前 runtime feature tie-out：100 个 prepared loans、12 期，8 个 dynamic fields 全部 matches，mismatch count = 0。

当前 period-1 transition benchmark：10,360 个 prepared loans，`max_abs_logit_diff = 3.52e-13`，`max_abs_probability_diff = 2.22e-16`。

当前 cashflow benchmark：20 loans × 100 paths × 12 periods 可以跑通并输出 workbook；sampling order 已按 roll-rate `status_to_roll` 对齐，CPR 使用 roll-rate PIF-balance 口径重算，metrics 和 absolute cashflows 都基本为浮点误差。

```text
max_abs_cpr_diff = 1.11e-15
max_abs_cdr_diff = 0.0000
max_abs_cgl_diff = ~0
max_abs_begin_balance_diff = 2.33e-10
max_abs_end_balance_diff   = 2.33e-10
```

## Run

From repo root:

```powershell
.\.venv\Scripts\python.exe docs\loan_simulation\examples\par_2025_1_generic_v4_prepped\feature_inventory.py --roll-rate-root C:\path\to\roll-rate-model
.\.venv\Scripts\python.exe docs\loan_simulation\examples\par_2025_1_generic_v4_prepped\runtime_feature_tieout.py --roll-rate-root C:\path\to\roll-rate-model
.\.venv\Scripts\python.exe docs\loan_simulation\examples\par_2025_1_generic_v4_prepped\period1_transition.py --roll-rate-root C:\path\to\roll-rate-model
.\.venv\Scripts\python.exe docs\loan_simulation\examples\par_2025_1_generic_v4_prepped\cashflow_benchmark.py --roll-rate-root C:\path\to\roll-rate-model
```

`ROLL_RATE_MODEL_ROOT` can be used instead of `--roll-rate-root`.

## Output

Default output:

```text
docs/loan_simulation/examples/par_2025_1_generic_v4_prepped/feature_inventory.xlsx
docs/loan_simulation/examples/par_2025_1_generic_v4_prepped/runtime_feature_tieout.xlsx
docs/loan_simulation/examples/par_2025_1_generic_v4_prepped/period1_transition.xlsx
docs/loan_simulation/examples/par_2025_1_generic_v4_prepped/cashflow_benchmark.xlsx
```

The workbooks are generated locally and ignored by git.
