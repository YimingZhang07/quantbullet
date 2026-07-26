# PAR_2025_1 + GENERIC_v4 Compare

这个 example 不 run simulation，只读取 QuantBullet 和 roll-rate 两边已经生成的 workbook，然后输出 comparison workbook。

## Inputs

- `../par_2025_1_generic_v4_quantbullet/quantbullet_cashflows.xlsx`
- `../par_2025_1_generic_v4_rollrate/rollrate_cashflows.xlsx`

先分别运行：

```powershell
.\.venv\Scripts\python.exe docs\loan_simulation\examples\par_2025_1_generic_v4_quantbullet\run.py
.\.venv\Scripts\python.exe docs\loan_simulation\examples\par_2025_1_generic_v4_rollrate\run.py --roll-rate-root C:\path\to\roll-rate-model
```

## Run

```powershell
.\.venv\Scripts\python.exe docs\loan_simulation\examples\par_2025_1_generic_v4_compare\compare.py
```

## Output

Default output:

```text
docs/loan_simulation/examples/par_2025_1_generic_v4_compare/comparison.xlsx
```

Workbook sheets:

- `summary`
- `metric_diff`
- `cashflow_diff`
- `quantbullet_metrics`
- `rollrate_metrics`
- `quantbullet_cashflows`
- `rollrate_cashflows`

Generated workbooks are local and ignored by git.
