# Loan Simulation Docs

这个目录保留当前可运行的 loan-simulation examples。Historical implementation
plans 和已被 core tests / production comparison 取代的 tie-out scripts 已移除。

## Examples

- `portfolio_example.py`: minimal portfolio demo，使用 constant transition table
  展示 cashflows 和 metrics。
- `model_backed_example.py`: generic model-backed demo，展示
  `CompositeTransitionModel`、macro features 和 path features。
- `examples/par_2025_1_generic_v4_quantbullet/`: production-oriented
  `PAR_2025_1 + GENERIC_v4` example，包含 runtime feature provider、parallel
  prepared-loan run 和 roll-rate production workbook comparison。

## Running Scripts

```powershell
.\.venv\Scripts\python.exe docs\loan_simulation\portfolio_example.py
.\.venv\Scripts\python.exe docs\loan_simulation\model_backed_example.py
.\.venv\Scripts\python.exe docs\loan_simulation\examples\par_2025_1_generic_v4_quantbullet\run.py
.\.venv\Scripts\python.exe docs\loan_simulation\examples\par_2025_1_generic_v4_quantbullet\compare_prod.py --rollrate-workbook C:\path\to\roll-rate-model\output\PAR_2025_1\base\sim_results.xlsx
```

Generated `.xlsx` workbooks stay local and are ignored by `.gitignore`.
