# PAR_2025_1 + GENERIC_v4 QuantBullet Demo

这个 example 是一个 self-contained QuantBullet run：不 import roll-rate code，不做 tie-out，只用本 folder 里的 coefficients、prepared loan sample 和 config 跑 cashflows / metrics。

## Inputs

- `input/coef/GENERIC_v4/from*.txt`: copied roll-rate GAM coefficient TSV files。
- `input/loans_prepped_sample.json`: 20-loan prepared sample copied from `PAR_2025_1/loans_prepped.json`。
- `feature_builder.py`: model/run-specific feature enrichment and runtime feature rolling。
- `run.py`: Python-native demo，使用 builder functions 构建 status config、transition model、payment policy、loans、simulator 和 workbook output。

## Run

From repo root:

```powershell
.\.venv\Scripts\python.exe docs\loan_simulation\examples\par_2025_1_generic_v4_quantbullet\run.py
```

## Output

Default output:

```text
docs/loan_simulation/examples/par_2025_1_generic_v4_quantbullet/quantbullet_cashflows.xlsx
```

Workbook sheets:

- `portfolio_metrics`
- `portfolio_cashflows`
- `path_cashflows`

Generated workbooks are local and ignored by git.
