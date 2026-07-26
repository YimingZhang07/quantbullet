# PAR_2025_1 + GENERIC_v4 QuantBullet Demo

这个 example 是一个 self-contained QuantBullet run：不 import roll-rate code，不做 tie-out，只用本 folder 里的 coefficients、prepared loan sample、macro inputs 和 config 跑 cashflows / metrics。

## Inputs

- `input/coef/GENERIC_v4/from*.txt`: copied roll-rate GAM coefficient TSV files。
- `input/loans_prepped_sample.json`: 20-loan prepared sample copied from `PAR_2025_1/loans_prepped.json`。
- `input/macro/*.csv`: copied CPI and FICO coupon inputs for runtime macro features。
- `feature_builder.py`: QuantBullet runtime provider plus the small GENERIC_v4 feature registry and roll functions。
- `run.py`: Python-native demo，使用 builder functions 构建 status config、transition model、payment policy、loans、simulator 和 workbook output。

Runtime features are example-local and intentionally kept in one file。Each feature is one roll function plus declared dependencies；the executor only walks the ordered rules and updates path state。`r_dt` 表示 transition model 的 as-of month，`period_date` 表示下一 cashflow month，provider 会 validate 两者 alignment；age 以 QuantBullet `LoanState` 为 source of truth。Macro files fail fast when missing，individual lookup gaps use explicit per-feature carry-forward，model-facing macro values round to 4 decimals。

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
