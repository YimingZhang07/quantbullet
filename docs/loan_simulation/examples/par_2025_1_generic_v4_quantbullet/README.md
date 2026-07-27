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

By default this uses the checked-in 20-loan sample. To run against full prepared loans:

```powershell
.\.venv\Scripts\python.exe docs\loan_simulation\examples\par_2025_1_generic_v4_quantbullet\run.py --loans C:\path\to\roll-rate-model\input\deals\PAR_2025_1\loans_prepped.json
```

## High-path subset diagnostic

This diagnostic uses a deterministic 100-loan sample with 2,000 paths and a
24-month horizon. From the QuantBullet repo root:

```powershell
$quantbulletRoot = (Get-Location).Path
$rollRateRoot = "C:\path\to\roll-rate-model"
$env:ROLL_RATE_MODEL_ROOT = $rollRateRoot
$subset = Join-Path $env:TEMP "par_2025_1_representative_100_loans.json"
```

Create the deterministic subset:

```powershell
@'
import json
import random
import os
from pathlib import Path

root = Path(os.environ["ROLL_RATE_MODEL_ROOT"])
loans = json.loads(
    (root / "input/deals/PAR_2025_1/loans_prepped.json").read_text()
)
active = [loan for loan in loans if float(loan.get("end_bal", 0) or 0) > 0.1]
selected = random.Random(20260726).sample(active, 100)
selected.sort(key=lambda loan: str(loan["loan_id"]))
(Path(os.environ["TEMP"]) / "par_2025_1_representative_100_loans.json").write_text(
    json.dumps(selected, indent=2)
)
'@ | .\.venv\Scripts\python.exe -
```

Run QuantBullet:

```powershell
.\.venv\Scripts\python.exe `
  docs\loan_simulation\examples\par_2025_1_generic_v4_quantbullet\run.py `
  --loans $subset `
  --workers 8 `
  --n-paths 2000 `
  --horizon 24 `
  --no-path-cashflows `
  --output docs\loan_simulation\examples\par_2025_1_generic_v4_quantbullet\quantbullet_subset100_paths2000.xlsx
```

Run roll-rate production:

```powershell
Push-Location $rollRateRoot
& "$quantbulletRoot\.venv\Scripts\python.exe" `
  python\run.py `
  --deal-name PAR_2025_1 `
  --coef-version GENERIC_v4 `
  --loans $subset `
  --n-per 24 `
  --dup 2000 `
  --seed 20260725 `
  --mode pool `
  --workers 8 `
  --output output\PAR_2025_1\base\prod_subset100_paths2000_dynamic_macro.xlsx
Pop-Location
```

Compare the production workbooks:

```powershell
.\.venv\Scripts\python.exe `
  docs\loan_simulation\examples\par_2025_1_generic_v4_quantbullet\compare_prod.py `
  --quantbullet-workbook docs\loan_simulation\examples\par_2025_1_generic_v4_quantbullet\quantbullet_subset100_paths2000.xlsx `
  --rollrate-workbook "$rollRateRoot\output\PAR_2025_1\base\prod_subset100_paths2000_dynamic_macro.xlsx" `
  --output docs\loan_simulation\examples\par_2025_1_generic_v4_quantbullet\prod_comparison.xlsx
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
