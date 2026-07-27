# Synthetic Loan Simulation Example

This is a self-contained, model-backed example of the complete loan simulation
flow: CSV inputs, monthly macro lookup, runtime feature construction, competing
probability models, Monte Carlo paths, cashflow projection, and Excel reporting.

## Inputs

- `input/loans.csv`: 100 deterministic synthetic loans. All loans originate on
  `2000-01-31`, start at age `0` in status `C`, and have `balance` equal to
  `original_balance`. Terms are limited to 3, 5, or 10 years. Each loan coupon
  is the market rate at origination plus a seeded spread of 1% to 4%.
- `input/hpi.csv`: monthly HPI index values.
- `input/rates.csv`: monthly annual base rates expressed as decimals.

Both macro files use month-end observations covering June 1999 through December
2010. HPI and rates follow seeded random processes, so regenerating the files
produces identical values.

## Features

`DataFrameMacroFeatureProvider` aligns the raw `hpi` and `market_rate` values to
each simulation month. `SyntheticFeatureProvider` then produces exactly three
model features:

- `age`: current loan age in months.
- `incentive`: `loan.annual_rate - market_rate`.
- `hpi`: current HPI index.

This keeps raw macro lookup, feature construction, and model scoring separate.

## Transition model

Active states are `C`, `D1`, and `D2`. `PIF` is prepaid and `CO` is charged off;
both are terminal. Normal scheduled paydown is not a separate state: the path
ends naturally when amortization reduces its balance to zero.

`ProbabilitySoftmaxTransitionModel` treats every configured edge value as an
independent binary probability. It converts each probability to odds
`p / (1 - p)` and normalizes all edge odds together with stay odds of `1`.

The two feature-backed probability models are deliberately elevated so their
effect is visible in a small demo:

- `C -> D1`:
  `clip(0.045 + 0.00015*age + 0.40*incentive - 0.0005*(hpi-100), 0.02, 0.12)`.
- `C -> PIF`:
  `clip(0.025 + 0.00025*age + 1.20*incentive + 0.0003*(hpi-100), 0.02, 0.18)`.

There is no direct `C -> CO` edge. The other competing probability specs are
constants:

- From `D1`: `C=0.30`, `D2=0.25`, `PIF=0.03`, `CO=0.02`.
- From `D2`: `C=0.10`, `D1=0.20`, `PIF=0.02`, `CO=0.25`.

Charge-off uses 60% severity and a three-month recovery lag. The payment matrix
uses normal roll-rate semantics: delinquency misses payments and cures collect
the required catch-up installments.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe docs\loan_simulation\examples\synthetic\run.py
```

The default run uses 50 paths, a fixed seed, and a 120-month horizon determined
from the longest loan. Common overrides include:

```powershell
.\.venv\Scripts\python.exe docs\loan_simulation\examples\synthetic\run.py `
  --n-paths 100 `
  --workers 4 `
  --include-path-cashflows
```

Default output:

```text
docs/loan_simulation/examples/synthetic/synthetic_cashflows.xlsx
```

The workbook contains `portfolio_metrics`, `portfolio_cashflows`, and
`loan_cashflows`. `path_cashflows` is optional because the path-level output can
be large.

## Regenerate inputs

```powershell
.\.venv\Scripts\python.exe docs\loan_simulation\examples\synthetic\generate_inputs.py
```
