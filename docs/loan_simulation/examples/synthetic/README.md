# Synthetic Loan Simulation Example

This folder will contain a self-contained, model-backed loan simulation example.
The current first step fixes the synthetic input data; the transition model and
simulation runner will be added separately.

## Inputs

- `input/loans.csv`: 100 deterministic synthetic loans. All loans originate on
  `2000-01-31`, start at age `0` in status `C`, and have `balance` equal to
  `original_balance`. Terms are limited to 3, 5, or 10 years. Each loan coupon
  is the market rate at origination plus a seeded spread of 1% to 4%.
- `input/hpi.csv`: monthly HPI index values.
- `input/rates.csv`: monthly annual market rates expressed as decimals.

Both macro files use month-end observations covering June 1999 through December
2010. HPI and rates follow seeded random processes, so regenerating the files
produces identical values.

## Regenerate

From the repository root:

```powershell
.\.venv\Scripts\python.exe docs\loan_simulation\examples\synthetic\generate_inputs.py
```
