# PAR_2025_1 + GENERIC_v4 Roll-Rate Run

这个 example 只运行 roll-rate-model reference，不 import QuantBullet code。

This example is not self-contained. It requires `--roll-rate-root` pointing to a local `roll-rate-model` checkout; inputs and roll-rate Python code are read from that checkout.

It uses:

- `PAR_2025_1/loans_prepped.json`
- `GENERIC_v4` coefficient TSV files
- roll-rate `run_cf_one`

## Run

From repo root:

```powershell
.\.venv\Scripts\python.exe docs\loan_simulation\examples\par_2025_1_generic_v4_rollrate\run.py --roll-rate-root C:\path\to\roll-rate-model
```

`ROLL_RATE_MODEL_ROOT` can be used instead of `--roll-rate-root`.

## Output

Default output:

```text
docs/loan_simulation/examples/par_2025_1_generic_v4_rollrate/rollrate_cashflows.xlsx
```

Workbook sheets:

- `run_config`
- `metrics`
- `cashflows`

Generated workbooks are local and ignored by git.
