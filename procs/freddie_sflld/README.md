# Freddie Mac SFLLD Standard Dataset: first-stage Parquet conversion

This process converts the downloaded Release 47 full Standard Dataset archive
to separate origination (`orig`) and monthly performance (`perf`) Parquet files,
partitioned by origination quarter. It does not create model features or a
joined loan-month table.

## Data layout

Set `FREDDIE_DATA_ROOT` to a directory outside this repository. Place the
downloaded archive at:

```text
<data-root>/raw/full_set_standard_historical_data.zip
```

The converter creates `parquet/orig/vintage=YYYYQn/`,
`parquet/perf/vintage=YYYYQn/`, `manifests/conversion.json`, and a temporary
`scratch/` directory underneath the same root. The source ZIP is never
modified. Only one quarter's text files are extracted at a time; the scratch
files are removed after that quarter succeeds or fails.

## Run

From the repository root on Windows:

```powershell
$env:FREDDIE_DATA_ROOT = Join-Path $env:USERPROFILE 'data\freddie_sflld'
.\.venv\Scripts\python.exe -m procs.freddie_sflld.build_parquet --quarter 2015Q1
.\.venv\Scripts\python.exe -m procs.freddie_sflld.build_parquet
```

The default range starts at `2015Q1` and ends at the last quarter in the
archive. Use `--start 2020Q1 --end 2021Q4` to process a range. The process
continues past individual quarter failures and exits with a nonzero status if
any quarter failed. Rerunning it skips quarters whose source ZIP hash and
Parquet output hashes match the manifest.

## Reading the output

Read `manifests/conversion.json` to locate the current `orig` and `perf`
Parquet path for each vintage. Previous outputs may briefly coexist while a
quarter is being updated, so avoid globbing all files in a partition. All
columns are stored as strings in this faithful first-stage conversion;
blank source fields become null. Sentinel codes (such as `999`, `9999`,
and `XX`) remain unchanged. Apply the official release-specific data
dictionary when constructing typed model features and prepayment labels.

The column names and order follow the [Release 47 SFLLD guide](https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf).

## Build a sampled loan-month panel

The sample process reads the current orig/perf paths from the conversion
manifest. It first applies any configured product filters to orig records,
then samples from **all eligible orig loans**, allocating the requested
count proportionally across the inclusive vintage range with the largest
remainder method based on each vintage's eligible count. Ties use vintage
order. Sampling is without replacement;
IDs are sorted before using a random seed derived from the configured seed
and vintage, so source row order does not affect the chosen set. Repeated
runs are reproducible in the same Python environment with unchanged inputs.

The example TOML selects 500,000 **30-year fixed-rate loans** from 2015Q1
through 2026Q1 with seed 42. Its paths use `${FREDDIE_DATA_ROOT}`. Set that variable to an external
directory, then run:

```powershell
.\.venv\Scripts\python.exe -m procs.freddie_sflld.build_sample_panel `
    --config procs/freddie_sflld/sample_panel.example.toml
```

Copy the example to an external local configuration to change paths,
`start_vintage`, `end_vintage`, `n_loans`, `seed`, or product filters:

```toml
[sample]
start_vintage = "2015Q1"
end_vintage = "2026Q1"
n_loans = 500000
seed = 42
amortization_type = "FRM"
original_loan_term = 360
```

`amortization_type` accepts `FRM` or `ARM`; `original_loan_term` is a positive
integer in months. Each condition is optional: omit it to impose no
restriction on that field. When both are present, both must match. Term
comparison is numeric even though the original field remains a String in
the output. The requested sample size is drawn entirely from eligible
loans; insufficient eligible population fails before clearing prior outputs.

Relative TOML paths are
resolved against the TOML file's directory; unset `${NAME}` variables fail
clearly. The output must be a dedicated external directory that does not
contain the input files. No new dependency is needed.

```text
<output-root>/
  sampled_loans.parquet
  panel/vintage=2015Q1/panel.parquet
  ...
  sampling_summary.json
```

`sampled_loans.parquet` retains every sampled loan's 31 static fields and
its vintage, including loans without performance observations. Each panel
partition retains the complete performance history of sampled loans and
joins their static fields many-to-one. It contains the 65 combined source
columns, `vintage` (String), and `month` (Date, first of month). Raw `period`
and all original fields remain strings with nulls and official codes intact.
There is no outcome filter, minimum history requirement, terminal-row
truncation, missing-month filling, or synthetic row for an orig-only loan.

The small summary records parameters and product filters, population and sample counts, vintage
quotas/fractions, panel rows, and loans with/without perf. Source paths are
relative to the Freddie data root. `population_loans` counts eligible loans,
while `raw_population_loans` counts orig loans before filtering, both in total
and per vintage. Quotas and sampling fractions use the eligible population.
The sample list has exactly the requested
number of globally unique loans; the panel can contain fewer loans when some
sampled orig loans have no perf. Those two counts plus the missing-perf count
reconcile in the summary.

Read every joined partition as one logical table; no additional static join
is needed:

```python
import os
from pathlib import Path
import polars as pl

output = Path(os.environ["FREDDIE_DATA_ROOT"]) / "samples" / "prepayment_500k"
panel = pl.scan_parquet(
    str(output / "panel" / "vintage=*" / "panel.parquet"),
    hive_partitioning=False,
)
loan_months = panel.select("loan_identifier", "vintage", "month", "current_actual_upb")
```

The process reads loan IDs separately and handles perf one quarter at a
time. Streaming writes retain all fields; a projection of sampled loan IDs
and months checks cardinality and duplicate loan-months before publication.
Only sampled performance histories enter the output, but the corresponding
source perf files still need scanning. Vintages with zero allocated quota do
not create a partition.

Every run rebuilds fixed outputs and removes this process's old sample
files/partitions so stale quarters cannot mix with the new sample. Original
input files are read-only. Single-quarter files are written temporarily then
replaced. A failure aborts the run and may leave completed quarter files;
`sampling_summary.json` is written only when all quarters succeed. Treat a
directory without that summary as incomplete and rerun the same command.
Run one writer against the output directory and avoid analyzing it during a
rebuild. There is no cache, build version, or schema version for this step.

Reusable functions are `filter_orig_loans`, `allocate_vintage_counts`, `sample_loan_ids`, and
`scan_loan_panel` in `quantbullet.data.freddie_sflld`. This stage does not
join macro data, define prepayment labels/features, or split training data.
