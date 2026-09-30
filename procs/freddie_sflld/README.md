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
