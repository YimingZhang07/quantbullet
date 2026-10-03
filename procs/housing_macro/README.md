# Housing macro process manual

Run commands from the repository root with the project `.venv`.

## Setup

```powershell
$env:MACRO_DATA_ROOT = '<external-data-root>\housing_macro'
```

The data root must be outside the repository. All commands also accept
`--data-root` to override the environment variable. No API key is required.

## Execution order

| Step | Process | Purpose | Output under the macro root |
| --- | --- | --- | --- |
| 1 | `download` | Download Zillow metro/state/ZIP ZHVI, national CPI and 30-year PMMS | `raw/` CSV snapshots and `manifests/downloads.json` |
| 2 | `build_parquet` | Standardize the current snapshots into monthly tables | `parquet/hpi.parquet`, `parquet/cpi.parquet`, `parquet/pmms.parquet` |
| 3 (optional) | `coverage` | Inspect history, geographic coverage and missing values | `reports/coverage.json`, `reports/coverage.md` |

```powershell
.\.venv\Scripts\python.exe -m procs.housing_macro.download
.\.venv\Scripts\python.exe -m procs.housing_macro.build_parquet
.\.venv\Scripts\python.exe -m procs.housing_macro.coverage
```

These monthly tables are inputs to
[Freddie panel preparation](../freddie_sflld/README.md).

## Updates and selected datasets

Default downloads reuse valid local snapshots. Add `--refresh` to fetch remote
updates, then rebuild Parquet. Run coverage again if updated reports are needed.

```powershell
# Update all sources and rebuild all tables.
.\.venv\Scripts\python.exe -m procs.housing_macro.download --refresh
.\.venv\Scripts\python.exe -m procs.housing_macro.build_parquet

# Update only PMMS.
.\.venv\Scripts\python.exe -m procs.housing_macro.download --dataset pmms_30y --refresh
.\.venv\Scripts\python.exe -m procs.housing_macro.build_parquet --dataset pmms
```

`download --dataset` accepts one or more of `zhvi_metro`, `zhvi_state`,
`zhvi_zip`, `cpi`, `pmms_30y`; the default is all five.
`build_parquet --dataset` accepts `hpi`, `cpi`, `pmms`; the default is all three.
Selected builds require their source snapshots and replace only their tables.
Use one writer at a time. Failed downloads preserve previous successful
snapshots; validation/conversion failures preserve existing Parquet outputs.

See [data rules](data_rules.md) for sources, schemas, monthly PMMS aggregation,
cutoff dates, validation and coverage definitions.
