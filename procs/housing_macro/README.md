# Housing macro data pipeline

Download complete-history CSV snapshots for monthly Zillow ZHVI (metro,
state, and ZIP5) and national CPI-U (`CPIAUCNS`, BLS via FRED). The metro
ZHVI CSV includes the national row. Zillow files use All Homes, mid-tier,
smoothed, seasonally adjusted ZHVI; CPI is not seasonally adjusted.

Set an external data directory, then run from the repository root:

```powershell
$env:MACRO_DATA_ROOT = '<external-data-root>\housing_macro'
.\.venv\Scripts\python.exe -m procs.housing_macro.download --dataset cpi
.\.venv\Scripts\python.exe -m procs.housing_macro.download
.\.venv\Scripts\python.exe -m procs.housing_macro.download --refresh
```

`--data-root` overrides the environment variable. `--dataset` accepts one
or more of `zhvi_metro`, `zhvi_state`, `zhvi_zip`, and `cpi`; the default is
all four, processed sequentially. The CLI rejects data roots inside this
repository. No API key or additional dependency is required.

Files are stored as `raw/<provider>/<dataset_id>/<sha256>.csv` under the data
root. `manifests/downloads.json` records each dataset's `current` snapshot
and its `versions`, including relative paths, hashes, sizes, original
filenames, URLs, metadata, download times, and available HTTP headers.
Download and check times describe local acquisition, not official release
dates or historical availability. No local absolute paths are written to
the manifest.

Default runs verify the current file's hash and reuse it (`cached`). Missing
or damaged files are downloaded again. `--refresh` fetches the complete
remote CSV: identical content produces `unchanged`; changed content creates
a new snapshot (`downloaded`) while keeping past snapshots. Readers should
locate current files through the manifest. Only one writer should run
against a data root at a time.

Downloads use bounded memory, a 30-second network timeout, and up to three
attempts for connection errors, HTTP 429, or HTTP 5xx (1- and 3-second retry
delays). A download or format failure preserves the previous manifest and
current snapshot for that dataset; remaining datasets continue. Any failure
makes the CLI exit with a nonzero status. Temporary `.part` files are cleaned
up after success or failure; forced process termination may leave a partial
file in `scratch/`, which is never selected by the manifest.

Validation checks a nonempty CSV, a matching first data row, and expected
headers. Zillow needs geographic columns and dated monthly columns; FRED
needs `observation_date` and the series column. This stage preserves original
data and does not assess missing values, geographic coverage, or mapping
quality. ZIP5 values cannot be directly joined to Freddie's disclosed ZIP3.

Sources: [Zillow housing data](https://www.zillow.com/research/data/) and
[CPIAUCNS on FRED](https://fred.stlouisfed.org/series/CPIAUCNS). Zillow download
paths can change; update the definitions in `quantbullet.data.zillow` when
necessary. The FRED CSV download endpoint is used here rather than its
credentialed observations API.

## Normalize current snapshots

```powershell
.\.venv\Scripts\python.exe -m procs.housing_macro.build_parquet
# Alternatively: --data-root '<external-data-root>\housing_macro'
```

The command verifies hashes and sizes of all four current download snapshots
before reading them. It writes two independent long tables with complete
history, including every source null. No interpolation or rebasing is applied.

| Table | Columns | Unique key |
|---|---|---|
| HPI | provider, metric, geography_level, region_id, region_name, month, value; source geography metadata | provider + metric + geography_level + region_id + month |
| CPI | provider, series_id, month, value | series_id + month |

HPI uses `provider="zillow"`, `metric="ZHVI"`, and original dollar values.
The name HPI describes the table's role; ZHVI has not been converted to an
index. Source `country` becomes `national`, `msa` becomes `metro`, and
`state` and `zip` remain unchanged. `region_id`, `region_name`, `region_type`,
`state_name`, `state`, `city`, `metro`, and `county_name` are strings; ZIP5
names retain leading zeros. `size_rank` is nullable Int64. Missing source
metadata columns become null. Zillow RegionID is not a Freddie MSA code.

CPI uses `provider="bls"` and `series_id="CPIAUCNS"`; FRED remains the
distributor in source metadata. `month` is Date at the first of the month,
and `value` is Float64 in both tables. Empty source values and FRED `.` are
null, including the current snapshot's missing CPI observation for 2025-10.
Normalized dates describe observation months, not publication dates.

```text
<external-data-root>/housing_macro/
  parquet/<build_id>/hpi.parquet
  parquet/<build_id>/cpi.parquet
  reports/<build_id>/coverage.json
  reports/<build_id>/coverage.md
  manifests/normalization.json
```

The normalization manifest selects `current` and retains successful
`versions`. Each build records schema version, source hashes and metadata,
relative output/report paths, hashes, sizes, and Parquet row counts. If
inputs, schema version, and all four artifact hashes match, a repeat run
returns `cached`. Changed or damaged artifacts trigger a new build directory.
Schema or transformation changes must increment `SCHEMA_VERSION`.

Both Parquet files and both reports must finish before the manifest is
atomically replaced. A failed build leaves the previous selected version
available. Temporary files are cleaned on ordinary failures; forced process
termination may leave unselected files in scratch or a version directory.
Run only one writer against a data root at a time.

`coverage.md` summarizes full history and 2015 onward. `coverage.json` also
contains monthly valid-region counts and each region's first/last valid
month, null counts, and missing months between valid bounds. The report
calendar spans the entire table's observed range, with 2015 onward clipped
to available history. Every source region appears, even if it has no valid
values in a reporting window. For a region:

- `missing_date_months`: calendar months with no original column/record.
- `null_values`: original month cells/records whose value is null.
- `missing_values_between_valid_bounds`: both kinds of missing values
  between the first and last nonnull month; the two components are also
  reported separately. These fields are null when no valid bounds exist.

Missing identifiers, duplicate normalized keys, invalid dates, unparseable
values, and nonfinite values fail the build. Nonpositive finite values are
retained and counted separately. Synthetic tests use local CSVs and mocked
downloads; they do not contact providers.

Library readers are `quantbullet.data.zillow.scan_zhvi_csv(path,
geography="metro" | "state" | "zip")` and
`quantbullet.data.fred.scan_cpi_csv(path, series_id="CPIAUCNS")`. They return
Polars LazyFrames. Project snapshot selection, table validation, reporting,
and publication belong to this workflow. Geographic mapping, returns, lag,
current LTV, and loan joins remain a subsequent modeling step.
