# Housing macro data rules

For execution order, update commands and output paths, see [the manual](README.md).

## Sources

Download complete-history CSV snapshots for monthly Zillow ZHVI (metro,
state, and ZIP5), national CPI-U (`CPIAUCNS`, BLS via FRED), and weekly
30-year PMMS mortgage rates (`MORTGAGE30US`, Freddie Mac via FRED). The metro
ZHVI CSV includes the national row. Zillow files use All Homes, mid-tier,
smoothed, seasonally adjusted ZHVI; CPI and PMMS are not seasonally adjusted.

## Download snapshots and validation

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
[CPIAUCNS on FRED](https://fred.stlouisfed.org/series/CPIAUCNS), and
[MORTGAGE30US on FRED](https://fred.stlouisfed.org/series/MORTGAGE30US). Zillow download
paths can change; update the definitions in `quantbullet.data.zillow` when
necessary. The FRED CSV download endpoint is used here rather than its
credentialed observations API.

## Monthly table definitions

| Table | Columns | Unique key |
|---|---|---|
| HPI | provider, metric, geography_level, region_id, region_name, month, value; source geography metadata | provider + metric + geography_level + region_id + month |
| CPI | provider, series_id, month, value | series_id + month |
| PMMS | provider, series_id, month, value | series_id + month |

HPI uses `provider="zillow"`, `metric="ZHVI"`, and original dollar values.
The name HPI describes the table's role; ZHVI has not been converted to an
index. Source `country` becomes `national`, `msa` becomes `metro`, and
`state` and `zip` remain unchanged. `region_id`, `region_name`, `region_type`,
`state_name`, `state`, `city`, `metro`, and `county_name` are strings; ZIP5
names retain leading zeros. `size_rank` is nullable Int64. Missing source
metadata columns become null. Zillow RegionID is not a Freddie MSA code.

CPI uses `provider="bls"` and `series_id="CPIAUCNS"`; FRED remains the
distributor in source metadata. `month` is Date at the first of the month,
and `value` is Float64 in all three tables. Empty source values and FRED `.` are
null; missing source observations are retained.
Normalized dates describe observation months, not publication dates.

PMMS uses `provider="freddie_mac"` and `series_id="MORTGAGE30US"`. Values
remain in percent: `6.5` means `6.5%`, not `0.065`. The original weekly CSV
and its dates are retained in `raw/fred/pmms_30y/`; no weekly Parquet is needed.
Each monthly value is the simple mean of nonnull weekly observations whose
original dates fall in that calendar month. It is not weighted by days.
All-null months remain null; absent weeks or months are not filled.

PMMS excludes the calendar month containing the earlier of the UTC build
date and the current snapshot's `downloaded_at_utc`, and all later months.
The snapshot acquisition cutoff prevents a cached partial month from aging
into a complete month when the build is rerun. `last_checked_at_utc` is not
used for this cutoff because a cached download updates that timestamp
without fetching remote data. Refresh the source before rebuilding to
obtain newly completed months.

This calendar cutoff does not guarantee provider completeness or an
official historical release date. Monthly means describe their observation
month and must not be treated as known at that month's beginning. The PMMS
methodology changed on 2022-11-17; preserve the continuous FRED series without
adjusting history. See the [official change notice](https://news.research.stlouisfed.org/2022/11/changes-to-freddie-mac-dataset-in-fred/).


## Build validation and publication

Missing identifiers, duplicate normalized keys, invalid dates, unparseable
values, nonfinite values, and invalid ZIP5 strings fail the build. Normal
nulls and nonpositive finite values are retained. The command prints row
counts and month ranges. PMMS also validates weekly dates, duplicate weekly
keys, and finite values before aggregation, including records in excluded
months. All selected temporary Parquet files must finish before
existing outputs are replaced, so validation or conversion failures preserve
existing files. Each file replacement is atomic; replacement of multiple
files is not a transaction. Run one writer against a data root at a time.

The build reads current selected CSVs directly and writes fixed Parquet paths.
It has no build cache or normalization manifest; source hash verification
belongs to the download command.

## Coverage definitions

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

Nonpositive values are counted separately in the report. Synthetic tests use
local CSVs and mocked downloads; they do not contact providers. Geographic
mapping, returns, lag, current LTV, and loan joins remain a subsequent step.

## Library entry points

Library readers are `quantbullet.data.zillow.scan_zhvi_csv(path,
geography="metro" | "state" | "zip")` and
`quantbullet.data.fred.scan_cpi_csv(path, series_id="CPIAUCNS")`, plus
`scan_pmms_csv(path, series_id="MORTGAGE30US")` and
`aggregate_pmms_monthly(weekly, as_of=...)`. The PMMS reader keeps Date-type
`observation_date`; the aggregation validates weekly records and returns
monthly values, excluding the `as_of` month. The functions return
Polars LazyFrames and define the actual column types. The build script
selects inputs, merges the ZHVI tables, validates, and saves the outputs.
