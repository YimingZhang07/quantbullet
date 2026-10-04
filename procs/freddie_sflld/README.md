# Freddie SFLLD process manual

Run commands from the repository root with the project `.venv`.
Data directories must be outside the repository.

## Setup

```powershell
$env:FREDDIE_DATA_ROOT = '<external-data-root>\freddie_sflld'
$env:MACRO_DATA_ROOT = '<external-data-root>\housing_macro'
```

Place the downloaded ZIP at
`$FREDDIE_DATA_ROOT/raw/full_set_standard_historical_data.zip`.
The final preparation step also requires the three Parquet files produced
by the [housing macro pipeline](../housing_macro/README.md).

## Execution order

| Step | Process | Purpose | Output |
| --- | --- | --- | --- |
| 1 | `build_parquet` | Convert source orig/perf files by vintage | `parquet/orig/`, `parquet/perf/`, `manifests/conversion.json` under the Freddie root |
| 2 | `build_sample_panel` | Apply product filters, sample loans and join static/performance history | `sampled_loans.parquet`, `panel/vintage=.../panel.parquet`, `sampling_summary.json` under its configured output root |
| 3 | `prepare_panel` | Clean the sampled panel and derive features, states and quality flags | One `panel.parquet` and `preparation_summary.json` under its configured output root |

```powershell
.\.venv\Scripts\python.exe -m procs.freddie_sflld.build_parquet
.\.venv\Scripts\python.exe -m procs.freddie_sflld.build_sample_panel --config procs/freddie_sflld/sample_panel.example.toml
.\.venv\Scripts\python.exe -m procs.freddie_sflld.prepare_panel --config procs/freddie_sflld/prepare_panel.example.toml
```

The final panel retains all raw observations. Targets and model-specific
filters are defined in subsequent modeling processes.

## Configuration and common options

- [Sample config](sample_panel.example.toml): vintage range, loan count, seed,
  optional `amortization_type` and `original_loan_term`, plus input/output roots.
  The example samples 500k 30-year FRM loans from 2015Q1 through 2026Q1.
- [Preparation config](prepare_panel.example.toml): sampled panel, macro and output roots.
  Copy examples to an external local TOML to customize them. `${NAME}` expands
  environment variables; relative paths resolve against the TOML directory.
- Conversion defaults to 2015Q1 through the latest available quarter.
  Use `--quarter 2015Q1` for one quarter or `--start 2020Q1 --end 2021Q4` for a range.
- Preparation accepts `--vintage 2015Q1` for a smaller run. It replaces the final
  output with that subset; use a separate output root to retain a full build.

## Reruns and outputs

Conversion reuses unchanged quarters. Sampling and preparation rebuild their
fixed outputs. Use dedicated output directories and one writer at a time.
Changes to derived feature formulas or names require step 3 and rebuilding
downstream modeling frames, fits and reports. Reuse the existing sampled panels
and macro Parquet files; conversion and sampling do not need to run again.
A failed sampling run may leave partial files; require a successful completion
and its summary before consuming the output. Preparation preserves the previous
output if conversion or validation fails.

Locate original Parquet files through `manifests/conversion.json`. Read the
final enriched table directly from the preparation output's `panel.parquet`.
See [panel preparation and feature construction](design/panel_preparation.md)
for the data flow, feature logic, status mappings and quality flags.
