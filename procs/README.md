# Process manual

`procs` contains runnable workflows; reusable data functions live in
`src/quantbullet/data/`. Run commands from the repository root using the
project `.venv`, with all data stored outside the repository.

## Execution order

| Step | Process | Purpose |
| --- | --- | --- |
| 1 | `housing_macro.download` | Download macro source CSVs |
| 2 | `housing_macro.build_parquet` | Build monthly HPI, CPI and PMMS tables |
| 3 | `freddie_sflld.build_parquet` | Convert the downloaded loan archive to orig/perf Parquet |
| 4 | `freddie_sflld.build_sample_panel` | Sample loans and join their full reported histories |
| 5 | `freddie_sflld.prepare_panel` | Clean and enrich the sampled panel, retaining all observations |

Macro download/build and Freddie conversion/sampling are independent.
Panel preparation requires both the sampled panel and the macro tables.
`housing_macro.coverage` optionally generates inspection reports after the
macro tables are built. Model targets and training filters are subsequent work.

## Detailed manuals

- [Housing macro](housing_macro/README.md): setup, commands, updates and outputs.
- [Freddie SFLLD](freddie_sflld/README.md): conversion, sampling and panel preparation.

Data definitions and design details are linked separately from each manual.
