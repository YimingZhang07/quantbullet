# Repository instructions

## Python environment

- Use `uv sync` to create or update the project's `.venv`, and `uv add` to add project dependencies. Do not use bare `pip install` for project dependencies.
- Run Python scripts and tests with the interpreter in `.venv`, not a global or system Python interpreter.
- On Windows, use `.\.venv\Scripts\python.exe`; on macOS/Linux, use `./.venv/bin/python`.

## Public repository hygiene

- Do not add personal information, user-specific absolute paths, credentials, or local dataset contents to tracked code, tests, documentation, or examples.
- Use environment variables or generic placeholders for local paths. Keep raw data, generated Parquet files, and manifests outside the repository.
- Before committing, inspect the staged file list and diff for personal paths, secrets, and generated data.

## Process documentation

- Use the repository skill [`process-readme`](.agents/skills/process-readme/SKILL.md) when writing or restructuring workflow READMEs; it can be invoked as `$process-readme`.
- Keep READMEs under `procs/` as concise operating manuals: execution order, purpose, commands, configuration, and outputs.
- Put data definitions, calculation rules, and design reasoning in separate Markdown documents linked from the manual.
