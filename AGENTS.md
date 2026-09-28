# Repository instructions

## Python environment

- Use `uv sync` to create or update the project's `.venv`, and `uv add` to add project dependencies. Do not use bare `pip install` for project dependencies.
- Run Python scripts and tests with the interpreter in `.venv`, not a global or system Python interpreter.
- On Windows, use `.\.venv\Scripts\python.exe`; on macOS/Linux, use `./.venv/bin/python`.
