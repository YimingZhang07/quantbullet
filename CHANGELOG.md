# Changelog

<!--next-version-placeholder-->

## v0.1.1

- Modernize `pyproject.toml`: add project metadata, classifiers, and repository URLs
- Bump Python requirement to `>=3.10` (3.9 is EOL)
- Update dependency version floors
- Remove unused `importlib-resources` dependency (stdlib in 3.10+)
- Remove conflicting `nbsphinx` and unused `sphinx-rtd-theme` dev dependencies
- Single-source package version via `importlib.metadata`

## v0.1.0 (19/09/2023)

- First release of `quantbullet`! This release includes:
  - Statistical Jump Model in Discrete state setting