# Contributing to EnTiSe

## Methods

Generator strategies subclass [`Method`](entise/core/base.py). Set a unique
`name`, supported `types`,
`required_keys` and `optional_keys`, `required_data` and `optional_data`,
and the `output_timeseries` and `output_summary` metadata.
Implement `generate(obj, data, results, ts_type)` so it returns a dictionary
with `Keys.TIMESERIES` and, when provided, `Keys.SUMMARY`.

Use `Columns`, `Objects`, and `Types` from [`entise.constants`](entise/constants/__init__.py)
for shared column names, object keys, and time series types. Name columns for
physical quantities using `name[unit]` with an optional height in metres, as in
`air_temperature[C]@2m`. Keep existing identifier and timestamp names.

Read the relevant coding constraints before editing [methods](entise/methods/README.md),
[core](entise/core/README.md), or [services](entise/services/README.md).

## Tests and changes

- Run the offline suite with `.venv/bin/python -m pytest tests --ignore=tests/installation`.
  Keep unit and integration tests offline by using fixtures or mocking external services.
- Run the full suite with `.venv/bin/python -m pytest tests`. The installation
  smoke test can access package indexes to install dependencies.
- Add an entry under `[Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md) when a change
  affects generated output or the public API.
- Never push directly to `main` or `develop`.
