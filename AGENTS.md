# Contributing to EnTiSe

## Methods

Generator strategies subclass [`Method`](entise/core/base.py). Set a unique
`name`, supported `types`,
`required_keys` and `optional_keys`, `required_data` and `optional_data`,
and the `output_timeseries` and `output_summary` metadata.
Implement `generate(self, obj=None, data=None, results=None, ts_type=..., **inputs)`
so it returns a dictionary with `Keys.TIMESERIES` and, when provided,
`Keys.SUMMARY`. Resolve the keyword inputs first with `self._process_inputs`
(or `_process_kwargs`), so every declared input can be passed by key or
Python name.

Use `Columns`, `Objects`, and `Types` from [`entise.constants`](entise/constants/__init__.py)
for shared column names, object keys, and time series types. Name columns for
physical quantities using `name[unit]` with an optional height in metres, as in
`air_temperature[C]@2m`. Keep existing identifier and timestamp names.

Read the relevant coding constraints before editing [methods](entise/methods/README.md),
[core](entise/core/README.md), or [services](entise/services/README.md).

## Tests and changes

Follow the [contribution workflow](docs/guides/CONTRIBUTING.md#workflow) from
scope and approval through review and publication. Record the agreed stopping
point for substantive work. Run tests from the repository root with your
development environment active.

- Run the offline suite with `pytest tests --ignore=tests/installation`.
  Keep unit and integration tests offline by using fixtures or mocking external services.
- Run the full suite with `pytest tests`. The installation
  smoke test can access package indexes to install dependencies.
- Add an entry under `[Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md) when a change
  affects generated output or the public API.
- Never push directly to `main` or `develop`.
