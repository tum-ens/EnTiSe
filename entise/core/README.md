# Core

This package defines method interfaces and runs the strategies configured for
each object.

## Coding constraints

- Resolve strategies through the [registry](registry.py). The
  [runner](runner.py) resolves a required input that names another configured
  strategy before calling the dependent strategy and passes prior results to
  `generate`.
- Preserve the result contract at the runner boundary. A strategy returns a
  dictionary containing `Keys.TIMESERIES`. The [generator](generator.py) merges
  any `Keys.SUMMARY` entries into the object's summary.
