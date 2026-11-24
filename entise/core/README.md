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
- A method that produces several time-series types from one calculation
  subclasses [`GroupedMethod`](grouped.py) and implements `generate_combined`.
  The runner calls its `generate_group` once for the types of `types` that
  select the method on the object, and expects a result for each of them. Keep
  method-specific rules out of the runner and out of `GroupedMethod`; the
  [RC models](../methods/hvac/_typed.py) add theirs in a subclass.
- Read timestamps through the [time layer](_time.py). Use `run_time(data)` to
  get the run's normalized tables, `site_zone(obj)` for the local calendar, and
  `RunTime.align` for auxiliary tables. Do not parse `datetime` columns in a
  method, and do not change the frames in `data`.
