# Methods

This package contains the strategies that generate EnTiSe time series and the
auxiliary calculations they use.

## Coding constraints

- For a generator strategy, implement the [`Method` contract](../core/base.py).
  Choose a name that is unique regardless of case because registration uses
  the lowercase name. Keep the declared inputs and output metadata in sync
  with `generate`. The [Ruhnau heat pump method](hp/ruhnau.py) is an example.
- Declare each optional key's default in `optional_keys`, a mapping from key
  to default, and resolve the object's keys with
  `self.get_relevant_objects(obj, ts_type)` instead of looking them up one by
  one. Keep table lookups in the existing helpers.
- Give `generate` the shared signature
  `generate(self, obj=None, data=None, results=None, ts_type=<type>, **inputs)` and
  resolve the keywords first with `self._process_kwargs(obj, data, ts_type, **inputs)`.
  Direct calls then accept every declared input by key or by Python name. Map a
  Python name kept from an earlier release to its key in `input_aliases`.
- For a method that produces several time-series types from one calculation,
  subclass [`GroupedMethod`](../core/grouped.py) together with `Method` and
  implement `generate_combined`. Route a direct call with `ts_type` through
  `group_result`. The [RC models](hvac/_typed.py) are an example.
- Add a built-in strategy to the [registry](../core/registry.py) and declare its
  install extra in [pyproject.toml](../../pyproject.toml), even when it has no
  optional dependencies. Include the method extra in its type group. Verify
  strategy discovery and the declared dependencies in fresh processes.
- Declare auxiliary dependencies in the extras of the main methods that use
  them. Auxiliaries have no separate install extra.
- Use [`AuxiliaryMethod`](../core/base_auxiliary.py) for supporting calculations
  that prepare inputs with `get_input_data` and return a DataFrame from `run`.
- Prefix private helper modules with `_`. Import optional dependencies in the
  code path that needs them. The [load profile generator](electricity/utils.py)
  imports PyLPG when it runs a chunk.
