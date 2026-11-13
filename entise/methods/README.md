# Methods

This package contains the strategies that generate EnTiSe time series and the
auxiliary calculations they use.

## Coding constraints

- For a generator strategy, implement the [`Method` contract](../core/base.py).
  Choose a name that is unique regardless of case because registration uses
  the lowercase name. Keep the declared inputs and output metadata in sync
  with `generate`. The [Ruhnau heat pump method](hp/ruhnau.py) is an example.
- Use [`AuxiliaryMethod`](../core/base_auxiliary.py) for supporting calculations
  that prepare inputs with `get_input_data` and return a DataFrame from `run`.
- Prefix private helper modules with `_`. Import optional dependencies in the
  code path that needs them. The [load profile generator](electricity/utils.py)
  imports PyLPG when it runs a chunk.
