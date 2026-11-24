---
name: add-method
description: "Add a time-series generator strategy to EnTiSe. Use for a new Method implementation that needs registry discovery, offline tests through Generator, and method reference documentation."
---

# Add an EnTiSe method

## Choose a model to follow

Read the applicable contributor guidance and find an existing strategy of the
same time-series type. For an example spanning implementation, tests, and
reference docs, see [Ruhnau](../../../entise/methods/hp/ruhnau.py),
[its unit tests](../../../tests/unit/methods/hp/test_ruhnau.py), and
[its method page](../../../docs/source/methods/hp/ruhnau.rst).

## Implement the method contract

- Subclass [`Method`](../../../entise/core/base.py). Choose a case-insensitive
  unique strategy name across all types. Declare the supported `Types`,
  required and optional object keys and data, and output metadata. Reuse
  [`Columns`, `Objects`, and `Types`](../../../entise/constants/__init__.py)
  and follow `name[unit]` with an optional `@height` for physical quantities.
- Implement `generate(self, obj=None, data=None, results=None, ts_type=...,
  **inputs)` with `Keys.TIMESERIES` and optional `Keys.SUMMARY` in the result.
  Resolve the keyword inputs first with `self._process_inputs` (or
  `_process_kwargs`), so every declared input can be passed by key or Python
  name. Declare each optional key's default in `optional_keys` (a mapping
  from key to default) and read the object's keys with
  `self.get_relevant_objects(obj, ts_type)`, which applies the type prefix,
  the plain key and the default in that order. Use the declared inputs and keep
  the output metadata consistent with the returned columns and units. If the
  method consumes another strategy's output, follow how the
  [runner](../../../entise/core/runner.py) resolves that dependency.
- Add the method to the [built-in registry](../../../entise/core/registry.py)
  and declare its extra and type group in [pyproject.toml](../../../pyproject.toml).
  Declare an empty extra when the method needs only core dependencies.
  Expose the class lazily through its type package if that is part of the public
  interface. Import optional dependencies where they are used and report the
  install command when a required top-level package is absent.
- Include auxiliary dependencies in the main method's extra. Auxiliaries
  are installed with their main methods.

## Verify and document

- Use small offline fixtures to check representative values, index alignment,
  output columns and units, and the physical bounds or other invariants that
  apply to this method. Check repeatability when the method uses randomness.
- In a fresh process, verify the method through
  [`Generator`](../../../entise/core/generator.py) without importing the new
  class first. Set the strategy name for its type on a test object, run with
  `raise_on_error=True`, and assert the result for that object and type. A
  direct import can register the class and mask a discovery problem. The
  subprocess should exercise the method with only its declared extra and check
  that it leaves other method modules unloaded. The
  [PV integration test](../../../tests/integration/pv_pvlib/test_pv_pvlib.py)
  shows how to exercise this entry point with local inputs.
- Document inputs, units, assumptions, and returned values in the method
  docstrings and metadata. The [Sphinx generator](../../../docs/source/generate_methods_docs.py)
  builds the method pages and indexes from these sources. If the docs environment
  is available, run `make -C docs html` and verify the new method appears in its
  type index. Inspect generated diffs for unrelated pages. Update an example
  when it helps explain how to use the method.
- Add an Unreleased entry in [`CHANGELOG.md`](../../../CHANGELOG.md) for the
  new public method. In the [method MR template](../../../.gitlab/merge_request_templates/Method_Merge_Request.md),
  describe the new method and report the focused test and documentation results.
