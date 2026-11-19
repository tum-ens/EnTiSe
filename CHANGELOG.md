
# Changelog

All notable changes to this project will be documented in this file.
See below for the format and guidelines for updating the changelog.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]
### Added
- Added a migration guide for 2.0.0 to the documentation. It lists every change that can alter results or break an existing setup, with the steps to adapt inputs. (`#116`)
- Added `Generator.generate(..., utc=False)`. By default every time series of an object is indexed in its site time zone (the `timezone` key, UTC if not set); `utc=True` indexes all of them in UTC. (`#116`)
- Added the time contract for 2.0.0 and the layer that implements it. Each run reads every table with timestamps once, as UTC instants in `datetime64[us]`; text timestamps must be ISO 8601. Naive timestamps are read as UTC with one warning per table, and errors name the table and timestamp. A new object key `timezone` sets the site's local calendar. Auxiliary tables align by timestamp, and a check warns when irradiance looks labelled at the interval end. The contract is documented in the workflow guide, and every method and the Open-Meteo weather service follow it. (`#116`)
- Added a shared contribution workflow that records approval scope and review
  evidence, with GitLab verification jobs for merge requests. (`#112`)
- Added contributor guidance for method contracts, offline testing, and method merge request review. (`#109`)

### Fixed
- Jordan-Vajen DHW no longer fails with `SeedSequence expects int` when a run mixes objects with and without `seed`. pandas stores the seed column as floats in that case; an integral float seed is now used as an integer. (`#116`)
- **Behavior change**: the RC models (1R0C, 1R1C, 5R1C, 7R2C), the Ruhnau heat pump, PV and wind follow the time contract. (`#116`)
  - They read the weather through the time layer, so mixed UTC offsets, named zones and typed or text timestamps give the same instants. The RC models, PV and wind require a regular time step.
  - Every output is indexed by the weather's instants in the object's `timezone`, or in UTC with `Generator.generate(..., utc=True)`. Before, the index depended on the method and on how the timestamps were written.
  - Each row is the interval that starts at its label. In the RC models, row 0 now holds the first interval's load and `temperature_in` is the temperature at the start of each interval, so row 0 is the initial temperature. A weather table with n rows gives n intervals instead of n − 1, and every RC model reports demand as the sum of load times step; 5R1C and 7R2C used the trapezoidal rule.
  - PV and the solar gains of the RC models take the sun's position at the midpoint of each interval instead of at its label. A warning names the weather table when its irradiance peaks about one step after solar noon, which indicates labels at the interval end; it appears once per run and longitude cell in each worker process.
  - Internal gains, ventilation and `table|column` references (also the legacy `<key>` with `<key>_COL` form) align with the weather by timestamp; before, a `table|column` power cap failed with `could not convert string to float`. A gains table read with `read_csv` gave NaN temperatures and 0 Wh heating in 5R1C and 7R2C; 1R0C and 1R1C used timestamped tables in row order and ignored their timestamps.
  - PV and wind use the table named by the object's `weather` key. Before, they always read `weather`.
  - Ruhnau's soil temperature estimate averages the last 24 hours of air temperature at any time step, not the last 24 rows.
  - The example weather (`examples/common_data/weather.csv`) is relabelled to interval starts: its irradiance columns move one row earlier and `solar_elevation_angle[degree]` is given at each interval's midpoint.
- Fixed outputs and examples that silently used the wrong values. (`#118`)
  - **Behavior change**: demandlib heat returned 2× the power at 30-minute steps and 4× at 15-minute steps, and half the power at 2-hour steps. Each step now carries the mean power of the hours it covers, starting at its own label, so a run over whole hours has the energy of the hourly run. Inputs at 30 or 15 minutes that end between two full hours no longer fail with a length mismatch; a run that starts or ends inside an hour still falls slightly short of the demand.
  - **Behavior change**: Jordan-Vajen DHW wrote the cold water temperature into `dhw:hot_water_temperature[C]`. The column now holds the hot water temperature.
  - **Behavior change**: an HVAC object that names a windows table missing from the data now raises a `ValueError`. Before, its solar gains were silently zero.
  - **Behavior change**: the 1R0C HVAC model dropped `latitude[degree]` and `longitude[degree]`, so its solar gains were always zero, even with windows. It now passes the location to the solar gains, as 1R1C does.
  - The 1R0C, 1R1C and 5R1C examples referenced a windows table named `windows_9buildings`, which does not exist, so their solar gains were zero. They now use `windows`.
  - The demandlib electricity example set `household_type`, which the method does not read, so every object used the default profile `h0_dyn`. It now sets `profile`.
  - The Jordan-Vajen DHW example and test objects set `temp_cold[C]` and `temp_hot[C]`, which the method does not read, so they ran with the default temperatures. They now set `cold_water_temperature[C]` and `hot_water_temperature[C]`.
  - The stored outputs of the 1R0C and Jordan-Vajen notebooks still show the old names until the notebooks are run again.
- Fixed four cases in which objects in one process affected each other or failed on shared input. (`#117`)
  - The Ruhnau heat pump no longer changes the caller's weather frame. It used to strip the column heights and replace the index, so a later wind object on the same frame failed with duplicate column labels.
  - The solar gains of the 1R1C, 5R1C and 7R2C HVAC models no longer fail with an `AttributeError` when the weather carries timestamps in a named time zone such as `Europe/Berlin`. The solar position is calculated from UTC instants, as before.
  - demandlib electricity no longer leaves the process with an error filter for all warnings. demandlib 0.2.2 sets one while it builds a standard load profile, which made a later object fail on any warning, for example HVAC parsing weather across a daylight saving change. EnTiSe now ignores that call and restores the caller's warning filters after demandlib runs. With pandas 3, this also lets demandlib electricity run despite demandlib's own deprecation warnings.
  - **Behavior change**: demandlib electricity and PyLPG read the datetimes frame passed in each call. They used to keep the first frame seen under a data key for the rest of the process, so a second run with different datetimes under the same key returned the earlier timestamps and values.
- **Behavior change**: `Generator.generate()` process workers now honor the mode
  set by `entise.set_accelerator()` on each call, including when joblib reuses
  workers. A forced Numba mode reports a missing dependency when the solver runs. (`#115`)
- Aligned the ventilation auxiliary output onto the model index in the 5R1C and 7R2C HVAC models. `_prepare_inputs` multiplied the auxiliary's series (carrying the user's table index) against the ventilation-split series (built on the weather index), so pandas aligned on the *union* of the two and silently produced an over-long, NaN-padded frame — `n+1` rows for a table with one extra timestamp, `2n` all-NaN rows for a table read from CSV with the default `RangeIndex`. The sensible solver reads `Hve_arr[t]` for `t < n` only and never noticed; the failure surfaced one call later in the latent-cooling post-pass as `ValueError: operands could not be broadcast together with shapes (n+1,) (n,)`. A new `align_auxiliary_series` helper in `entise/core/utils.py` now reindexes a `DatetimeIndex` result by label (a table not covering the full weather index is a hard error naming the gap, rather than silent NaN) and aligns an index-less result positionally after a length check — matching how R1C1 already consumes the same auxiliaries. (`#106`)
- Made ventilation timeseries usable on the 5R1C and 7R2C HVAC models; previously ventilation could only ever be a scalar there and any table reference raised. Three gaps stacked up: (1) both models read `ventilation_column`, `gains_internal_column`, `latitude[degree]` and `longitude[degree]` in `_get_relevant_objects` but declared none of them in `optional_keys`, so `Method._process_kwargs` — which filters against `required_keys + optional_keys` — silently stripped them on the keyword API, after which `VentilationTimeSeries` fell back to `str(object_id)` as the column name and raised a `Warning` pointing at the caller; (2) `_prepare_data_tables` filed the resolved table under the *canonical* key (`ventilation[W K-1]`) while the auxiliary strategies look it up by the *user's* key, so they received `None` and died on `None.loc` — tables are now reachable under both names, as they already were on R1C1; (3) `R7C2.optional_data` omitted `O.VENTILATION` entirely, so on that model the table was never resolved out of `data` at all. (`#106`)
- Fixed the solar-gain caches in `SolarGainsPVLib` (used by the 1R1C and 7R2C HVAC models, and through `SolarGainsISO13790` by 5R1C) returning results that belonged to another object or another weather frame. The caches were keyed by the position rounded to 0.1° while the solar position was calculated at the *exact* position of whichever object in that cell came first, so an object's gains depended on the order of the objects, and with `workers > 1` on how they were batched across processes. The solar position is now calculated at the centre of the 0.1° cell, so all objects in a cell share one result regardless of order. The caches were also keyed by a weather summary (length, first and last timestamp, mean GHI), so two weather frames with the same time grid and mean GHI but different irradiance shared plane-of-array irradiance; they are now kept per weather frame, identified by object identity as in `WeatherCache`. Added `clear_solar_cache()` to `entise.methods.auxiliary.solar.strategies`. **Behavior change**: solar gains change slightly for every object whose position is not a cell centre, because its solar position is now taken up to about 5 km away; results that previously depended on object order are now reproducible. Added an integration test that requires bit-identical time series for 1R1C, Ruhnau, Jordan-Vajen (with a seed) and pvlib across object order, worker count and batch size. (`#108`)

### Changed
- **Behavior change**: the Open-Meteo weather service returns data on the time contract. (`#116`)
  - Each row is labelled at the start of its hour. Open-Meteo labels hourly aggregates (radiation, precipitation, rain, snowfall, evapotranspiration, sunshine duration and gusts) at the end of the hour, so the service moves them one hour earlier and requests one more day so the last requested hour stays complete. Instantaneous values keep their labels.
  - `solar_elevation_angle[degree]` is the elevation at the middle of each hour instead of at its label.
  - `timezone=None` returns UTC days with UTC timestamps; before, it returned the location's local days with UTC timestamps. `timezone="auto"` returns timestamps in the zone Open-Meteo assigns to the location; before, it failed with `UnknownTimeZoneError`. Any other value must be an IANA time zone name and is checked before the request.
- **Behavior change**: the `file` method returns its table on the object's time axis. A table with timestamps is indexed by its instants in the object's `timezone`, or in UTC with `Generator.generate(..., utc=True)`. If the object names a `weather` table, the file is aligned to the weather's instants and must cover them; a table without timestamps is then used row by row at equal length, with a warning. A table without timestamps for an object without `weather` raises an error. Before, `file` returned the table unchanged, for example with a `RangeIndex` and a text `datetime` column. (`#116`)
- **Behavior change**: the calendar methods follow the local calendar of each object's new `timezone` key (UTC if unset), and their outputs are indexed by the run's instants in that zone, or in UTC with `Generator.generate(..., utc=True)`. (`#116`)
  - demandlib electricity, demandlib heat, DistrictHeatingSim and PyLPG read their engines at the site's legal wall clock. They used to take the first row's UTC offset as the clock for the whole run, so with the example weather every summer-time hour carried the value of the hour before, and with UTC input every hour was one to two hours off. On the 23-hour day of the change to summer time the skipped hour is not read; on the 25-hour day the repeated hour is read twice. Each row is the mean over the interval that starts at its label, also for steps that are not whole hours or that start between two engine labels.
  - `demand[kWh]` is the energy over the simulated run for demandlib electricity and DistrictHeatingSim, as it already was for demandlib heat. Electricity used to take the run's share of an annual profile, and DistrictHeatingSim scaled every calendar year to the full demand.
  - Jordan-Vajen DHW draws water per local day. Each event type's share of the mean daily volume comes from the weekly mean of its `probability_day`, and within the type `probability_day` sets a factor per weekday whose weekly mean is 1 (holidays use Sunday's factor). Weekday totals therefore vary, and the annual event mix follows the weekly-mean shares (Bath about 25 %). Before, the event mix used only Monday's `probability_day` (Bath about 15 %), and the per-day weights were lost, as were the seasonal weights: `seasonal_variation` changed nothing. The number of events per day and type is a Poisson draw and each event keeps its own size, so baths stay about 140 l. The volume over the run is the sum of the days' expected volumes; before, it was scaled to 365 days of demand whatever the run length. The seasonal phase uses the actual length of each year, and activity times between two rows of the run are counted in the row that contains them.
  - GeoMA and PHT read the electricity series by its timestamps, so they now run after the built-in electricity methods (before: `KeyError: 'datetime'`). The night window and the hold until 09:00 use the local clock, also on the days when the clocks change. The night hold starts once occupancy has lasted the new optional key `night_schedule_duration[h]` (default 1 h) instead of four samples, so it no longer depends on the time step.
  - A run whose first step crosses a clock change, that starts between two engine labels, that ends between two hours, or that touches a calendar year with a single row now runs. So does daily data labelled at local midnight, whose days last 23 or 25 hours when the clocks change: the calendar methods accept a step that is regular in elapsed time or on the site's local clock. Any other change of the step, such as a gap, raises an error naming the table and the instant instead of shifting all later rows. PyLPG raises if the engine leaves minutes without output instead of filling them with zero.
  - `Generator.generate` warns once per run about objects without `timezone`. The examples now set `timezone` to the zone of each object's holidays location, `Europe/Berlin` for the German sites; the heat example's US object now uses German holidays, because its temperatures come from the German example weather.
- **Behavior change**: the default install now provides the EnTiSe core. Install individual methods with extras such as `pip install "entise[pv-pvlib,hvac-1r1c]"`, or use `pip install "entise[all]"` for the previous feature coverage. Methods load on demand, and missing optional packages report the relevant install command. The weather service and example tools have their own extras. (`#111`)
- Documented how to contribute time-series methods in the shared `add-method` skill. (`#110`)

## [1.3.0] Latent cooling load and other HVAC improvements - 2026-07-05
### Added
- Added latent cooling load to the RC HVAC models (1R1C, 5R1C, 7R2C). The cooling output now carries three timeseries columns — `cooling:load[W]` (total = sensible + latent), `cooling:sensible_load[W]`, and `cooling:latent_load[W]` — while `cooling:demand[Wh]` and `cooling:load_max[W]` in the summary are the corresponding totals. Latent load is computed as a vectorised post-pass over the sensible-solver result, using outdoor relative humidity + surface pressure from the weather (both default in OpenMeteo pulls) plus an optional latent-internal-gains series. Since 1R1C carries no humidity state the post-pass runs entirely outside the numba-accelerated kernel. `power_cooling[W]` is now interpreted as the **total nameplate cap** (sensible + latent) with sensible-priority clipping to match how DX coils operate. Two new optional keys: `target_humidity_rel[1]` (default `0.5`, ASHRAE Standard 55 comfort-band midpoint) and `gains_internal_latent[W]` (default `0.0`, mirror of `gains_internal[W]`). Weather without humidity → latent forced to zero + one warning per missing column. **Behavior change**: users whose weather carries humidity + pressure will see their reported cooling demand increase to reflect the previously-missing latent portion — this is the intended physics fix. Well-behaved existing tests (weather without humidity) pass unchanged. (`#103`)

### Changed
- Clarified the semantic of `gains_internal[W]` as **sensible-only** (not total). Its latent companion is the new `gains_internal_latent[W]`. Users migrating from a pre-#103 setup who lumped a total metabolic gain (~130 W per seated occupant per ASHRAE) into `gains_internal[W]` should split it (~75 W sensible + ~55 W latent) to avoid double-counting the latent portion once latent cooling activates. No numerical change for weather without humidity or with `gains_internal_latent[W] = 0`. Docstring clarifications in `entise/constants/objects.py`, `entise/methods/hvac/R1C1.py`, `entise/methods/hvac/_latent_cooling.py`, and `entise/methods/auxiliary/internal/strategies.py`. (`#103`)
- Consolidated the specific-heat-of-air (`CP_AIR = 1000 J/(kg·K)`) and air-density (`RHO_AIR = 1.2 kg/m³`) constants into `entise/constants/constants.py`. Previously each lived as a magic number in `entise/methods/auxiliary/ventilation/strategies.py`, `R5C1.py`, `R7C2.py`, and the new `_latent_cooling.py`. The mass-flow reconstruction `m_dot = H_ve / c_p` in the latent post-pass now provably matches whatever the ventilation strategy uses. Historical `AIR_DENSITY` / `HEAT_CAPACITY` names in `ventilation.strategies` are kept as re-exports for backward compatibility. (`#103`)
- Added a 1R0C model for quick validations (`#93`, `!65`)
- Added an optional numba-accelerated solver path for the 1R1C HVAC model. Install with `pip install entise[numba]` and the accelerated path is used automatically. Behavior is controlled by the `ENTISE_ACCELERATOR` environment variable (`auto` | `numba` | `none`) or the `entise.set_accelerator()` API. Delivers a ~100–300× speedup on 8760-hour runs; output matches the numpy path to within a few ULPs.
- `power_heating[W]` and `power_cooling[W]` in the 1R1C HVAC model now accept a `pd.Series` aligned to the weather index in addition to a scalar float. Passing a series lets users define heating- and cooling-off periods (set to `0`) or seasonally varying power caps without splitting the simulation externally. Scalar behavior is unchanged. A series whose index does not match the weather index raises `ValueError`. (`#100`)
- Added an optional `deadband[K]` thermostat dead band (hysteresis) parameter to all three RC HVAC models (1R1C, 5R1C, 7R2C). When set, heating fires only when `T_in < T_min` and, once firing, keeps firing until `T_in ≥ T_min + deadband`; cooling mirrors around `T_max`. Default `0.0` collapses the state machine to aim-for-setpoint and preserves current output bit-for-bit. Useful for producing thermostat switching statistics that match real actuator cycling. (`#102`)
- Vectorized the impulse-response precomputation in the 1R1C solver so `G_tot`, `decay`, `gain`, and `T_ss_pas` are computed as numpy arrays once per run instead of per step; the Python loop now contains only the temperature recursion and the controller inversion. Output is byte-exact vs. the previous implementation; ~1.5× faster on the example set.
- Eliminated a per-object `.astype("datetime64[ns]")` on the full weather DATETIME column in R1C0 and R1C1 (only the delta between the first two timestamps is used to compute `Δt`). This alone cut per-object wall time from ~26 ms to ~2.4 ms in a 100-object benchmark on the standard example set — ~11× total speedup on the full pipeline with the numba accelerator enabled. Also preserved float32 dtypes through `SolarGainsPVLib`, `SolarGainsISO13790`, `InternalTimeSeries`, and `VentilationTimeSeries` to avoid follow-on float64↔float32 conversions in downstream solvers.

### Fixed
- Replaced every `pd.Timedelta(seconds=…)` / `pd.Timedelta(days=…)` / `pd.Timedelta("Xh")`-style call across the codebase with `pd.Timedelta(N, unit="…")`. On numpy 2.5 the keyword and string forms internally trigger the *"generic unit for NumPy timedelta is deprecated"* warning from pandas' Cython path, which pytest 9 escalates to a hard test failure — three tests (including the `electricity_demandlib` integration end-to-end) were failing on Python 3.12/3.13 in tox. The positional-value-plus-explicit-`unit` form is deprecation-free. Applied to `entise/methods/electricity/{demandlib,pylpg}.py`, `entise/methods/heat/{demandlib,districtheatingsim}.py`, `entise/methods/occupancy/utils.py`, `entise/services/weather/openmeteo.py`, the affected test files under `tests/**`, and the two `examples/electricity_*/runme.py` scripts. Full test suite is green on py310/py311/py312/py313.
- Restored the auto-generated method reference pages on Read the Docs. The eager submodule walkers in `entise.methods.__init__` and `entise.core.registry` were pulling in the private `_R1C1_numba` accelerator at import time; without the optional `numba` dependency this crashed the `entise.methods` package import, silently aborting the `generate_methods_docs` Sphinx hook and the `entise.methods.occupancy` autodoc entries. Both walkers now skip underscore-prefixed submodules (the accelerator stays lazy-imported by its dispatcher). Also cleared the remaining Sphinx warnings: reformatted the R1C1, R5C1, GeoMA, PHT class docstrings so docutils parses their bullet lists cleanly; de-roled the `entise.get_accelerator` / private-module cross-refs in `calculate_timeseries_1r1c`; fixed short title underlines in `docs/source/api/*.rst` and `services/weather.rst`; and injected a default Python3 kernelspec into copied example notebooks so `myst-nb` no longer emits "No source code lexer found" per cell. Local build is now warning-free.
- Fixed the docs build by adding method rsts directly and fixing the notebooks titles (`#96`, `!70`)
- Replaced the explicit-Euler integrator in the 1R1C HVAC model with the analytical exponential update, fixing the numerical instability that produced fictitious summer heating/cooling demand for low-R envelopes at hourly resolution (Δt/τ > 2). Well-behaved cases (Δt/τ ≪ 1) are numerically indistinguishable from the previous solver. (`#97`, `!71`)
- Replaced the per-module `_WEATHER_CACHE` dicts (keyed by weather-key name) with a shared `WeatherCache` utility keyed by the identity of the input DataFrame. Callers passing different DataFrames under the same weather-key (including the default `O.WEATHER`) no longer silently get each other's preprocessed weather. Applied to R1C0, R1C1, R5C1, R7C2 (HVAC) and `heat.demandlib`, `heat.districtheatingsim`. Also fixed a latent bug in the two heat modules where the cache was consulted but never populated, so every call was a miss. (`#99`)
- Wired the `active_ventilation`, `active_gains_internal`, and `active_gains_solar` flags into the pipeline for all three HVAC models (R1C1, R5C1, R7C2). Previously the flags were stored on the resolved object but never consulted; setting them to `False` had no effect. Now the flags gate the respective auxiliary via the `SolarGains`, `InternalGains`, and `Ventilation` selectors, and via matching call-site checks in R5C1 (`SolarGainsISO13790`, `VentilationTimeSeries`) and R7C2 (`VentilationTimeSeries`) which bypass the selectors. **Behavior change**: users who set any of these flags to `False` in their objects will now see the corresponding auxiliary zeroed out. Defaults are unchanged (all True) so callers who never touched the flags are unaffected. (`#98`)


## [1.2.0] New methods, batching and benchmarking script - 2026-02-26
### Added
- Added benchmarking script for comparing different methods (`#87`, `!59`)
- Added heating method based on demandlib's BDEW method (`#70`, `!63`)
- Added occupancy detection method GeoMA (`#58`, `!43`)
- Added occupancy detection method PHT (`#66`, `!46`)
- Added electricity method based on demandlib's BDEW method (`#56`, `!48`)
- Added optional batching to reduce parallel overhead and improve throughput (`#75`, `!67`)
- Added heating method based on districtheatingsim's BDEW method - only works with python 3.11 (`#78`, `!60`)
- Added electricity method based on pyLPG (`#62`, `!52`)

### Changed
- Changed internal naming of core methods and parameters for improved clarity (`#90`, `!62`)
- Update documentation to reflect new methods and changes (`#89`, `!61`)

### Fixed
- Ensure arrays are 1‑D via `.ravel()` in 7R2C and 5R1C HVAC models to prevent shape errors during computation. (b6d26cd)
- Handle Windows-specific edge case in `_silence_fds` to avoid `WinError 1`; allow override via environment variable. (93b300b)
- Handle warnings gracefully in `examples/benchmark.py` to prevent run interruptions; warnings are logged. (b85ab33)
- Ensure all methods are loaded before accessing or listing strategies in `core/registry.py`. (7adb130)
- Correct naming for `POWER_COOLING` and `POWER_HEATING` in `constants/objects.py`. (4c0ef9d)

## [1.1.0] New HVAC models and performance improvements - 2025-12-16
### Added
- Added 5R1C HVAC model based on ISO 13790 (`#80`, `!54`)
- Added 7R2C HVAC model based on VDI 6007 (`#81`, `!55`)

### Changed
- Improved computational speed of the 1R1C HVAC model by roughly 10x hitting architectural limits (`#83`, `!56`)

## [1.0.0] New naming scheme and methods - 2025-10-28
### Added
- Added a weather service to download weather data directly (`#33`,`!31`)
- Added wind power generation method based on windpowerlib (`#35`, `!33`)
- Added a ventilation auxiliary class similar to internal gains to allow for time series (`#43`, `!38`)
- Added heat pump COP time series generation method based on Ruhnau et al. (`#37`,`!34`)

### Changed
- Renamed weather and other columns to use the following convention: name[unit]@height (e.g., temp[C]@2m) (`#41`, `!44`)
- Changed timezone format in SolarGainsPVLib to IANA-based timezone handling due to breaking PVLib changes (`#77`, `!51`)

### Fixed
- Fixed issue that failed 1R1C HVAC model when weather time series was not under "weather" key (`#75`, `!49`)

## [0.2.1] Hotfix - 2025-06-25
### Fixed
- Fixed exposure of methods which was taken out by the githooks (`#38`, `!35`)

## [0.2.0] New architecture, methods and packaging - 2025-06-04
### Added
- Added a simple RC model for HVAC time series (`#12`, `!6`)
- Added simpler functionality for dependent methods (`#18`, `!12`)
- Added basic documentation of the package (`#13`, `!8`)
- Added a dhw method based on the method by Jordan et. al. in DHWCalc (`#16`, `!15`)
- Added a PV generation method based on pvlib (`#29`, `!22`)
- Added direct access methods to provide two ways for interacting with the tool (batch & singular) (`#31`, `!26`)
- Converted the project into a proper Python package with modern tooling (`#32`, `!27`)
- Added support for automatic versioning using git tags with hatch-vcs (`#32`, `!27`)
- Added CI/CD configurations for both GitHub and GitLab (`#32`, `!27`)
- Added support for Python 3.10-3.13 (`#32`, `!27`)

### Changed
- Restructured entire architecture towards a pipeline- and strategy-based approach to make methods more flexible (`#18`, `!12`)
- Replaced setuptools with hatchling for modern build system (`#32`, `!27`)
- Replaced flake8, black, and isort with ruff for faster linting and formatting (`#32`, `!27`)
- Replaced pip with uv for faster dependency management (`#32`, `!27`)
- Updated pvlib to version 0.12.0 with different (IANA-based) timezone handling (`#32`, `!27`)
- Updated Python version requirement to 3.10 or newer (`#32`, `!27`)

## [0.1.0] Initial Release - 2024-11-04
### Added
- Initial setup of the project with initial architecture (no methods added yet)

---

# Guidelines for Updating the Changelog
## [Version X.X.X] - YYYY-MM-DD
### Added
- Description of newly implemented features or functions, with a reference to the issue or MR number if applicable (e.g., `#42`).

### Changed
- Description of changes or improvements made to existing functionality, where relevant.

### Fixed
- Explanation of bugs or issues that have been resolved.

### Deprecated
- Note any features that are marked for future removal.

### Removed
- List of any deprecated features that have been fully removed.

---

## Example Entries

- **Added**: `Added feature to analyze time-series data from smart meters. Closes #10.`
- **Changed**: `Refined energy demand forecast model for better accuracy.`
- **Fixed**: `Resolved error in database connection handling in simulation module.`
- **Deprecated**: `Marked support for legacy data formats as deprecated.`
- **Removed**: `Removed deprecated API endpoints no longer in use.`

---

## Versioning Guidelines

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):
- **Major** (X): Significant changes, likely with breaking compatibility.
- **Minor** (Y): New features that are backward-compatible.
- **Patch** (Z): Bug fixes and minor improvements.

**Example Versions**:
- **[2.1.0]** for a backward-compatible new feature.
- **[2.0.1]** for a minor fix that doesn’t break existing functionality.

## Best Practices

1. **One Entry per Change**: Each update, bug fix, or new feature should have its own entry.
2. **Be Concise**: Keep descriptions brief and informative.
3. **Link Issues or MRs**: Where possible, reference related issues or merge requests for easy tracking.
4. **Date Each Release**: Add the release date in `YYYY-MM-DD` format for each version.
5. **Organize Unreleased Changes**: Document ongoing changes under the `[Unreleased]` section, which can be merged into the next release version.
