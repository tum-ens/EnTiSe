
# Changelog

All notable changes to this project will be documented in this file.
See below for the format and guidelines for updating the changelog.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]
### Added
- Added defaults to method input declarations and one resolver for an object's keys. (`#119`)
  - `optional_keys` may map each key to its default; a list still declares keys without defaults. `Method.key_defaults()` returns every declared key with its default.
  - `Method.get_relevant_objects(obj, ts_type)` resolves each declared key from the key prefixed with the type, then the plain key, then the default. `None`, NaN and `pd.NA` count as missing, `False` and `0` are values, and `id` and `timezone` are read without a prefix. It returns only declared keys that have a value.
  - It also resolves several types at once, as one calculation for heating and cooling needs. A key given under two of them with different values raises a `ValueError`.
  - Every built-in method resolves its keys through it and declares its defaults in `optional_keys`. 5R1C and 7R2C no longer keep a separate list of object keys.
- Added iteration and value membership to `Columns`, `Objects`, `Types` and `Keys` in `entise.constants`. Their constants retain their plain-string types and values. `len()` counts constants and `items()` yields name-value pairs in declaration order. **Behavior change**: these classes cannot be instantiated, and their attributes are read-only. (`#59`)
- Added a migration guide for 2.0.0 to the documentation. It lists every change that can alter results or break an existing setup, with the steps to adapt inputs. (`#116`)
- Added a run seed, `Generator.generate(..., seed=None)`, that makes Jordan-Vajen DHW reproducible without a `seed` per object. Each object and type draws its own stream, derived from the run seed, the object's `id` and the type, so results do not depend on the order of the objects, the number of workers, the batch size or the process. An object's own `seed` takes precedence over the run seed. Seeded Jordan-Vajen results still differ from 1.3.0, because `#116` changed how the method draws water. Without any seed, results still differ between runs. PyLPG is not seeded: pylpg chooses its household members with Python's global random numbers. (`#17`)
- Added `Generator.generate(..., utc=False)`. By default every time series of an object is indexed in its site time zone (the `timezone` key, UTC if not set); `utc=True` indexes all of them in UTC. (`#116`)
- Added the time contract for 2.0.0 and the layer that implements it. Each run reads every table with timestamps once, as UTC instants in `datetime64[us]`; text timestamps must be ISO 8601. Naive timestamps are read as UTC with one warning per table, and errors name the table and timestamp. A new object key `timezone` sets the site's local calendar. Auxiliary tables align by timestamp, and a check warns when irradiance looks labelled at the interval end. The contract is documented in the workflow guide, and every method and the Open-Meteo weather service follow it. (`#116`)
- Added a shared contribution workflow that records approval scope and review
  evidence, with GitLab verification jobs for merge requests. (`#112`)
- Added contributor guidance for method contracts, offline testing, and method merge request review. (`#109`)

### Fixed
- **Behavior change**: 7R2C weights its equivalent outdoor temperature as VDI 6007 does: each element's share is its own `U·A` over the exterior envelope's `U·A`. (`#131`, `#133`)
  - Before, the window share added `u_value × area` over every row of the windows table. With the shared example table, that is 9 buildings' windows, 761 W/K.
  - That sum exceeded the envelope's conductance, so the opaque share fell to zero. The equivalent temperature became the air temperature amplified, for example 3.4 × `T_air` for `SFH_mid`.
  - Windows are now matched by `id`, as for the solar gains.
  - The envelope's conductance is `1 / (R_1_AW + R_rest_AW + R_in)` (VDI 6007 eq. 27/28), with `R_in` the inner surface resistance of the exterior walls, recovered from `R_alpha_star_IL`, `R_alpha_star_AW` and `R_alpha_star_IW`. Before, it was `1 / (R_1_AW + R_rest_AW)`, which understated the window share. The window share is capped at 1.
  - On the 7R2C example, cooling falls from 2.9–34.7 MWh to 0.5–4.7 MWh per building, and heating changes from 7.9–87.5 MWh to 17.0–83.8 MWh.
  - `T_eq_alpha_SW[1]` and `T_eq_h_o[W m-2 K-1]` now affect objects whose windows table has `u_value`; before, the zero opaque share removed their effect.
- **Behavior change**: the 1R0C HVAC model keeps the indoor temperature between `min_temperature[C]` and `max_temperature[C]`, as 1R1C, 5R1C and 7R2C do. (`#129`)
  - In each interval the zone takes its free-running temperature `T_out + gains / H`. Heating lifts it to the minimum and cooling lowers it to the maximum, within their power limits.
  - Before, 1R0C held the indoor temperature at `init_temperature[C]` in every step. It heated only when the outdoor air was at or below the minimum, and cooled only when the outdoor air was at or above the maximum.
  - Cooling therefore missed hours in which gains pushed the zone above the maximum, and cooled down to `init_temperature[C]` instead of the maximum.
  - `init_temperature[C]` now has no effect, because the zone has no thermal mass.
  - On the 1R0C example (20–25 °C), heating is unchanged, and cooling rises from 2.1–8.8 MWh to 4.7–20.3 MWh per building.
  - The example no longer sets `set_temperature[C]` and `capacitance[J K-1]`, which 1R0C does not read. The stored outputs of its notebook show the old values until it is run again.
- **Behavior change**: the RC models and WPLib use the inputs they declare. (`#119`)
  - 1R0C, 1R1C, 5R1C and 7R2C use an object's `gains_solar[W]` in place of calculated solar gains: a constant in W, `table|column`, or a table whose column is the object's `id` (or whose only column holds the gains). Before, it was accepted and ignored. In 1R1C, 5R1C and 7R2C, `active_gains_solar=False` still switches solar gains off.
  - 5R1C and 7R2C use `H_ve[W K-1]` in place of the ventilation input, in the same forms; `ventilation_split` then divides it. Before, it was ignored.
  - 5R1C uses `ventilation_split`. Before, it always used 0. The split matters only when `supply_temperature[C]` differs from the outdoor air.
  - WPLib uses `wind_model`: a mapping of windpowerlib ModelChain settings, such as `{"wind_speed_model": "hellman"}`, or the name of one in the data. It replaces the matching defaults; an unknown setting raises a `ValueError`, and so does a setting that needs a variable at two heights the weather does not have. Before, it was declared but never read, and the defaults always applied. Two objects of the wind example name its `data/system.json`; their generation falls by 1.9 % and 2.8 %. That file now sets `temperature_model` to `linear_gradient`, because the example weather has the air temperature at one height and `interpolation_extrapolation` failed there.
  - 1R1C declares `latitude[degree]` and `longitude[degree]`, which it already read for solar gains. Direct calls accept `latitude=` and `longitude=`, and `input_names()` lists them.
- **Behavior change**: 1R0C honors `active_ventilation=False` for both scalar and timestamped ventilation inputs. The flag defaults to `True`. (`#123`)
- **Behavior change**: a ventilation table column named with EnTiSe's own unit spelling, such as `typical[W K-1]`, `typical[m3 h-1]` or `typical[h-1]`, is read in that unit. Before, only `W/K`, `m3/h` and `1/h` were recognised, and any other unit was read as air changes per hour, so 61 W/K in a 140 m² zone became about 7,400 W/K. Both spellings are now accepted. An unrecognised unit is still read as air changes per hour, with a warning that names the unit; a column without a unit, such as an object id, is read as air changes per hour as before. (`#107`)
- **Behavior change**: the 1R0C HVAC model converts a ventilation table in air changes per hour with the object's `area[m2]` and `height[m]`, as 1R1C, 5R1C and 7R2C do. Before, it dropped both keys, so the conversion always used the default zone of 1 m² and 2.5 m, whatever the object's size. Objects without area or height still use those defaults. Ventilation given as a number or in `W K-1` or `m3 h-1` is unchanged. (`#121`)
- **Behavior change**: the Open-Meteo weather service names `terrestrial_radiation` `toa_incoming_shortwave_flux[W m-2]` and `terrestrial_radiation_instant` `toa_incoming_shortwave_flux_instant[W m-2]`. Both hold the solar radiation at the top of the atmosphere. Before, both were named `surface_upwelling_longwave_flux_in_air[W m-2]`, a different quantity, so requesting both gave two columns with the same name. The hourly mean still moves to the start of its hour and the instantaneous value keeps its label. The example weather (`examples/common_data/weather.csv`), which the service produced, carries the new column name; no method reads this column. (`#122`)
- **Behavior change**: when two types of one object report the same summary column, such as `heating:demand[Wh]` from an HVAC model and from demandlib heat, both values are kept, each prefixed with its type (`hvac:heating:demand[Wh]`, `heating:heating:demand[Wh]`), with a warning. Before, the type that ran last silently overwrote the other. Columns that only one type reports keep their names. (`#116`)
- Jordan-Vajen DHW no longer fails with `SeedSequence expects int` when a run mixes objects with and without `seed`. pandas stores the seed column as floats in that case; an integral float seed is now used as an integer. (`#116`)
- **Behavior change**: the RC models (1R0C, 1R1C, 5R1C, 7R2C), the Ruhnau heat pump, PV and wind follow the time contract. (`#116`)
  - They read the weather through the time layer, so mixed UTC offsets, named zones and typed or text timestamps give the same instants. The RC models, PV and wind require a regular time step.
  - Every output is indexed by the weather's instants in the object's `timezone`, or in UTC with `Generator.generate(..., utc=True)`. Before, the index depended on the method and on how the timestamps were written.
  - Each row is the interval that starts at its label. In the RC models, row 0 now holds the first interval's load and `temperature_in` is the temperature at the start of each interval, so row 0 is the initial temperature. A weather table with n rows gives n intervals instead of n − 1, and every RC model reports demand as the sum of load times step; 5R1C and 7R2C used the trapezoidal rule.
  - PV and the solar gains of the RC models take the sun's position at the midpoint of each interval instead of at its label. A warning names the weather table when its irradiance peaks about one step after solar noon, which indicates labels at the interval end; it appears once per run and longitude cell in each worker process.
  - Internal gains, ventilation and `table|column` references align with the weather by timestamp; before, a `table|column` power cap failed with `could not convert string to float`. A gains table read with `read_csv` gave NaN temperatures and 0 Wh heating in 5R1C and 7R2C; 1R0C and 1R1C used timestamped tables in row order and ignored their timestamps.
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
- The method pages list the summary keys that pvlib, wplib, Jordan-Vajen DHW, demandlib heat and DistrictHeatingSim report. Their `output_summary` declared other names, such as `generation[Wh]_pv` for `pv:generation[Wh]`, `dhw_volume_total` for `dhw:volume_total[l]`, and `heating:demand[kWh]` for `heating:demand[Wh]`. A test now checks the declared keys against each example's summary. (`#130`)
- **Behavior change**: an HVAC object that names a windows table but would get no solar gains from it raises a `ValueError`. Before, its solar gains were silently zero. (`#134`)
  - This happens when the object has no `latitude[degree]` and `longitude[degree]`, or when the table has no row with its `id`, including an `id` of `"1"` against a table id `1`. The error names the object and table and, for an `id`, shows both types.
  - Objects with `active_gains_solar` set to False or with supplied `gains_solar[W]` are not checked.
  - The 7R2C example named its windows without a location, so it ran without window solar gains. It now sets `latitude[degree]` and `longitude[degree]`; its cooling rises from 0.5–4.7 MWh to 3.5–14.0 MWh per building and its heating falls from 17.0–83.8 MWh to 12.7–71.2 MWh. Its stored notebook outputs show the old values until it is run again.
- A `Method` may declare inputs that differ only in their height, such as `wind_speed[m s-1]@10m` and `wind_speed[m s-1]@100m`, which EnTiSe's column names allow. Their shared Python name is dropped and they are set by their keys; using the short name raises a `TypeError` that lists them. Before, such a class failed at definition. (`#134`)
- The examples set only keys their methods read: the 7R2C example drops `alpha_sc`, the wind example drops `manufacturer`, and 7R2C declares `windows`, which it already read. A test checks the objects of every example; before, it checked two. (`#134`)
- **Behavior change**: Jordan-Vajen DHW reads the activity and demand tables an object names with `dhw_activity` and `dhw_demand_per_size[m2]`. Before, it ignored the names: it read the activity table only from `data["dhw_activity"]`, and it always used the bundled demand table. A named table that is missing from the data raises a `ValueError`. Objects that name no table use the bundled tables, as before. (`#134`)
- The warning about irradiance that peaks after solar noon names both common causes: labels at the interval end, or local timestamps without an offset, which are read as UTC. Before, it advised only shifting the irradiance. (`#134`)

### Changed
- Every method takes the same arguments when called directly: `generate(obj=None, data=None, results=None, ts_type=..., **inputs)`. (`#86`)
  - Any input the method declares can be passed as a keyword, by its key (`**{"area[m2]": 140}`) or by its Python name, which is the key without its unit and height (`area=140`). The names of earlier releases keep working: `temp_init`, `temp_min` and `temp_max` in 1R0C and 1R1C, `temp_water_cold` and `temp_water_hot` in Jordan-Vajen DHW, `demand_kwh` in demandlib electricity, `annual_demand_kwh` in demandlib heat and DistrictHeatingSim, `temp_sink` and `temp_water` in Ruhnau, and `file` in the file loader.
  - Call the method's `input_names()` to list accepted names and the input keys they set.
  - Keywords override the same keys in `obj`, also a key prefixed with the type, such as `hvac:area[m2]`. A DataFrame or mapping for a table input, such as `weather=df`, goes into `data`, and the object refers to it by its key; before, `weather=df` failed with `unhashable type: 'DataFrame'`. A string names a table already in `data`. GeoMA and PHT take the electricity series as `electricity=df` in place of the electricity result. `obj` can also be a row of an objects table (a Series); it is not changed.
  - **Behavior change**: an unknown keyword raises a `TypeError` that names the method and lists its inputs, and so does one input passed under two names. Before, 5R1C and 7R2C silently ignored unknown keywords, and demandlib electricity silently ignored `weather`, which it does not use; the other methods already raised a `TypeError` for a keyword outside their parameter list.
  - **Behavior change**: keywords that were accepted but never reached the method now take effect, so direct calls that pass them give different results: `latitude`, `longitude`, `power`, `azimuth`, `tilt` and `altitude` in pvlib; `power` and `hub_height` in wind; `dwelling_size`, `dhw_demand_per_size`, `temp_water_cold` and `temp_water_hot` in Jordan-Vajen DHW; the demand in demandlib electricity, demandlib heat and DistrictHeatingSim; `temp_sink`, `temp_water`, `hp_system` and `cop_coefficients` in Ruhnau; `file` in the file loader; and in 1R0C and 1R1C every keyword except `weather` and the `active_*` flags. Runs through the `Generator` give the same results as before.
  - **Behavior change** for custom `Method` subclasses:
    - `_process_kwargs(obj, data, ts_type, **inputs)` takes the type as its third argument.
    - Like `_process_inputs`, it raises a `TypeError` for a keyword the method does not declare. Before, it ignored such keywords and accepted a declared input only under its exact key.
    - Defining a subclass raises a `TypeError` when two declared inputs share a Python name, or when an alias names an undeclared input.
  - Ruhnau declares `cop_coefficients`, which it already read from the object, as an optional key. PyLPG, demandlib electricity and 1R0C declare their tables in `required_data` and `optional_data` instead of `required_timeseries` and `optional_timeseries`, which nothing read, so `datetimes=df` and `weather=df` work there too.
- **Behavior change**: EnTiSe requires Python 3.11 to 3.14 (`>=3.11,<3.15`); before, it supported 3.10 to 3.13. Python 3.10 has reached its end of life, and pvlib 0.16 no longer supports it. The main pipeline tests 3.11 to 3.14. (`#120`)
- EnTiSe supports pandas 2 and pandas 3 (`pandas>=2.2.2,<4`); before, it required pandas below 3. The merge request pipeline tests both pandas lines, and the main pipeline also tests the oldest supported pandas. Results are identical on pandas 2. (`#94`)
  - The methods' own code no longer emits pandas deprecation warnings on pandas 3, and the tests fail on any `FutureWarning`, or pandas 3 deprecation, raised from EnTiSe code.
  - **Behavior change**: importing the PV method no longer switches off every warning in the process (`warnings.filterwarnings("ignore")`).
  - **Behavior change**: the PV summary values are plain integers; before, each was a one-element Series, so code that reads them with `.iloc[0]` fails.
  - The example scripts and notebooks run on pandas 2 and 3 and with current matplotlib (the `examples` extra now needs matplotlib 3.9 or newer), and `examples.utils.run_simulation` raises an object's error instead of returning an error column.
- **Behavior change**: an auxiliary table with timestamps (internal gains, ventilation, `file`) that holds instants between the weather's time steps, such as 15-minute values on hourly weather, is averaged over each weather step, with a warning. Before, it was sampled at the weather's steps, which changed its energy. It must still contain every weather instant, so a table shifted against the weather's steps raises an error. Rows before the first or after the last weather step are still ignored. (`#116`)
- **Behavior change**: the Open-Meteo weather service returns data on the time contract. (`#116`)
  - Each row is labelled at the start of its hour. Open-Meteo labels hourly aggregates (radiation, precipitation, rain, snowfall, evapotranspiration, sunshine duration and gusts) at the end of the hour, so the service moves them one hour earlier and requests one more day so the last requested hour stays complete. Instantaneous values keep their labels.
  - `solar_elevation_angle[degree]` is the elevation at the middle of each hour instead of at its label.
  - `timezone=None` returns UTC days with UTC timestamps; before, it returned the location's local days with UTC timestamps. `timezone="auto"` returns timestamps in the zone Open-Meteo assigns to the location; before, it failed with `UnknownTimeZoneError`. Any other value must be an IANA time zone name; case and surrounding spaces are ignored, and the name is checked before the request.
- **Behavior change**: the `file` method returns its table on the object's time axis. A table with timestamps is indexed by its instants in the object's `timezone`, or in UTC with `Generator.generate(..., utc=True)`. If the object names a `weather` table, the file is aligned to the weather's instants and must cover them; a table without timestamps is then used row by row at equal length, with a warning. A table without timestamps for an object without `weather` is returned unchanged, with a warning. Before, `file` returned the table unchanged, for example with a `RangeIndex` and a text `datetime` column. (`#116`)
- **Behavior change**: the calendar methods follow the local calendar of each object's new `timezone` key (UTC if unset), and their outputs are indexed by the run's instants in that zone, or in UTC with `Generator.generate(..., utc=True)`. (`#116`)
  - demandlib electricity, demandlib heat, DistrictHeatingSim and PyLPG read their engines at the site's legal wall clock. They used to take the first row's UTC offset as the clock for the whole run, so with the example weather every summer-time hour carried the value of the hour before, and with UTC input every hour was one to two hours off. On the 23-hour day of the change to summer time the skipped hour is not read; on the 25-hour day the repeated hour is read twice. Each row is the mean over the interval that starts at its label, also for steps that are not whole hours or that start between two engine labels.
  - `demand[kWh]` is the energy over the simulated run for demandlib electricity and DistrictHeatingSim, as it already was for demandlib heat. Electricity used to take the run's share of an annual profile, and DistrictHeatingSim scaled every calendar year to the full demand.
  - Jordan-Vajen DHW draws water per local day. Each event type's share of the mean daily volume comes from the weekly mean of its `probability_day`, and within the type `probability_day` sets a factor per weekday whose weekly mean is 1 (holidays use Sunday's factor). Weekday totals therefore vary, and the annual event mix follows the weekly-mean shares (Bath about 25 %). Before, the event mix used only Monday's `probability_day` (Bath about 15 %), and the per-day weights were lost, as were the seasonal weights: `seasonal_variation` changed nothing. The number of events per day and type is a Poisson draw and each event keeps its own size, so baths stay about 140 l. The volume over the run is the sum of the days' expected volumes; before, it was scaled to 365 days of demand whatever the run length. The seasonal phase uses the actual length of each year, and activity times between two rows of the run are counted in the row that contains them.
  - GeoMA and PHT read the electricity series by its timestamps, so they now run after the built-in electricity methods (before: `KeyError: 'datetime'`). The night window and the hold until 09:00 use the local clock, also on the days when the clocks change. The night hold starts once occupancy has lasted the new optional key `night_schedule_duration[h]` (default 1 h) instead of four samples, so it no longer depends on the time step.
  - A run whose first step crosses a clock change, that starts between two engine labels, that ends between two hours, or that touches a calendar year with a single row now runs. So does daily data labelled at local midnight, whose days last 23 or 25 hours when the clocks change: the calendar methods accept a step that is regular in elapsed time or on the site's local clock. Any other change of the step, such as a gap, raises an error naming the table and the instant instead of shifting all later rows. PyLPG raises if the engine leaves minutes without output instead of filling them with zero.
  - `Generator.generate` warns once per run about objects without `timezone`. The examples now set `timezone` to the zone of each object's holidays location, `Europe/Berlin` for the German sites; the heat example's US object now uses German holidays, because its temperatures come from the German example weather.
- **Behavior change**: the default install now provides the EnTiSe core. Install individual methods with extras such as `pip install "entise[pv-pvlib,hvac-1r1c]"`, or use `pip install "entise[all]"` for the previous feature coverage. Methods load on demand, and missing optional packages report the relevant install command. The weather service and example tools have their own extras. (`#111`)
- Documented how to contribute time-series methods in the shared `add-method` skill. (`#110`)
- **Behavior change**: the Ruhnau heat pump reads keys prefixed with `hp:` before the plain keys, like the other methods. Before, it ignored them. (`#119`)
- **Behavior change**: `Method.get_relevant_objects` resolves the declared keys with defaults (see *Added*). Before, it returned every key prefixed with the type, with the prefix removed, and the declared keys without defaults; no built-in method called it. Built-in methods declare `optional_keys` as a mapping, so code that adds lists to it, such as `PVLib.optional_keys + [...]`, must use `list(...)` or `get_requirements()`, which still returns lists. (`#119`)

### Removed
- **Behavior change**: removed the `<key>_COL` object key that named the column of a table reference, such as `power_heating[W]_COL`. Write the reference as `table|column` instead, for example `power_heating[W]: caps|ph`. A table reference without a column now raises an error that shows this form. `gains_internal_column` and `ventilation_column` are unchanged. (`#119`)
- Removed unused domestic hot water reference tables from the repository and packages. Jordan-Vajen retains its two active input tables and applies the 10 °C cold-water default directly. (`#126`)

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
