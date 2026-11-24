Migrating to 2.0.0
==================

EnTiSe 2.0.0 changes how time is read and written, how some methods count
energy, and how EnTiSe is installed. This page lists every change that can
alter your results or break an existing setup, with what to do about it. The
full list is in the changelog; the time rules are described in
:ref:`time-contract`.

Installation
------------

**Before:** ``pip install entise`` installed every method and its dependencies.

**Now:** the default install is the EnTiSe core. Install the methods you use
as extras, or everything at once:

.. code-block:: bash

    pip install "entise[pv-pvlib,hvac-1r1c]"
    pip install "entise[all]"

A method whose optional package is missing tells you which extra to install.
The weather service and the example tools have their own extras.

**Python and pandas.** EnTiSe 2.0.0 runs on Python 3.11 to 3.14; Python 3.10
is no longer supported. It works with pandas 2 (2.2.2 or newer) and pandas 3.

Direct method calls
-------------------

Keep ``obj``, ``data``, ``results`` and ``ts_type`` as the first four
arguments to ``generate()``. Pass method-specific inputs by keyword.
Earlier keyword names such as ``temp_init`` for 1R0C and 1R1C remain
available. Call ``input_names()`` on the method to see the accepted names.

Some keywords that were previously ignored now take effect. For example,
``power=5000`` sets the rated power for PVLib even if the object row already
contains a power value. Remove an old keyword if you want to use the object
value or the method default. Unknown keywords raise ``TypeError``, including
the unused ``weather`` keyword on demandlib electricity. See
:ref:`direct_input_names` for an example that combines an object row and
keywords.

Custom methods
--------------

This section applies if you subclass ``Method`` yourself.

**Unknown keywords raise** ``TypeError``. ``_process_kwargs`` and
``_process_inputs`` accept only the inputs the method declares, by key or by
Python name, and raise for any other keyword. Before, ``_process_kwargs``
ignored keywords it did not know, and accepted a declared input only under its
exact key. So ``power=...`` for ``power[W]`` was ignored and now takes effect.

``_process_kwargs`` **takes** ``ts_type`` **as its third argument**, before the
inputs: ``_process_kwargs(obj, data, ts_type, **inputs)``. Pass the inputs by
keyword. Built-in methods follow this pattern:

.. code-block:: python

    # Before
    def generate(self, obj=None, data=None, results=None, ts_type=None, *, power=None, scale=None):
        obj, data = self._process_kwargs(obj, data, power=power, scale=scale)

    # Now
    def generate(self, obj=None, data=None, results=None, ts_type="pv", **inputs):
        obj, data = self._process_kwargs(obj, data, ts_type, **inputs)

Read any setting that is not an input, such as ``scale`` above, from your own
argument before the call. Alternatively, declare it in ``optional_keys``.

**Declarations are checked when the class is defined**, including classes with
``register = False``. A ``TypeError`` is raised if two declared inputs have the
same Python name, which is the key without its unit and height, for example
``area[m2]`` and ``area[ft2]``. Inputs that differ only in their height, such as
``wind_speed[m s-1]@10m`` and ``wind_speed[m s-1]@100m``, are allowed: their
shared Python name is dropped, and they are set by their keys. A ``TypeError``
is also raised if an entry in ``input_aliases`` names an input that is not
declared or reuses another input's name. Before, declarations were not checked.

Constants
---------

``Columns``, ``Objects``, ``Types`` and ``Keys`` in ``entise.constants`` are
read-only namespaces. Their values are still plain strings. ``Types()`` and the
other classes cannot be instantiated; that raises ``TypeError``. Assigning an
attribute raises ``AttributeError``, also through
``mock.patch.object(Types, "DHW", ...)``. In tests, patch the name where your
code looks it up instead, for example ``mock.patch("mypackage.module.Types", ...)``.

Timestamps
----------

**Text timestamps must be ISO 8601.** ``2022-07-01 12:00:00+02:00``,
``2022-07-01T10:00:00Z`` and ``2022-07-01 12:00`` work. Formats such as
``01/07/2022 12:00``, a time without a date, or numbers raise an error that
names the table. Convert them once with pandas:

.. code-block:: python

    df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%m/%Y %H:%M")

**Naive timestamps are read as UTC.** Before, some methods read them as UTC and
others as local time, and Jordan-Vajen DHW failed on them. EnTiSe now logs one
warning per table. If your table holds local times, localize it before passing
it:

.. code-block:: python

    df["datetime"] = df["datetime"].dt.tz_localize("Europe/Berlin", ambiguous="infer")

**A repeated or missing timestamp raises an error** that names the table and
the timestamp. Rows out of order are sorted by time.

**Gaps raise an error** in the methods that need a regular time step:

- the RC models (1R0C, 1R1C, 5R1C, 7R2C), PV and wind need a step that is
  regular in elapsed time;
- demandlib electricity and heat, DistrictHeatingSim, PyLPG and Jordan-Vajen
  accept a step that is regular in elapsed time or on the object's local
  clock. Daily data labelled at local midnight therefore works across daylight
  saving changes, although those days last 23 or 25 hours.

Ruhnau, GeoMA, PHT and ``file`` do not check the step. Before, gaps were
accepted and most calendar methods shifted every later row onto a regular
grid.

The site time zone
------------------

Add a ``timezone`` column to your objects, with an IANA name such as
``Europe/Berlin``:

.. code-block:: text

    id,hvac,weather,timezone,...
    house1,1R1C,weather,Europe/Berlin,...

It sets the object's local calendar (hour, weekday, holidays and daylight
saving days) and the time zone of its outputs. It does not change how tables
are read. Without it, the calendar is UTC and EnTiSe warns once per run, listing
the objects.

**Before:** the calendar methods (demandlib electricity and heat,
DistrictHeatingSim, PyLPG, Jordan-Vajen, GeoMA and PHT) took the UTC offset of
the first row as their clock for the whole run. With timestamps carrying
``+01:00`` in winter and ``+02:00`` in summer, every summer hour got the value
of the hour before; with UTC timestamps, every hour was one to two hours off.

**Now:** each instant is read at the local wall clock of ``timezone``. On the
23-hour day of the change to summer time the skipped hour is not read; on the
25-hour day the repeated hour is read twice.

Interval labels
---------------

Every value is labelled at the **start** of its interval. Interval quantities,
such as irradiance, precipitation or power, are means or sums over
``[t, t + step)``. Instantaneous quantities, such as air temperature, are
values at ``t``.

**Weather downloaded from Open-Meteo directly, or from a similar source,**
labels its hourly aggregates at the end of the hour: radiation means,
precipitation, rain and snowfall sums, evapotranspiration, sunshine duration
and gusts. Shift exactly those columns one row earlier, and leave
instantaneous values, including the ``*_instant`` radiation variables, at
their labels. A table saved from EnTiSe's weather service before 2.0.0 names
the top-of-atmosphere radiation ``surface_upwelling_longwave_flux_in_air[W m-2]``;
rename it to ``toa_incoming_shortwave_flux[W m-2]`` first (see `Other changes`_).
With EnTiSe's column names:

.. code-block:: python

    aggregates = [
        "global_horizontal_irradiance[W m-2]",
        "direct_horizontal_irradiance[W m-2]",
        "diffuse_horizontal_irradiance[W m-2]",
        "direct_normal_irradiance[W m-2]",
        "global_tilted_irradiance[W m-2]",
        "toa_incoming_shortwave_flux[W m-2]",
        "precipitation_amount[kg m-2]",
        "rainfall_amount[kg m-2]",
        "thickness_of_snowfall_amount[m]",
        "water_evapotranspiration_amount[kg m-2]",
        "duration_of_sunshine[s]",
        "wind_speed_of_gust[m s-1]@10m",
    ]
    columns = [c for c in aggregates if c in weather.columns]
    weather[columns] = weather[columns].shift(-1)

The last row then has no value for those columns; drop it, or fetch one more
hour. For your own data, the rule is the same: shift every column that holds a
mean, sum or maximum over the interval that ends at its label; values measured
at the label stay where they are.

EnTiSe warns when irradiance in a weather table peaks about one step after
solar noon. That indicates end labels, or local timestamps without an offset,
which are read as UTC; the warning names both. The check needs at least 14 days
of hourly or longer data.

**EnTiSe's Open-Meteo service** now returns start-labelled data: it moves the
interval quantities one hour earlier, requests one more day so the last hour is
complete, and computes ``solar_elevation_angle[degree]`` at the middle of each
hour. ``timezone=None`` now means UTC days and UTC timestamps. Before, it meant
the location's local days with UTC timestamps. ``timezone="auto"`` returns the
timestamps in the zone Open-Meteo assigns to the location; before, it failed.

**The example weather** (``examples/common_data/weather.csv``) is relabelled
the same way.

**PV and the solar gains** of the HVAC models use the sun's position at the
middle of each interval. Annual PV yield changes by a few per cent on east- and
west-facing arrays; south-facing arrays change little.

Outputs
-------

**Every time series of an object shares one index:** the instants of its input
in the object's ``timezone``. Before, the index depended on the method and on
how the timestamps were written (object dtype, a fixed offset, naive UTC or a
``RangeIndex``). For UTC timestamps on every output, ask for them once per run:

.. code-block:: python

    summary, timeseries = generator.generate(data, utc=True)

The index uses ``datetime64[us]``. Joins with your own ``datetime64[ns]`` data
still align.

**The** ``file`` **method** returns its table on that index. A table with
timestamps keeps its instants; if the object also names a ``weather`` table,
the file is aligned to the weather's instants and must cover them. A table
without timestamps is used row by row, one row per weather step, with a
warning. Without a ``weather`` key, such a table is returned unchanged, with a
warning. A ``weather`` key that names a missing table raises an error. The ``datetime`` column becomes
the index. Before, ``file`` returned the table unchanged.

**Summary columns two types share.** If two types of one object report the
same summary column, for example ``heating:demand[Wh]`` from an HVAC model and
from a heat profile, both values are kept: each column is prefixed with the
type that wrote it (``hvac:heating:demand[Wh]`` and
``heating:heating:demand[Wh]``), and a warning names them. Before, the type that
ran last overwrote the other's value. Columns that only one type reports keep
their names.

**PV summary values are numbers.** Each ``pv:*`` summary value is now a plain
number. Before, it was a one-element Series. Drop any ``.iloc[0]`` you used to
read it.

HVAC (1R0C, 1R1C, 5R1C, 7R2C)
-----------------------------

**Row 0.** Before, row 0 held the initial temperature and zero load, and a
weather table with n rows simulated n − 1 intervals. Now row ``t`` is the
interval that starts at ``t``: its loads are the means over that interval, and
``indoor_temperature[C]`` is the temperature at its start. Row 0 therefore
holds the initial temperature and the first interval's load, and n rows give n
intervals.

**Energy.** All four models report demand as the sum of load times step. 5R1C
and 7R2C used the trapezoidal rule.

**1R0C indoor temperature.** 1R0C keeps the indoor temperature between
``min_temperature[C]`` and ``max_temperature[C]``, like the other RC models.
Without heating or cooling, the zone takes ``T_out + gains / H`` in each
interval. Before, it held ``init_temperature[C]`` in every step. It heated
only when the outdoor air was at or below the minimum and cooled only when it
was at or above the maximum. Cooling demand rises, and heating stays the same
when ``init_temperature[C]`` equals the minimum. ``init_temperature[C]`` now
has no effect in 1R0C. Row 0 is the first interval's temperature, not an
initial state. If you used ``init_temperature[C]`` as the setpoint, set
``min_temperature[C]`` to that value instead.

**Auxiliary tables.** Internal gains, ventilation and ``table|column``
references align with the weather by timestamp. A table without timestamps is
used row by row only if it has one row per weather step, with a warning; any
other mismatch raises an error. A table that is finer than the weather, such as
15-minute gains on hourly weather, is averaged over each weather step, with a
warning. It must still contain every weather instant, so a table shifted
against the weather's steps raises an error. Rows before the first or after the
last weather step are ignored. Before, a finer table was sampled at the
weather's steps, which changed its energy. Before, 5R1C and 7R2C gave NaN temperatures and
0 Wh heating for a gains table read with ``read_csv``, 1R0C and 1R1C ignored the
timestamps of a table, and ``table|column`` power caps failed.

**Windows.** An object that names a windows table missing from the data raises
an error; before, its solar gains were silently zero. So does an object whose
windows would get no solar gains: it has no ``latitude[degree]`` and
``longitude[degree]``, or the table has no row with its ``id``. An ``id`` of
``"1"`` does not match ``1``; the error shows both types. Add the location,
remove ``windows``, or set ``active_gains_solar`` to ``False``. Objects that
supply ``gains_solar[W]`` are not checked. 1R0C now uses its
``latitude[degree]`` and ``longitude[degree]`` for solar gains; before, they
were dropped, so its solar gains were always zero.

**Ventilation units.** A ventilation table column named with EnTiSe's unit
spelling, such as ``typical[W K-1]``, ``typical[m3 h-1]`` or ``typical[h-1]``,
is now read in that unit. Before, only ``W/K``, ``m3/h`` and ``1/h`` were
recognised, and every other unit was read as air changes per hour. Both
spellings are accepted. 1R0C now converts air changes per hour with the
object's ``area[m2]`` and ``height[m]``; before, it always used the default
zone of 1 m² and 2.5 m.

**1R0C ventilation switch.** ``active_ventilation=False`` now switches off
ventilation in 1R0C, as it already did in 1R1C, 5R1C and 7R2C. Before, 1R0C
ignored the flag and always applied ventilation. The flag still defaults to
``True``.

**Weather key.** PV and wind now use the table named by the object's
``weather`` key, like the other methods. Before, they always read ``weather``.

Demand and calendars
--------------------

``demand[kWh]`` **is the demand over the simulated run.** demandlib
electricity and DistrictHeatingSim now scale so that the series over the run
adds up to ``demand[kWh]``, as demandlib heat already did. Before, electricity
took the run's share of a yearly profile scaled to ``demand[kWh]``, and
DistrictHeatingSim scaled each calendar year of the run to the full
``demand[kWh]``. For a run
that is not one calendar year, scale your input:

.. code-block:: python

    run_demand_kwh = annual_demand_kwh * run_hours / 8760

**demandlib heat at other time steps.** Before, its power was 2× too high at
30 minutes, 4× at 15 minutes and half at 2 hours. Each step now carries the
mean power of the hours it covers.

**Jordan-Vajen DHW** draws water per local day:

- Weekday, holiday and seasonal weights now reach their own day. Before, only
  Monday's ``probability_day`` was used and ``seasonal_variation`` had no
  effect.
- Each event type's share of the daily volume is the weekly mean of its
  ``probability_day``; within the type, ``probability_day`` scales the
  expected volume per weekday. With the shipped activity table, baths make up
  about 25 % of the annual volume instead of about 15 %.
- The run's volume is the sum of the daily demand over the simulated days.
  Before, it was scaled to 365 days of demand whatever the run length.
- ``dhw:hot_water_temperature[C]`` now holds the hot water temperature;
  before, it repeated the cold one.
- A table named by ``dhw_activity`` or ``dhw_demand_per_size[m2]`` is used,
  and a named table missing from the data raises an error. Before, the names
  were ignored, and the bundled demand table was always used.

**GeoMA and PHT.** The night window and the hold until 09:00 use the local
clock. The hold starts once occupancy has lasted ``night_schedule_duration[h]``
(new, default 1 h) instead of four samples, which lasted four hours on hourly
data.

Method inputs
-------------

**Solar gains you supply.** 1R0C, 1R1C, 5R1C and 7R2C now use an object's
``gains_solar[W]`` instead of calculating solar gains. It can be a constant in
W, ``table|column``, or the name of a table whose column is the object's
``id`` (or whose only column holds the gains). Before, the value was accepted
and ignored. In 1R1C, 5R1C and 7R2C, ``active_gains_solar=False`` still
switches solar gains off. Remove ``gains_solar[W]`` from your objects if you
want the calculated gains.

**Ventilation in 5R1C and 7R2C.** ``H_ve[W K-1]`` now takes the place of the
ventilation input, in the same forms as supplied solar gains, and ``ventilation_split`` divides it between supply air and
infiltration. 5R1C now uses ``ventilation_split``; before, it always used 0.
The split changes results only when ``supply_temperature[C]`` differs from the
outdoor air. Before, ``H_ve[W K-1]`` was ignored in both models.

**Table columns.** A key with the suffix ``_COL`` no longer names the column of
a table reference. Write the reference as ``table|column``:

.. code-block:: python

    # Before
    {"power_heating[W]": "caps", "power_heating[W]_COL": "ph"}
    # Now
    {"power_heating[W]": "caps|ph"}

A key that names a table without a column, where a column is needed, raises an
error that shows this form. ``gains_internal_column`` and
``ventilation_column`` are unchanged.

**Prefixed keys for the heat pump.** Ruhnau now reads ``hp:correction_factor``
before ``correction_factor``, as every other method reads its own prefix.
Before, it ignored prefixed keys. Objects that set both now use the prefixed
value.

**Wind model settings.** WPLib now uses ``wind_model``: a mapping of
windpowerlib ModelChain settings, such as ``{"wind_speed_model": "hellman"}``,
or the name of one in ``data``. It replaces the matching defaults of
``MODELCHAIN_PARAMS``, and an unknown setting raises ``ValueError``. Before, the
key was declared but never read. Remove it from objects that should keep the
defaults. An ``interpolation_extrapolation`` model needs its variable at two or
more heights in the weather; otherwise EnTiSe raises ``ValueError``.

**1R1C location.** 1R1C now declares ``latitude[degree]`` and
``longitude[degree]``, which it already read, so direct calls accept
``latitude=`` and ``longitude=``.

**Your own methods.** A method may declare ``optional_keys`` as a mapping from
each key to its default; a list still works and means no defaults. Resolve the
keys with ``self.get_relevant_objects(obj, ts_type)``: it reads the prefixed
key, then the plain key, then the default, and returns only declared keys that
have a value. Before, ``get_relevant_objects`` returned every key with the
type's prefix, prefix removed, and the declared keys without defaults. The
built-in methods now declare ``optional_keys`` as mappings, so code such as
``PVLib.optional_keys + [...]`` must use ``list(PVLib.optional_keys)`` or
``get_requirements()``, which still returns lists.

Other changes
-------------

- ``entise.set_accelerator()`` now applies to process workers on every call,
  including reused ones. A forced ``"numba"`` mode reports a missing Numba
  when the accelerated solver runs, also in workers; before, workers could fall
  back to NumPy silently.
- PyLPG raises an error if the engine leaves minutes of the run without
  output. Before, they were filled with zero.
- Importing the PV method no longer switches off every warning in the process.
  Before, it called ``warnings.filterwarnings("ignore")`` for the whole
  process. Warnings from your own code and other libraries that it hid now
  appear again.
- Solar gains of objects in the same 0.1° cell share one solar position, taken
  at the cell centre, so results no longer depend on object order. Gains change
  slightly for objects away from a cell centre.
- demandlib electricity and PyLPG read the datetimes passed in each call;
  before, a second run under the same data key reused the first run's
  timestamps.
- The Open-Meteo service returns ``terrestrial_radiation`` as
  ``toa_incoming_shortwave_flux[W m-2]`` and ``terrestrial_radiation_instant``
  as ``toa_incoming_shortwave_flux_instant[W m-2]``: the solar radiation at the
  top of the atmosphere. Before, both were named
  ``surface_upwelling_longwave_flux_in_air[W m-2]``. Rename that column in
  weather tables you saved from the service.
- The heat pump (Ruhnau) no longer changes the weather frame you pass in, and
  its soil temperature estimate averages the last 24 hours rather than the last
  24 rows.
