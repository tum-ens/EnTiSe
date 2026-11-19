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
their labels. With EnTiSe's column names:

.. code-block:: python

    aggregates = [
        "global_horizontal_irradiance[W m-2]",
        "direct_horizontal_irradiance[W m-2]",
        "diffuse_horizontal_irradiance[W m-2]",
        "direct_normal_irradiance[W m-2]",
        "global_tilted_irradiance[W m-2]",
        # Open-Meteo's terrestrial_radiation; its _instant variant has the same name, so leave
        # this one out if your table holds the instantaneous values.
        "surface_upwelling_longwave_flux_in_air[W m-2]",
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
solar noon, which indicates end labels. The check needs at least 14 days of
hourly or longer data.

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
an error; before, its solar gains were silently zero. 1R0C now uses its
``latitude[degree]`` and ``longitude[degree]`` for solar gains; before, they
were dropped, so its solar gains were always zero.

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

**GeoMA and PHT.** The night window and the hold until 09:00 use the local
clock. The hold starts once occupancy has lasted ``night_schedule_duration[h]``
(new, default 1 h) instead of four samples, which lasted four hours on hourly
data.

Other changes
-------------

- ``entise.set_accelerator()`` now applies to process workers on every call,
  including reused ones. A forced ``"numba"`` mode reports a missing Numba
  when the accelerated solver runs, also in workers; before, workers could fall
  back to NumPy silently.
- PyLPG raises an error if the engine leaves minutes of the run without
  output. Before, they were filled with zero.
- Solar gains of objects in the same 0.1° cell share one solar position, taken
  at the cell centre, so results no longer depend on object order. Gains change
  slightly for objects away from a cell centre.
- demandlib electricity and PyLPG read the datetimes passed in each call;
  before, a second run under the same data key reused the first run's
  timestamps.
- The heat pump (Ruhnau) no longer changes the weather frame you pass in, and
  its soil temperature estimate averages the last 24 hours rather than the last
  24 rows.
