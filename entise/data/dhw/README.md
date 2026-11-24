# Jordan–Vajen DHW reference data

The `jordanvajen` method uses two bundled CSVs. Both are in `jordan_vajen/`.

| File | Use |
| --- | --- |
| `dhw_activity.csv` | Default activity parameters by weekday, time and draw-off category. The table gives probabilities, durations in seconds and flow rates in litres per second. |
| `dhw_demand_by_dwelling.csv` | Annual demand per square metre of floor area and the relative standard deviation used to sample daily demand. |

See the [column names and units](jordan_vajen/units.txt) for both tables.

The original EnTiSe data notes credit Ulrike Jordan and Klaus Vajen's
[Realistic Domestic Hot-Water Profiles in Different Time Scales](https://sel.me.wisc.edu/trnsys/trnlib/iea-shc-task26/iea-shc-task26-load-profiles-description-jordan.pdf).
Their [DHWcalc paper](https://solar-publikationen.umwelt-uni-kassel.de/uploads/2005%20ISES-SWC%20Jordan%20und%20Vajen%20Program%20to%20Generate%20Domestic%20Hot%20Water%20Profiles%20with%20Statistical%20Means%20for%20User%20Defined%20Conditions.pdf)
describes the statistical method. The exact derivation and redistribution
terms for the two EnTiSe tables are still to be established.
