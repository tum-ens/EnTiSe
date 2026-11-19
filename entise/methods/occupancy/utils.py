import math
from zoneinfo import ZoneInfo

import pandas as pd

from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core._time import UTC, normalize_table

# Occupancy must last this long within the evening window to trigger the night hold.
DEFAULT_NIGHT_SCHEDULE_DURATION_H = 1.0
# The night hold lasts until this local time on the next day.
NIGHT_HOLD_END = pd.Timedelta(9, unit="h")


def electricity_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an electricity time series indexed by its UTC instants.

    Built-in electricity methods index their output by time; a table passed
    through the ``file`` method may carry a ``datetime`` column instead.
    """
    table, _ = normalize_table(frame, table="electricity")
    if table is None:
        raise ValueError("The electricity time series for occupancy needs timestamps.")
    return table


def apply_nightly_schedule(
    df_occ_schedule: pd.DataFrame,
    start_hour: str,
    end_hour: str,
    zone: ZoneInfo = UTC,
    duration_h: float = DEFAULT_NIGHT_SCHEDULE_DURATION_H,
) -> pd.DataFrame:
    """Hold occupancy until 09:00 local time once it has lasted long enough in the evening.

    The evening window ``start_hour``–``end_hour`` is read on the site's local
    clock. On each local day, once occupancy has lasted ``duration_h`` hours
    without a break inside the window, it is held from that time until 09:00
    of the next local day, including on the days when the clocks change.

    Args:
        df_occ_schedule: Occupancy (column ``O.OCCUPANCY``) indexed by tz-aware
            instants at a regular step.
        start_hour: Start of the evening window, for example "20:00".
        end_hour: End of the evening window, for example "23:59".
        zone: The site's time zone.
        duration_h: Occupied duration in hours that triggers the hold.
    """
    df_result = df_occ_schedule.copy()
    if len(df_result) < 2:
        return df_result

    local = pd.DatetimeIndex(df_result.index).tz_convert(zone)
    step = pd.Series(local).diff().median()
    samples = max(1, math.ceil(pd.Timedelta(duration_h, unit="h") / step - 1e-9))

    occupancy = pd.Series(df_result[O.OCCUPANCY].to_numpy(), index=local)
    evening = occupancy.between_time(start_hour, end_hour)

    for day, day_occ in evening.groupby(evening.index.date):
        lasted = day_occ.rolling(window=samples, min_periods=samples).sum() == samples
        if not lasted.any():
            continue
        start_time = lasted[lasted].index[0]
        end_time = (pd.Timestamp(day) + pd.Timedelta(1, unit="D") + NIGHT_HOLD_END).tz_localize(
            zone, nonexistent="shift_forward"
        )
        df_result.loc[(local >= start_time) & (local <= end_time), O.OCCUPANCY] = 1

    df_result.index.name = C.DATETIME
    return df_result
