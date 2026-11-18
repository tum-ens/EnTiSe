"""The night rule of GeoMA and PHT runs on the site's local clock and on durations."""

from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from entise.constants import Objects as O
from entise.methods.occupancy.utils import apply_nightly_schedule

BERLIN = ZoneInfo("Europe/Berlin")


def _occupancy(day: str, freq: str, occupied_from: str, occupied_for: str) -> pd.DataFrame:
    index = pd.date_range(day, periods=int(pd.Timedelta("2D") / pd.Timedelta(freq)), freq=freq, tz=BERLIN)
    start = pd.Timestamp(f"{day} {occupied_from}", tz=BERLIN)
    occupied = (index >= start) & (index < start + pd.Timedelta(occupied_for))
    return pd.DataFrame({O.OCCUPANCY: occupied.astype(int)}, index=index.tz_convert("UTC"))


def _span(result: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    occupied = result.index[result[O.OCCUPANCY] == 1].tz_convert(BERLIN)
    return occupied[0], occupied[-1]


@pytest.mark.parametrize(
    "day, hold_end",
    [
        ("2022-01-10", "2022-01-11 09:00"),
        ("2022-07-10", "2022-07-11 09:00"),
        ("2022-03-26", "2022-03-27 09:00"),
        ("2022-03-27", "2022-03-28 09:00"),
        ("2022-10-29", "2022-10-30 09:00"),
        ("2022-10-30", "2022-10-31 09:00"),
    ],
)
def test_an_evening_trigger_holds_until_nine_local_time(day, hold_end):
    occupancy = _occupancy(day, "15min", "20:00", "1h")

    result = apply_nightly_schedule(occupancy, "20:00", "23:59", zone=BERLIN)

    first, last = _span(result)
    assert first == pd.Timestamp(f"{day} 20:00", tz=BERLIN)
    assert last == pd.Timestamp(hold_end, tz=BERLIN)


def test_the_trigger_is_a_duration_at_any_time_step():
    hourly = _occupancy("2022-01-10", "1h", "20:00", "1h")
    minutes = _occupancy("2022-01-10", "1min", "20:00", "4min")

    assert _span(apply_nightly_schedule(hourly, "20:00", "23:59", zone=BERLIN))[1] == pd.Timestamp(
        "2022-01-11 09:00", tz=BERLIN
    )
    assert apply_nightly_schedule(minutes, "20:00", "23:59", zone=BERLIN)[O.OCCUPANCY].sum() == 4
