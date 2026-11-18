"""Jordan-Vajen DHW draws each local day's water from that day's own calendar."""

import os

import numpy as np
import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.methods.dhw.jordan_vajen import JordanVajen
from entise.methods.dhw.jordanvajen.activity import _get_activity_data

pytest.importorskip("scipy")

WEATHER_CSV = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "integration", "common_data", "weather.csv"
)
BERLIN = "Europe/Berlin"
VOLUME = f"{Types.DHW}{SEP}volume[l]"


def _weather(first: int = 0, rows: int = 24 * 28) -> pd.DataFrame:
    return pd.read_csv(WEATHER_CSV).iloc[first : first + rows].reset_index(drop=True)


def _obj(**extra) -> dict:
    return {O.ID: "home", O.DWELLING_SIZE: 100, O.DATETIMES: "weather", O.SEED: 7, O.TIMEZONE: BERLIN, **extra}


def _volume(obj: dict, data: dict) -> pd.Series:
    return JordanVajen().generate(obj, data)["timeseries"][VOLUME]


def _daily(volume: pd.Series) -> pd.Series:
    local = volume.index.tz_convert(BERLIN)
    return volume.groupby(local.strftime("%Y-%m-%d")).sum()


def test_weekday_schedules_reach_their_own_day():
    activity = _get_activity_data("jordan_vajen").copy()
    activity[C.PROBABILITY] = 0.0
    activity.loc[(activity[C.DAY_OF_WEEK] == 6) & (activity[C.TIME] == "12:00"), C.PROBABILITY] = 1.0
    activity.loc[(activity[C.DAY_OF_WEEK] < 6) & (activity[C.TIME] == "06:00"), C.PROBABILITY] = 1.0

    volume = _volume(_obj(), {"weather": _weather(), O.DHW_ACTIVITY: activity})

    local = volume.index.tz_convert(BERLIN)
    noon, morning = volume[local.hour == 12], volume[local.hour == 6]
    assert noon[noon.index.tz_convert(BERLIN).day_of_week == 6].sum() == pytest.approx(noon.sum())
    assert morning[morning.index.tz_convert(BERLIN).day_of_week == 6].sum() == 0
    assert volume.sum() == pytest.approx(noon.sum() + morning.sum())


def test_seasonal_variation_scales_each_local_day():
    data = {"weather": _weather(0, 24 * 60)}

    flat = _daily(_volume(_obj(**{O.SEASONAL_VARIATION: 0.0}), data))
    seasonal = _daily(_volume(_obj(**{O.SEASONAL_VARIATION: 0.5}), data))

    days = pd.DatetimeIndex(flat.index)
    factor = 1 + 0.5 * np.cos(2 * np.pi * (days.day_of_year - 15) / 365)
    np.testing.assert_allclose((seasonal / flat).to_numpy(), factor, rtol=2e-3)


def test_the_seasonal_phase_uses_the_actual_length_of_each_year():
    weather = pd.DataFrame({C.DATETIME: pd.date_range("2024-12-30", periods=24 * 4, freq="h", tz=BERLIN)})

    flat = _daily(_volume(_obj(**{O.SEASONAL_VARIATION: 0.0}), {"weather": weather}))
    seasonal = _daily(_volume(_obj(**{O.SEASONAL_VARIATION: 0.5}), {"weather": weather}))

    ratio = seasonal / flat
    assert ratio["2024-12-31"] == pytest.approx(1 + 0.5 * np.cos(2 * np.pi * (366 - 15) / 366), rel=2e-3)
    assert ratio["2025-01-01"] == pytest.approx(1 + 0.5 * np.cos(2 * np.pi * (1 - 15) / 365), rel=2e-3)


def test_the_same_instants_give_the_same_volume():
    weather = _weather(2000, 24 * 7)
    utc = pd.to_datetime(weather[C.DATETIME], utc=True)
    runs = [
        _volume(_obj(), {"weather": w})
        for w in (
            weather,
            weather.assign(**{C.DATETIME: utc}),
            weather.assign(**{C.DATETIME: utc.dt.tz_convert(BERLIN)}),
        )
    ]

    for other in runs[1:]:
        pd.testing.assert_series_equal(other, runs[0])


# -- the per-day sampler ----------------------------------------------------------------------------------------------


def _sample(first_day: str, days: int, activity: pd.DataFrame | None = None) -> pd.Series:
    from entise.methods.dhw.jordanvajen.calculation import sample_volume

    axis = pd.date_range(first_day, periods=days * 24 + 1, freq="h", tz=BERLIN).tz_convert("UTC")
    axis = axis[axis < pd.Timestamp(first_day, tz=BERLIN) + pd.DateOffset(days=days)]
    local_days = pd.date_range(first_day, periods=days, freq="D")
    daily = pd.Series(120.0, index=local_days)
    activity = _get_activity_data("jordan_vajen") if activity is None else activity
    from zoneinfo import ZoneInfo

    return sample_volume(
        axis, ZoneInfo(BERLIN), pd.Timedelta(1, unit="h"), activity, daily, np.random.default_rng(3), frozenset()
    )


@pytest.mark.parametrize("first_day", ["2022-03-25", "2022-10-28"], ids=["spring", "fall"])
def test_each_local_day_gets_its_daily_volume(first_day):
    volume = _sample(first_day, 5)

    np.testing.assert_allclose(_daily(volume).to_numpy(), 120.0)
    assert len(_daily(volume)) == 5


def test_draws_at_a_skipped_or_repeated_time_land_on_one_instant():
    activity = _get_activity_data("jordan_vajen").copy()
    activity[C.PROBABILITY] = np.where(activity[C.TIME] == "02:23", 1.0, 0.0)

    spring = _sample("2022-03-27", 1, activity)
    fall = _sample("2022-10-30", 1, activity)

    # 02:23 does not exist on 27 March and moves forward to the 03:00 hour, which starts at 01:00 UTC.
    assert list(spring[spring > 0].index) == [pd.Timestamp("2022-03-27 01:00", tz="UTC")]
    # 02:23 happens twice on 30 October; the first occurrence is in summer time, at 00:23 UTC.
    assert list(fall[fall > 0].index) == [pd.Timestamp("2022-10-30 00:00", tz="UTC")]
