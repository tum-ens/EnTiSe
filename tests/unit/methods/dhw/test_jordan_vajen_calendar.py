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


def _daily_volume(days: pd.DatetimeIndex, variation: float, seed: int = 7) -> pd.Series:
    from entise.methods.dhw.jordanvajen.calculation import daily_volume

    return daily_volume(
        days,
        365 * 120.0,
        0.2,
        _get_activity_data("jordan_vajen"),
        variation,
        15,
        frozenset(),
        np.random.default_rng(seed),
    )


def test_seasonal_variation_scales_each_local_day():
    days = pd.date_range("2022-01-01", periods=60, freq="D")

    ratio = _daily_volume(days, 0.5) / _daily_volume(days, 0.0)

    np.testing.assert_allclose(ratio.to_numpy(), 1 + 0.5 * np.cos(2 * np.pi * (days.day_of_year - 15) / 365))


def test_the_seasonal_phase_uses_the_actual_length_of_each_year():
    days = pd.date_range("2024-12-30", periods=4, freq="D")

    ratio = _daily_volume(days, 0.5) / _daily_volume(days, 0.0)

    assert ratio["2024-12-31"] == pytest.approx(1 + 0.5 * np.cos(2 * np.pi * (366 - 15) / 366))
    assert ratio["2025-01-01"] == pytest.approx(1 + 0.5 * np.cos(2 * np.pi * (1 - 15) / 365))


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


def test_draws_at_a_skipped_or_repeated_time_land_on_one_instant():
    activity = _get_activity_data("jordan_vajen").copy()
    activity[C.PROBABILITY] = np.where(activity[C.TIME] == "02:23", 1.0, 0.0)

    spring = _sample("2022-03-27", 1, activity)
    fall = _sample("2022-10-30", 1, activity)

    # 02:23 does not exist on 27 March and moves forward to the 03:00 hour, which starts at 01:00 UTC.
    assert list(spring[spring > 0].index) == [pd.Timestamp("2022-03-27 01:00", tz="UTC")]
    # 02:23 happens twice on 30 October; the first occurrence is in summer time, at 00:23 UTC.
    assert list(fall[fall > 0].index) == [pd.Timestamp("2022-10-30 00:00", tz="UTC")]


# -- event sizes, run total and event mix ------------------------------------------------------------------------------


def _weekly_mean_shares(activity: pd.DataFrame) -> pd.Series:
    weekly = activity.groupby([C.EVENT, C.DAY_OF_WEEK])[C.PROBABILITY_DAY].first().groupby(level=0).mean()
    return weekly / weekly.sum()


def _year_of_draws(seed: int, years: int = 1, freq: str = "h", by_event: bool = False):
    from zoneinfo import ZoneInfo

    from entise.methods.dhw.jordanvajen.calculation import daily_volume, sample_volume

    activity = _get_activity_data("jordan_vajen")
    axis = pd.date_range("2022-01-01", f"{2022 + years}-01-01", freq=freq, tz=BERLIN, inclusive="left").tz_convert(
        "UTC"
    )
    days = pd.date_range("2022-01-01", f"{2022 + years}-01-01", freq="D", inclusive="left")
    rng = np.random.default_rng(seed)
    daily = daily_volume(days, 365 * 130.0, 0.2, activity, 0.0, 15, frozenset(), rng)
    step = pd.Timedelta(freq) if freq != "h" else pd.Timedelta(1, unit="h")
    volume = sample_volume(axis, ZoneInfo(BERLIN), step, activity, daily, rng, frozenset(), by_event=by_event)
    return daily, volume


def test_draws_keep_the_size_of_each_event():
    # A bath is about 140 l; at 15 minutes some row of a year holds a whole bath.
    weather = pd.DataFrame(
        {C.DATETIME: pd.date_range("2022-01-01", "2023-01-01", freq="15min", tz=BERLIN, inclusive="left")}
    )

    volume = _volume(_obj(), {"weather": weather})

    assert volume.max() > 100


def test_the_run_total_equals_the_daily_volumes():
    daily, volume = _year_of_draws(seed=1)

    assert volume.sum() == pytest.approx(daily.sum(), rel=1e-9)


def test_weekday_volumes_follow_the_day_weights():
    activity = _get_activity_data("jordan_vajen")
    weights = activity.groupby([C.DAY_OF_WEEK, C.EVENT])[C.PROBABILITY_DAY].first().unstack()
    shares = _weekly_mean_shares(activity)
    expected = (weights / weights.mean()).mul(shares, axis=1).sum(axis=1)

    per_weekday = []
    for seed in range(4):
        _, volume = _year_of_draws(seed, years=2)
        local = volume.index.tz_convert(BERLIN)
        daily = volume.groupby(local.date).sum()
        per_weekday.append(daily.groupby(pd.DatetimeIndex(daily.index).day_of_week).mean())
    observed = pd.concat(per_weekday, axis=1).mean(axis=1)

    np.testing.assert_allclose(
        (observed / observed.mean()).to_numpy(), (expected / expected.mean()).to_numpy(), rtol=0.08
    )


def test_the_annual_event_mix_follows_the_weekly_mean_weights():
    shares = []
    for seed in range(3):
        _, by_event = _year_of_draws(seed, years=2, by_event=True)
        shares.append(by_event.sum() / by_event.sum().sum())
    observed = pd.concat(shares, axis=1).mean(axis=1)

    expected = _weekly_mean_shares(_get_activity_data("jordan_vajen"))
    np.testing.assert_allclose(observed[expected.index].to_numpy(), expected.to_numpy(), atol=0.03)


def test_daily_data_at_local_midnight_runs_across_clock_changes():
    days = pd.date_range("2022-03-20", "2022-11-10", freq="D", tz=BERLIN)

    volume = _volume(_obj(), {"weather": pd.DataFrame({C.DATETIME: days})})

    assert len(volume) == len(days)
    assert (volume.index.tz_convert(BERLIN).hour == 0).all()
