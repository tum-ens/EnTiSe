"""The time layer reads every table as UTC instants and checks it once per run."""

import logging
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core import _time as T

WEATHER_CSV = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "integration", "common_data", "weather.csv")


def _weather(first: int = 0, rows: int = 48) -> pd.DataFrame:
    return pd.read_csv(WEATHER_CSV).iloc[first : first + rows].reset_index(drop=True)


def _instants(frame: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(frame[C.DATETIME], utc=True)).as_unit("us")


# -- parsing ----------------------------------------------------------------------------------------------------------


def _representations(weather: pd.DataFrame) -> dict[str, pd.Series]:
    utc = pd.to_datetime(weather[C.DATETIME], utc=True)
    return {
        "offset strings": weather[C.DATETIME],
        "Z strings": utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "typed UTC": utc,
        "typed Europe/Berlin": utc.dt.tz_convert("Europe/Berlin"),
        "Timestamps with mixed offsets": pd.Series([pd.Timestamp(s) for s in weather[C.DATETIME]], dtype=object),
    }


@pytest.mark.parametrize("name", list(_representations(_weather())))
def test_every_representation_gives_the_same_utc_instants(name):
    # 2022-03-27 crosses the change to summer time, so the offsets are mixed.
    weather = _weather(2030, 24)
    values = _representations(weather)[name]

    axis, naive = T.parse_instants(values, table="weather")

    assert not naive
    assert axis.equals(_instants(weather))
    assert str(axis.tz) == "UTC"
    assert axis.unit == "us"


def test_fractional_offsets_are_kept():
    values = pd.Series(["2022-01-01 05:30:00+05:30", "2022-01-01 06:30:00+05:30"])

    axis, _ = T.parse_instants(values, table="t")

    assert list(axis) == [pd.Timestamp("2022-01-01 00:00", tz="UTC"), pd.Timestamp("2022-01-01 01:00", tz="UTC")]


@pytest.mark.parametrize("typed", [False, True], ids=["strings", "typed"])
def test_naive_timestamps_are_read_as_utc(typed):
    values = pd.Series(["2022-07-01 12:00:00", "2022-07-01 13:00:00"])
    if typed:
        values = pd.to_datetime(values)

    axis, naive = T.parse_instants(values, table="t")

    assert naive
    assert axis[0] == pd.Timestamp("2022-07-01 12:00", tz="UTC")


def test_mixing_naive_and_aware_timestamps_raises():
    values = pd.Series(["2022-07-01 12:00:00+02:00", "2022-07-01 13:00:00"])

    with pytest.raises(ValueError, match=r"'t'.*2022-07-01 13:00:00"):
        T.parse_instants(values, table="t")


def test_a_missing_timestamp_raises():
    values = pd.Series(["2022-07-01 12:00:00+02:00", None])

    with pytest.raises(ValueError, match=r"'t'.*row 1"):
        T.parse_instants(values, table="t")


def test_unreadable_timestamps_raise():
    with pytest.raises(ValueError, match="'t'"):
        T.parse_instants(pd.Series(["not a time", "2022-01-01"]), table="t")


# -- tables -----------------------------------------------------------------------------------------------------------


def test_normalized_table_is_indexed_by_instants_and_leaves_the_input_unchanged():
    weather = _weather(0, 24)
    original = weather.copy()

    table, naive = T.normalize_table(weather, table="weather")

    pd.testing.assert_frame_equal(weather, original)
    assert not naive
    assert table.index.equals(_instants(weather))
    assert table.index.name == C.DATETIME
    assert C.DATETIME not in table.columns
    pd.testing.assert_frame_equal(table.reset_index(drop=True), weather.drop(columns=C.DATETIME).reset_index(drop=True))


@pytest.mark.parametrize("kind", ["DatetimeIndex", "string index"])
def test_timestamps_in_the_index_are_used(kind):
    weather = _weather(0, 24)
    frame = weather.set_index(C.DATETIME)
    if kind == "DatetimeIndex":
        frame.index = pd.to_datetime(frame.index, utc=True)

    table, _ = T.normalize_table(frame, table="weather")

    assert table.index.equals(_instants(weather))


def test_column_and_index_must_identify_the_same_instants():
    weather = _weather(0, 24)
    agreeing = weather.set_index(pd.DatetimeIndex(_instants(weather)), drop=False)
    disagreeing = agreeing.set_index(agreeing.index + pd.Timedelta(3600, unit="s"))

    T.normalize_table(agreeing, table="weather")
    with pytest.raises(ValueError, match=r"'weather'.*index"):
        T.normalize_table(disagreeing, table="weather")


def test_a_table_without_timestamps_is_not_a_time_table():
    assert T.normalize_table(pd.DataFrame({"a": [1, 2]}), table="windows") == (None, False)


def test_a_repeated_timestamp_raises_with_a_hint_for_local_times():
    # A naive local-time file repeats 02:00 on the change back to winter time.
    frame = pd.DataFrame(
        {C.DATETIME: ["2022-10-30 01:00", "2022-10-30 02:00", "2022-10-30 02:00", "2022-10-30 03:00"], "x": range(4)}
    )

    with pytest.raises(ValueError, match=r"(?s)'weather'.*2022-10-30 02:00.*ambiguous=\"infer\""):
        T.normalize_table(frame, table="weather")


def test_rows_are_sorted_by_instant():
    frame = pd.DataFrame({C.DATETIME: ["2022-01-01 01:00Z", "2022-01-01 00:00Z"], "x": [1, 2]})

    table, _ = T.normalize_table(frame, table="weather")

    assert list(table["x"]) == [2, 1]
    assert table.index.is_monotonic_increasing


def test_regular_step():
    axis, _ = T.parse_instants(_weather(0, 5)[C.DATETIME], table="weather")

    assert T.regular_step(axis, table="weather") == pd.Timedelta(3600, unit="s")
    with pytest.raises(ValueError, match=r"'weather'.*step"):
        T.regular_step(axis.delete(2), table="weather")


# -- site zone --------------------------------------------------------------------------------------------------------


def test_site_zone_reads_the_timezone_key():
    assert T.site_zone({O.ID: 1, O.TIMEZONE: "Europe/Berlin"}) == ZoneInfo("Europe/Berlin")


@pytest.mark.parametrize("value", [None, np.nan, ""])
def test_site_zone_defaults_to_utc(value):
    assert T.site_zone({O.ID: 1, O.TIMEZONE: value}) == ZoneInfo("UTC")
    assert T.site_zone({O.ID: 1}) == ZoneInfo("UTC")


@pytest.mark.parametrize("value", ["Mars/Olympus", "+01:00", 1.0])
def test_an_unknown_site_zone_raises(value):
    with pytest.raises(ValueError, match=r"'house'.*IANA"):
        T.site_zone({O.ID: "house", O.TIMEZONE: value})


# -- run time and auxiliary tables ------------------------------------------------------------------------------------


def _axis(rows: int = 24) -> pd.DatetimeIndex:
    return T.parse_instants(_weather(0, rows)[C.DATETIME], table="weather")[0]


def _gains(rows: int = 48) -> pd.DataFrame:
    weather = _weather(0, rows)
    return pd.DataFrame({C.DATETIME: weather[C.DATETIME], "residential": np.arange(rows, dtype=float)})


def test_timestamped_auxiliary_tables_align_by_instant():
    gains = _gains()
    shuffled = gains.sample(frac=1.0, random_state=1)
    as_utc = gains.assign(**{C.DATETIME: pd.to_datetime(gains[C.DATETIME], utc=True)})
    run = T.RunTime({"ordered": gains, "shuffled": shuffled, "utc": as_utc})
    axis = _axis()

    expected = run.align("ordered", axis)

    assert expected.index.equals(axis)
    assert list(expected["residential"]) == list(range(24))
    pd.testing.assert_frame_equal(run.align("shuffled", axis), expected)
    pd.testing.assert_frame_equal(run.align("utc", axis), expected)


def test_an_auxiliary_table_that_misses_an_instant_raises():
    gains = _gains().drop(index=5)
    run = T.RunTime({"gains": gains})
    first_missing = _axis()[5]

    with pytest.raises(ValueError, match=rf"'gains'.*{first_missing:%Y-%m-%d %H:%M}"):
        run.align("gains", _axis())


def test_an_auxiliary_table_without_timestamps_aligns_by_position_with_one_warning(caplog):
    gains = _gains(24).drop(columns=C.DATETIME)
    run = T.RunTime({"gains": gains})

    with caplog.at_level(logging.WARNING, logger=T.__name__):
        first = run.align("gains", _axis())
        run.align("gains", _axis())

    assert first.index.equals(_axis())
    assert list(first["residential"]) == list(range(24))
    assert len([r for r in caplog.records if "'gains'" in r.getMessage()]) == 1


def test_an_auxiliary_table_without_timestamps_and_another_length_raises():
    run = T.RunTime({"gains": _gains(23).drop(columns=C.DATETIME)})

    with pytest.raises(ValueError, match=r"'gains'.*23.*24"):
        run.align("gains", _axis())


def test_run_time_warns_once_per_naive_table(caplog):
    naive = pd.DataFrame({C.DATETIME: pd.date_range("2022-01-01", periods=3, freq="h"), "x": range(3)})

    with caplog.at_level(logging.WARNING, logger=T.__name__):
        run = T.RunTime({"naive": naive, "aware": _weather(0, 3)})
        run.table("naive")
        run.table("naive")

    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1
    assert "'naive'" in messages[0] and "tz_localize" in messages[0]


def test_a_broken_table_raises_only_when_it_is_used():
    broken = pd.DataFrame({C.DATETIME: ["2022-01-01 00:00Z", "2022-01-01 00:00Z"], "x": [1, 2]})
    run = T.RunTime({"broken": broken, "weather": _weather(0, 3)})

    assert len(run.table("weather")) == 3
    with pytest.raises(ValueError, match="'broken'"):
        run.table("broken")


def test_run_time_is_built_for_direct_calls_and_reused_in_a_run():
    data = {"weather": _weather(0, 3)}

    built = T.run_time(data)

    assert isinstance(built, T.RunTime)
    assert T.RUN_TIME_KEY not in data
    assert T.run_time({**data, T.RUN_TIME_KEY: built}) is built


# -- irradiance labels ------------------------------------------------------------------------------------------------

LONGITUDE = 11.06  # the example weather's site


def _irradiance(first: int, rows: int, shift: int = 0) -> pd.DataFrame:
    weather = _weather(first, rows + 1)
    ghi = weather[C.SOLAR_GHI].shift(-shift)
    frame = pd.DataFrame({C.SOLAR_GHI: ghi.to_numpy()}, index=_instants(weather))
    return frame.iloc[:rows]


def test_end_labelled_irradiance_is_reported(caplog):
    # The example weather comes from Open-Meteo, which labels hourly means at the interval end.
    with caplog.at_level(logging.WARNING, logger=T.__name__):
        offset = T.check_irradiance_labels(_irradiance(4000, 24 * 14), LONGITUDE, table="weather")

    assert offset > 0.5
    assert any("'weather'" in r.getMessage() and "shift" in r.getMessage() for r in caplog.records)


def test_start_labelled_irradiance_is_not_reported(caplog):
    with caplog.at_level(logging.WARNING, logger=T.__name__):
        offset = T.check_irradiance_labels(_irradiance(4000, 24 * 14, shift=1), LONGITUDE, table="weather")

    assert abs(offset) < 0.25
    assert not caplog.records


@pytest.mark.parametrize("rows, freq", [(24 * 3, "h"), (24 * 4 * 14, "15min")], ids=["three days", "15 min"])
def test_irradiance_labels_are_not_judged_on_short_or_subhourly_data(rows, freq):
    index = pd.date_range("2022-06-01", periods=rows, freq=freq, tz="UTC")
    frame = pd.DataFrame({C.SOLAR_GHI: np.ones(rows)}, index=index)

    assert T.check_irradiance_labels(frame, LONGITUDE, table="weather") is None


# -- review findings --------------------------------------------------------------------------------------------------


def test_date_only_values_are_naive_and_may_mix_with_times():
    values = pd.Series(["2022-01-01", "2022-01-01 01:00:00", "2022-01-01 02:00:00"])

    axis, naive = T.parse_instants(values, table="t")

    assert naive
    assert axis[0] == pd.Timestamp("2022-01-01 00:00", tz="UTC")


@pytest.mark.parametrize(
    "values",
    [
        ["05/01/2022 00:00", "13/01/2022 00:00"],
        ["12:00", "13:00"],
        ["2022-01-01 00:00 UTC", "2022-01-01 01:00 UTC"],
        ["2022-01-01 00:00 EST", "2022-01-01 01:00 EST"],
    ],
    ids=["day first", "time only", "zone name UTC", "zone name EST"],
)
def test_text_that_is_not_iso_8601_raises(values):
    with pytest.raises(ValueError, match=r"'t'.*ISO 8601"):
        T.parse_instants(pd.Series(values), table="t")


def test_numbers_are_not_timestamps():
    with pytest.raises(ValueError, match=r"'t'.*numbers"):
        T.parse_instants(pd.Series([1_640_995_200, 1_640_998_800]), table="t")


def test_an_unreadable_table_that_no_object_uses_does_not_stop_the_run():
    unused = pd.DataFrame({C.DATETIME: ["2022-01-01 00:00 UTC", "2022-01-01 01:00 UTC"], "x": [1, 2]})

    run = T.RunTime({"unused": unused, "weather": _weather(0, 3)})

    assert len(run.table("weather")) == 3
    with pytest.raises(ValueError, match="'unused'"):
        run.table("unused")


def test_changing_a_returned_table_does_not_change_the_run():
    run = T.RunTime({"weather": _weather(0, 3)})
    column = C.SOLAR_GHI

    first = run.table("weather")
    first.loc[first.index[0], column] = -1.0
    first[column] *= 10

    assert run.table("weather")[column].iloc[0] != -1.0
    assert (run.table("weather")[column] == run.table("weather")[column]).all()
    pd.testing.assert_series_equal(
        run.table("weather")[column], T.RunTime({"weather": _weather(0, 3)}).table("weather")[column]
    )


@pytest.mark.parametrize("value", [pd.NA, "   ", np.float32("nan")])
def test_site_zone_treats_missing_values_as_utc(value):
    assert T.site_zone({O.ID: 1, O.TIMEZONE: value}) == ZoneInfo("UTC")


def test_a_zone_directory_is_not_a_time_zone():
    with pytest.raises(ValueError, match=r"'house'.*IANA"):
        T.site_zone({O.ID: "house", O.TIMEZONE: "Europe"})


def test_align_requires_an_aware_axis():
    run = T.RunTime({"gains": _gains()})

    with pytest.raises(ValueError, match="tz-aware"):
        run.align("gains", _axis().tz_localize(None))


def test_ten_days_of_irradiance_are_not_judged():
    assert T.check_irradiance_labels(_irradiance(4000, 24 * 10), LONGITUDE, table="weather") is None
