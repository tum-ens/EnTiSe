"""The file method returns its table on the object's time axis."""

import logging

import numpy as np
import pandas as pd
import pytest

from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.constants import Types
from entise.core.generator import Generator
from entise.methods.multiple.file import FileLoader

BERLIN = "Europe/Berlin"


def _instants(periods: int, start: str = "2022-03-26 22:00") -> pd.DatetimeIndex:
    # Crosses the change to summer time in Berlin.
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


def _table(periods: int = 6, *, timestamps: bool = True) -> pd.DataFrame:
    table = pd.DataFrame({"load[W]": np.arange(periods, dtype=float)})
    if timestamps:
        table.insert(0, C.DATETIME, _instants(periods).tz_convert(BERLIN).strftime("%Y-%m-%d %H:%M:%S%z"))
    return table


def _weather(periods: int = 6) -> pd.DataFrame:
    return pd.DataFrame({C.DATETIME: _instants(periods), "air_temperature[C]": 0.0})


def _object(**keys) -> dict:
    return {O.ID: "house", O.FILE: "table", O.TIMEZONE: BERLIN, **keys}


def test_a_timestamped_table_is_indexed_by_its_instants_in_the_site_zone():
    result = FileLoader().generate(_object(), {"table": _table()})["timeseries"]

    assert str(result.index.tz) == BERLIN
    assert result.index.tz_convert("UTC").equals(_instants(6).as_unit("us"))
    assert result.index.name == C.DATETIME
    assert list(result.columns) == ["load[W]"]


def test_a_utc_run_indexes_the_table_in_utc():
    generator = Generator(raise_on_error=True)
    generator.add_objects([{O.ID: "house", Types.ELECTRICITY: "file", O.FILE: "table", O.TIMEZONE: BERLIN}])

    _, timeseries = generator.generate({"table": _table()}, workers=1, show_progress=False, utc=True)

    assert str(timeseries["house"][Types.ELECTRICITY].index.tz) == "UTC"


def test_with_weather_a_timestamped_table_is_aligned_to_the_weather():
    table = _table(8).sample(frac=1.0, random_state=0)

    result = FileLoader().generate(_object(**{O.WEATHER: "weather"}), {"table": table, "weather": _weather(6)})[
        "timeseries"
    ]

    assert result.index.tz_convert("UTC").equals(_instants(6).as_unit("us"))
    assert list(result["load[W]"]) == list(range(6))


def test_with_weather_a_table_that_misses_an_instant_raises():
    table = _table(6).drop(index=2)

    with pytest.raises(ValueError, match="'table'"):
        FileLoader().generate(_object(**{O.WEATHER: "weather"}), {"table": table, "weather": _weather(6)})


def test_with_weather_a_table_without_timestamps_is_used_row_by_row_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        result = FileLoader().generate(
            _object(**{O.WEATHER: "weather"}), {"table": _table(timestamps=False), "weather": _weather(6)}
        )["timeseries"]

    assert result.index.tz_convert("UTC").equals(_instants(6).as_unit("us"))
    assert any("'table'" in record.getMessage() for record in caplog.records)


def test_with_weather_a_table_without_timestamps_and_another_length_raises():
    with pytest.raises(ValueError, match=r"'table'.*5.*6"):
        FileLoader().generate(
            _object(**{O.WEATHER: "weather"}), {"table": _table(5, timestamps=False), "weather": _weather(6)}
        )


def test_without_weather_a_table_without_timestamps_is_returned_as_is_with_a_warning(caplog):
    table = _table(timestamps=False)

    with caplog.at_level(logging.WARNING):
        result = FileLoader().generate(_object(), {"table": table})["timeseries"]

    pd.testing.assert_frame_equal(result, table)
    assert any("'table'" in r.getMessage() and "timestamps" in r.getMessage() for r in caplog.records)


def test_the_callers_table_is_unchanged():
    table = _table()
    original = table.copy()

    FileLoader().generate(_object(), {"table": table})

    pd.testing.assert_frame_equal(table, original)


def test_a_weather_key_that_names_a_missing_table_raises():
    with pytest.raises(ValueError, match=r"'house'.*'site_b'"):
        FileLoader().generate(_object(**{O.WEATHER: "site_b"}), {"table": _table()})


def test_a_finer_table_is_averaged_to_the_weather_step_with_a_warning(caplog):
    # A 15-minute table on hourly weather: each hour gets the mean of its four quarter hours.
    index = pd.date_range("2022-03-26 22:00", periods=24, freq="15min", tz="UTC")
    table = pd.DataFrame({C.DATETIME: index, "load[W]": np.arange(24, dtype=float)})

    with caplog.at_level(logging.WARNING):
        result = FileLoader().generate(_object(**{O.WEATHER: "weather"}), {"table": table, "weather": _weather(6)})[
            "timeseries"
        ]

    assert list(result["load[W]"]) == [1.5, 5.5, 9.5, 13.5, 17.5, 21.5]
    assert len([r for r in caplog.records if "'table'" in r.getMessage() and "averag" in r.getMessage()]) == 1


def test_a_table_shifted_against_the_weather_steps_raises():
    # Values at half past the hour would be moved by half a step if they were averaged.
    index = pd.date_range("2022-03-26 22:30", periods=6, freq="h", tz="UTC")
    table = pd.DataFrame({C.DATETIME: index, "load[W]": np.arange(6, dtype=float)})

    with pytest.raises(ValueError, match=r"'table'.*does not cover"):
        FileLoader().generate(_object(**{O.WEATHER: "weather"}), {"table": table, "weather": _weather(6)})


def test_rows_outside_the_weather_span_are_ignored():
    index = pd.date_range("2022-03-26 18:00", periods=16, freq="h", tz="UTC")
    table = pd.DataFrame({C.DATETIME: index, "load[W]": np.arange(16, dtype=float)})

    result = FileLoader().generate(_object(**{O.WEATHER: "weather"}), {"table": table, "weather": _weather(6)})[
        "timeseries"
    ]

    assert list(result["load[W]"]) == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
