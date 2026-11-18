"""Calendar methods read the site's local wall clock at every instant of the run."""

import logging
import os
import sys
import types
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core.generator import Generator

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
BERLIN = "Europe/Berlin"
ELEC = f"{Types.ELECTRICITY}{SEP}{C.LOAD}[W]"
HEAT = f"{Types.HEATING}{SEP}{C.LOAD}[W]"
ELEC_DEMAND = f"{Types.ELECTRICITY}{SEP}{C.DEMAND}[Wh]"
HEAT_DEMAND = f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"


def _weather(first: int = 0, rows: int = 8760) -> pd.DataFrame:
    weather = pd.read_csv(os.path.join(HERE, "common_data", "weather.csv"))
    return weather.iloc[first : first + rows].reset_index(drop=True)


def _representations(weather: pd.DataFrame) -> dict[str, pd.DataFrame]:
    utc = pd.to_datetime(weather[C.DATETIME], utc=True)
    return {
        "offset strings": weather,
        "typed UTC": weather.assign(**{C.DATETIME: utc}),
        "typed Berlin": weather.assign(**{C.DATETIME: utc.dt.tz_convert(BERLIN)}),
    }


def _run(objects: list[dict], data: dict) -> tuple[pd.DataFrame, dict]:
    generator = Generator(raise_on_error=True)
    generator.add_objects(objects)
    return generator.generate(data, workers=1, show_progress=False)


def _wall(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index).tz_convert(BERLIN).tz_localize(None)


def _electricity(demand_kwh: float = 3000.0, **extra) -> dict:
    return {
        O.ID: "electricity",
        Types.ELECTRICITY: "demandlib_electricity",
        O.DATETIMES: O.WEATHER,
        O.DEMAND_KWH: demand_kwh,
        O.PROFILE: "h0",
        O.TIMEZONE: BERLIN,
        **extra,
    }


def _heat(demand_kwh: float = 20000.0) -> dict:
    return {
        O.ID: "heat",
        Types.HEATING: "demandlib_heat",
        O.WEATHER: O.WEATHER,
        O.DEMAND_KWH: demand_kwh,
        O.TIMEZONE: BERLIN,
    }


def _slp_reference(index: pd.DatetimeIndex, profile: str, energy_wh: float) -> np.ndarray:
    """BDEW SLP at each instant's Berlin wall clock, hourly mean, scaled to the run's energy."""
    pytest.importorskip("demandlib")
    from demandlib import bdew

    from entise.methods.utils._demandlib import contain_warning_filters
    from entise.methods.utils.holidays import get_holidays

    wall = _wall(index)
    years = sorted(set(wall.year))
    with contain_warning_filters():
        slp = pd.concat(
            [
                bdew.ElecSlp(y, holidays=get_holidays(holidays_location=None, years=[y])).get_scaled_power_profiles(
                    {profile: 1.0}
                )[profile]
                for y in years
            ]
        )
    hourly = slp.resample("1h").mean().reindex(wall).to_numpy()
    return hourly * energy_wh / hourly.sum()


# -- M1: the local clock ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["offset strings", "typed UTC", "typed Berlin"])
def test_electricity_reads_the_slp_at_the_local_wall_clock(name):
    weather = _representations(_weather())[name]

    summary, timeseries = _run([_electricity()], {O.WEATHER: weather})

    out = timeseries["electricity"][Types.ELECTRICITY]
    expected = _slp_reference(out.index, "h0", 3_000_000.0)
    mismatches = int((np.abs(out[ELEC].to_numpy() - expected) > 0.51).sum())
    assert mismatches == 0
    assert str(out.index.tz) == BERLIN


def test_heat_reads_the_engine_at_the_local_wall_clock():
    pytest.importorskip("demandlib")
    from demandlib import bdew

    seen = []
    original = bdew.HeatBuilding.get_bdew_profile

    def spy(self, *args, **kwargs):
        profile = original(self, *args, **kwargs)
        seen.append(profile.copy())
        return profile

    with patch.object(bdew.HeatBuilding, "get_bdew_profile", spy):
        summary, timeseries = _run([_heat()], {O.WEATHER: _weather()})

    out = timeseries["heat"][Types.HEATING]
    profile = seen[0]
    profile.index = pd.DatetimeIndex(profile.index).tz_localize(None)
    engine = profile.reindex(_wall(out.index)).to_numpy(dtype=float)
    expected = engine * 20_000_000.0 / engine.sum()
    assert int((np.abs(out[HEAT].to_numpy() - expected) > 0.51).sum()) == 0


@pytest.mark.parametrize("method", ["electricity", "heat"])
def test_the_same_instants_give_the_same_output(method):
    obj = _electricity() if method == "electricity" else _heat()
    column = ELEC if method == "electricity" else HEAT
    ts_type = Types.ELECTRICITY if method == "electricity" else Types.HEATING
    outputs = []
    for weather in _representations(_weather(2000, 24 * 14)).values():
        _, timeseries = _run([obj], {O.WEATHER: weather})
        outputs.append(timeseries[obj[O.ID]][ts_type][column])

    for other in outputs[1:]:
        pd.testing.assert_series_equal(other, outputs[0])


# -- 23-hour and 25-hour days -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("day, rows", [("2022-03-27", 23), ("2022-10-30", 25)])
def test_electricity_on_daylight_saving_days(day, rows):
    weather = _weather(0, 8760)
    local = pd.to_datetime(weather[C.DATETIME], utc=True).dt.tz_convert(BERLIN)
    weather = weather[(local >= pd.Timestamp(day, tz=BERLIN) - pd.Timedelta(1, unit="D"))].iloc[: 24 * 3]

    summary, timeseries = _run([_electricity()], {O.WEATHER: weather})

    out = timeseries["electricity"][Types.ELECTRICITY]
    on_day = out[out.index.strftime("%Y-%m-%d") == day]
    assert len(on_day) == rows
    expected = _slp_reference(out.index, "h0", 3_000_000.0)
    assert int((np.abs(out[ELEC].to_numpy() - expected) > 0.51).sum()) == 0
    wall_hours = list(_wall(on_day.index).hour)
    if rows == 23:
        assert 2 not in wall_hours
    else:
        assert wall_hours.count(2) == 2
        repeated = on_day[_wall(on_day.index).hour == 2][ELEC]
        assert repeated.iloc[0] == repeated.iloc[1]


# -- M10: edges and cadence -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("first", [2041, 7249], ids=["spring", "fall"])
@pytest.mark.parametrize("method", ["electricity", "heat"])
def test_a_first_step_across_a_clock_change_runs(method, first):
    obj = _electricity() if method == "electricity" else _heat()
    ts_type = Types.ELECTRICITY if method == "electricity" else Types.HEATING
    weather = _weather(first, 6)

    _, timeseries = _run([obj], {O.WEATHER: weather})

    assert len(timeseries[obj[O.ID]][ts_type]) == 6


def test_a_run_that_starts_between_two_profile_labels_keeps_its_energy():
    index = pd.date_range("2022-02-01 00:05", periods=4 * 24 * 7, freq="15min", tz=BERLIN)
    weather = pd.DataFrame({C.DATETIME: index})

    summary, timeseries = _run([_electricity(demand_kwh=60.0)], {O.WEATHER: weather})

    out = timeseries["electricity"][Types.ELECTRICITY]
    assert len(out) == len(index)
    assert abs(out[ELEC].sum() * 0.25 - 60_000) <= 0.5 * 0.25 * len(index)


def test_heat_at_30_minutes_ending_between_two_hours_keeps_every_row():
    weather = _weather(0, 25)
    instants = pd.to_datetime(weather[C.DATETIME], utc=True)
    index = pd.date_range(instants.iloc[0], periods=48, freq="30min")
    temperature = np.interp(
        index.as_unit("ns").asi8, pd.DatetimeIndex(instants).as_unit("ns").asi8, weather["air_temperature[C]@2m"]
    )
    half_hourly = pd.DataFrame({C.DATETIME: index, "air_temperature[C]@2m": temperature})

    _, timeseries = _run([_heat()], {O.WEATHER: half_hourly})

    assert len(timeseries["heat"][Types.HEATING]) == 48


def test_a_gap_in_the_datetimes_raises():
    weather = _weather(0, 48).drop(index=10)

    with pytest.raises(ValueError, match="time step that is regular"):
        _run([_electricity()], {O.WEATHER: weather})


def test_electricity_energy_equals_the_demand_over_the_run():
    summary, timeseries = _run([_electricity(demand_kwh=1000.0)], {O.WEATHER: _weather(1000, 168)})

    out = timeseries["electricity"][Types.ELECTRICITY]
    assert abs(out[ELEC].sum() - 1_000_000) <= 0.5 * 168
    assert abs(summary.loc["electricity", ELEC_DEMAND] - 1_000_000) <= 0.5 * 168


# -- PyLPG and DistrictHeatingSim through stub engines ----------------------------------------------------------------


def _stub_lpg(obj_id, year, start, end, households, occupants_per_household, energy_intensity_name):
    """Minute energies whose power is the local wall-clock hour plus one, so a reading reveals its hour."""
    minutes = pd.date_range(start, end, freq="min", inclusive="left", name=C.DATETIME)
    return pd.DataFrame({"Electricity HH1": (minutes.hour.to_numpy(dtype=float) + 1) / 60_000.0}, index=minutes)


def _pylpg(**extra) -> dict:
    return {
        O.ID: "lpg",
        Types.ELECTRICITY: "pylpg",
        O.DATETIMES: O.WEATHER,
        O.HOUSEHOLDS: 1,
        O.OCCUPANTS_PER_HOUSEHOLD: 2,
        O.TIMEZONE: BERLIN,
        **extra,
    }


@pytest.mark.parametrize("first", [2030, 7240], ids=["spring", "fall"])
def test_pylpg_reads_the_engine_at_the_local_wall_clock(first):
    with patch("entise.methods.electricity.pylpg.run_year_chunk", _stub_lpg):
        _, timeseries = _run([_pylpg()], {O.WEATHER: _weather(first, 36)})

    out = timeseries["lpg"][Types.ELECTRICITY]
    assert list(out[ELEC]) == list(_wall(out.index).hour + 1)


def test_pylpg_reads_a_year_with_a_single_row():
    weather = pd.DataFrame({C.DATETIME: pd.date_range("2022-12-31 22:00", periods=3, freq="h", tz=BERLIN)})

    with patch("entise.methods.electricity.pylpg.run_year_chunk", _stub_lpg):
        _, timeseries = _run([_pylpg()], {O.WEATHER: weather})

    assert list(timeseries["lpg"][Types.ELECTRICITY][ELEC]) == [23, 24, 1]


@pytest.fixture
def stub_districtheatingsim(monkeypatch):
    """A DistrictHeatingSim stand-in that returns the TRY hour (1-24) as hourly energy."""

    def calculate(JWB_kWh, profiletype, subtype, TRY_file_path, year, real_ww_share=None):  # noqa: N803
        hours = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left")
        rows = np.loadtxt(TRY_file_path, skiprows=34, usecols=(2, 3, 4))
        assert len(rows) == len(hours)
        total = rows[:, 2].astype(float)
        return hours.to_numpy(), total, None, None, None

    module = types.ModuleType("districtheatingsim.heat_requirement.heat_requirement_BDEW")
    module.calculate = calculate
    for name in ("districtheatingsim", "districtheatingsim.heat_requirement"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_districtheatingsim_reads_the_engine_at_the_local_wall_clock(stub_districtheatingsim):
    obj = {O.ID: "dhs", Types.HEATING: "districtheatingsim", O.WEATHER: O.WEATHER, O.TIMEZONE: BERLIN}

    summary, timeseries = _run([obj], {O.WEATHER: _weather(2030, 36)})

    out = timeseries["dhs"][Types.HEATING]
    total = out[HEAT] + out[f"{Types.DHW}{SEP}{C.LOAD}[W]"]
    engine = _wall(out.index).hour.to_numpy() + 1.0
    expected = engine / engine.sum() * total.sum()
    assert np.abs(total.to_numpy() - expected).max() <= 1.01


# -- missing time zone ------------------------------------------------------------------------------------------------


def test_a_run_warns_once_about_objects_without_a_time_zone(caplog):
    objects = [{**_electricity(), O.ID: name} for name in ("north", "east", "west")]
    for obj in objects[1:]:
        del obj[O.TIMEZONE]

    with caplog.at_level(logging.WARNING, logger="entise.core.generator"):
        _run(objects, {O.WEATHER: _weather(0, 24)})

    messages = [r.getMessage() for r in caplog.records if f"'{O.TIMEZONE}'" in r.getMessage()]
    assert len(messages) == 1
    assert "east" in messages[0] and "west" in messages[0] and "north" not in messages[0]


@pytest.mark.parametrize(
    "example", sorted(d for d in os.listdir(os.path.join(ROOT, "examples")) if d[0].isalpha() and "_" in d)
)
def test_example_objects_declare_their_time_zone(example):
    path = os.path.join(ROOT, "examples", example, "objects.csv")
    if not os.path.exists(path):
        pytest.skip("no objects table")
    objects = pd.read_csv(path)

    assert O.TIMEZONE in objects.columns
    assert (objects[O.TIMEZONE] == BERLIN).all()


# -- M15 and M6: occupancy after a built-in electricity method --------------------------------------------------------


@pytest.mark.parametrize("method", ["GeoMA", "PHT"])
def test_occupancy_runs_on_a_built_in_electricity_method(method):
    obj = {**_electricity(), O.ID: "home", Types.OCCUPANCY: method, f"{Types.OCCUPANCY}:{O.NIGHT_SCHEDULE}": True}

    _, timeseries = _run([obj], {O.WEATHER: _weather(0, 24 * 7)})

    occupancy = timeseries["home"][Types.OCCUPANCY]
    assert occupancy.index.equals(timeseries["home"][Types.ELECTRICITY].index)
    assert str(occupancy.index.tz) == BERLIN


# -- review fixes ---------------------------------------------------------------------------------------------------


def test_daily_electricity_at_local_midnight_runs_across_clock_changes():
    days = pd.date_range("2022-03-20", "2022-11-10", freq="D", tz=BERLIN)

    summary, timeseries = _run([_electricity(demand_kwh=1000.0)], {O.WEATHER: pd.DataFrame({C.DATETIME: days})})

    out = timeseries["electricity"][Types.ELECTRICITY]
    assert len(out) == len(days)
    hours = np.diff(np.append(days.as_unit("ns").asi8, pd.Timestamp("2022-11-11", tz=BERLIN).value)) / 3.6e12
    assert abs(float((out[ELEC].to_numpy() * hours).sum()) - 1_000_000) <= 0.5 * hours.sum()
    assert abs(summary.loc["electricity", ELEC_DEMAND] - 1_000_000) <= 0.5 * hours.sum()


def test_a_time_step_irregular_on_both_clocks_names_the_instant():
    index = pd.DatetimeIndex(["2022-01-01 00:00", "2022-01-01 01:00", "2022-01-01 03:00"]).tz_localize(BERLIN)

    with pytest.raises(ValueError, match=r"'weather'.*2022-01-01 03:00"):
        _run([_electricity()], {O.WEATHER: pd.DataFrame({C.DATETIME: index})})


def test_pylpg_reports_missing_engine_output():
    def with_a_hole(*args, **kwargs):
        frame = _stub_lpg(*args, **kwargs)
        return frame.drop(frame.index[100:160])

    with (
        patch("entise.methods.electricity.pylpg.run_year_chunk", with_a_hole),
        pytest.raises(ValueError, match="2022-01-01 01:40"),
    ):
        _run([_pylpg()], {O.WEATHER: _weather(0, 24)})
