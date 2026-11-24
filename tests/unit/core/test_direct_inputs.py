"""Calling a method directly: one signature, every declared input by name (issue #86)."""

import inspect
import os

import pandas as pd
import pytest

from entise.constants import SEP, VALID_TYPES, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core.base import Method, input_name
from entise.core.generator import Generator
from entise.core.registry import _BUILTIN_STRATEGIES, get_strategy

WEATHER_CSV = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "integration", "common_data", "weather.csv")

# Generator methods; the auxiliary strategies keep their own interface.
GENERATORS = [name for name, strategy in _BUILTIN_STRATEGIES.items() if not strategy.module.startswith("auxiliary.")]

# Python names that predate the derived names and stay accepted.
ALIASES = [
    ("jordanvajen", "temp_water_cold", O.TEMP_WATER_COLD),
    ("jordanvajen", "temp_water_hot", O.TEMP_WATER_HOT),
    ("demandlib_electricity", "demand_kwh", O.DEMAND_KWH),
    ("demandlib_heat", "annual_demand_kwh", O.DEMAND_KWH),
    ("districtheatingsim", "annual_demand_kwh", O.DEMAND_KWH),
    ("ruhnau", "temp_sink", O.TEMP_SINK),
    ("ruhnau", "temp_water", O.TEMP_WATER),
    ("1r1c", "temp_init", O.TEMP_INIT),
    ("1r1c", "temp_min", O.TEMP_MIN),
    ("1r1c", "temp_max", O.TEMP_MAX),
    ("1r0c", "temp_init", O.TEMP_INIT),
    ("1r0c", "temp_min", O.TEMP_MIN),
    ("1r0c", "temp_max", O.TEMP_MAX),
    ("file", "file", O.FILE),
]


class _Stop(Exception):
    """Ends a call once its inputs are resolved."""


def _declared(cls) -> list:
    return list(dict.fromkeys([*cls.required_keys, *cls.optional_keys, *cls.required_data, *cls.optional_data]))


def _resolved_all(monkeypatch, cls, **inputs) -> tuple[dict, dict, dict]:
    """Return the obj, data and results a direct call resolves its keywords to."""
    seen = {}
    resolve = Method._process_inputs

    def spy(self, obj=None, data=None, results=None, ts_type=None, **kwargs):
        seen["out"] = resolve(self, obj, data, results, ts_type, **kwargs)
        raise _Stop

    monkeypatch.setattr(cls, "_process_inputs", spy)
    with pytest.raises(_Stop):
        cls().generate(**inputs)
    return seen["out"]


def _resolved(monkeypatch, cls, **inputs) -> tuple[dict, dict]:
    """Return the obj and data a direct call resolves its keywords to."""
    return _resolved_all(monkeypatch, cls, **inputs)[:2]


def _placed(obj: dict, data: dict, key: str):
    if key in obj and isinstance(obj[key], str) and obj[key] == key and key in data:
        return data[key]  # a table the object refers to by its key
    return obj[key] if key in obj else data[key]


def _weather(hours: int = 7 * 24) -> pd.DataFrame:
    return pd.read_csv(WEATHER_CSV).iloc[:hours].reset_index(drop=True)


def _generator_run(obj: dict, data: dict, ts_type: str) -> tuple[dict, pd.DataFrame]:
    gen = Generator(raise_on_error=True)
    gen.add_objects(obj)
    summary, timeseries = gen.generate(data, workers=1, show_progress=False)
    row = summary.loc[obj[O.ID]].dropna().to_dict()
    return row, timeseries[obj[O.ID]][ts_type]


def _assert_same(direct: dict, summary: dict, timeseries: pd.DataFrame):
    pd.testing.assert_frame_equal(direct["timeseries"], timeseries, check_exact=True)
    assert {k: direct["summary"][k] for k in summary} == summary


# --- Names -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, name",
    [
        ("area[m2]", "area"),
        ("power_heating[W]", "power_heating"),
        ("capacitance[J K-1]", "capacitance"),
        ("air_temperature[C]@2m", "air_temperature"),
        ("T_eq[C]_column", "T_eq_column"),
        ("temp_init", "temp_init"),
        ("weather", "weather"),
    ],
)
def test_input_name_drops_unit_and_height(key, name):
    assert input_name(key) == name


@pytest.mark.parametrize("method", GENERATORS)
def test_generate_has_the_uniform_signature(method):
    cls = get_strategy(method)
    params = list(inspect.signature(cls.generate).parameters.values())

    assert [p.name for p in params] == ["self", "obj", "data", "results", "ts_type", "inputs"]
    assert [p.kind for p in params[1:]] == [inspect.Parameter.POSITIONAL_OR_KEYWORD] * 4 + [
        inspect.Parameter.VAR_KEYWORD
    ]
    assert [p.default for p in params[1:4]] == [None, None, None]
    assert params[4].default is None or params[4].default in cls.types


@pytest.mark.parametrize("method", GENERATORS)
def test_every_declared_input_has_its_own_python_name(method):
    cls = get_strategy(method)
    names = cls.input_names()

    for key in _declared(cls):
        assert names[key] == key
        assert input_name(key).isidentifier()
        assert names[input_name(key)] == key, f"{input_name(key)!r} is taken by {names[input_name(key)]!r}"


@pytest.mark.parametrize("method", GENERATORS)
def test_every_declared_input_is_accepted_by_python_name_and_full_key(monkeypatch, method):
    cls = get_strategy(method)

    for key in _declared(cls):
        for name in dict.fromkeys([input_name(key), key]):
            if key in VALID_TYPES:  # a result of another type
                table = pd.DataFrame({"x": [1]})
                _, _, results = _resolved_all(monkeypatch, cls, **{name: table})
                assert results[key]["timeseries"] is table
                continue
            obj, data = _resolved(monkeypatch, cls, **{name: "given"})
            assert _placed(obj, data, key) == "given", f"{cls.__name__}: {name!r} did not reach {key!r}"


@pytest.mark.parametrize("method, alias, key", ALIASES)
def test_existing_python_names_still_work(monkeypatch, method, alias, key):
    cls = get_strategy(method)

    obj, data = _resolved(monkeypatch, cls, **{alias: 12})

    assert _placed(obj, data, key) == 12
    assert alias not in obj


# --- Errors ----------------------------------------------------------------------------


@pytest.mark.parametrize("method", GENERATORS)
def test_unknown_keyword_raises_and_lists_valid_names(method):
    cls = get_strategy(method)
    some_name = input_name(_declared(cls)[0])

    with pytest.raises(TypeError) as err:
        cls().generate(not_an_input=1)

    message = str(err.value)
    assert cls.__name__ in message
    assert "not_an_input" in message
    assert some_name in message


@pytest.mark.parametrize(
    "method, inputs",
    [
        ("1r1c", {"area": 140, O.AREA: 150}),
        ("5r1c", {"area": 140, O.AREA: 150}),
        ("jordanvajen", {"temp_water_cold": 10, "cold_water_temperature": 12}),
        ("demandlib_heat", {"annual_demand_kwh": 1000, "demand": 1000}),
    ],
)
def test_two_names_for_one_input_raise(method, inputs):
    with pytest.raises(TypeError, match="multiple values"):
        get_strategy(method)().generate(**inputs)


# --- Where the values go ---------------------------------------------------------------


def test_keywords_override_the_object(monkeypatch):
    obj = {O.AREA: 100, f"{Types.HVAC}:{O.HEIGHT}": 2.5, O.TEMP_INIT: 18}
    cls = get_strategy("1r1c")

    out, _ = _resolved(monkeypatch, cls, obj=obj, area=140, height=3.0, temp_init=21)

    assert out[O.AREA] == 140
    assert out[O.HEIGHT] == 3.0
    assert f"{Types.HVAC}:{O.HEIGHT}" not in out
    assert out[O.TEMP_INIT] == 21
    assert obj == {O.AREA: 100, f"{Types.HVAC}:{O.HEIGHT}": 2.5, O.TEMP_INIT: 18}


def test_none_counts_as_not_given(monkeypatch):
    out, _ = _resolved(monkeypatch, get_strategy("1r1c"), obj={O.AREA: 100}, area=None)

    assert out[O.AREA] == 100


def test_table_for_a_data_input_goes_into_data(monkeypatch):
    weather = _weather(24)
    windows = pd.DataFrame({O.ID: ["b"]})

    obj, data = _resolved(monkeypatch, get_strategy("1r1c"), obj={O.WEATHER: "other"}, weather=weather, windows=windows)

    assert data[O.WEATHER] is weather
    assert obj[O.WEATHER] == O.WEATHER
    assert data[O.WINDOWS] is windows
    assert obj[O.WINDOWS] == O.WINDOWS


def test_table_reference_by_name_stays_in_the_object(monkeypatch):
    obj, data = _resolved(monkeypatch, get_strategy("pvlib"), data={"w": _weather(24)}, weather="w")

    assert obj[O.WEATHER] == "w"
    assert O.WEATHER not in data


# --- Direct calls give the Generator's results -----------------------------------------


def test_1r1c_direct_call_equals_generator():
    weather = _weather()
    keys = {O.CAPACITANCE: 1e8, O.RESISTANCE: 0.005, O.AREA: 120.0, O.HEIGHT: 2.6, O.TEMP_INIT: 19.0}
    summary, timeseries = _generator_run(
        {O.ID: "b", Types.HVAC: "1R1C", O.WEATHER: O.WEATHER, **keys}, {O.WEATHER: weather}, Types.HVAC
    )

    direct = get_strategy("1r1c")().generate(
        capacitance=1e8, resistance=0.005, area=120.0, height=2.6, temp_init=19.0, weather=weather
    )
    _assert_same(direct, summary, timeseries)

    by_key = get_strategy("1r1c")().generate(**keys, weather=weather)
    _assert_same(by_key, summary, timeseries)


def test_5r1c_direct_call_equals_generator():
    weather = _weather()
    keys = {
        O.C_M: 5e7,
        O.H_TR_IS: 500.0,
        O.H_TR_MS: 1000.0,
        O.H_TR_W: 30.0,
        O.H_TR_EM: 800.0,
        O.AREA: 150.0,
        O.LAT: 49.0,
        O.LON: 11.0,
    }
    summary, timeseries = _generator_run(
        {O.ID: "b", Types.HVAC: "5R1C", O.WEATHER: O.WEATHER, **keys}, {O.WEATHER: weather}, Types.HVAC
    )

    # The solar gains look up the object's windows by its id.
    direct = get_strategy("5r1c")().generate(
        {O.ID: "b"},
        C_m=5e7, H_tr_is=500.0, H_tr_ms=1000.0, H_tr_w=30.0, H_tr_em=800.0, area=150.0, latitude=49.0, longitude=11.0,
        weather=weather,
    )  # fmt: skip
    _assert_same(direct, summary, timeseries)

    # Full keys, as 5R1C took them before.
    by_key = get_strategy("5r1c")().generate({O.ID: "b"}, {O.WEATHER: weather}, **keys, **{O.WEATHER: O.WEATHER})
    _assert_same(by_key, summary, timeseries)


def test_pvlib_direct_call_equals_generator():
    weather = _weather()
    keys = {O.LAT: 48.1, O.LON: 11.6, O.POWER: 5000, O.AZIMUTH: 180, O.TILT: 30, O.ALTITUDE: 520}
    summary, timeseries = _generator_run(
        {O.ID: "p", Types.PV: "pvlib", O.WEATHER: O.WEATHER, **keys}, {O.WEATHER: weather}, Types.PV
    )

    direct = get_strategy("pvlib")().generate(
        latitude=48.1, longitude=11.6, power=5000, azimuth=180, tilt=30, altitude=520, weather=weather
    )
    _assert_same(direct, summary, timeseries)


def test_jordanvajen_direct_call_equals_generator():
    weather = _weather()
    keys = {O.DWELLING_SIZE: 90, O.TEMP_WATER_COLD: 11.0, O.SEED: 42}
    summary, timeseries = _generator_run(
        {O.ID: "d", Types.DHW: "jordanvajen", O.DATETIMES: O.WEATHER, **keys}, {O.WEATHER: weather}, Types.DHW
    )

    direct = get_strategy("jordanvajen")().generate(datetimes=weather, dwelling_size=90, temp_water_cold=11.0, seed=42)
    _assert_same(direct, summary, timeseries)


def test_demandlib_electricity_direct_call_equals_generator():
    datetimes = _weather()[[C.DATETIME]]
    keys = {O.DEMAND_KWH: 3000.0, O.PROFILE: "h0"}
    summary, timeseries = _generator_run(
        {O.ID: "e", Types.ELECTRICITY: "demandlib_electricity", O.DATETIMES: O.DATETIMES, **keys},
        {O.DATETIMES: datetimes},
        Types.ELECTRICITY,
    )

    method = get_strategy("demandlib_electricity")()
    direct = method.generate(data={O.DATETIMES: datetimes}, datetimes=O.DATETIMES, demand=3000.0, profile="h0")
    _assert_same(direct, summary, timeseries)

    old_name = method.generate(data={O.DATETIMES: datetimes}, datetimes=O.DATETIMES, demand_kwh=3000.0, profile="h0")
    _assert_same(old_name, summary, timeseries)

    as_table = method.generate(datetimes=datetimes, demand=3000.0, profile="h0")
    _assert_same(as_table, summary, timeseries)


def _fake_lpg_chunk(*, start, end, households, **_):
    """Stand in for the LPG engine: a fixed minute energy per household."""
    minutes = pd.date_range(start, end, freq="min", inclusive="left")
    return pd.DataFrame(
        {f"Electricity HH{hh}": (1 + minutes.minute % 7) * 1e-3 * hh for hh in range(1, households + 1)},
        index=minutes,
    )


def test_pylpg_direct_call_with_a_datetimes_table_equals_generator(monkeypatch):
    import entise.methods.electricity.pylpg as pylpg

    monkeypatch.setattr(pylpg, "run_year_chunk", _fake_lpg_chunk)
    datetimes = _weather(48)[[C.DATETIME]]
    keys = {O.HOUSEHOLDS: 2, O.OCCUPANTS_PER_HOUSEHOLD: 3}
    summary, timeseries = _generator_run(
        {O.ID: "l", Types.ELECTRICITY: "pylpg", O.DATETIMES: O.DATETIMES, **keys},
        {O.DATETIMES: datetimes},
        Types.ELECTRICITY,
    )

    direct = get_strategy("pylpg")().generate({O.ID: "l"}, datetimes=datetimes, households=2, occupants_per_household=3)
    _assert_same(direct, summary, timeseries)


def test_1r0c_direct_call_with_a_weather_table_equals_generator():
    weather = _weather()
    summary, timeseries = _generator_run(
        {O.ID: "b", Types.HVAC: "1R0C", O.WEATHER: O.WEATHER, O.RESISTANCE: 0.005, O.TEMP_INIT: 19.0},
        {O.WEATHER: weather},
        Types.HVAC,
    )

    direct = get_strategy("1r0c")().generate(resistance=0.005, temp_init=19.0, weather=weather)
    _assert_same(direct, summary, timeseries)


@pytest.mark.parametrize("timestamped", [False, True], ids=["scalar", "timestamped"])
def test_1r0c_direct_ventilation_keyword_overrides_prefixed_object_value(timestamped):
    weather = pd.DataFrame({C.DATETIME: pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"), C.TEMP_AIR: 0.0})
    obj = {O.ID: "b", Types.HVAC: "1R0C", O.WEATHER: O.WEATHER, O.RESISTANCE: 2.0, O.TEMP_MIN: 22.0}
    data = {O.WEATHER: weather}
    ventilation = 50.0
    if timestamped:
        column = "scheduled[W K-1]"
        ventilation = "ventilation_schedule"
        data[ventilation] = pd.DataFrame({C.DATETIME: weather[C.DATETIME], column: [25.0, 50.0, 75.0]})
        obj[O.VENTILATION_COL] = column

    flag = f"{Types.HVAC}{SEP}{O.ACTIVE_VENTILATION}"
    obj[O.VENTILATION] = ventilation
    summary, timeseries = _generator_run({**obj, flag: False}, data, Types.HVAC)

    method = get_strategy("1r0c")()
    assert method.input_names()["active_ventilation"] == O.ACTIVE_VENTILATION
    row = pd.Series({**obj, flag: True})
    direct = method.generate(obj=row, data=data, active_ventilation=False)
    _assert_same(direct, summary, timeseries)
    assert row[flag] is True

    heating = f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"
    assert summary[heating] == 33
    assert method.generate(obj=row, data=data, active_ventilation=None)["summary"][heating] == 3333


# --- Declarations ----------------------------------------------------------------------


@pytest.mark.parametrize("method", GENERATORS)
def test_tables_are_declared_as_data(method):
    cls = get_strategy(method)

    assert not hasattr(cls, "required_timeseries")
    assert not hasattr(cls, "optional_timeseries")


@pytest.mark.parametrize("method", GENERATORS)
def test_only_occupancy_requires_another_type_as_data(method):
    # The runner generates a required data input first when it names a configured type.
    cls = get_strategy(method)
    expected = [Types.ELECTRICITY] if method in ("geoma", "pht") else []

    assert [key for key in cls.required_data if key in VALID_TYPES] == expected


# --- Tables passed by keyword take effect ----------------------------------------------

EXAMPLES = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "examples")


def _example_row(folder: str, **drop) -> dict:
    row = pd.read_csv(os.path.join(EXAMPLES, folder, "objects.csv")).iloc[0].dropna().to_dict()
    return {k: v for k, v in row.items() if k not in drop and k not in VALID_TYPES}


def _example_table(folder: str, name: str):
    path = os.path.join(EXAMPLES, folder, "data", name)
    if name.endswith(".json"):
        import json

        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return pd.read_csv(path)


def _same_result(a: dict, b: dict):
    pd.testing.assert_frame_equal(a["timeseries"], b["timeseries"], check_exact=True)
    assert a["summary"] == b["summary"]


def test_7r2c_windows_by_keyword_equal_windows_by_reference():
    weather = _weather()
    windows = _example_table("hvac_7r2c", "windows.csv")
    obj = _example_row("hvac_7r2c", windows=None)
    method = get_strategy("7r2c")()

    by_reference = method.generate({**obj, O.WINDOWS: "win"}, {O.WEATHER: weather, "win": windows})
    without = method.generate(obj, {O.WEATHER: weather})
    assert not by_reference["timeseries"].equals(without["timeseries"])

    _same_result(method.generate(obj, {O.WEATHER: weather}, windows=windows), by_reference)
    _same_result(method.generate(obj, {O.WEATHER: weather, "win": windows}, windows="win"), by_reference)


def test_table_name_by_keyword_does_not_replace_the_table_in_data(monkeypatch):
    windows = pd.DataFrame({O.ID: ["b"]})

    obj, data = _resolved(monkeypatch, get_strategy("7r2c"), data={O.WINDOWS: windows, "win": windows}, windows="win")

    assert obj[O.WINDOWS] == "win"
    assert data[O.WINDOWS] is windows


def test_1r1c_tables_by_keyword_equal_tables_by_reference():
    weather = _weather()
    windows = _example_table("hvac_1r1c", "windows.csv")
    gains = pd.DataFrame({C.DATETIME: weather[C.DATETIME], "g": [100.0 + i % 24 * 10 for i in range(len(weather))]})
    obj = {**_example_row("hvac_1r1c", windows=None, weather=None), O.GAINS_INTERNAL_COL: "g"}
    obj.pop(O.GAINS_INTERNAL)
    method = get_strategy("1r1c")()

    by_reference = method.generate(
        {**obj, O.WEATHER: "w", O.WINDOWS: "win", O.GAINS_INTERNAL: "gains"},
        {"w": weather, "win": windows, "gains": gains},
    )
    by_keyword = method.generate(obj, weather=weather, windows=windows, gains_internal=gains)
    _same_result(by_keyword, by_reference)


def test_ruhnau_system_mapping_by_keyword_equals_reference():
    weather = _weather()
    system = _example_table("hp_ruhnau", "system.json")
    obj = _example_row("hp_ruhnau", weather=None)
    method = get_strategy("ruhnau")()

    by_reference = method.generate(
        {**obj, O.WEATHER: O.WEATHER, O.HP_SYSTEM: "system"}, {O.WEATHER: weather, "system": system}
    )
    _same_result(method.generate(obj, weather=weather, hp_system=system), by_reference)


def test_pvlib_arrays_mapping_by_keyword_equals_reference():
    weather = _weather()
    arrays = _example_table("pv_pvlib", "system.json")
    obj = _example_row("pv_pvlib")
    method = get_strategy("pvlib")()

    by_reference = method.generate({**obj, O.PV_ARRAYS: "arrays"}, {O.WEATHER: weather, "arrays": arrays})
    _same_result(method.generate(obj, {O.WEATHER: weather}, pv_arrays=arrays), by_reference)


def test_file_table_by_keyword_equals_reference():
    table = _weather(24)[[C.DATETIME, C.TEMP_AIR + "@2m"]]
    method = get_strategy("file")()

    by_reference = method.generate({O.ID: "f", O.FILE: "tbl"}, {"tbl": table})
    _same_result(method.generate({O.ID: "f"}, filename=table), by_reference)
    _same_result(method.generate({O.ID: "f"}, file=table), by_reference)


@pytest.mark.parametrize("method", ["geoma", "pht"])
def test_occupancy_electricity_by_keyword_equals_result(method):
    datetimes = _weather()[[C.DATETIME]]
    electricity = get_strategy("demandlib_electricity")().generate(datetimes=datetimes, demand=3000.0)["timeseries"]
    cls = get_strategy(method)

    by_result = cls().generate(results={Types.ELECTRICITY: {"timeseries": electricity}})
    _same_result(cls().generate(electricity=electricity), by_result)
    _same_result(cls().generate(results={}, electricity=electricity), by_result)


# --- obj as a row of an objects table --------------------------------------------------


def test_object_row_as_obj_with_keywords():
    weather = _weather()
    row = pd.Series({O.ID: "b", O.CAPACITANCE: 1e8, O.RESISTANCE: 0.005, O.WEATHER: O.WEATHER, O.TEMP_INIT: 18.0})
    method = get_strategy("1r1c")()

    as_dict = method.generate({**row.to_dict(), O.TEMP_INIT: 15.0}, {O.WEATHER: weather})
    _same_result(method.generate(row, {O.WEATHER: weather}, temp_init=15), as_dict)
    assert row[O.TEMP_INIT] == 18.0
    _same_result(method.generate(row, {O.WEATHER: weather}), method.generate(row.to_dict(), {O.WEATHER: weather}))


# --- Name collisions in a method's declarations ----------------------------------------


def test_two_keys_with_one_python_name_are_rejected_at_class_creation():
    with pytest.raises(TypeError, match="area"):

        class _Clash(Method):
            types = [Types.HVAC]
            name = "_clash_keys"
            register = False
            required_keys = ["area[m2]", "area[ha]"]

            def generate(self, obj=None, data=None, results=None, ts_type=None, **inputs):
                return {}


def test_alias_that_shadows_another_input_is_rejected_at_class_creation():
    with pytest.raises(TypeError, match="height"):

        class _Shadow(Method):
            types = [Types.HVAC]
            name = "_clash_alias"
            register = False
            required_keys = ["area[m2]", "height[m]"]
            input_aliases = {"height": "area[m2]"}

            def generate(self, obj=None, data=None, results=None, ts_type=None, **inputs):
                return {}


def test_alias_to_an_undeclared_key_is_rejected_at_class_creation():
    with pytest.raises(TypeError, match="not_declared"):

        class _Dangling(Method):
            types = [Types.HVAC]
            name = "_clash_dangling"
            register = False
            required_keys = ["area[m2]"]
            input_aliases = {"old": "not_declared"}

            def generate(self, obj=None, data=None, results=None, ts_type=None, **inputs):
                return {}
