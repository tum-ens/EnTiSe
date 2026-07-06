import numpy as np
import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.methods.hvac.R5C1 import R5C1


@pytest.fixture
def minimal_weather():
    """Minimal weather data for testing."""
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            C.DATETIME: index,
            C.TEMP_AIR: np.full(24, 5.0),
            C.SOLAR_GHI: np.full(24, 100.0),
            C.SOLAR_DHI: np.full(24, 20.0),
            C.SOLAR_DNI: np.full(24, 80.0),
        },
        index=index,
    )


@pytest.fixture
def basic_5r1c_object():
    """Basic 5R1C object with required parameters."""
    return {
        O.ID: "test_obj",
        O.C_M: 5e7,
        O.H_TR_IS: 500.0,
        O.H_TR_MS: 1000.0,
        O.H_TR_W: 30.0,
        O.H_TR_EM: 800.0,
        O.VENTILATION: 100.0,
        O.TEMP_INIT: 20.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 24.0,
        O.AREA: 150.0,
        O.HEIGHT: 2.7,
        O.LAT: 49.0,
        O.LON: 11.0,
    }


def test_r5c1_basic_generation(basic_5r1c_object, minimal_weather):
    """Test basic R5C1 generation with minimal inputs."""
    obj = basic_5r1c_object
    data = {O.WEATHER: minimal_weather}

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    assert "summary" in result
    assert "timeseries" in result
    assert isinstance(result["timeseries"], pd.DataFrame)
    assert len(result["timeseries"]) == 24


def test_r5c1_output_columns(basic_5r1c_object, minimal_weather):
    """Test that R5C1 outputs contain expected columns."""
    obj = basic_5r1c_object
    data = {O.WEATHER: minimal_weather}

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    ts = result["timeseries"]
    expected_cols = [
        C.TEMP_IN,
        f"{Types.HEATING}{SEP}{C.LOAD}[W]",
        f"{Types.COOLING}{SEP}{C.LOAD}[W]",
    ]
    assert all(col in ts.columns for col in expected_cols)


def test_r5c1_summary_keys(basic_5r1c_object, minimal_weather):
    """Test that R5C1 summary contains expected keys."""
    obj = basic_5r1c_object
    data = {O.WEATHER: minimal_weather}

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    summary = result["summary"]
    expected_keys = [
        f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]",
        f"{Types.HEATING}{SEP}{O.LOAD_MAX}[W]",
        f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]",
        f"{Types.COOLING}{SEP}{O.LOAD_MAX}[W]",
    ]
    assert all(key in summary for key in expected_keys)


def test_r5c1_heating_demand_cold_weather(basic_5r1c_object):
    """Test heating demand with cold outdoor temperature."""
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    cold_weather = pd.DataFrame(
        {
            C.DATETIME: index,
            C.TEMP_AIR: np.full(24, -5.0),
            C.SOLAR_GHI: np.zeros(24),
            C.SOLAR_DHI: np.zeros(24),
            C.SOLAR_DNI: np.zeros(24),
        },
        index=index,
    )

    obj = basic_5r1c_object
    data = {O.WEATHER: cold_weather}

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    heating_demand = result["summary"][f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"]
    cooling_demand = result["summary"][f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]"]

    assert heating_demand > 0, "Should have heating demand in cold weather"
    assert cooling_demand == 0, "Should have no cooling demand in cold weather"


def test_r5c1_heating_disabled(basic_5r1c_object, minimal_weather):
    """Test that heating can be disabled."""
    obj = basic_5r1c_object.copy()
    obj[O.ACTIVE_HEATING] = False
    obj[O.TEMP_MIN] = 20.0

    # Set very cold weather
    data = {O.WEATHER: minimal_weather.copy()}
    data[O.WEATHER][C.TEMP_AIR] = -10.0

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    heating_load = result["timeseries"][f"{Types.HEATING}{SEP}{C.LOAD}[W]"]
    assert heating_load.sum() == 0, "Heating should be zero when disabled"


def test_r5c1_cooling_disabled(basic_5r1c_object, minimal_weather):
    """Test that cooling can be disabled."""
    obj = basic_5r1c_object.copy()
    obj[O.ACTIVE_COOLING] = False
    obj[O.TEMP_MAX] = 24.0

    # Set hot weather
    data = {O.WEATHER: minimal_weather.copy()}
    data[O.WEATHER][C.TEMP_AIR] = 35.0

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    cooling_load = result["timeseries"][f"{Types.COOLING}{SEP}{C.LOAD}[W]"]
    assert cooling_load.sum() == 0, "Cooling should be zero when disabled"


def test_r5c1_power_limits(basic_5r1c_object, minimal_weather):
    """Test that power limits are respected."""
    obj = basic_5r1c_object.copy()
    obj[O.POWER_HEATING] = 1000.0
    obj[O.POWER_COOLING] = 500.0

    # Set extreme cold weather
    data = {O.WEATHER: minimal_weather.copy()}
    data[O.WEATHER][C.TEMP_AIR] = -20.0

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    heating_load = result["timeseries"][f"{Types.HEATING}{SEP}{C.LOAD}[W]"]
    assert heating_load.max() <= obj[O.POWER_HEATING], "Heating load should not exceed limit"


def test_r5c1_temperature_bounds(basic_5r1c_object, minimal_weather):
    """Test that indoor temperature stays within bounds when HVAC is active."""
    obj = basic_5r1c_object.copy()
    obj[O.TEMP_MIN] = 20.0
    obj[O.TEMP_MAX] = 24.0
    obj[O.POWER_HEATING] = 10000.0
    obj[O.POWER_COOLING] = 10000.0

    data = {O.WEATHER: minimal_weather}

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    temp_in = result["timeseries"][C.TEMP_IN]
    # Allow small tolerance for numerical precision
    assert temp_in.min() >= obj[O.TEMP_MIN] - 0.1, "Temperature should not drop below minimum"
    assert temp_in.max() <= obj[O.TEMP_MAX] + 0.1, "Temperature should not exceed maximum"


def test_r5c1_constant_internal_gains(basic_5r1c_object, minimal_weather):
    """Test with constant internal gains as scalar."""
    obj = basic_5r1c_object.copy()
    obj[O.ACTIVE_GAINS_INTERNAL] = True
    obj[O.GAINS_INTERNAL] = 500.0

    data = {O.WEATHER: minimal_weather}

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    assert "summary" in result
    assert "timeseries" in result


def test_r5c1_solar_gains_disabled(basic_5r1c_object, minimal_weather):
    """Test with solar gains disabled."""
    obj = basic_5r1c_object.copy()
    obj[O.ACTIVE_GAINS_SOLAR] = False

    data = {O.WEATHER: minimal_weather}

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    assert "summary" in result
    assert "timeseries" in result


def test_r5c1_ventilation_disabled(basic_5r1c_object, minimal_weather):
    """Test with ventilation disabled."""
    obj = basic_5r1c_object.copy()
    obj[O.ACTIVE_VENTILATION] = False

    data = {O.WEATHER: minimal_weather}

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    assert "summary" in result
    assert "timeseries" in result


def test_r5c1_gain_splits(basic_5r1c_object, minimal_weather):
    """Test with custom gain split parameters."""
    obj = basic_5r1c_object.copy()
    obj[O.SIGMA_SURFACE] = 0.6
    obj[O.FRAC_CONV_INTERNAL] = 0.4
    obj[O.FRAC_RAD_SURFACE] = 0.8
    obj[O.FRAC_RAD_MASS] = 0.2
    obj[O.ACTIVE_GAINS_INTERNAL] = True
    obj[O.GAINS_INTERNAL] = 300.0

    data = {O.WEATHER: minimal_weather}

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    assert "summary" in result
    assert "timeseries" in result


def test_r5c1_thermal_mass_variation(basic_5r1c_object, minimal_weather):
    """Test with different thermal mass values."""
    obj_low = basic_5r1c_object.copy()
    obj_low[O.C_M] = 1e7

    obj_high = basic_5r1c_object.copy()
    obj_high[O.C_M] = 1e8

    data = {O.WEATHER: minimal_weather.copy()}
    data[O.WEATHER][C.TEMP_AIR] = -5.0

    r5c1 = R5C1()
    result_low = r5c1.generate(obj_low, data, Types.HVAC)
    result_high = r5c1.generate(obj_high, data, Types.HVAC)

    # Higher thermal mass should generally result in more stable temperatures
    temp_std_low = result_low["timeseries"][C.TEMP_IN].std()
    temp_std_high = result_high["timeseries"][C.TEMP_IN].std()

    # Both should produce valid results
    assert temp_std_low >= 0
    assert temp_std_high >= 0


def test_r5c1_resistance_variation(basic_5r1c_object, minimal_weather):
    """Test with different thermal resistance values."""
    obj_low_r = basic_5r1c_object.copy()
    obj_low_r[O.H_TR_EM] = 1500.0  # Lower resistance (higher heat transfer)

    obj_high_r = basic_5r1c_object.copy()
    obj_high_r[O.H_TR_EM] = 400.0  # Higher resistance (lower heat transfer)

    data = {O.WEATHER: minimal_weather.copy()}
    data[O.WEATHER][C.TEMP_AIR] = -5.0

    r5c1 = R5C1()
    result_low_r = r5c1.generate(obj_low_r, data, Types.HVAC)
    result_high_r = r5c1.generate(obj_high_r, data, Types.HVAC)

    # Lower resistance should result in higher heating demand
    demand_low_r = result_low_r["summary"][f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"]
    demand_high_r = result_high_r["summary"][f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"]

    assert demand_low_r > demand_high_r, "Lower resistance should have higher heating demand"


def test_r5c1_missing_required_parameter():
    """Test that missing required parameters raise appropriate errors."""
    obj = {
        O.ID: "test_obj",
        # Missing C_m, H_tr_is, etc.
    }
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            C.DATETIME: index,
            C.TEMP_AIR: np.full(24, 5.0),
            C.SOLAR_GHI: np.full(24, 100.0),
            C.SOLAR_DHI: np.full(24, 20.0),
            C.SOLAR_DNI: np.full(24, 80.0),
        },
        index=index,
    )
    data = {O.WEATHER: weather}

    r5c1 = R5C1()
    with pytest.raises((KeyError, ValueError, TypeError)):
        r5c1.generate(obj, data, Types.HVAC)


def test_r5c1_with_windows(basic_5r1c_object, minimal_weather):
    """Test with window data for solar gains."""
    obj = basic_5r1c_object.copy()
    obj[O.ACTIVE_GAINS_SOLAR] = True

    windows = pd.DataFrame(
        [
            {
                O.ID: "test_obj",
                C.AREA: 15.0,
                C.G_VALUE: 0.7,
                C.SHADING: 1.0,
                C.TILT: 90.0,
                C.ORIENTATION: 180.0,
            }
        ]
    )

    data = {O.WEATHER: minimal_weather, O.WINDOWS: windows}

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    assert "summary" in result
    assert "timeseries" in result


def test_r5c1_numerical_stability(basic_5r1c_object, minimal_weather):
    """Test numerical stability with extreme but valid parameters."""
    obj = basic_5r1c_object.copy()
    obj[O.C_M] = 1e8  # Very high thermal mass
    obj[O.H_TR_EM] = 50.0  # Very low heat transfer
    obj[O.POWER_HEATING] = 50000.0
    obj[O.POWER_COOLING] = 50000.0

    data = {O.WEATHER: minimal_weather}

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    # Check that results are finite and reasonable
    temp_in = result["timeseries"][C.TEMP_IN]
    assert np.all(np.isfinite(temp_in)), "Temperature values should be finite"
    assert temp_in.min() > -50, "Temperature should be reasonable"
    assert temp_in.max() < 100, "Temperature should be reasonable"


def test_r5c1_zero_ventilation(basic_5r1c_object, minimal_weather):
    """Test with zero ventilation rate."""
    obj = basic_5r1c_object.copy()
    obj[O.VENTILATION] = 0.0

    data = {O.WEATHER: minimal_weather}

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    assert "summary" in result
    assert "timeseries" in result


def test_r5c1_method_attributes():
    """Test that R5C1 has correct method attributes."""
    r5c1 = R5C1()

    assert Types.HVAC in r5c1.types
    assert r5c1.name == "5R1C"
    assert O.C_M in r5c1.required_keys
    assert O.H_TR_IS in r5c1.required_keys
    assert O.H_TR_MS in r5c1.required_keys
    assert O.H_TR_W in r5c1.required_keys
    assert O.H_TR_EM in r5c1.required_keys
    assert O.WEATHER in r5c1.required_data


def test_r5c1_output_summary_values(basic_5r1c_object, minimal_weather):
    """Test that summary values are non-negative and reasonable."""
    obj = basic_5r1c_object
    data = {O.WEATHER: minimal_weather}

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    summary = result["summary"]

    # All values should be non-negative
    assert summary[f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"] >= 0
    assert summary[f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]"] >= 0
    assert summary[f"{Types.HEATING}{SEP}{O.LOAD_MAX}[W]"] >= 0
    assert summary[f"{Types.COOLING}{SEP}{O.LOAD_MAX}[W]"] >= 0


def test_r5c1_output_timeseries_values(basic_5r1c_object, minimal_weather):
    """Test that timeseries values are finite and reasonable."""
    obj = basic_5r1c_object
    data = {O.WEATHER: minimal_weather}

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    ts = result["timeseries"]

    # Check temperature is finite and reasonable
    temp_in = ts[C.TEMP_IN]
    assert np.all(np.isfinite(temp_in))
    assert temp_in.min() >= -50
    assert temp_in.max() <= 100

    # Check loads are non-negative and finite
    heating_load = ts[f"{Types.HEATING}{SEP}{C.LOAD}[W]"]
    cooling_load = ts[f"{Types.COOLING}{SEP}{C.LOAD}[W]"]
    assert np.all(heating_load >= 0)
    assert np.all(cooling_load >= 0)
    assert np.all(np.isfinite(heating_load))
    assert np.all(np.isfinite(cooling_load))


def test_r5c1_multiple_objects_same_weather(basic_5r1c_object, minimal_weather):
    """Test processing multiple objects with same weather data."""
    obj1 = basic_5r1c_object.copy()
    obj1[O.ID] = "obj1"

    obj2 = basic_5r1c_object.copy()
    obj2[O.ID] = "obj2"
    obj2[O.C_M] = 8e7  # Different thermal mass

    data = {O.WEATHER: minimal_weather}

    r5c1 = R5C1()
    result1 = r5c1.generate(obj1, data, Types.HVAC)
    result2 = r5c1.generate(obj2, data, Types.HVAC)

    assert "summary" in result1
    assert "summary" in result2
    # Results should be different due to different parameters
    demand1 = result1["summary"][f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"]
    demand2 = result2["summary"][f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"]
    assert demand1 != demand2


def test_r5c1_initial_temperature(basic_5r1c_object, minimal_weather):
    """Test that initial temperature is respected."""
    obj = basic_5r1c_object.copy()
    obj[O.TEMP_INIT] = 15.0
    obj[O.TEMP_MIN] = 10.0
    obj[O.TEMP_MAX] = 30.0

    data = {O.WEATHER: minimal_weather}

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    temp_in = result["timeseries"][C.TEMP_IN]
    # First temperature should be close to initial (within reason given timestep)
    assert abs(temp_in.iloc[0] - obj[O.TEMP_INIT]) < 5.0


def test_r5c1_area_m_and_area_tot(basic_5r1c_object, minimal_weather):
    """Test with explicit area_m and area_tot parameters."""
    obj = basic_5r1c_object.copy()
    obj[O.AREA_M] = 300.0
    obj[O.AREA_TOT] = 400.0
    obj[O.ACTIVE_GAINS_INTERNAL] = True
    obj[O.GAINS_INTERNAL] = 300.0

    data = {O.WEATHER: minimal_weather}

    r5c1 = R5C1()
    result = r5c1.generate(obj, data, Types.HVAC)

    assert "summary" in result
    assert "timeseries" in result


# --- Thermostat dead band / hysteresis (issue #102) --------------------------
# Same semantics as R1C1: default 0 preserves current output; deadband > 0
# makes the controller land steady-state T_in on the upper/lower band edge
# instead of on T_min/T_max.


def _cold_weather(periods: int = 96, temp_c: float = -5.0) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            C.DATETIME: idx,
            C.TEMP_AIR: np.full(periods, temp_c),
            C.SOLAR_GHI: np.zeros(periods),
            C.SOLAR_DHI: np.zeros(periods),
            C.SOLAR_DNI: np.zeros(periods),
        },
        index=idx,
    )


def _hot_weather(periods: int = 96, temp_c: float = 35.0) -> pd.DataFrame:
    idx = pd.date_range("2025-06-01", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            C.DATETIME: idx,
            C.TEMP_AIR: np.full(periods, temp_c),
            C.SOLAR_GHI: np.zeros(periods),
            C.SOLAR_DHI: np.zeros(periods),
            C.SOLAR_DNI: np.zeros(periods),
        },
        index=idx,
    )


def test_r5c1_deadband_default_zero_matches_omitted(basic_5r1c_object):
    """Explicit deadband=0 must produce the same output as omitting it —
    bit-for-bit. Strict equality guards against future refactors that leak
    even sub-ULP drift into the no-op path."""
    weather = _cold_weather()
    obj_no = {**basic_5r1c_object, O.ID: "r5_no_db"}
    obj_zero = {**basic_5r1c_object, O.ID: "r5_db_zero", O.DEADBAND: 0.0}

    ts_no = R5C1().generate(obj_no, {O.WEATHER: weather.copy()}, Types.HVAC)["timeseries"]
    ts_zero = R5C1().generate(obj_zero, {O.WEATHER: weather.copy()}, Types.HVAC)["timeseries"]

    for col in ts_no.columns:
        assert np.array_equal(
            ts_no[col].to_numpy(), ts_zero[col].to_numpy()
        ), f"deadband=0 diverged from omitted on column {col}"


def test_r5c1_heating_with_deadband_targets_upper_band(basic_5r1c_object):
    """Cold ambient with active heating and deadband=2 K: T_in must settle
    at T_min + 2 (upper band edge)."""
    deadband = 2.0
    T_min = 20.0
    weather = _cold_weather(periods=240, temp_c=-10.0)  # long run to reach steady state

    obj = {
        **basic_5r1c_object,
        O.ID: "r5_heat_db",
        O.TEMP_MIN: T_min,
        O.TEMP_MAX: 30.0,
        O.DEADBAND: deadband,
        O.ACTIVE_COOLING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
    }
    ts = R5C1().generate(obj, {O.WEATHER: weather}, Types.HVAC)["timeseries"]

    # Steady state at upper band edge, not T_min.
    assert ts[C.TEMP_IN].iloc[-1] == pytest.approx(T_min + deadband, abs=0.15)
    assert ts[f"{Types.HEATING}{SEP}{C.LOAD}[W]"].iloc[-1] > 0


def test_r5c1_cooling_with_deadband_targets_lower_band(basic_5r1c_object):
    """Mirror: hot ambient, active cooling, deadband=2 K → T settles at T_max−2."""
    deadband = 2.0
    T_max = 24.0
    weather = _hot_weather(periods=240, temp_c=35.0)

    obj = {
        **basic_5r1c_object,
        O.ID: "r5_cool_db",
        O.TEMP_MIN: 15.0,
        O.TEMP_MAX: T_max,
        O.DEADBAND: deadband,
        O.ACTIVE_HEATING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
    }
    ts = R5C1().generate(obj, {O.WEATHER: weather}, Types.HVAC)["timeseries"]

    assert ts[C.TEMP_IN].iloc[-1] == pytest.approx(T_max - deadband, abs=0.15)
    assert ts[f"{Types.COOLING}{SEP}{C.LOAD}[W]"].iloc[-1] > 0


def test_r5c1_deadband_latches_across_in_band_transient(basic_5r1c_object):
    """State-machine correctness test. The distinguishing behavior of
    hysteresis vs. aim-for-setpoint is what happens when Ta_free lands
    *inside* the band AFTER a firing step. Without deadband the heater
    switches off (Ta_free ≥ T_min); with deadband the heater must stay
    firing (heating_on latched, Ta_free < T_min + deadband).

    Scenario: an initial cold spike forces the heater on; then ambient
    settles just above T_min. In the tail, the no-deadband heater is off
    while the deadband heater keeps firing thanks to the latch."""
    periods = 72
    T_min = 20.0
    deadband = 3.0
    idx = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    # 6 h cold to force heater on, then mild ambient just above T_min.
    temp_air = np.concatenate([np.full(6, -20.0), np.full(periods - 6, T_min + 0.5)])
    weather = pd.DataFrame(
        {
            C.DATETIME: idx,
            C.TEMP_AIR: temp_air,
            C.SOLAR_GHI: np.zeros(periods),
            C.SOLAR_DHI: np.zeros(periods),
            C.SOLAR_DNI: np.zeros(periods),
        },
        index=idx,
    )

    def _obj(db):
        return {
            **basic_5r1c_object,
            O.ID: f"r5_latch_{db}",
            O.C_M: 5e5,  # low thermal mass → responds within a step
            O.TEMP_INIT: T_min,
            O.TEMP_MIN: T_min,
            O.TEMP_MAX: 40.0,
            O.DEADBAND: db,
            O.ACTIVE_COOLING: False,
            O.ACTIVE_GAINS_SOLAR: False,
            O.ACTIVE_GAINS_INTERNAL: False,
        }

    p_no = (
        R5C1()
        .generate(_obj(0.0), {O.WEATHER: weather.copy()}, Types.HVAC)["timeseries"][f"{Types.HEATING}{SEP}{C.LOAD}[W]"]
        .to_numpy()
    )
    p_db = (
        R5C1()
        .generate(_obj(deadband), {O.WEATHER: weather.copy()}, Types.HVAC)["timeseries"][
            f"{Types.HEATING}{SEP}{C.LOAD}[W]"
        ]
        .to_numpy()
    )

    # After the cold spike, in the mild tail the deadband case must fire
    # strictly more than the no-deadband case (the latch keeps working).
    tail = slice(-24, None)
    assert (
        p_db[tail].sum() > p_no[tail].sum() * 1.5
    ), f"Deadband heater tail sum {p_db[tail].sum()} not meaningfully greater than baseline {p_no[tail].sum()}."
    # And a broken state machine that ignored the latch would match the
    # baseline exactly in the tail: assert at least one tail step where
    # the two paths diverge.
    assert not np.array_equal(
        p_no[tail], p_db[tail]
    ), "Deadband produced identical output to baseline in the mild tail — state machine not exercised."


def test_r5c1_wide_deadband_does_not_double_fire(basic_5r1c_object):
    """Wide-band mutex regression. Same claim as the R1C1 counterpart:
    a deadband wider than the setpoint band must not allow heating and
    cooling to fire simultaneously in the same step."""
    periods = 48
    T_min = 20.0
    T_max = 24.0
    deadband = 10.0
    idx = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    temp_air = np.where(np.arange(periods) % 2 == 0, -20.0, 30.0)
    weather = pd.DataFrame(
        {
            C.DATETIME: idx,
            C.TEMP_AIR: temp_air,
            C.SOLAR_GHI: np.zeros(periods),
            C.SOLAR_DHI: np.zeros(periods),
            C.SOLAR_DNI: np.zeros(periods),
        },
        index=idx,
    )
    obj = {
        **basic_5r1c_object,
        O.ID: "r5_wide_db",
        O.TEMP_MIN: T_min,
        O.TEMP_MAX: T_max,
        O.DEADBAND: deadband,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
    }
    ts = R5C1().generate(obj, {O.WEATHER: weather}, Types.HVAC)["timeseries"]
    p_heat = ts[f"{Types.HEATING}{SEP}{C.LOAD}[W]"].to_numpy()
    p_cool = ts[f"{Types.COOLING}{SEP}{C.LOAD}[W]"].to_numpy()

    both = (p_heat > 0) & (p_cool > 0)
    assert (
        not both.any()
    ), f"Heating and cooling fired simultaneously on {int(both.sum())} steps — wide-band mutex broken."
    assert (p_heat > 0).any() or (p_cool > 0).any()


# --- Latent cooling (issue #103) ---------------------------------------------
# Weather without humidity → the existing tests above guarantee bit-exact
# output. These target the humidity-present path.


def _wet_weather_5r1c(temp_c: float, rh: float, periods: int, p_pa: float = 101325.0) -> pd.DataFrame:
    """R5C1 needs solar columns too; a wet-summer fixture with humidity."""
    index = pd.date_range("2025-06-01", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            C.DATETIME: index,
            C.TEMP_AIR: np.full(periods, temp_c),
            C.HUMIDITY_REL: np.full(periods, rh),
            C.SURFACE_AIR_PRESSURE: np.full(periods, p_pa),
            C.SOLAR_GHI: np.zeros(periods),
            C.SOLAR_DHI: np.zeros(periods),
            C.SOLAR_DNI: np.zeros(periods),
        },
        index=index,
    )


def test_r5c1_output_has_sensible_and_latent_columns(basic_5r1c_object):
    """After #103 R5C1 emits three cooling columns: total, sensible, latent."""
    weather = _wet_weather_5r1c(30.0, 0.70, 24)
    result = R5C1().generate(basic_5r1c_object, {O.WEATHER: weather}, Types.HVAC)
    ts = result["timeseries"]

    assert f"{Types.COOLING}{SEP}{C.LOAD}[W]" in ts.columns
    assert f"{Types.COOLING}{SEP}sensible_{C.LOAD}[W]" in ts.columns
    assert f"{Types.COOLING}{SEP}latent_{C.LOAD}[W]" in ts.columns


def test_r5c1_wet_summer_produces_positive_latent_load(basic_5r1c_object):
    """R5C1 latent-cooling integration: hot humid outdoor + cool indoor →
    latent > 0 in steady state, total == sensible + latent."""
    weather = _wet_weather_5r1c(30.0, 0.70, 96)
    obj = {**basic_5r1c_object, O.TEMP_INIT: 24.0, O.ACTIVE_HEATING: False}
    result = R5C1().generate(obj, {O.WEATHER: weather}, Types.HVAC)
    ts = result["timeseries"]

    sens_col = f"{Types.COOLING}{SEP}sensible_{C.LOAD}[W]"
    lat_col = f"{Types.COOLING}{SEP}latent_{C.LOAD}[W]"
    total_col = f"{Types.COOLING}{SEP}{C.LOAD}[W]"

    assert ts[sens_col].iloc[-1] > 0
    assert ts[lat_col].iloc[-1] > 0
    assert ts[total_col].iloc[-1] == pytest.approx(ts[sens_col].iloc[-1] + ts[lat_col].iloc[-1], abs=1)


def test_r5c1_active_cooling_false_zeros_latent(basic_5r1c_object):
    """`active_cooling=False` → zero cooling of any kind, even on humid days."""
    weather = _wet_weather_5r1c(35.0, 0.80, 48)
    obj = {**basic_5r1c_object, O.ACTIVE_COOLING: False, O.ACTIVE_HEATING: False, O.TEMP_INIT: 24.0}
    result = R5C1().generate(obj, {O.WEATHER: weather}, Types.HVAC)
    ts = result["timeseries"]

    assert ts[f"{Types.COOLING}{SEP}latent_{C.LOAD}[W]"].max() == 0
    assert ts[f"{Types.COOLING}{SEP}sensible_{C.LOAD}[W]"].max() == 0


def test_r5c1_power_cooling_caps_total_load(basic_5r1c_object):
    """`power_cooling[W]` is now the TOTAL nameplate cap; sensible has
    priority. Total must not exceed the cap on any step."""
    weather = _wet_weather_5r1c(35.0, 0.80, 96)
    obj = {
        **basic_5r1c_object,
        O.TEMP_INIT: 24.0,
        O.ACTIVE_HEATING: False,
        O.POWER_COOLING: 800.0,
    }
    result = R5C1().generate(obj, {O.WEATHER: weather}, Types.HVAC)
    ts = result["timeseries"]

    total_col = f"{Types.COOLING}{SEP}{C.LOAD}[W]"
    assert ts[total_col].max() <= 800.0 + 1
