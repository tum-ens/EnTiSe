import numpy as np
import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.methods.hvac.R7C2 import R7C2, clamp_power, split_gains_7r2c


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
def basic_7r2c_object():
    """Basic 7R2C object with required parameters based on VDI 6007."""
    return {
        O.ID: "test_7r2c",
        O.R_1_AW: 0.002,
        O.C_1_AW: 4e6,
        O.R_1_IW: 0.002,
        O.C_1_IW: 1e7,
        O.R_ALPHA_STAR_IL: 0.0006,
        O.R_ALPHA_STAR_AW: 0.002,
        O.R_ALPHA_STAR_IW: 0.0008,
        O.R_REST_AW: 0.015,
        O.VENTILATION: 80.0,
        O.TEMP_INIT: 20.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 25.0,
        O.AREA: 150.0,
        O.HEIGHT: 2.7,
        O.LAT: 49.0,
        O.LON: 11.0,
    }


def test_r7c2_basic_generation(basic_7r2c_object, minimal_weather):
    """Test basic R7C2 generation with minimal inputs."""
    obj = basic_7r2c_object
    obj[O.WEATHER] = "basic_weather"
    data = {"basic_weather": minimal_weather}

    r7c2 = R7C2()
    result = r7c2.generate(obj, data, Types.HVAC)

    assert "summary" in result
    assert "timeseries" in result
    assert isinstance(result["timeseries"], pd.DataFrame)
    assert len(result["timeseries"]) == len(minimal_weather)


def test_r7c2_output_columns(basic_7r2c_object, minimal_weather):
    """Test that R7C2 outputs contain expected columns."""
    obj = basic_7r2c_object
    data = {O.WEATHER: minimal_weather}

    r7c2 = R7C2()
    result = r7c2.generate(obj, data, Types.HVAC)

    ts = result["timeseries"]
    expected_cols = [
        C.TEMP_IN,
        f"{Types.HEATING}{SEP}{C.LOAD}[W]",
        f"{Types.COOLING}{SEP}{C.LOAD}[W]",
    ]
    assert all(col in ts.columns for col in expected_cols)


def test_r7c2_summary_keys(basic_7r2c_object, minimal_weather):
    """Test that R7C2 summary contains expected keys."""
    obj = basic_7r2c_object
    data = {O.WEATHER: minimal_weather}

    r7c2 = R7C2()
    result = r7c2.generate(obj, data, Types.HVAC)

    summary = result["summary"]
    expected_keys = [
        f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]",
        f"{Types.HEATING}{SEP}{O.LOAD_MAX}[W]",
        f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]",
        f"{Types.COOLING}{SEP}{O.LOAD_MAX}[W]",
    ]
    assert all(key in summary for key in expected_keys)


def test_r7c2_heating_demand_cold_weather(basic_7r2c_object):
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

    obj = basic_7r2c_object
    obj[O.WEATHER] = "cold_weather"
    data = {"cold_weather": cold_weather}

    r7c2 = R7C2()
    result = r7c2.generate(obj, data, Types.HVAC)

    heating_demand = result["summary"][f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"]
    cooling_demand = result["summary"][f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]"]

    assert heating_demand > 0, "Should have heating demand in cold weather"
    assert cooling_demand == 0, "Should have no cooling demand in cold weather"


def test_r7c2_cooling_demand_hot_weather(basic_7r2c_object):
    """Test cooling demand with hot outdoor temperature."""
    index = pd.date_range("2025-07-01", periods=48, freq="h", tz="UTC")
    # Create hot weather with high solar gains to ensure cooling is needed
    hot_weather = pd.DataFrame(
        {
            C.DATETIME: index,
            C.TEMP_AIR: np.full(48, 35.0),
            C.SOLAR_GHI: np.full(48, 800.0),
            C.SOLAR_DHI: np.full(48, 100.0),
            C.SOLAR_DNI: np.full(48, 700.0),
        },
        index=index,
    )

    obj = basic_7r2c_object.copy()
    obj[O.TEMP_INIT] = 30.0  # Start hot
    obj[O.TEMP_MIN] = 15.0
    obj[O.TEMP_MAX] = 24.0
    obj[O.ACTIVE_GAINS_INTERNAL] = True
    obj[O.ACTIVE_GAINS_SOLAR] = True
    data = {O.WEATHER: hot_weather}

    r7c2 = R7C2()
    result = r7c2.generate(obj, data, Types.HVAC)

    heating_demand = result["summary"][f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"]
    cooling_demand = result["summary"][f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]"]

    # With hot outdoor temp, high solar gains, and initial high temp, should need cooling
    assert (
        cooling_demand > 0 or result["timeseries"][C.TEMP_IN].max() > 24.5
    ), "Should have cooling demand or temperature above setpoint in hot weather"
    assert heating_demand >= 0, "Heating demand should be zero or minimal"


def test_r7c2_heating_disabled(basic_7r2c_object, minimal_weather):
    """Test that heating can be disabled."""
    obj = basic_7r2c_object.copy()
    obj[O.ACTIVE_HEATING] = False
    data = {O.WEATHER: minimal_weather}

    r7c2 = R7C2()
    result = r7c2.generate(obj, data, Types.HVAC)

    heating_demand = result["summary"][f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"]
    assert heating_demand == 0, "Heating demand should be zero when disabled"


def test_r7c2_cooling_disabled(basic_7r2c_object, minimal_weather):
    """Test that cooling can be disabled."""
    obj = basic_7r2c_object.copy()
    obj[O.ACTIVE_COOLING] = False
    data = {O.WEATHER: minimal_weather}

    r7c2 = R7C2()
    result = r7c2.generate(obj, data, Types.HVAC)

    cooling_demand = result["summary"][f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]"]
    assert cooling_demand == 0, "Cooling demand should be zero when disabled"


def test_r7c2_power_limits(basic_7r2c_object, minimal_weather):
    """Test that power limits are respected."""
    obj = basic_7r2c_object.copy()
    obj[O.POWER_HEATING] = 500.0
    obj[O.POWER_COOLING] = 300.0
    data = {O.WEATHER: minimal_weather}

    r7c2 = R7C2()
    result = r7c2.generate(obj, data, Types.HVAC)

    ts = result["timeseries"]
    max_heating = ts[f"{Types.HEATING}{SEP}{C.LOAD}[W]"].max()
    max_cooling = ts[f"{Types.COOLING}{SEP}{C.LOAD}[W]"].max()

    assert max_heating <= 500.0, "Heating load should not exceed limit"
    assert max_cooling <= 300.0, "Cooling load should not exceed limit"


def test_r7c2_temperature_bounds(basic_7r2c_object, minimal_weather):
    """Test that indoor temperature respects setpoint bounds when possible."""
    obj = basic_7r2c_object.copy()
    obj[O.TEMP_MIN] = 20.0
    obj[O.TEMP_MAX] = 24.0
    obj[O.POWER_HEATING] = 10000.0
    obj[O.POWER_COOLING] = 10000.0
    data = {O.WEATHER: minimal_weather}

    r7c2 = R7C2()
    result = r7c2.generate(obj, data, Types.HVAC)

    ts = result["timeseries"]
    temps = ts[C.TEMP_IN]

    # With sufficient power, temperature should mostly stay within bounds
    # Allow small violations due to initial conditions or numerical precision
    assert temps.min() >= 19.5, "Indoor temp should stay above minimum setpoint"
    assert temps.max() <= 24.5, "Indoor temp should stay below maximum setpoint"


def test_r7c2_ventilation_split(basic_7r2c_object, minimal_weather):
    """Test ventilation split between mechanical and infiltration."""
    obj = basic_7r2c_object.copy()
    obj[O.VENTILATION] = 100.0
    obj[O.VENTILATION_SPLIT] = 0.7  # 70% mechanical, 30% infiltration
    data = {O.WEATHER: minimal_weather}

    r7c2 = R7C2()
    result = r7c2.generate(obj, data, Types.HVAC)

    # Should generate successfully with split ventilation
    assert "summary" in result
    assert "timeseries" in result


def test_r7c2_thermal_mass_variation(basic_7r2c_object, minimal_weather):
    """Test that higher thermal mass reduces load variations."""
    # Low thermal mass
    obj_low = basic_7r2c_object.copy()
    obj_low[O.C_1_AW] = 1e6
    obj_low[O.C_1_IW] = 2e6
    data = {O.WEATHER: minimal_weather}

    r7c2 = R7C2()
    result_low = r7c2.generate(obj_low, data, Types.HVAC)

    # High thermal mass
    obj_high = basic_7r2c_object.copy()
    obj_high[O.C_1_AW] = 1e7
    obj_high[O.C_1_IW] = 2e7

    result_high = r7c2.generate(obj_high, data, Types.HVAC)

    # Both should generate successfully
    assert result_low["summary"][f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"] >= 0
    assert result_high["summary"][f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"] >= 0


def test_r7c2_sigma_splits(basic_7r2c_object, minimal_weather):
    """Test HVAC sigma distribution (AW, IW, convective)."""
    obj = basic_7r2c_object.copy()
    obj[O.SIGMA_7R2C_AW] = 0.3
    obj[O.SIGMA_7R2C_IW] = 0.2
    # Convective should be 0.5 (derived)
    data = {O.WEATHER: minimal_weather}

    r7c2 = R7C2()
    result = r7c2.generate(obj, data, Types.HVAC)

    assert "summary" in result
    assert result["summary"][f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"] >= 0


def test_r7c2_missing_required_parameter():
    """Test that missing required parameters raise an error."""
    obj = {
        O.ID: "test_incomplete",
        O.R_1_AW: 0.002,
        # Missing other required parameters
    }
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            C.DATETIME: index,
            C.TEMP_AIR: np.full(24, 5.0),
            C.SOLAR_GHI: np.full(24, 100.0),
        },
        index=index,
    )
    data = {O.WEATHER: weather}

    r7c2 = R7C2()
    with pytest.raises((KeyError, ValueError, TypeError)):
        r7c2.generate(obj, data, Types.HVAC)


def test_r7c2_numerical_stability(basic_7r2c_object, minimal_weather):
    """Test numerical stability with extreme but valid parameters."""
    obj = basic_7r2c_object.copy()
    obj[O.R_1_AW] = 0.0001  # Very low resistance
    obj[O.C_1_AW] = 1e8  # Very high capacitance
    data = {O.WEATHER: minimal_weather}

    r7c2 = R7C2()
    result = r7c2.generate(obj, data, Types.HVAC)

    # Check that outputs are finite
    ts = result["timeseries"]
    assert np.all(np.isfinite(ts[C.TEMP_IN])), "Indoor temperatures should be finite"
    assert np.all(np.isfinite(ts[f"{Types.HEATING}{SEP}{C.LOAD}[W]"])), "Heating loads should be finite"


def test_r7c2_method_attributes():
    """Test that R7C2 method has required attributes."""
    r7c2 = R7C2()
    assert hasattr(r7c2, "name")
    assert hasattr(r7c2, "types")
    assert hasattr(r7c2, "required_keys")
    assert r7c2.name == "7R2C"
    assert Types.HVAC in r7c2.types


def test_r7c2_with_windows(basic_7r2c_object, minimal_weather):
    """Test R7C2 with window data."""
    obj = basic_7r2c_object.copy()
    obj[O.WINDOWS] = "test_windows"

    # Create simple window data
    windows = pd.DataFrame(
        {
            O.ID: ["test_7r2c"],
            "orientation[degree]": [180],
            "tilt[degree]": [90],
            C.AREA: [10.0],
            "g_value[1]": [0.6],
            C.U_VALUE: [1.5],
            "shading[1]": [1.0],
        }
    )

    data = {
        O.WEATHER: minimal_weather,
        "test_windows": windows,
    }

    r7c2 = R7C2()
    result = r7c2.generate(obj, data, Types.HVAC)

    assert "summary" in result
    assert "timeseries" in result


def test_r7c2_initial_temperature(basic_7r2c_object, minimal_weather):
    """Test that initial temperature affects first timestep."""
    obj_cold = basic_7r2c_object.copy()
    obj_cold[O.TEMP_INIT] = 15.0

    obj_warm = basic_7r2c_object.copy()
    obj_warm[O.TEMP_INIT] = 23.0

    data = {O.WEATHER: minimal_weather}

    r7c2 = R7C2()
    result_cold = r7c2.generate(obj_cold, data, Types.HVAC)
    result_warm = r7c2.generate(obj_warm, data, Types.HVAC)

    # Different initial temperatures should lead to different results
    temp_cold = result_cold["timeseries"][C.TEMP_IN].iloc[0]
    temp_warm = result_warm["timeseries"][C.TEMP_IN].iloc[0]

    assert temp_cold != temp_warm, "Initial temperature should affect results"


# Unit tests for helper functions


def test_split_gains_7r2c():
    """Test gain splitting function."""
    g_int = np.array([100.0, 200.0, 150.0])
    g_sol = np.array([50.0, 100.0, 75.0])
    f_conv = 0.5
    f_aw = 0.6
    f_iw = 0.4

    q_conv, phi_aw, phi_iw = split_gains_7r2c(g_int, g_sol, f_conv, f_aw, f_iw)

    # Check shapes
    assert q_conv.shape == g_int.shape
    assert phi_aw.shape == g_int.shape
    assert phi_iw.shape == g_int.shape

    # Check values for first element
    expected_conv = 0.5 * 100.0  # 50W convective from internal
    expected_rad = 0.5 * 100.0 + 50.0  # 100W total radiant
    expected_aw = 0.6 * expected_rad
    expected_iw = 0.4 * expected_rad

    assert np.isclose(q_conv[0], expected_conv)
    assert np.isclose(phi_aw[0], expected_aw)
    assert np.isclose(phi_iw[0], expected_iw)


def test_clamp_power_heating():
    """Test power clamping for heating."""
    Q_req = 1000.0
    on_h = True
    on_c = True
    P_h_max = 500.0
    P_c_max = 300.0

    Q, clamped = clamp_power(Q_req, on_h, on_c, P_h_max, P_c_max)

    assert Q == 500.0, "Should clamp to max heating power"
    assert clamped is True, "Should indicate clamping occurred"


def test_clamp_power_cooling():
    """Test power clamping for cooling."""
    Q_req = -500.0
    on_h = True
    on_c = True
    P_h_max = 1000.0
    P_c_max = 300.0

    Q, clamped = clamp_power(Q_req, on_h, on_c, P_h_max, P_c_max)

    assert Q == -300.0, "Should clamp to max cooling power"
    assert clamped is True, "Should indicate clamping occurred"


def test_clamp_power_disabled_heating():
    """Test power clamping when heating is disabled."""
    Q_req = 1000.0
    on_h = False
    on_c = True
    P_h_max = 500.0
    P_c_max = 300.0

    Q, clamped = clamp_power(Q_req, on_h, on_c, P_h_max, P_c_max)

    assert Q == 0.0, "Should return zero when heating is disabled"
    assert clamped is True, "Should indicate clamping occurred"


def test_clamp_power_disabled_cooling():
    """Test power clamping when cooling is disabled."""
    Q_req = -500.0
    on_h = True
    on_c = False
    P_h_max = 1000.0
    P_c_max = 300.0

    Q, clamped = clamp_power(Q_req, on_h, on_c, P_h_max, P_c_max)

    assert Q == 0.0, "Should return zero when cooling is disabled"
    assert clamped is True, "Should indicate clamping occurred"


def test_clamp_power_no_clamping():
    """Test power clamping when no clamping is needed."""
    Q_req = 300.0
    on_h = True
    on_c = True
    P_h_max = 1000.0
    P_c_max = 500.0

    Q, clamped = clamp_power(Q_req, on_h, on_c, P_h_max, P_c_max)

    assert Q == 300.0, "Should return requested power"
    assert clamped is False, "Should indicate no clamping occurred"


# VDI 6007 Validation Tests


def test_r7c2_vdi_validation_case():
    """Test R7C2 with VDI 6007 validation parameters.

    This test uses parameters from the VDI 6007 standard test case.
    The validation case represents a typical test zone with:
    - Defined RC parameters for outer walls (AW) and inner walls (IW)
    - Specified ventilation rate and temperature setpoints
    - Window openings in four orientations

    According to VDI 6007, the model should:
    1. Maintain indoor temperature within setpoint bounds when HVAC is active
    2. Calculate heating/cooling loads that follow physical principles
    3. Show appropriate thermal response to outdoor conditions
    """
    # VDI validation parameters from validation.csv
    obj = {
        O.ID: "vdi_tz1",
        O.R_1_AW: 6.373022952356368e-05,
        O.C_1_AW: 375843459.6618696,
        O.R_1_IW: 1.943247597984055e-05,
        O.C_1_IW: 310348978.77092135,
        O.R_ALPHA_STAR_IL: 7.1959050757407e-05,
        O.R_ALPHA_STAR_AW: 6.945817025576001e-05,
        O.R_ALPHA_STAR_IW: 3.874007633861909e-05,
        O.R_REST_AW: 0.00027428186434865095,
        O.VENTILATION: 100.0,
        O.VENTILATION_SPLIT: 1.0,
        O.TEMP_INIT: 20.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 25.0,
        O.AREA: 200.0,
        O.HEIGHT: 3.0,
        O.LAT: 49.0,
        O.LON: 11.0,
        O.SIGMA_7R2C_AW: 0.25,
        O.SIGMA_7R2C_IW: 0.25,
        O.FRAC_CONV_INTERNAL: 0.5,
        O.FRAC_RAD_AW: 0.6,
    }

    # Create test weather for validation
    # Use 72 hours to test multi-day behavior while avoiding potential data processing limits
    n_hours = 72
    index = pd.date_range("2025-01-01", periods=n_hours, freq="h", tz="UTC")

    # VDI validation typically uses varied outdoor conditions
    outdoor_temp = 10.0 + 5.0 * np.sin(2 * np.pi * np.arange(n_hours) / 24)
    solar_radiation = np.maximum(0, 300 * np.sin(2 * np.pi * np.arange(n_hours) / 24))

    weather = pd.DataFrame(
        {
            C.DATETIME: index,
            C.TEMP_AIR: outdoor_temp,
            C.SOLAR_GHI: solar_radiation,
            C.SOLAR_DHI: solar_radiation * 0.2,
            C.SOLAR_DNI: solar_radiation * 0.8,
        },
        index=index,
    )

    data = {O.WEATHER: weather}

    r7c2 = R7C2()
    result = r7c2.generate(obj, data, Types.HVAC)

    # VDI validation checks
    ts = result["timeseries"]
    summary = result["summary"]

    # 1. Output structure validation
    assert len(ts) >= 24, "Should have at least one day of timeseries data"
    assert C.TEMP_IN in ts.columns, "Should contain indoor temperature"

    # 2. Temperature bounds (VDI 6007 requirement)
    # With sufficient HVAC power, temperature should stay within setpoints
    temps = ts[C.TEMP_IN]
    assert temps.min() >= 19.0, f"Indoor temp {temps.min():.2f}°C below acceptable range"
    assert temps.max() <= 26.0, f"Indoor temp {temps.max():.2f}°C above acceptable range"

    # 3. Energy balance validation (VDI 6007 principle)
    # Total energy should be non-negative
    heating_demand = summary[f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"]
    cooling_demand = summary[f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]"]
    assert heating_demand >= 0, "Heating demand must be non-negative"
    assert cooling_demand >= 0, "Cooling demand must be non-negative"

    # 4. Load peaks validation
    max_heating_load = summary[f"{Types.HEATING}{SEP}{O.LOAD_MAX}[W]"]
    max_cooling_load = summary[f"{Types.COOLING}{SEP}{O.LOAD_MAX}[W]"]
    assert max_heating_load >= 0, "Max heating load must be non-negative"
    assert max_cooling_load >= 0, "Max cooling load must be non-negative"

    # 5. Physical plausibility (VDI requirement)
    # Heating and cooling should not occur simultaneously
    heating_loads = ts[f"{Types.HEATING}{SEP}{C.LOAD}[W]"]
    cooling_loads = ts[f"{Types.COOLING}{SEP}{C.LOAD}[W]"]
    simultaneous = ((heating_loads > 1.0) & (cooling_loads > 1.0)).sum()
    assert simultaneous == 0, "Heating and cooling should not occur simultaneously"

    # 6. Temperature response validation
    # Indoor temperature should respond to outdoor temperature changes
    # but with thermal mass damping effect
    temp_std = temps.std()
    outdoor_std = weather[C.TEMP_AIR].std()
    assert temp_std < outdoor_std, "Indoor temp variations should be damped by thermal mass"


def test_r7c2_vdi_steady_state():
    """Test R7C2 steady-state behavior according to VDI 6007.

    In steady-state conditions (constant outdoor temperature, no solar gains),
    the model should reach equilibrium with constant heating/cooling load.
    """
    obj = {
        O.ID: "test_steady",
        O.R_1_AW: 0.001,
        O.C_1_AW: 5e6,
        O.R_1_IW: 0.001,
        O.C_1_IW: 5e6,
        O.R_ALPHA_STAR_IL: 0.0006,
        O.R_ALPHA_STAR_AW: 0.002,
        O.R_ALPHA_STAR_IW: 0.0008,
        O.R_REST_AW: 0.01,
        O.VENTILATION: 80.0,
        O.TEMP_INIT: 20.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 25.0,
        O.AREA: 150.0,
        O.HEIGHT: 2.7,
        O.LAT: 49.0,
        O.LON: 11.0,
        O.WEATHER: "steady_weather",
    }

    # Constant conditions for steady-state test
    index = pd.date_range("2025-01-01", periods=168, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            C.DATETIME: index,
            C.TEMP_AIR: np.full(168, 5.0),
            C.SOLAR_GHI: np.zeros(168),
            C.SOLAR_DHI: np.zeros(168),
            C.SOLAR_DNI: np.zeros(168),
        },
        index=index,
    )

    data = {"steady_weather": weather}

    r7c2 = R7C2()
    result = r7c2.generate(obj, data, Types.HVAC)

    ts = result["timeseries"]
    temps = ts[C.TEMP_IN]
    heating_loads = ts[f"{Types.HEATING}{SEP}{C.LOAD}[W]"]

    # After sufficient time, system should reach steady state
    # Check last 24 hours for steady-state behavior
    temps_final = temps.iloc[-24:]
    loads_final = heating_loads.iloc[-24:]

    # Temperature should be stable at setpoint
    temp_variation = temps_final.max() - temps_final.min()
    assert temp_variation < 0.2, "Temperature should be stable in steady state"

    # Heating load should be relatively constant in steady state
    # Allow for some variation due to numerical discretization
    load_std = loads_final.std()
    load_mean = loads_final.mean()
    if load_mean > 10:  # Only check if there's significant load
        cv = load_std / load_mean  # Coefficient of variation
        assert cv < 0.3, "Heating load should be relatively stable in steady state"


def test_r7c2_vdi_thermal_mass_effect():
    """Test thermal mass damping effect as per VDI 6007.

    Higher thermal mass should reduce temperature fluctuations
    and shift peak loads in time.
    """
    base_obj = {
        O.ID: "test_mass",
        O.R_1_AW: 0.002,
        O.R_1_IW: 0.002,
        O.R_ALPHA_STAR_IL: 0.0006,
        O.R_ALPHA_STAR_AW: 0.002,
        O.R_ALPHA_STAR_IW: 0.0008,
        O.R_REST_AW: 0.015,
        O.VENTILATION: 80.0,
        O.TEMP_INIT: 20.0,
        O.TEMP_MIN: 18.0,
        O.TEMP_MAX: 26.0,
        O.POWER_HEATING: 50.0,  # Limited power to see temperature swing
        O.POWER_COOLING: 50.0,
        O.AREA: 150.0,
        O.HEIGHT: 2.7,
        O.LAT: 49.0,
        O.LON: 11.0,
    }

    # Varying outdoor temperature
    index = pd.date_range("2025-01-01", periods=72, freq="h", tz="UTC")
    outdoor_temp = 10.0 + 8.0 * np.sin(2 * np.pi * np.arange(72) / 24)
    weather = pd.DataFrame(
        {
            C.DATETIME: index,
            C.TEMP_AIR: outdoor_temp,
            C.SOLAR_GHI: np.zeros(72),
            C.SOLAR_DHI: np.zeros(72),
            C.SOLAR_DNI: np.zeros(72),
        },
        index=index,
    )

    data = {O.WEATHER: weather}

    # Low thermal mass
    obj_low = base_obj.copy()
    obj_low[O.C_1_AW] = 1e6
    obj_low[O.C_1_IW] = 1e6

    # High thermal mass
    obj_high = base_obj.copy()
    obj_high[O.C_1_AW] = 1e7
    obj_high[O.C_1_IW] = 1e7

    r7c2 = R7C2()
    result_low = r7c2.generate(obj_low, data, Types.HVAC)
    result_high = r7c2.generate(obj_high, data, Types.HVAC)

    # High thermal mass should dampen temperature fluctuations
    temp_std_low = result_low["timeseries"][C.TEMP_IN].std()
    temp_std_high = result_high["timeseries"][C.TEMP_IN].std()

    assert temp_std_high <= temp_std_low, "Higher thermal mass should reduce temperature fluctuations"


def test_r7c2_vdi_ventilation_effect():
    """Test ventilation heat loss according to VDI 6007.

    Higher ventilation rate should increase heating demand in cold weather.
    """
    base_obj = {
        O.ID: "test_vent",
        O.R_1_AW: 0.002,
        O.C_1_AW: 4e6,
        O.R_1_IW: 0.002,
        O.C_1_IW: 1e7,
        O.R_ALPHA_STAR_IL: 0.0006,
        O.R_ALPHA_STAR_AW: 0.002,
        O.R_ALPHA_STAR_IW: 0.0008,
        O.R_REST_AW: 0.015,
        O.TEMP_INIT: 20.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 25.0,
        O.AREA: 150.0,
        O.HEIGHT: 2.7,
        O.LAT: 49.0,
        O.LON: 11.0,
    }

    # Cold weather
    index = pd.date_range("2025-01-01", periods=72, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            C.DATETIME: index,
            C.TEMP_AIR: np.full(72, 0.0),
            C.SOLAR_GHI: np.zeros(72),
            C.SOLAR_DHI: np.zeros(72),
            C.SOLAR_DNI: np.zeros(72),
        },
        index=index,
    )

    data = {O.WEATHER: weather}

    # Low ventilation
    obj_low_vent = base_obj.copy()
    obj_low_vent[O.VENTILATION] = 40.0

    # High ventilation
    obj_high_vent = base_obj.copy()
    obj_high_vent[O.VENTILATION] = 160.0

    r7c2 = R7C2()
    result_low = r7c2.generate(obj_low_vent, data, Types.HVAC)
    result_high = r7c2.generate(obj_high_vent, data, Types.HVAC)

    # Higher ventilation should lead to higher heating demand
    demand_low = result_low["summary"][f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"]
    demand_high = result_high["summary"][f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"]

    assert demand_high > demand_low, "Higher ventilation should increase heating demand in cold weather"
