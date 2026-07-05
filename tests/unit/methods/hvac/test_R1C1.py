import numpy as np
import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.methods.hvac.R1C1 import R1C1, _calculate_timeseries_numpy
from entise.perf import set_accelerator


# Every test in this file runs under both accelerator paths. `numba` is skipped
# when numba is not installed. Autouse so no existing test has to opt in.
@pytest.fixture(autouse=True, params=["none", "numba"])
def accelerator(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    set_accelerator(request.param)
    yield request.param
    set_accelerator("auto")


@pytest.fixture
def dummy_inputs():
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    weather = pd.DataFrame(
        {
            C.DATETIME: index,
            C.TEMP_AIR: np.full(24, 0.0),
            C.SOLAR_GHI: np.full(24, 100.0),
            C.SOLAR_DHI: np.full(24, 20.0),
            C.SOLAR_DNI: np.full(24, 80.0),
        },
        index=index,
    )

    windows = pd.DataFrame(
        [{O.ID: "obj1", C.AREA: 10.0, C.G_VALUE: 0.7, C.SHADING: 1.0, C.TILT: 90.0, C.ORIENTATION: 180.0}]
    )

    internal_gains = pd.DataFrame({"obj1": np.arange(24)}, index=pd.date_range("2025-01-01", periods=24, freq="h"))

    obj = {
        O.ID: "obj1",
        O.CAPACITANCE: 1e5,
        O.RESISTANCE: 2.0,
        O.TEMP_INIT: 20.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 24.0,
        O.POWER_HEATING: 3000.0,
        O.POWER_COOLING: 3000.0,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL_COL: "obj1",  # Specifies the column in the timeseries
        O.GAINS_INTERNAL: "internal_gains",  # Points to timeseries in `data`
        O.WEATHER: f"{O.WEATHER}_dummy",
    }

    data = {f"{O.WEATHER}_dummy": weather, O.WINDOWS: windows, "internal_gains": internal_gains}

    return obj, data


def test_r1c1_outputs(dummy_inputs):
    obj, data = dummy_inputs
    r1c1 = R1C1()
    result = r1c1.generate(obj, data, Types.HVAC)

    assert "timeseries" in result
    ts = result["timeseries"]
    assert all(
        col in ts.columns for col in [C.TEMP_IN, f"{Types.HEATING}{SEP}{C.LOAD}[W]", f"{Types.COOLING}{SEP}{C.LOAD}[W]"]
    )
    assert len(ts) == 24


@pytest.fixture
def minimal_weather():
    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            C.DATETIME: index,
            C.TEMP_AIR: np.zeros(24),
            C.SOLAR_GHI: np.full(24, 100.0),
            C.SOLAR_DHI: np.full(24, 20.0),
            C.SOLAR_DNI: np.full(24, 80.0),
        },
        index=index,
    )


@pytest.fixture
def dummy_windows():
    return pd.DataFrame(
        [{O.ID: "obj1", "area": 10.0, "transmittance": 0.7, "shading": 1.0, O.TILT: 90.0, O.ORIENTATION: 180.0}]
    )


def test_r1c1_constant_internal_gains(minimal_weather, dummy_windows):
    obj = {
        O.CAPACITANCE: 1e5,
        O.RESISTANCE: 2.0,
        O.TEMP_INIT: 20.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 24.0,
        O.POWER_HEATING: 3000.0,
        O.POWER_COOLING: 3000.0,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 100,  # Numeric → triggers InternalConstant
    }
    data = {
        O.WEATHER: minimal_weather,
        O.WINDOWS: dummy_windows,
    }

    r1c1 = R1C1()
    result = r1c1.generate(obj, data, Types.HVAC)

    assert "summary" in result
    assert "timeseries" in result
    assert isinstance(result["timeseries"], pd.DataFrame)
    assert C.TEMP_IN in result["timeseries"]


# --- Regression + solver-correctness tests -----------------------------------
# These target the bug where explicit Euler becomes unstable when Δt/τ > 2.
# See issue "1R1C net_transfer produces unphysical temperature overshoot and
# oscillation at Δt/τ > 2" (Stengel, TH Rosenheim, BauSIM 2026 evaluation).
#
# Each test uses a unique weather-key name to bypass the module-level
# _WEATHER_CACHE in R1C1.py, which would otherwise cross-contaminate tests
# that share the default O.WEATHER key.


def _flat_weather(temp_c: float, periods: int) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {C.DATETIME: index, C.TEMP_AIR: np.full(periods, temp_c, dtype=np.float64)},
        index=index,
    )


def test_r1c1_no_spurious_demand_at_steady_state():
    """T_out constant at 24 °C, low-R envelope: T_in must settle at 24 °C
    without either heating or cooling firing.

    This is Martin Stengel's reduced reproducer. With explicit Euler and
    Δt/τ ≈ 4 the current code overshoots to ~36 °C in one step, triggers
    cooling, then oscillates between T_min and T_max forever."""
    weather_key = "weather_regression_steady"
    weather = _flat_weather(24.0, 24 * 7)
    obj = {
        O.ID: "b_low_r",
        O.CAPACITANCE: 593083.5840,
        O.RESISTANCE: 0.0015,
        O.TEMP_INIT: 20.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 28.0,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 0.0,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: weather_key,
    }

    result = R1C1().generate(obj, {weather_key: weather}, Types.HVAC)
    ts = result["timeseries"]
    summary = result["summary"]

    # After warmup, indoor temperature must sit at outdoor temperature.
    assert ts[C.TEMP_IN].iloc[-1] == pytest.approx(24.0, abs=1e-2)
    # And must never exceed the outdoor temperature via passive transfer.
    assert ts[C.TEMP_IN].max() <= 24.0 + 1e-3
    # No demand at all in this scenario.
    assert summary[f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"] == 0
    assert summary[f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]"] == 0


def test_r1c1_passive_decay_matches_analytical():
    """With controllers disabled, ventilation zeroed and no gains, indoor
    temperature must follow the closed-form exponential decay
        T(t) = T_out + (T0 - T_out) · exp(-t/τ)   with τ = R·C
    to within a small tolerance. Δt/τ ≈ 4 here — this is the regime where
    explicit Euler overshoots and oscillates, so any implementation that
    passes this test cannot be the current one."""
    periods = 24
    T_out = 24.0
    T0 = 20.0
    R_val = 0.0015
    C_val = 593083.5840
    dt = 3600.0
    tau = R_val * C_val  # ≈ 890 s → Δt/τ ≈ 4.04

    weather_key = "weather_passive_decay"
    weather = _flat_weather(T_out, periods)
    obj = {
        O.ID: "b_passive",
        O.CAPACITANCE: C_val,
        O.RESISTANCE: R_val,
        O.TEMP_INIT: T0,
        # Setpoints far outside the analytical range so controllers stay idle.
        O.TEMP_MIN: -100.0,
        O.TEMP_MAX: 100.0,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 0.0,
        O.ACTIVE_HEATING: False,
        O.ACTIVE_COOLING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: weather_key,
    }

    result = R1C1().generate(obj, {weather_key: weather}, Types.HVAC)
    temp_in = result["timeseries"][C.TEMP_IN].to_numpy()

    t_seconds = np.arange(periods) * dt
    T_analytic = T_out + (T0 - T_out) * np.exp(-t_seconds / tau)

    # 0.1 K tolerance — captures float32 round-off, rejects any oscillation.
    assert np.abs(temp_in - T_analytic).max() < 0.1
    # T_in must monotonically approach T_out from below (no overshoot).
    assert temp_in.max() <= T_out + 1e-3


def test_r1c1_heating_holds_setpoint_in_steady_state():
    """Cold ambient with active heating: T_in must sit at T_min once
    settled, and the steady-state heating load must equal the steady-state
    loss to ambient (T_min − T_out)/R. Verifies controller inversion."""
    periods = 96  # 4 days at hourly resolution
    T_out = -10.0
    T_min = 20.0
    R_val = 0.01
    C_val = 5e6

    weather_key = "weather_heating_setpoint"
    weather = _flat_weather(T_out, periods)
    obj = {
        O.ID: "b_heating",
        O.CAPACITANCE: C_val,
        O.RESISTANCE: R_val,
        O.TEMP_INIT: T_min,
        O.TEMP_MIN: T_min,
        O.TEMP_MAX: 30.0,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 0.0,
        O.ACTIVE_COOLING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: weather_key,
    }

    result = R1C1().generate(obj, {weather_key: weather}, Types.HVAC)
    ts = result["timeseries"]

    # After settling, indoor temp sits at T_min.
    assert ts[C.TEMP_IN].iloc[-1] == pytest.approx(T_min, abs=0.05)
    # Steady-state heating load = (T_min - T_out) / R  (ventilation disabled).
    expected_load = (T_min - T_out) / R_val
    actual_load = ts[f"{Types.HEATING}{SEP}{C.LOAD}[W]"].iloc[-1]
    assert actual_load == pytest.approx(expected_load, rel=0.02)
    # Cooling never fires.
    assert ts[f"{Types.COOLING}{SEP}{C.LOAD}[W]"].max() == 0


def test_r1c1_cooling_holds_setpoint_in_steady_state():
    """Mirror of the heating test: hot ambient with active cooling. T_in
    must settle at T_max, cooling load = (T_out − T_max) / R, heating idle."""
    periods = 96
    T_out = 35.0
    T_max = 24.0
    R_val = 0.01
    C_val = 5e6

    weather_key = "weather_cooling_setpoint"
    weather = _flat_weather(T_out, periods)
    obj = {
        O.ID: "b_cooling",
        O.CAPACITANCE: C_val,
        O.RESISTANCE: R_val,
        O.TEMP_INIT: T_max,
        O.TEMP_MIN: 15.0,
        O.TEMP_MAX: T_max,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 0.0,
        O.ACTIVE_HEATING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: weather_key,
    }

    result = R1C1().generate(obj, {weather_key: weather}, Types.HVAC)
    ts = result["timeseries"]

    assert ts[C.TEMP_IN].iloc[-1] == pytest.approx(T_max, abs=0.05)
    expected_load = (T_out - T_max) / R_val
    actual_load = ts[f"{Types.COOLING}{SEP}{C.LOAD}[W]"].iloc[-1]
    assert actual_load == pytest.approx(expected_load, rel=0.02)
    assert ts[f"{Types.HEATING}{SEP}{C.LOAD}[W]"].max() == 0


def test_r1c1_undersized_heater_clips_and_underheats():
    """When P_heat_max is smaller than the steady-state heat loss, the
    heater must run at exactly P_heat_max and T_in must settle below T_min
    at T_out + R·P_heat_max (steady-state energy balance)."""
    periods = 96
    T_out = -10.0
    T_min = 20.0
    R_val = 0.01
    C_val = 5e6
    P_max = 2000.0  # required load is (20-(-10))/0.01 = 3000 W, so we clip

    weather_key = "weather_undersized_heater"
    weather = _flat_weather(T_out, periods)
    obj = {
        O.ID: "b_undersized",
        O.CAPACITANCE: C_val,
        O.RESISTANCE: R_val,
        O.TEMP_INIT: T_min,
        O.TEMP_MIN: T_min,
        O.TEMP_MAX: 30.0,
        O.POWER_HEATING: P_max,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 0.0,
        O.ACTIVE_COOLING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: weather_key,
    }

    result = R1C1().generate(obj, {weather_key: weather}, Types.HVAC)
    ts = result["timeseries"]

    # After settling, indoor temp floats at T_out + R·P_max.
    expected_temp = T_out + R_val * P_max
    assert ts[C.TEMP_IN].iloc[-1] == pytest.approx(expected_temp, abs=0.05)
    assert ts[C.TEMP_IN].iloc[-1] < T_min
    # Heater runs at exactly P_max, never above.
    assert ts[f"{Types.HEATING}{SEP}{C.LOAD}[W]"].iloc[-1] == pytest.approx(P_max, rel=0.01)
    assert ts[f"{Types.HEATING}{SEP}{C.LOAD}[W]"].max() <= P_max + 1


def test_r1c1_internal_gains_trigger_cooling_at_mild_ambient():
    """Cool outdoor with constant internal gains that would push T above
    T_max in steady state: cooling must fire and land T_in on T_max."""
    periods = 96
    T_out = 18.0
    T_max = 24.0
    R_val = 0.01
    C_val = 5e6
    gain_w = 1200.0  # steady-state passive T would be T_out + R·gain = 30 °C

    weather_key = "weather_gains_cooling"
    weather = _flat_weather(T_out, periods)
    obj = {
        O.ID: "b_gains",
        O.CAPACITANCE: C_val,
        O.RESISTANCE: R_val,
        O.TEMP_INIT: T_max,
        O.TEMP_MIN: 15.0,
        O.TEMP_MAX: T_max,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: gain_w,  # constant scalar → InternalConstant strategy
        O.VENTILATION: 0.0,
        O.ACTIVE_HEATING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.WEATHER: weather_key,
    }

    result = R1C1().generate(obj, {weather_key: weather}, Types.HVAC)
    ts = result["timeseries"]

    assert ts[C.TEMP_IN].iloc[-1] == pytest.approx(T_max, abs=0.05)
    # Steady-state cooling = gain − (T_max − T_out)/R = 1200 − 600 = 600 W.
    expected_load = gain_w - (T_max - T_out) / R_val
    actual_load = ts[f"{Types.COOLING}{SEP}{C.LOAD}[W]"].iloc[-1]
    assert actual_load == pytest.approx(expected_load, rel=0.02)


def test_r1c1_tracks_sine_ambient_at_low_R_without_oscillation():
    """The Δt/τ > 2 regime from Martin's report but with a diurnal sine on
    T_out. Verify T_in tracks smoothly (no runaway) and demand stays
    bounded — regression against explicit-Euler instability under
    time-varying forcing."""
    periods = 24 * 3  # 3 days
    T_mean = 22.0
    T_amp = 6.0
    index = pd.date_range("2025-06-01", periods=periods, freq="h", tz="UTC")
    hours = np.arange(periods)
    temp_air = T_mean + T_amp * np.sin(2 * np.pi * hours / 24 - np.pi / 2)
    weather = pd.DataFrame({C.DATETIME: index, C.TEMP_AIR: temp_air}, index=index)

    weather_key = "weather_sine"
    obj = {
        O.ID: "b_sine",
        O.CAPACITANCE: 593083.5840,
        O.RESISTANCE: 0.0015,  # Δt/τ ≈ 4 — the pathological regime
        O.TEMP_INIT: T_mean,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 28.0,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 0.0,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: weather_key,
    }

    result = R1C1().generate(obj, {weather_key: weather}, Types.HVAC)
    ts = result["timeseries"]
    temp_in = ts[C.TEMP_IN].to_numpy()

    # T_in stays inside the T_out envelope + small setpoint margin.
    assert temp_in.min() >= min(T_mean - T_amp, 20.0) - 0.5
    assert temp_in.max() <= max(T_mean + T_amp, 28.0) + 0.5
    # No step-to-step swing exceeding the ambient swing → no oscillation.
    max_step = np.abs(np.diff(temp_in)).max()
    assert max_step < 2 * T_amp


# --- Accelerator path agreement ---------------------------------------------
# Direct comparison of the numpy and numba solvers on the same inputs, to
# catch drift between the two implementations. Runs once (not parametrized).


def test_numpy_and_numba_paths_agree():
    """The numba solver must reproduce the numpy solver's output within a
    few ULPs. Any daylight between them is a numba-path bug."""
    numba = pytest.importorskip("numba")  # noqa: F841
    from entise.methods.auxiliary.internal.selector import InternalGains
    from entise.methods.auxiliary.solar.selector import SolarGains
    from entise.methods.auxiliary.ventilation.selector import Ventilation
    from entise.methods.hvac._R1C1_numba import calculate_timeseries_1r1c as _numba
    from entise.methods.hvac.defaults import (
        DEFAULT_ACTIVE_COOLING,
        DEFAULT_ACTIVE_HEATING,
        DEFAULT_POWER_COOLING,
        DEFAULT_POWER_HEATING,
    )

    periods = 168
    index = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    T_out = 5 + 10 * np.sin(2 * np.pi * np.arange(periods) / 24.0)
    weather = pd.DataFrame({C.DATETIME: index, C.TEMP_AIR: T_out}, index=index)

    obj = {
        O.ID: "b",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.005,
        O.TEMP_INIT: 20.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 25.0,
        O.POWER_HEATING: DEFAULT_POWER_HEATING,
        O.POWER_COOLING: DEFAULT_POWER_COOLING,
        O.ACTIVE_HEATING: DEFAULT_ACTIVE_HEATING,
        O.ACTIVE_COOLING: DEFAULT_ACTIVE_COOLING,
        O.GAINS_INTERNAL: 100.0,
        O.VENTILATION: 50.0,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.WEATHER: "w_agree",
    }
    data = {"w_agree": weather}
    data[O.WEATHER] = weather
    data[O.GAINS_INTERNAL] = InternalGains().generate(obj, data)
    data[O.GAINS_SOLAR] = SolarGains().generate(obj, data)
    data[O.VENTILATION] = Ventilation().generate(obj, data)

    dt = 3600.0
    t_np, ph_np, pc_np = _calculate_timeseries_numpy(obj, data, dt)
    t_nb, ph_nb, pc_nb = _numba(obj, data, dt)

    # Tolerances chosen tight enough to catch a bug but loose enough to
    # tolerate float32 evaluation-order differences between the two paths.
    assert np.max(np.abs(t_np - t_nb)) < 1e-3, "temperature drift between paths"
    assert np.max(np.abs(ph_np - ph_nb)) < 0.5, "heating power drift between paths"
    assert np.max(np.abs(pc_np - pc_nb)) < 0.5, "cooling power drift between paths"
