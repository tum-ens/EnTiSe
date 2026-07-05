import logging

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


# --- power_heating / power_cooling as pd.Series (issue #100) -----------------
# Users need to define heating- and cooling-off periods by passing a time
# series for P_max instead of a scalar. Scalar behavior must be preserved
# bit-for-bit. Series inputs must be aligned to the weather index; a
# mismatched index must raise. See issue #100.


def test_r1c1_power_heating_scalar_unchanged():
    """Regression: passing power_heating as a scalar float produces the
    same output as before the timeseries change. Guards against accidental
    behavior drift when the resolve_ts_or_scalar path is added."""
    periods = 96
    T_out = -10.0
    T_min = 20.0
    R_val = 0.01
    C_val = 5e6
    P_max = 2000.0

    weather_key = "weather_scalar_regression"
    weather = _flat_weather(T_out, periods)
    obj = {
        O.ID: "b_scalar",
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

    # Steady-state indoor temp = T_out + R·P_max (the undersized-heater
    # scenario). Any drift here would mean the scalar path was silently
    # rewritten.
    assert ts[C.TEMP_IN].iloc[-1] == pytest.approx(T_out + R_val * P_max, abs=0.05)
    assert ts[f"{Types.HEATING}{SEP}{C.LOAD}[W]"].iloc[-1] == pytest.approx(P_max, rel=0.01)


def test_r1c1_power_heating_accepts_series_off_period_yields_zero_demand():
    """The core new feature: pass power_heating as a pd.Series with 0 during
    an 'off' period, finite during an 'on' period. Demand must be exactly
    zero during off, and the heater must fire during on."""
    periods = 96
    T_out = -10.0
    T_min = 20.0
    index = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")

    # Off for the first 48 h, then unlimited for the second 48 h.
    p_heat_series = pd.Series([0.0] * 48 + [float("inf")] * 48, index=index)

    weather_key = "weather_series_off_period"
    weather = _flat_weather(T_out, periods)
    weather.index = index  # align to the series
    weather[C.DATETIME] = index
    obj = {
        O.ID: "b_series_off",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.01,
        O.TEMP_INIT: T_min,
        O.TEMP_MIN: T_min,
        O.TEMP_MAX: 30.0,
        O.POWER_HEATING: p_heat_series,
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
    p_heat = ts[f"{Types.HEATING}{SEP}{C.LOAD}[W]"].to_numpy()

    # First 48 h: heater is forbidden.
    assert p_heat[:48].max() == 0
    # Second 48 h: heater must fire (T_out is well below T_min).
    assert p_heat[48:].max() > 0
    # Once heating is re-enabled, T_in returns to T_min at steady state.
    assert ts[C.TEMP_IN].iloc[-1] == pytest.approx(T_min, abs=0.1)


def test_r1c1_power_cooling_accepts_series_off_period_yields_zero_demand():
    """Mirror of the heating-series test: cooling off first, then on."""
    periods = 96
    T_out = 35.0
    T_max = 24.0
    index = pd.date_range("2025-06-01", periods=periods, freq="h", tz="UTC")

    p_cool_series = pd.Series([0.0] * 48 + [float("inf")] * 48, index=index)

    weather_key = "weather_cool_series_off_period"
    weather = _flat_weather(T_out, periods)
    weather.index = index
    weather[C.DATETIME] = index
    obj = {
        O.ID: "b_cool_series_off",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.01,
        O.TEMP_INIT: T_max,
        O.TEMP_MIN: 15.0,
        O.TEMP_MAX: T_max,
        O.POWER_COOLING: p_cool_series,
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
    p_cool = ts[f"{Types.COOLING}{SEP}{C.LOAD}[W]"].to_numpy()

    assert p_cool[:48].max() == 0
    assert p_cool[48:].max() > 0
    assert ts[C.TEMP_IN].iloc[-1] == pytest.approx(T_max, abs=0.1)


def test_r1c1_power_heating_series_index_mismatch_raises():
    """A power_heating series whose index does not match the weather index
    must fail loudly with a specific message rather than silently
    misaligning. Match the exact error string so a future refactor that
    changes which line raises does not accidentally pass this test with
    an unrelated KeyError."""
    periods = 24
    weather_key = "weather_mismatch"
    weather = _flat_weather(-5.0, periods)

    # Wrong length — half the weather index.
    bad_index = pd.date_range("2025-01-01", periods=periods // 2, freq="h", tz="UTC")
    bad_series = pd.Series(np.zeros(periods // 2), index=bad_index)

    obj = {
        O.ID: "b_mismatch",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.01,
        O.TEMP_INIT: 20.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 24.0,
        O.POWER_HEATING: bad_series,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 0.0,
        O.ACTIVE_COOLING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: weather_key,
    }
    with pytest.raises(ValueError, match="does not match the target index"):
        R1C1().generate(obj, {weather_key: weather}, Types.HVAC)


def test_r1c1_power_heating_series_clips_to_finite_cap():
    """When the series carries finite caps (not just 0/inf), the heater
    must run at exactly the cap when the required load exceeds it."""
    periods = 96
    T_out = -10.0
    T_min = 20.0
    R_val = 0.01
    C_val = 5e6
    P_cap = 2000.0  # required = (20-(-10))/0.01 = 3000, so we clip to 2000
    index = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")

    p_heat_series = pd.Series(np.full(periods, P_cap), index=index)

    weather_key = "weather_series_finite_cap"
    weather = _flat_weather(T_out, periods)
    weather.index = index
    weather[C.DATETIME] = index
    obj = {
        O.ID: "b_series_cap",
        O.CAPACITANCE: C_val,
        O.RESISTANCE: R_val,
        O.TEMP_INIT: T_min,
        O.TEMP_MIN: T_min,
        O.TEMP_MAX: 30.0,
        O.POWER_HEATING: p_heat_series,
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

    # Steady-state T_in = T_out + R·P_cap = 10 °C (below T_min).
    assert ts[C.TEMP_IN].iloc[-1] == pytest.approx(T_out + R_val * P_cap, abs=0.05)
    assert ts[f"{Types.HEATING}{SEP}{C.LOAD}[W]"].iloc[-1] == pytest.approx(P_cap, rel=0.01)
    assert ts[f"{Types.HEATING}{SEP}{C.LOAD}[W]"].max() <= P_cap + 1


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


# --- Latent cooling (issue #103) ---------------------------------------------
# Weather without humidity → the existing regression tests above already
# guarantee bit-exact output. These tests exercise the humidity-present path
# and the reinterpretation of `power_cooling[W]` as total nameplate capacity.


def _wet_weather(temp_c: float, rh: float, periods: int, p_pa: float = 101325.0) -> pd.DataFrame:
    """Weather fixture with humidity + pressure columns so the latent
    post-pass has everything it needs to compute a non-zero latent load."""
    index = pd.date_range("2025-06-01", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            C.DATETIME: index,
            C.TEMP_AIR: np.full(periods, temp_c, dtype=np.float64),
            C.HUMIDITY_REL: np.full(periods, rh, dtype=np.float64),
            C.SURFACE_AIR_PRESSURE: np.full(periods, p_pa, dtype=np.float64),
        },
        index=index,
    )


def test_r1c1_cooling_output_carries_sensible_and_latent_columns():
    """After #103 the timeseries has three cooling columns: total, sensible,
    latent. Their names are part of the API — assert they exist."""
    periods = 24
    weather_key = "weather_latent_cols"
    weather = _wet_weather(30.0, 0.7, periods)

    obj = {
        O.ID: "b_cols",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.01,
        O.TEMP_INIT: 24.0,
        O.TEMP_MIN: 15.0,
        O.TEMP_MAX: 24.0,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 100.0,
        O.ACTIVE_HEATING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: weather_key,
    }
    result = R1C1().generate(obj, {weather_key: weather}, Types.HVAC)
    ts = result["timeseries"]

    assert f"{Types.COOLING}{SEP}{C.LOAD}[W]" in ts.columns
    assert f"{Types.COOLING}{SEP}sensible_{C.LOAD}[W]" in ts.columns
    assert f"{Types.COOLING}{SEP}latent_{C.LOAD}[W]" in ts.columns


def test_r1c1_wet_summer_produces_positive_latent_load():
    """Hot humid outdoor + cool indoor setpoint → sensible AND latent must be
    positive in steady state. Sanity check that the post-pass fires."""
    periods = 96
    weather_key = "weather_wet_summer"
    weather = _wet_weather(t_out_c := 30.0, rh_out := 0.70, periods)

    obj = {
        O.ID: "b_wet",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.01,
        O.TEMP_INIT: 24.0,
        O.TEMP_MIN: 15.0,
        O.TEMP_MAX: 24.0,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 100.0,
        O.ACTIVE_HEATING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: weather_key,
    }
    result = R1C1().generate(obj, {weather_key: weather}, Types.HVAC)
    ts = result["timeseries"]

    sens_col = f"{Types.COOLING}{SEP}sensible_{C.LOAD}[W]"
    lat_col = f"{Types.COOLING}{SEP}latent_{C.LOAD}[W]"
    total_col = f"{Types.COOLING}{SEP}{C.LOAD}[W]"

    # Steady state: sensible > 0, latent > 0, total == sensible + latent.
    assert ts[sens_col].iloc[-1] > 0
    assert ts[lat_col].iloc[-1] > 0
    assert ts[total_col].iloc[-1] == pytest.approx(ts[sens_col].iloc[-1] + ts[lat_col].iloc[-1], abs=1)
    # Latent should be non-trivial for these ambient conditions
    # (30 °C 70 % RH → wet-bulb ~25 °C, moisture removal is real).
    assert ts[lat_col].max() > 50
    _ = t_out_c, rh_out  # silence-unused hint


def test_r1c1_dry_summer_produces_zero_latent_load():
    """Hot but dry outdoor → sensible > 0, latent = 0. Confirms the
    dry-outdoor sign-flip clip works end-to-end."""
    periods = 96
    weather_key = "weather_dry_summer"
    # 40 °C 10 % RH: outdoor ω ~ 0.005 kg/kg, indoor target at 24 °C 50 % RH
    # is ~0.0094 kg/kg. Ventilation moisture load is negative → latent 0.
    weather = _wet_weather(40.0, 0.10, periods)

    obj = {
        O.ID: "b_dry",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.01,
        O.TEMP_INIT: 24.0,
        O.TEMP_MIN: 15.0,
        O.TEMP_MAX: 24.0,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 100.0,
        O.ACTIVE_HEATING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: weather_key,
    }
    result = R1C1().generate(obj, {weather_key: weather}, Types.HVAC)
    ts = result["timeseries"]

    sens_col = f"{Types.COOLING}{SEP}sensible_{C.LOAD}[W]"
    lat_col = f"{Types.COOLING}{SEP}latent_{C.LOAD}[W]"
    assert ts[sens_col].iloc[-1] > 0
    assert ts[lat_col].max() == 0


def test_r1c1_active_cooling_false_forces_zero_latent():
    """When `active_cooling` is False, there is no cooling at all — sensible
    and latent must both be zero even on a hot, humid day."""
    periods = 48
    weather_key = "weather_cooling_off"
    weather = _wet_weather(35.0, 0.80, periods)

    obj = {
        O.ID: "b_cool_off",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.01,
        O.TEMP_INIT: 24.0,
        O.TEMP_MIN: 15.0,
        O.TEMP_MAX: 24.0,
        O.ACTIVE_COOLING: False,
        O.ACTIVE_HEATING: False,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 100.0,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: weather_key,
    }
    result = R1C1().generate(obj, {weather_key: weather}, Types.HVAC)
    ts = result["timeseries"]

    lat_col = f"{Types.COOLING}{SEP}latent_{C.LOAD}[W]"
    sens_col = f"{Types.COOLING}{SEP}sensible_{C.LOAD}[W]"
    assert ts[lat_col].max() == 0
    assert ts[sens_col].max() == 0


def test_r1c1_power_cooling_caps_total_load_sensible_priority():
    """`power_cooling[W]` is now the TOTAL nameplate cap. When the physics
    would demand more than the cap, sensible is served first (thermostat
    priority) and latent is clipped to the remainder. Steady-state sensible
    must equal the cap; latent must be zero if sensible alone hits the cap."""
    periods = 96
    weather_key = "weather_undersized_total"
    weather = _wet_weather(35.0, 0.80, periods)

    # (T_out - T_max)/R = (35-24)/0.01 = 1100 W sensible steady-state.
    # Cap set below that → sensible is clipped, latent should be zero.
    P_cap = 800.0

    obj = {
        O.ID: "b_undersized_total",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.01,
        O.TEMP_INIT: 24.0,
        O.TEMP_MIN: 15.0,
        O.TEMP_MAX: 24.0,
        O.POWER_COOLING: P_cap,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 100.0,
        O.ACTIVE_HEATING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: weather_key,
    }
    result = R1C1().generate(obj, {weather_key: weather}, Types.HVAC)
    ts = result["timeseries"]

    sens_col = f"{Types.COOLING}{SEP}sensible_{C.LOAD}[W]"
    lat_col = f"{Types.COOLING}{SEP}latent_{C.LOAD}[W]"
    total_col = f"{Types.COOLING}{SEP}{C.LOAD}[W]"

    # Total load never exceeds the cap.
    assert ts[total_col].max() <= P_cap + 1
    # In steady state, sensible saturates at the cap; latent goes to zero.
    assert ts[sens_col].iloc[-1] == pytest.approx(P_cap, rel=0.02)
    assert ts[lat_col].iloc[-1] == 0


def test_r1c1_gains_internal_latent_adds_to_latent_load():
    """Positive `gains_internal_latent[W]` must add to the latent load
    additively on top of the ventilation-driven part."""
    periods = 96
    weather_key = "weather_int_latent"
    weather = _wet_weather(28.0, 0.55, periods)

    base_obj = {
        O.ID: "b_int_lat",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.01,
        O.TEMP_INIT: 24.0,
        O.TEMP_MIN: 15.0,
        O.TEMP_MAX: 24.0,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 100.0,
        O.ACTIVE_HEATING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: weather_key,
    }

    # First run: no internal latent gains.
    ts_base = R1C1().generate({**base_obj, O.ID: "b_lat_0"}, {weather_key: weather}, Types.HVAC)["timeseries"]
    # Second run: 300 W of internal latent gains.
    ts_gain = R1C1().generate(
        {**base_obj, O.ID: "b_lat_300", O.GAINS_INTERNAL_LATENT: 300.0},
        {weather_key: weather},
        Types.HVAC,
    )["timeseries"]

    lat_col = f"{Types.COOLING}{SEP}latent_{C.LOAD}[W]"
    # Steady-state latent should differ by exactly 300 W (± rounding).
    delta = ts_gain[lat_col].iloc[-1] - ts_base[lat_col].iloc[-1]
    assert delta == pytest.approx(300, abs=2)


def test_r1c1_cooling_load_equals_sensible_plus_latent_per_row():
    """The `cooling:load[W]` column must equal `cooling:sensible_load[W]` +
    `cooling:latent_load[W]` on every row. Regression against a rounding
    bug where independently-rounded sensible + latent could differ from
    the rounded total by ±1 W."""
    periods = 168  # a week of hourly steps → high chance of straddling .5
    weather_key = "weather_invariant"
    weather = _wet_weather(28.0, 0.55, periods)
    obj = {
        O.ID: "b_inv",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.01,
        O.TEMP_INIT: 24.0,
        O.TEMP_MIN: 15.0,
        O.TEMP_MAX: 24.0,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 100.0,
        O.ACTIVE_HEATING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: weather_key,
    }
    ts = R1C1().generate(obj, {weather_key: weather}, Types.HVAC)["timeseries"]

    total = ts[f"{Types.COOLING}{SEP}{C.LOAD}[W]"]
    sens = ts[f"{Types.COOLING}{SEP}sensible_{C.LOAD}[W]"]
    lat = ts[f"{Types.COOLING}{SEP}latent_{C.LOAD}[W]"]
    assert (total == sens + lat).all(), "cooling:load[W] must equal sensible + latent per-row"


def test_r1c1_negative_gains_internal_latent_acts_as_moisture_sink():
    """A negative `gains_internal_latent[W]` is a valid input — it models a
    separate moisture-removing appliance (e.g. a dedicated dehumidifier).
    Its magnitude offsets the positive ventilation moisture load, and if
    the offset drives net moisture below zero the AC's latent contribution
    is clipped to zero (the AC can't add moisture)."""
    periods = 96
    weather_key = "weather_neg_gains"
    weather = _wet_weather(28.0, 0.55, periods)
    base_obj = {
        O.ID: "b_neg_gains",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.01,
        O.TEMP_INIT: 24.0,
        O.TEMP_MIN: 15.0,
        O.TEMP_MAX: 24.0,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 100.0,
        O.ACTIVE_HEATING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: weather_key,
    }

    ts_no_sink = R1C1().generate({**base_obj, O.ID: "b_no_sink"}, {weather_key: weather}, Types.HVAC)["timeseries"]
    ts_with_sink = R1C1().generate(
        {**base_obj, O.ID: "b_sink", O.GAINS_INTERNAL_LATENT: -200.0},
        {weather_key: weather},
        Types.HVAC,
    )["timeseries"]

    lat_col = f"{Types.COOLING}{SEP}latent_{C.LOAD}[W]"
    # A -200 W moisture sink must reduce the AC's steady-state latent load
    # by 200 W (as long as the resulting value stays non-negative).
    delta = ts_no_sink[lat_col].iloc[-1] - ts_with_sink[lat_col].iloc[-1]
    assert delta == pytest.approx(200, abs=2), "moisture sink must offset ventilation-driven latent"
    # Latent still non-negative — the max(0, ...) floor holds.
    assert ts_with_sink[lat_col].min() >= 0


def test_r1c1_missing_humidity_warns_and_zeros_latent(caplog):
    """Weather without humidity/pressure → latent = 0 everywhere, one
    warning per missing column. Regression against silent zero-output."""
    periods = 24
    # Deliberately no humidity or pressure column.
    index = pd.date_range("2025-06-01", periods=periods, freq="h", tz="UTC")
    weather = pd.DataFrame({C.DATETIME: index, C.TEMP_AIR: np.full(periods, 30.0, dtype=np.float64)}, index=index)
    weather_key = "weather_no_humidity"

    obj = {
        O.ID: "b_no_h",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.01,
        O.TEMP_INIT: 24.0,
        O.TEMP_MIN: 15.0,
        O.TEMP_MAX: 24.0,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 100.0,
        O.ACTIVE_HEATING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: weather_key,
    }

    with caplog.at_level(logging.WARNING, logger="entise.methods.hvac._latent_cooling"):
        result = R1C1().generate(obj, {weather_key: weather}, Types.HVAC)

    ts = result["timeseries"]
    lat_col = f"{Types.COOLING}{SEP}latent_{C.LOAD}[W]"
    sens_col = f"{Types.COOLING}{SEP}sensible_{C.LOAD}[W]"
    total_col = f"{Types.COOLING}{SEP}{C.LOAD}[W]"

    assert ts[lat_col].max() == 0
    # Total should equal sensible (no latent contribution).
    assert (ts[total_col] == ts[sens_col]).all()
    # Warning emitted once, mentioning both missing columns.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a warning about missing humidity/pressure columns"


# --- Thermostat dead band / hysteresis (issue #102) --------------------------
# The controller switches heating/cooling on/off at exact T_min/T_max. Real
# thermostats cycle around a small band. Add a `deadband[K]` parameter that,
# when > 0, makes the controller: (a) only turn on at the setpoint edge as
# before, (b) once firing, aim for the opposite band edge (T_min+deadband for
# heating, T_max-deadband for cooling), (c) once T lands in the band from
# outside, stay off until the setpoint edge is crossed again.
#
# Default 0 must preserve current behavior bit-for-bit.


def test_r1c1_deadband_default_zero_matches_omitted():
    """Explicit deadband=0 produces the same output as omitting the parameter.
    This is the backward-compatibility guarantee for the whole feature."""
    periods = 96
    T_out = -10.0
    T_min = 20.0

    weather = _flat_weather(T_out, periods)
    obj_no_db = {
        O.ID: "b_no_db",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.01,
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
        O.WEATHER: "weather_no_db",
    }
    obj_db_zero = {**obj_no_db, O.ID: "b_db_zero", O.DEADBAND: 0.0, O.WEATHER: "weather_db_zero"}

    ts_no_db = R1C1().generate(obj_no_db, {"weather_no_db": weather}, Types.HVAC)["timeseries"]
    ts_db_zero = R1C1().generate(obj_db_zero, {"weather_db_zero": weather}, Types.HVAC)["timeseries"]

    # Bit-exact equality of every column.
    for col in ts_no_db.columns:
        assert (
            ts_no_db[col].to_numpy() == ts_db_zero[col].to_numpy()
        ).all(), f"deadband=0 diverged from omitted deadband on column {col}"


def test_r1c1_heating_with_deadband_targets_upper_band():
    """Continuous cold ambient with deadband=2 K: T_in must settle at T_min+2,
    not at T_min. Verifies the controller aims for the upper band edge while
    heating is on."""
    periods = 96
    T_out = -10.0
    T_min = 20.0
    deadband = 2.0

    weather = _flat_weather(T_out, periods)
    obj = {
        O.ID: "b_heat_db",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.01,
        O.TEMP_INIT: T_min,
        O.TEMP_MIN: T_min,
        O.TEMP_MAX: 30.0,
        O.DEADBAND: deadband,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 0.0,
        O.ACTIVE_COOLING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: "weather_heat_db",
    }
    ts = R1C1().generate(obj, {"weather_heat_db": weather}, Types.HVAC)["timeseries"]

    # Steady state sits on the upper band edge, not on T_min.
    assert ts[C.TEMP_IN].iloc[-1] == pytest.approx(T_min + deadband, abs=0.05)
    # Sanity: heater is firing (steady loss > 0).
    assert ts[f"{Types.HEATING}{SEP}{C.LOAD}[W]"].iloc[-1] > 0


def test_r1c1_cooling_with_deadband_targets_lower_band():
    """Mirror of the heating case: continuous hot ambient with deadband=2 K.
    T_in must settle at T_max−2 while cooling is on."""
    periods = 96
    T_out = 35.0
    T_max = 24.0
    deadband = 2.0

    weather = _flat_weather(T_out, periods)
    obj = {
        O.ID: "b_cool_db",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.01,
        O.TEMP_INIT: T_max,
        O.TEMP_MIN: 15.0,
        O.TEMP_MAX: T_max,
        O.DEADBAND: deadband,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 0.0,
        O.ACTIVE_HEATING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: "weather_cool_db",
    }
    ts = R1C1().generate(obj, {"weather_cool_db": weather}, Types.HVAC)["timeseries"]

    assert ts[C.TEMP_IN].iloc[-1] == pytest.approx(T_max - deadband, abs=0.05)
    assert ts[f"{Types.COOLING}{SEP}{C.LOAD}[W]"].iloc[-1] > 0


def test_r1c1_deadband_reduces_switching_count_on_transient():
    """A transient scenario known to cause per-step on/off switching without
    hysteresis must switch strictly fewer times with a non-zero dead band.

    The forcing: cool ambient just below T_min plus a square-wave internal
    gain that alternates zero (heating needed) and enough gain to push the
    passive T_next above T_min (heating not needed). We deliberately use
    Δt/τ ≫ 1 (τ ≈ 100 s, Δt = 3600 s) so each step reaches steady state
    within itself — that guarantees per-step on/off toggling without
    hysteresis. With deadband, once heating fires it aims for the upper
    band edge and stays on across the gain step (still in-band), so the
    switching count drops sharply."""
    periods = 48
    T_out = 19.0  # just below T_min → loss overcomes when gain is off
    T_min = 20.0
    index = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    # Alternating gain: 0 W (heating needed) vs. 1500 W (comfortably overshoots).
    gain = pd.DataFrame({"b_toggle": np.where(np.arange(periods) % 2 == 0, 0.0, 1500.0)}, index=index)

    weather = _flat_weather(T_out, periods)
    weather.index = index
    weather[C.DATETIME] = index

    def _build_obj(deadband, key_suffix):
        return {
            O.ID: "b_toggle",
            # Fast decay (τ ≈ 100 s) so each hourly step reaches steady state
            # and the controller sees crisp on/off transitions.
            O.CAPACITANCE: 1e5,
            O.RESISTANCE: 0.001,
            O.TEMP_INIT: T_min,
            O.TEMP_MIN: T_min,
            O.TEMP_MAX: 30.0,
            O.DEADBAND: deadband,
            O.LAT: 48.1,
            O.LON: 11.6,
            O.GAINS_INTERNAL: f"gains_{key_suffix}",
            O.GAINS_INTERNAL_COL: "b_toggle",
            O.VENTILATION: 0.0,
            O.ACTIVE_COOLING: False,
            O.ACTIVE_GAINS_SOLAR: False,
            O.WEATHER: f"weather_{key_suffix}",
        }

    def _switch_count(load):
        # Count 0↔non-zero transitions.
        active = load > 0
        return int(np.sum(active[1:] != active[:-1]))

    ts_no_db = R1C1().generate(
        _build_obj(0.0, "swno"),
        {"weather_swno": weather, "gains_swno": gain},
        Types.HVAC,
    )["timeseries"]
    ts_db = R1C1().generate(
        _build_obj(2.0, "swdb"),
        {"weather_swdb": weather, "gains_swdb": gain},
        Types.HVAC,
    )["timeseries"]

    p_no_db = ts_no_db[f"{Types.HEATING}{SEP}{C.LOAD}[W]"].to_numpy()
    p_db = ts_db[f"{Types.HEATING}{SEP}{C.LOAD}[W]"].to_numpy()

    # Sanity: the no-deadband case must actually switch (otherwise the test
    # doesn't prove anything).
    assert _switch_count(p_no_db) > 5
    # With the band, strictly fewer switches.
    assert _switch_count(p_db) < _switch_count(p_no_db)


def test_r1c1_deadband_stays_off_when_entering_band_from_above():
    """Start with T_in above T_min+deadband, gently drift down into the band.
    The state machine must stay in the "off" branch as long as the passive
    T_next remains above T_min — i.e., there must be at least one step where
    the recorded (after-HVAC) temperature lies inside the band but the
    heater did NOT fire. Once the passive T_next drops below T_min the
    heater fires; from then on the "on" branch runs different physics, so
    we assert only up to the first fire."""
    periods = 48
    T_min = 20.0
    deadband = 2.0
    # τ ≈ R·C = 3600 s so Δt/τ = 1: the room drifts about 63 % of the way
    # toward T_out per step. That lets T land inside the band on the first
    # step (still above T_min) and cross T_min on the second — exercising
    # the "off, entering band from above" branch before the state machine
    # flips to "on".
    T_out = T_min - 1.0

    weather = _flat_weather(T_out, periods)
    obj = {
        O.ID: "b_from_above",
        O.CAPACITANCE: 1e5,
        O.RESISTANCE: 0.036,
        O.TEMP_INIT: T_min + deadband + 0.5,  # start above the band
        O.TEMP_MIN: T_min,
        O.TEMP_MAX: 30.0,
        O.DEADBAND: deadband,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 0.0,
        O.ACTIVE_COOLING: False,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: "weather_from_above",
    }
    ts = R1C1().generate(obj, {"weather_from_above": weather}, Types.HVAC)["timeseries"]

    temp = ts[C.TEMP_IN].to_numpy()
    heat = ts[f"{Types.HEATING}{SEP}{C.LOAD}[W]"].to_numpy()

    # Locate the first step where the heater fired.
    fired = heat > 0
    assert fired.any(), "Test scenario never fires the heater — bad params."
    first_fire = int(np.argmax(fired))

    # Before that, at least one step must show T inside the band (proving
    # we did enter the band from above without firing).
    entered_band = (temp[:first_fire] >= T_min) & (temp[:first_fire] <= T_min + deadband + 1e-3)
    assert (
        entered_band.any()
    ), "T never crossed into the band before firing — cannot verify off-branch hysteresis with this scenario."
    # Every pre-fire step has heat == 0 by construction of `first_fire`.
    assert (heat[:first_fire] == 0).all()


def test_r1c1_wide_deadband_does_not_double_fire():
    """Regression for the wide-band mutex bug. When `deadband > (T_max-T_min)/2`
    the intervals `[T_min, T_min+deadband]` and `[T_max-deadband, T_max]`
    overlap. Without the extreme-guard both `heating_on` and `cooling_on`
    can stay latched at the same step, so heating and cooling would fire
    simultaneously on the same time step — physically nonsensical.

    Setup: fast decay (τ ≪ Δt) so T_pas nearly reaches T_out each step,
    and alternating T_out that crosses both setpoints hourly. Confirms
    that at no step do heating and cooling fire together."""
    periods = 48
    T_min = 20.0
    T_max = 24.0
    deadband = 10.0  # deliberately wider than T_max − T_min = 4
    index = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    # Alternating cold / hot ambient.
    temp_air = np.where(np.arange(periods) % 2 == 0, -20.0, 26.0)
    weather = pd.DataFrame({C.DATETIME: index, C.TEMP_AIR: temp_air}, index=index)

    obj = {
        O.ID: "b_wide_db",
        # Fast decay: τ ≈ 100 s → Δt/τ = 36; T settles near T_out per step.
        O.CAPACITANCE: 1e5,
        O.RESISTANCE: 0.001,
        O.TEMP_INIT: (T_min + T_max) / 2,
        O.TEMP_MIN: T_min,
        O.TEMP_MAX: T_max,
        O.DEADBAND: deadband,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 0,
        O.VENTILATION: 0.0,
        O.ACTIVE_GAINS_SOLAR: False,
        O.ACTIVE_GAINS_INTERNAL: False,
        O.WEATHER: "weather_wide_db",
    }
    ts = R1C1().generate(obj, {"weather_wide_db": weather}, Types.HVAC)["timeseries"]
    p_heat = ts[f"{Types.HEATING}{SEP}{C.LOAD}[W]"].to_numpy()
    p_cool = ts[f"{Types.COOLING}{SEP}{C.LOAD}[W]"].to_numpy()

    # The primary claim: no step ever fires both.
    both = (p_heat > 0) & (p_cool > 0)
    assert (
        not both.any()
    ), f"Heating and cooling fired simultaneously on {int(both.sum())} steps — wide-band mutex broken."
    # Sanity: at least *some* actuator fires during the run.
    assert (p_heat > 0).any() or (p_cool > 0).any()
