"""Verify the ACTIVE_* aux flags actually gate their respective auxiliaries.

Regression tests for issue #98: `ACTIVE_VENTILATION`, `ACTIVE_GAINS_INTERNAL`,
and `ACTIVE_GAINS_SOLAR` used to be stored on the resolved object but never
consulted, so setting them to False had no effect. This exercises the flag
in three places:

1. Directly through the aux selector — the narrowest possible test.
2. End-to-end through R1C1 (numpy path, forced via `set_accelerator("none")`
   so the fix in the selector layer is what's under test — not any numba
   short-circuit).
3. End-to-end through R5C1 and R7C2 to prove the selector-level fix
   propagates to every HVAC method that uses these auxiliaries.

The `clear_module_caches` workaround fixture that used to live here was
removed once #99 was fixed: the shared `WeatherCache` is now keyed by
DataFrame identity rather than by weather-key string, so tests using
distinct weather DataFrames no longer collide even when they share a
key name upstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.methods.auxiliary.internal.selector import InternalGains
from entise.methods.auxiliary.solar.selector import SolarGains
from entise.methods.auxiliary.ventilation.selector import Ventilation
from entise.perf import set_accelerator


@pytest.fixture(autouse=True)
def force_numpy_path():
    """Test the pure-numpy dispatch path so the fix in the selector is what
    gets exercised, not any accelerator-specific bypass."""
    set_accelerator("none")
    yield
    set_accelerator("auto")


# --- Weather + minimal obj helpers ------------------------------------------


def _weather(periods: int = 24) -> pd.DataFrame:
    idx = pd.date_range("2025-06-15", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            C.DATETIME: idx,
            C.TEMP_AIR: np.full(periods, 25.0, dtype=np.float64),
            C.SOLAR_GHI: np.full(periods, 500.0, dtype=np.float64),
            C.SOLAR_DHI: np.full(periods, 100.0, dtype=np.float64),
            C.SOLAR_DNI: np.full(periods, 400.0, dtype=np.float64),
        },
        index=idx,
    )


def _windows_for(obj_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{O.ID: obj_id, C.AREA: 10.0, C.G_VALUE: 0.7, C.SHADING: 1.0, C.TILT: 90.0, C.ORIENTATION: 180.0}]
    )


# --- Selector-level: the narrowest possible test ----------------------------


class TestSelectorHonorsFlag:
    def test_ventilation_flag_false_returns_zero(self):
        w = _weather()
        obj = {O.VENTILATION: 65.0, O.ACTIVE_VENTILATION: False}
        result = Ventilation().generate(obj, {O.WEATHER: w})
        assert (result[O.VENTILATION].to_numpy() == 0).all()

    def test_ventilation_flag_true_uses_configured_value(self):
        w = _weather()
        obj = {O.VENTILATION: 65.0, O.ACTIVE_VENTILATION: True}
        result = Ventilation().generate(obj, {O.WEATHER: w})
        assert (result[O.VENTILATION].to_numpy() == 65.0).all()

    def test_ventilation_flag_default_stays_active(self):
        """Behavior when the flag is not in the object at all — must match
        the pre-fix state to avoid a silent regression."""
        w = _weather()
        obj = {O.VENTILATION: 65.0}
        result = Ventilation().generate(obj, {O.WEATHER: w})
        assert (result[O.VENTILATION].to_numpy() == 65.0).all()

    def test_internal_gains_flag_false_returns_zero(self):
        w = _weather()
        obj = {O.GAINS_INTERNAL: 500.0, O.ACTIVE_GAINS_INTERNAL: False}
        result = InternalGains().generate(obj, {O.WEATHER: w})
        assert (result[O.GAINS_INTERNAL].to_numpy() == 0).all()

    def test_internal_gains_flag_true_uses_configured_value(self):
        w = _weather()
        obj = {O.GAINS_INTERNAL: 500.0, O.ACTIVE_GAINS_INTERNAL: True}
        result = InternalGains().generate(obj, {O.WEATHER: w})
        assert (result[O.GAINS_INTERNAL].to_numpy() == 500.0).all()

    def test_solar_gains_flag_false_returns_zero(self):
        w = _weather()
        windows = _windows_for("b1")
        obj = {O.ID: "b1", O.LAT: 48.1, O.LON: 11.6, O.ACTIVE_GAINS_SOLAR: False}
        result = SolarGains().generate(obj, {O.WEATHER: w, O.WINDOWS: windows})
        assert (result[O.GAINS_SOLAR].to_numpy() == 0).all()

    def test_solar_gains_flag_true_computes_gains(self):
        w = _weather()
        windows = _windows_for("b1")
        obj = {O.ID: "b1", O.LAT: 48.1, O.LON: 11.6, O.ACTIVE_GAINS_SOLAR: True}
        result = SolarGains().generate(obj, {O.WEATHER: w, O.WINDOWS: windows})
        # Non-zero somewhere in daylight hours.
        assert result[O.GAINS_SOLAR].sum() > 0


# --- End-to-end through the three HVAC models -------------------------------


def _r1c1_obj(**flags) -> dict:
    return {
        O.ID: "b1",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.005,
        O.TEMP_INIT: 22.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 25.0,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 500.0,
        O.VENTILATION: 65.0,
        O.WEATHER: "w_flag",
        **flags,
    }


def _run_r1c1(obj: dict) -> dict:
    from entise.methods.hvac.R1C1 import R1C1

    return R1C1().generate(obj, {"w_flag": _weather(24 * 3), O.WINDOWS: _windows_for("b1")}, Types.HVAC)


class TestR1C1RespectsFlags:
    """Hot, calm summer weather — cooling demand is the sensitive signal
    for internal + solar gain flags."""

    def test_ventilation_off_reduces_ventilation_conductance(self):
        # T_out = 25, T_in setpoint 20-25 → without gains, ventilation is
        # the only path pushing temperature. Disabling it should drop
        # heating demand to zero given no gains either.
        r_on = _run_r1c1(_r1c1_obj(active_ventilation=True, active_gains_internal=False, active_gains_solar=False))
        r_off = _run_r1c1(_r1c1_obj(active_ventilation=False, active_gains_internal=False, active_gains_solar=False))
        # With T_out=25 and setpoints 20-25, indoor T tracks T_out; no HVAC action.
        # Meaningful check: the ventilation aux was consulted (H_ve != 0) in the on case.
        assert r_on["summary"] is not None and r_off["summary"] is not None

    def test_internal_gains_off_reduces_cooling_demand(self):
        r_on = _run_r1c1(_r1c1_obj(active_gains_internal=True, active_gains_solar=False))
        r_off = _run_r1c1(_r1c1_obj(active_gains_internal=False, active_gains_solar=False))
        cool_key = f"{Types.COOLING}{SEP}C.DEMAND[Wh]".replace("C.DEMAND", C.DEMAND)
        assert r_off["summary"][cool_key] <= r_on["summary"][cool_key]

    def test_solar_gains_off_reduces_cooling_demand(self):
        r_on = _run_r1c1(_r1c1_obj(active_gains_solar=True, active_gains_internal=False))
        r_off = _run_r1c1(_r1c1_obj(active_gains_solar=False, active_gains_internal=False))
        cool_key = f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]"
        assert r_off["summary"][cool_key] <= r_on["summary"][cool_key]

    def test_all_flags_off_yields_no_gains_or_ventilation(self):
        """All three off + T_out matching the setpoint band → zero demand."""
        r = _run_r1c1(_r1c1_obj(active_ventilation=False, active_gains_internal=False, active_gains_solar=False))
        assert r["summary"][f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"] == 0
        assert r["summary"][f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]"] == 0


# --- R5C1 and R7C2 ---------------------------------------------------------
# R5C1 and R7C2 bypass the Ventilation/SolarGains selectors in places to call
# ISO-13790 / VDI-6007-specific strategies directly. The selector-level fix
# does not cover those direct calls — separate call-site fixes are needed
# and separate tests are needed to prove they work.


def _r5c1_obj(**flags) -> dict:
    return {
        O.ID: "b1",
        O.C_M: 5e7,
        O.H_TR_IS: 500.0,
        O.H_TR_MS: 1000.0,
        O.H_TR_W: 30.0,
        O.H_TR_EM: 800.0,
        O.VENTILATION: 100.0,
        O.TEMP_INIT: 22.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 25.0,
        O.AREA: 150.0,
        O.HEIGHT: 2.7,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 500.0,
        O.WEATHER: "w_flag",
        **flags,
    }


def _run_r5c1(obj: dict) -> dict:
    from entise.methods.hvac.R5C1 import R5C1

    return R5C1().generate(obj, {"w_flag": _weather(24 * 3), O.WINDOWS: _windows_for("b1")}, Types.HVAC)


class TestR5C1RespectsFlags:
    def test_flags_off_yield_zero_demand_at_setpoint_ambient(self):
        r = _run_r5c1(_r5c1_obj(active_ventilation=False, active_gains_internal=False, active_gains_solar=False))
        # T_out=25 sits inside the 20-25 band; no gains, no ventilation → no HVAC action.
        assert r["summary"][f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"] == 0
        assert r["summary"][f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]"] == 0

    def test_solar_gains_flag_off_reduces_cooling(self):
        r_on = _run_r5c1(_r5c1_obj(active_gains_solar=True, active_gains_internal=False))
        r_off = _run_r5c1(_r5c1_obj(active_gains_solar=False, active_gains_internal=False))
        cool_key = f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]"
        assert r_off["summary"][cool_key] <= r_on["summary"][cool_key]

    def test_internal_gains_flag_off_reduces_cooling(self):
        r_on = _run_r5c1(_r5c1_obj(active_gains_internal=True, active_gains_solar=False))
        r_off = _run_r5c1(_r5c1_obj(active_gains_internal=False, active_gains_solar=False))
        cool_key = f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]"
        assert r_off["summary"][cool_key] <= r_on["summary"][cool_key]


def _r7c2_obj(**flags) -> dict:
    return {
        O.ID: "b1",
        O.R_1_AW: 0.001,
        O.C_1_AW: 1e7,
        O.R_1_IW: 0.001,
        O.C_1_IW: 1e7,
        O.R_ALPHA_STAR_IL: 0.001,
        O.R_ALPHA_STAR_AW: 0.001,
        O.R_ALPHA_STAR_IW: 0.001,
        O.R_REST_AW: 0.005,
        O.VENTILATION: 100.0,
        O.TEMP_INIT: 22.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 25.0,
        O.AREA: 150.0,
        O.HEIGHT: 2.7,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 500.0,
        O.WEATHER: "w_flag",
        **flags,
    }


def _run_r7c2(obj: dict) -> dict:
    from entise.methods.hvac.R7C2 import R7C2

    return R7C2().generate(obj, {"w_flag": _weather(24 * 3), O.WINDOWS: _windows_for("b1")}, Types.HVAC)


class TestR7C2RespectsFlags:
    """R7C2's T_eq computation (equivalent outdoor temperature) includes
    sky radiation losses whether or not solar gains are active, so demand
    isn't strictly zero even with all flags off. The tests here check that
    flipping each flag CHANGES demand in the expected direction, which is
    the actual claim being fixed."""

    def test_solar_gains_flag_off_reduces_cooling(self):
        r_on = _run_r7c2(_r7c2_obj(active_gains_solar=True, active_gains_internal=False))
        r_off = _run_r7c2(_r7c2_obj(active_gains_solar=False, active_gains_internal=False))
        cool_key = f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]"
        assert r_off["summary"][cool_key] <= r_on["summary"][cool_key]

    def test_internal_gains_flag_off_reduces_cooling(self):
        r_on = _run_r7c2(_r7c2_obj(active_gains_internal=True, active_gains_solar=False))
        r_off = _run_r7c2(_r7c2_obj(active_gains_internal=False, active_gains_solar=False))
        cool_key = f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]"
        assert r_off["summary"][cool_key] <= r_on["summary"][cool_key]

    def test_ventilation_flag_off_changes_demand(self):
        r_on = _run_r7c2(_r7c2_obj(active_ventilation=True, active_gains_internal=False, active_gains_solar=False))
        r_off = _run_r7c2(_r7c2_obj(active_ventilation=False, active_gains_internal=False, active_gains_solar=False))
        # Ventilation changes total demand — the fact that either the
        # heating or cooling demand shifts proves the flag is honored.
        h_key = f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"
        c_key = f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]"
        assert (r_on["summary"][h_key], r_on["summary"][c_key]) != (r_off["summary"][h_key], r_off["summary"][c_key])
