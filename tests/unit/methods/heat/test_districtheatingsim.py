"""
Unit tests for EnTiSe DistrictHeatingSim heat (BDEW).

This suite focuses on the behavior of the Districtheatingsim heating method and
validates the following contracts:
- Correct output typing, shape, and value domain for hourly and subhourly input
- Linear scaling of energy with annual demand for a fixed horizon
- Input validation: positive annual demand, presence of required weather fields
- Error propagation for invalid BDEW configuration parameters (profile_type)
- Public API: generate() returns a schema-consistent timeseries and summary

The tests are intentionally implementation-aware regarding resampling and the
summary calculation to catch regressions in time handling and energy accounting.
"""

import numpy as np
import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.methods.heat.districtheatingsim import DistrictHeatSim, calculate_timeseries

OUT_COL = f"{Types.HEATING}{SEP}{C.LOAD}[W]"
SUM_DEMAND_WH = f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"
SUM_LOAD_MAX_W = f"{Types.HEATING}{SEP}{O.LOAD_MAX}[W]"


@pytest.fixture(scope="module")
def _require_districtheatsim():
    """Ensure demandlib is available; otherwise skip this module's tests.

    This keeps the suite optional for environments that do not install the
    external 'demandlib' package while still validating our integration when
    it is present.
    """
    pytest.importorskip("districtheatingsim")


def _make_weather(start: str, periods: int, freq: str = "h") -> pd.DataFrame:
    """Create a minimal weather DataFrame for tests.

    Args:
        start: Start timestamp (UTC, any pandas-parsable string).
        periods: Number of periods to generate.
        freq: Pandas frequency string (e.g., "h", "15min").

    Returns:
        DataFrame with UTC datetime column and a synthetic air temperature ramp.
    """
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            C.DATETIME: idx,
            C.TEMP_AIR: np.linspace(-5.0, 5.0, len(idx), dtype=np.float32),
        }
    )


def _energy_wh_from_power_series(power_w: pd.Series, dt_s: float) -> float:
    """Compute energy in Wh given a power series and its timestep in seconds.

    This helper integrates the discrete power series by multiplying the sum of
    samples by the timestep (dt_s) and converting seconds to hours.

    Args:
        power_w: Power values [W] sampled at a fixed timestep.
        dt_s: Timestep in seconds between samples.

    Returns:
        Total energy in watt-hours (Wh).
    """
    return float(power_w.sum()) * (dt_s / 3600.0)


def _find_split_cols(ts: pd.DataFrame) -> tuple[str | None, str | None]:
    """Best-effort detection of optional DHW + space-heating split columns.

    We do NOT hardcode names to avoid coupling tests to a specific naming choice.
    Instead, we look for columns containing 'dhw' and 'space' (case-insensitive).
    """
    cols = list(ts.columns)
    dhw_col = next((c for c in cols if "dhw" in str(c).lower()), None)
    space_col = next((c for c in cols if "space" in str(c).lower()), None)
    return dhw_col, space_col


def test_calculate_timeseries_hourly_returns_series_and_no_nans(_require_districtheatsim):
    """Hourly input yields an integer Series with same length, no NaNs, non-negative.

    Verifies the core hourly-native path of calculate_timeseries: output type,
    NaN-free, non-negative values, and length equality with hourly weather.
    """
    weather = _make_weather("2022-01-01 00:00:00", periods=24, freq="h")

    obj = {
        O.ID: "obj_1",
        O.DEMAND_KWH: 10_000.0,
        O.PROFILE_TYPE: "HEF03",
        O.DHW_SHARE: 0.2,
        O.WEATHER: O.WEATHER,
    }
    data = {O.WEATHER: weather}

    m = DistrictHeatSim()
    obj_in, data_in = m._get_input_data(obj, data, Types.HEATING)
    s = calculate_timeseries(obj_in, data_in)

    assert isinstance(s, pd.Series)
    assert len(s) == len(weather)
    assert s.isna().sum() == 0
    assert (s >= 0).all()
    assert pd.api.types.is_integer_dtype(s.dtype)


def test_calculate_timeseries_subhourly_15min_length_and_no_nans(_require_districtheatsim):
    """15-min input follows resample→interpolate path, remains valid and int.

    The demandlib heat algorithm is hourly-native. For subhourly inputs, we
    resample to hourly and back to the user timestep. This test ensures:
    - The output is a Series with no NaNs and non-negative integer values
    - Length is not greater than input (boundaries may differ by pandas version)
    - The naive start timestamp aligns with the input start
    """
    weather = _make_weather("2022-01-01 00:00:00", periods=24 * 4, freq="15min")

    obj = {
        O.ID: "obj_1",
        O.DEMAND_KWH: 10_000.0,
        O.PROFILE_TYPE: "HEF03",
        O.DHW_SHARE: 0.2,
        O.WEATHER: O.WEATHER,
    }
    data = {O.WEATHER: weather}

    m = DistrictHeatSim()
    obj_in, data_in = m._get_input_data(obj, data, Types.HEATING)
    s = calculate_timeseries(obj_in, data_in)

    assert isinstance(s, pd.Series)

    # Due to hourly → subhourly resample, output may be <= input length
    assert 0 < len(s) <= len(weather)

    assert s.isna().sum() == 0
    assert (s >= 0).all()
    assert pd.api.types.is_integer_dtype(s.dtype)

    # Check naive start alignment
    assert s.index[0] == pd.Timestamp("2022-01-01 00:00:00")


def test_demand_scaling_is_linear_for_fixed_horizon(_require_districtheatsim):
    """Doubling annual demand approximately doubles total energy at output dt.

    Checks proportionality of total Wh computed from the generated power series
    while keeping weather and horizon constant. Uses a tolerance to allow for
    rounding when converting to integer watt values.
    """
    weather = _make_weather("2022-01-01 00:00:00", periods=24 * 4, freq="15min")
    data = {O.WEATHER: weather}
    m = DistrictHeatSim()

    base_obj = {
        O.ID: "obj_1",
        O.PROFILE_TYPE: "HEF03",
        O.DHW_SHARE: 0.2,
        O.WEATHER: O.WEATHER,
    }

    obj1 = {**base_obj, O.DEMAND_KWH: 8_000.0}
    obj2 = {**base_obj, O.DEMAND_KWH: 16_000.0}

    o1, d1 = m._get_input_data(obj1, data, Types.HEATING)
    o2, d2 = m._get_input_data(obj2, data, Types.HEATING)

    s1 = calculate_timeseries(o1, d1)
    s2 = calculate_timeseries(o2, d2)

    dt_s = float(
        pd.Series(pd.DatetimeIndex(pd.to_datetime(d1[O.WEATHER][C.DATETIME], utc=True)))
        .diff()
        .dt.total_seconds()
        .median()
    )

    e1_wh = _energy_wh_from_power_series(s1.astype(float), dt_s)
    e2_wh = _energy_wh_from_power_series(s2.astype(float), dt_s)

    assert e2_wh == pytest.approx(2.0 * e1_wh, rel=1e-2, abs=2000.0)


def test_profile_type_too_short_raises(_require_districtheatsim):
    """Invalid profile_type should error early."""
    weather = _make_weather("2022-01-01 00:00:00", periods=24, freq="h")

    obj = {
        O.ID: "obj_1",
        O.DEMAND_KWH: 10_000.0,
        O.PROFILE_TYPE: "H",
        O.DHW_SHARE: 0.2,
        O.WEATHER: O.WEATHER,
    }
    data = {O.WEATHER: weather}

    m = DistrictHeatSim()
    with pytest.raises(ValueError, match=r"profile_type must be"):
        m._get_input_data(obj, data, Types.HEATING)


def test_generate_end_to_end_schema_and_summary_consistency(_require_districtheatsim):
    """Public API generate() returns expected schema and consistent summary.

    Asserts that:
    - result contains both 'summary' and 'timeseries'
    - timeseries contains the HEATING|load[W] convention (total)
    - no NaNs are present
    - summary demand[Wh] equals the integral of the total timeseries at its dt
    - summary max load equals the max of the total timeseries

    If the method additionally provides DHW/space split columns, also validate:
    - both split columns are valid (no NaNs, non-negative ints)
    - space + dhw == total (within small rounding tolerance)
    """
    weather = _make_weather("2022-01-01 00:00:00", periods=24, freq="h")

    obj = {
        O.ID: "obj_1",
        O.DEMAND_KWH: 10_000.0,
        O.PROFILE_TYPE: "HEF03",
        O.DHW_SHARE: 0.0,
        O.WEATHER: O.WEATHER,
    }
    data = {O.WEATHER: weather}

    m = DistrictHeatSim()
    result = m.generate(obj=obj, data=data, ts_type=Types.HEATING)

    assert "summary" in result
    assert "timeseries" in result

    ts = result["timeseries"]
    assert isinstance(ts, pd.DataFrame)
    assert OUT_COL in ts.columns
    assert ts[OUT_COL].isna().sum() == 0

    # Optional: validate DHW + space split if present
    dhw_col, space_col = _find_split_cols(ts)
    if dhw_col is not None and space_col is not None:
        assert ts[dhw_col].isna().sum() == 0
        assert ts[space_col].isna().sum() == 0
        assert (ts[dhw_col] >= 0).all()
        assert (ts[space_col] >= 0).all()
        assert pd.api.types.is_integer_dtype(ts[dhw_col].dtype)
        assert pd.api.types.is_integer_dtype(ts[space_col].dtype)

        # Sum consistency (allow +/- 1 W due to integer rounding)
        summed = ts[dhw_col].astype(float) + ts[space_col].astype(float)
        diff = (summed - ts[OUT_COL].astype(float)).abs().max()
        assert float(diff) <= 1.0

    idx_user = pd.DatetimeIndex(ts.index)
    timestep_s = float(pd.Series(idx_user).diff().dt.total_seconds().median())
    total_wh = float(ts[OUT_COL].sum()) * timestep_s / 3600.0

    assert result["summary"][SUM_DEMAND_WH] == int(round(total_wh))
    assert result["summary"][SUM_LOAD_MAX_W] == int(round(float(ts[OUT_COL].max())))


def test_too_short_weather_raises(_require_districtheatsim):
    """Weather with <2 points should fail in dt computation or downstream logic."""
    weather = _make_weather("2022-01-01 00:00:00", periods=1, freq="h")

    obj = {
        O.ID: "obj_1",
        O.DEMAND_KWH: 10_000.0,
        O.PROFILE_TYPE: "HEF03",
        O.DHW_SHARE: 0.2,
        O.WEATHER: O.WEATHER,
    }
    data = {O.WEATHER: weather}

    m = DistrictHeatSim()
    o, d = m._get_input_data(obj, data, Types.HEATING)

    with pytest.raises((IndexError, ValueError)):
        calculate_timeseries(o, d)
