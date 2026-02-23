import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.methods.electricity.pylpg import PyLPG, calculate_timeseries

OUT_COL = f"{Types.ELECTRICITY}{SEP}{C.LOAD}[W]"


@pytest.fixture(autouse=True)
def _clear_dts_cache():
    """Clear the module-level DTS cache before each test to avoid cross-test state leakage."""
    import entise.methods.electricity.pylpg as m

    m._DTS_CACHE.clear()


def _make_datetimes(start: str, periods: int, freq: str = "h", tz: str = "UTC") -> pd.DataFrame:
    """Helper to build a minimal datetimes dataframe with a single C.DATETIME column.

    Args:
        start: Start timestamp string parseable by pandas.
        periods: Number of periods to generate.
        freq: Pandas offset alias (e.g., "h", "15min").
        tz: Time zone for the generated index (default: "UTC").

    Returns:
        DataFrame with one column C.DATETIME containing a DatetimeIndex.
    """
    idx = pd.date_range(start, periods=periods, freq=freq, tz=tz)
    return pd.DataFrame({C.DATETIME: idx})


def _energy_wh_from_power(power_w: pd.Series, dt_s: float) -> float:
    """Compute energy in Wh by integrating a power series at resolution dt_s seconds."""
    return float(power_w.sum()) * (dt_s / 3600.0)


def test_calculate_timeseries_hourly_runs_and_no_nans():
    """Hourly path: returns DataFrame with expected column, length match, tz-naive index, no NaNs, non-negative."""
    dts = _make_datetimes("2022-01-01 00:00:00", periods=6, freq="h")

    obj = {
        O.ID: "obj_1",
        O.HOUSEHOLDS: 1,
        O.OCCUPANTS_PER_HOUSEHOLD: 2,
        O.DATETIMES: O.DATETIMES,
    }
    data = {O.DATETIMES: dts}

    m = PyLPG()
    obj_in, data_in = m._get_input_data(obj, data, Types.ELECTRICITY)
    ts = calculate_timeseries(obj_in, data_in)

    assert isinstance(ts, pd.DataFrame)
    assert "pylpg" in ts.columns

    assert len(ts) == len(dts)
    assert ts.index.tz is None
    assert ts.index[0] == pd.Timestamp("2022-01-01 00:00:00")
    assert ts["pylpg"].isna().sum() == 0
    assert (ts["pylpg"] >= 0).all()


def test_generate_end_to_end_schema_and_summary_consistency():
    """Public API: returns summary and timeseries; summary’s Wh/max W match recomputed values from ts."""
    dts = _make_datetimes("2022-01-01 00:00:00", periods=8, freq="15min")

    obj = {
        O.ID: "obj_1",
        O.HOUSEHOLDS: 1,
        O.OCCUPANTS_PER_HOUSEHOLD: 2,
        O.DATETIMES: O.DATETIMES,
    }
    data = {O.DATETIMES: dts}

    m = PyLPG()
    result = m.generate(obj=obj, data=data, ts_type=Types.ELECTRICITY)

    assert "summary" in result
    assert "timeseries" in result

    ts = result["timeseries"]
    assert isinstance(ts, pd.DataFrame)
    assert OUT_COL in ts.columns
    assert ts[OUT_COL].isna().sum() == 0

    idx = pd.DatetimeIndex(ts.index)
    dt_s = float(pd.Series(idx).diff().dt.total_seconds().median())
    total_wh = _energy_wh_from_power(ts[OUT_COL].astype(float), dt_s)

    assert result["summary"][f"{Types.ELECTRICITY}{SEP}{C.DEMAND}[Wh]"] == int(round(total_wh))
    assert result["summary"][f"{Types.ELECTRICITY}{SEP}{O.LOAD_MAX}[W]"] == int(round(float(ts[OUT_COL].max())))


def test_multiyear_concat_runs():
    """Cross-year horizon: handles Dec→Jan boundary with correct length, tz-naive index, and no NaNs."""
    dts = _make_datetimes("2021-12-31 23:00:00", periods=5, freq="h")

    obj = {
        O.ID: "obj_1",
        O.HOUSEHOLDS: 1,
        O.OCCUPANTS_PER_HOUSEHOLD: 2,
        O.DATETIMES: O.DATETIMES,
    }
    data = {O.DATETIMES: dts}

    m = PyLPG()
    obj_in, data_in = m._get_input_data(obj, data, Types.ELECTRICITY)
    ts = calculate_timeseries(obj_in, data_in)

    assert len(ts) == 5
    assert "pylpg" in ts.columns
    assert ts.index.tz is None
    assert ts.index[0] == pd.Timestamp("2021-12-31 23:00:00")
    assert ts["pylpg"].isna().sum() == 0


def test_resampling_behaviour_preserves_energy_in_summary():
    """Verify that summary Wh matches recomputed Wh from the emitted timeseries for multiple dt_s."""
    for freq in ["60min", "15min", "5min"]:
        periods = int(pd.Timedelta("2h") / pd.Timedelta(freq))
        dts = _make_datetimes("2022-01-01 00:00:00", periods=periods, freq=freq)

        obj = {
            O.ID: f"obj_{freq}",
            O.HOUSEHOLDS: 1,
            O.OCCUPANTS_PER_HOUSEHOLD: 2,
            O.DATETIMES: O.DATETIMES,
        }
        data = {O.DATETIMES: dts}

        res = PyLPG().generate(obj=obj, data=data, ts_type=Types.ELECTRICITY)
        ts = res["timeseries"]

        dt_s = float(pd.Series(pd.DatetimeIndex(ts.index)).diff().dt.total_seconds().median())
        e_wh = float(ts[OUT_COL].astype(float).sum()) * (dt_s / 3600.0)

        assert res["summary"][f"{Types.ELECTRICITY}{SEP}{C.DEMAND}[Wh]"] == int(round(e_wh))


def test_non_monotonic_datetimes_are_normalized_but_length_preserved():
    """Non-monotonic datetimes: implementation normalizes to a regular range; ensure output length matches input."""
    idx = pd.to_datetime(
        ["2022-01-01 00:00:00Z", "2022-01-01 02:00:00Z", "2022-01-01 01:00:00Z"]  # out of order
    )
    dts = pd.DataFrame({C.DATETIME: idx})

    obj = {
        O.ID: "obj_1",
        O.HOUSEHOLDS: 1,
        O.OCCUPANTS_PER_HOUSEHOLD: 2,
        O.DATETIMES: O.DATETIMES,
    }
    data = {O.DATETIMES: dts}

    res = PyLPG().generate(obj=obj, data=data, ts_type=Types.ELECTRICITY)
    ts = res["timeseries"]

    assert len(ts) == len(dts)
    assert ts[OUT_COL].isna().sum() == 0


@pytest.mark.parametrize("freq", ["30s", "90s"])
def test_unsupported_timestep_is_rejected(freq):
    """dt_s must be >=60s and a multiple of 60s."""
    dts = _make_datetimes("2022-01-01 00:00:00", periods=5, freq=freq)

    obj = {
        O.ID: "obj_bad_dt",
        O.HOUSEHOLDS: 1,
        O.OCCUPANTS_PER_HOUSEHOLD: 2,
        O.DATETIMES: O.DATETIMES,
    }
    data = {O.DATETIMES: dts}

    with pytest.raises(ValueError, match="Unsupported timestep"):
        PyLPG().generate(obj=obj, data=data, ts_type=Types.ELECTRICITY)
