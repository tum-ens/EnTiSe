import numpy as np
import pandas as pd
import pytest

from entise.constants import Objects as O
from entise.methods.auxiliary.ventilation.selector import Ventilation


# Fixtures for minimal data setup
@pytest.fixture
def minimal_weather():
    index = pd.date_range("2025-01-01", periods=24, freq="h")
    return pd.DataFrame(index=index)


@pytest.fixture
def timeseries_with_column():
    index = pd.date_range("2025-01-01", periods=24, freq="h")
    return pd.DataFrame({"object_1": np.arange(24)}, index=index)


# Tests
def test_ventilation_inactive_strategy(minimal_weather):
    """Test the VentilationInactive strategy."""
    obj = {}
    data = {O.WEATHER: minimal_weather}
    ventilation = Ventilation()

    result = ventilation.generate(obj, data)

    assert isinstance(result, pd.DataFrame)
    assert O.VENTILATION in result.columns
    assert (result[O.VENTILATION] == 0).all()


def test_ventilation_constant_strategy(minimal_weather):
    """Test the VentilationConstant strategy."""
    obj = {O.VENTILATION: 100}
    data = {O.WEATHER: minimal_weather}
    ventilation = Ventilation()

    result = ventilation.generate(obj, data)

    assert isinstance(result, pd.DataFrame)
    assert O.VENTILATION in result.columns
    assert (result[O.VENTILATION] == 100).all()


def test_ventilation_timeseries_strategy(minimal_weather, timeseries_with_column):
    """Test the VentilationTimeSeries strategy."""
    obj = {O.ID: "object_1", O.VENTILATION_COL: "object_1", O.VENTILATION: O.VENTILATION}
    data = {
        O.VENTILATION: timeseries_with_column,
        O.WEATHER: minimal_weather,
    }
    ventilation = Ventilation()

    result = ventilation.generate(obj, data)

    assert isinstance(result, pd.DataFrame)
    assert O.VENTILATION in result.columns
    expected = (
        np.arange(24) * 1 * 2.5 * 1.2 * 1000 / 3600
    )  # Apply the same transformation as in VentilationTimeSeries.run()
    assert np.allclose(result[O.VENTILATION], expected)


def test_ventilation_string_reference(minimal_weather, timeseries_with_column):
    """Test that a string reference to a time series is correctly handled."""
    obj = {O.VENTILATION: "ventilation_data"}
    data = {
        "ventilation_data": timeseries_with_column,
        O.WEATHER: minimal_weather,
    }
    ventilation = Ventilation()

    # This should raise an error because ventilation_col is not provided
    with pytest.raises(Warning):
        ventilation.generate(obj, data)


def test_ventilation_with_column_name(minimal_weather, timeseries_with_column):
    """Test that a column name is correctly used."""
    obj = {O.ID: "wrong_id", O.VENTILATION: "ventilation_data", O.VENTILATION_COL: "object_1"}
    data = {
        "ventilation_data": timeseries_with_column,
        O.WEATHER: minimal_weather,
    }
    ventilation = Ventilation()

    result = ventilation.generate(obj, data)

    assert isinstance(result, pd.DataFrame)
    assert O.VENTILATION in result.columns
    expected = (
        np.arange(24) * 1 * 2.5 * 1.2 * 1000 / 3600
    )  # Apply the same transformation as in VentilationTimeSeries.run()
    assert np.allclose(result[O.VENTILATION], expected)


def test_ventilation_with_object_id(minimal_weather, timeseries_with_column):
    """Test that the object ID is used as column name if ventilation_col is not provided."""
    obj = {O.ID: "object_1", O.VENTILATION: "ventilation_data"}
    data = {
        "ventilation_data": timeseries_with_column,
        O.WEATHER: minimal_weather,
    }
    ventilation = Ventilation()

    result = ventilation.generate(obj, data)

    assert isinstance(result, pd.DataFrame)
    assert O.VENTILATION in result.columns
    expected = (
        np.arange(24) * 1 * 2.5 * 1.2 * 1000 / 3600
    )  # Apply the same transformation as in VentilationTimeSeries.run()
    assert np.allclose(result[O.VENTILATION], expected)


def test_ventilation_column_not_found(minimal_weather, timeseries_with_column):
    """Test that an error is raised if the column is not found."""
    obj = {O.ID: "object_1", O.VENTILATION: "ventilation_data", O.VENTILATION_COL: "non_existent_column"}
    data = {
        "ventilation_data": timeseries_with_column,
        O.WEATHER: minimal_weather,
    }
    ventilation = Ventilation()

    with pytest.raises(Warning):
        ventilation.generate(obj, data)


def test_ventilation_timeseries_with_numeric_value(minimal_weather):
    """Test that VentilationTimeSeries falls back to VentilationConstant when given a numeric value."""
    # Create an object that would normally use VentilationTimeSeries
    obj = {O.ID: "object_1", O.VENTILATION_COL: "object_1", O.VENTILATION: 100}
    data = {O.WEATHER: minimal_weather}

    # Create a VentilationTimeSeries instance directly to test its behavior
    from entise.methods.auxiliary.ventilation.strategies import VentilationTimeSeries

    ventilation_ts = VentilationTimeSeries()

    result = ventilation_ts.generate(obj, data)

    assert isinstance(result, pd.DataFrame)
    assert O.VENTILATION in result.columns
    assert (result[O.VENTILATION] == 100).all()


# Unit spellings in the ventilation column name (issue #107)
AREA, HEIGHT = 140.0, 2.6
AIR_CHANGE_TO_W_PER_K = AREA * HEIGHT * 1.2 * 1000 / 3600
VOLUME_FLOW_TO_W_PER_K = 1.2 * 1000 / 3600


def _run_with_column(col, value=61.0):
    from entise.methods.auxiliary.ventilation.strategies import VentilationTimeSeries

    index = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
    table = pd.DataFrame({col: np.full(24, value)}, index=index)
    return VentilationTimeSeries().run(
        **{
            O.ID: "x",
            O.AREA: AREA,
            O.HEIGHT: HEIGHT,
            O.VENTILATION_COL: col,
            O.VENTILATION: table,
            O.VENTILATION_FACTOR: 0.5,
        }
    )[O.VENTILATION]


@pytest.mark.parametrize(
    "col, factor",
    [
        ("typical[W K-1]", 1.0),
        ("typical[W/K]", 1.0),
        ("typical[m3 h-1]", VOLUME_FLOW_TO_W_PER_K),
        ("typical[m3/h]", VOLUME_FLOW_TO_W_PER_K),
        ("typical[h-1]", AIR_CHANGE_TO_W_PER_K),
        ("typical[1/h]", AIR_CHANGE_TO_W_PER_K),
        ("typical[W / K]", 1.0),
        ("typical[w/k]", 1.0),
        ("typical[m3h-1]", VOLUME_FLOW_TO_W_PER_K),
        ("typical[m³/h]", VOLUME_FLOW_TO_W_PER_K),
        ("typical[m^3/h]", VOLUME_FLOW_TO_W_PER_K),
        ("typical[1 / h]", AIR_CHANGE_TO_W_PER_K),
        ("typical[1/H]", AIR_CHANGE_TO_W_PER_K),
    ],
)
def test_ventilation_unit_spellings_are_equivalent(col, factor, caplog):
    """Both the slash and the exponent spelling of each unit give the same conductance."""
    with caplog.at_level("WARNING"):
        result = _run_with_column(col)

    np.testing.assert_allclose(result, 61.0 * factor, rtol=1e-6)
    assert not caplog.records


def test_an_unrecognised_ventilation_unit_is_named_in_the_warning(caplog):
    """An unknown unit falls back to air changes per hour and the warning names it."""
    with caplog.at_level("WARNING"):
        result = _run_with_column("typical[l s-1]")

    np.testing.assert_allclose(result, 61.0 * AIR_CHANGE_TO_W_PER_K, rtol=1e-6)
    assert any("l s-1" in record.getMessage() for record in caplog.records)


def test_a_column_without_unit_is_read_in_air_changes_per_hour(caplog):
    """A bare column name, such as an object id, keeps the fallback to air changes per hour."""
    with caplog.at_level("WARNING"):
        result = _run_with_column("object_1")

    np.testing.assert_allclose(result, 61.0 * AIR_CHANGE_TO_W_PER_K, rtol=1e-6)
    assert caplog.records
