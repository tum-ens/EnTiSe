
import pytest
import pandas as pd
import numpy as np
from src.methods.R1C1 import R1C1
from src.constants import Keys as K, Objects as O, Columns as C, Types as Types

ID = 'test_obj'


# Fixture for weather data with dummy solar irradiance columns (required for pvlib)
@pytest.fixture
def weather_data():
    index = pd.date_range(start='2024-01-01', periods=10, freq='H', tz='UTC')
    temp_out = pd.Series([0, -2, -3, 5, 10, 12, 15, 18, 20, 22], index=index)
    weather = pd.DataFrame({
        C.DATETIME: index,
        C.TEMP_OUT: temp_out,
        C.SOLAR_GHI: np.zeros(len(index)),  # Dummy GHI values
        C.SOLAR_DNI: np.zeros(len(index)),  # Dummy DNI values
        C.SOLAR_DHI: np.zeros(len(index)),  # Dummy DHI values
    })
    weather.set_index(C.DATETIME, inplace=True, drop=False)
    return weather

@pytest.fixture
def windows_data():
    data = {
        C.ID: [ID, ID, ID],
        C.AREA: [10.0, 5.0, 8.0],
        C.TRANSMITTANCE: [0.6, 0.5, 0.55],
        C.ORIENTATION: [180, 90, 200],
        C.TILT: [90, 90, 90],
        C.SHADING: [0.8, 0.9, 0.85]
    }
    return pd.DataFrame(data)



# Fixture for building parameters
@pytest.fixture
def parameters():
    obj = {
        O.ID: ID,
        O.RESISTANCE: 0.01,
        O.CAPACITANCE: 20_000_000,
        O.TEMP_INIT: 20,
        O.TEMP_MIN: 19.5,
        O.TEMP_MAX: 22,
        O.WEATHER: O.WEATHER,
        O.LAT: 40.0,
        O.LON: -80.0,
    }
    return obj


# Test that heating and cooling loads behave correctly based on outdoor temperatures.
def test_calculate_heating_cooling_basic(weather_data, windows_data, parameters):
    obj = parameters
    data = {O.WEATHER: weather_data, O.WINDOWS: windows_data}
    summary, df = R1C1().generate(obj, data)
    df.set_index(C.DATETIME, inplace=True, drop=False)
    # Check that the output DataFrame has the expected columns and index
    assert f'{C.LOAD}_{Types.HEATING}' in df.columns
    assert f'{C.LOAD}_{Types.COOLING}' in df.columns

    heating = df[f'{C.LOAD}_{Types.HEATING}']
    cooling = df[f'{C.LOAD}_{Types.COOLING}']

    # No heating when outdoor temp is at or above the minimum setpoint
    assert (heating[weather_data[C.TEMP_OUT] >= obj[O.TEMP_MIN]] == 0).all()
    # No cooling when outdoor temp is at or below the maximum setpoint
    assert (cooling[weather_data[C.TEMP_OUT] <= obj[O.TEMP_MAX]] == 0).all()


# Test the evolution of indoor temperature when outdoor temperature is low (heating active)
def test_temp_evolution(weather_data, windows_data, parameters):
    weather_data[C.TEMP_OUT] = 15  # Constant outdoor temperature below the heating setpoint
    data = {O.WEATHER: weather_data, O.WINDOWS: windows_data}
    summary, df = R1C1().generate(parameters, data)

    final_temp = df[f'{C.TEMP_IN}'].iloc[-1]
    # With heating active, the final indoor temperature should be at least the minimum setpoint.
    assert final_temp >= parameters[O.TEMP_MIN]


# Test that the output DataFrame has the expected structure.
def test_generate_output_structure(weather_data, parameters):
    data = {O.WEATHER: weather_data}
    summary, df = R1C1().generate(parameters, data)
    expected_columns = [C.DATETIME, f'{C.TEMP_IN}', f'{C.LOAD}_{Types.HEATING}', f'{C.LOAD}_{Types.COOLING}']
    for col in expected_columns:
        assert col in df.columns


if __name__ == "__main__":
    pytest.main()
