import pandas as pd
from src.constants import Objects, Columns


def get_mock_timeseries_data():
    return {
        Objects.WEATHER: pd.DataFrame({
            Columns.DATETIME: [1, 2, 3],
            Columns.TEMP_OUT: [15, 10, 5]
        }),
        Objects.OCCUPATION: pd.DataFrame({
            Columns.DATETIME: [1, 2, 3],
            Columns.OCCUPATION: [0, 1, 1]
        })
    }


def get_valid_timeseries():
    """Returns a mock valid timeseries for testing."""
    return {
        Objects.WEATHER: pd.DataFrame({
            Columns.DATETIME: pd.date_range("2023-01-01", periods=3),
            Columns.TEMP_OUT: [15.0, 10.0, 5.0],
        })
    }
