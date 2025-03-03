import pandas as pd
import numpy as np


class Columns:
    """Column names used in timeseries data."""
    DATETIME = "datetime"  # Timestamps
    DEMAND = "demand"  # Heating or cooling demand
    LOAD = "load"  # Load values (e.g., electricity load)
    OCCUPATION = "occupation"  # Occupancy data
    SOLAR_GHI = "solar_ghi"  # Global horizontal irradiance
    TEMP_OUT = "temp_out"  # Outdoor temperature
    DTYPES = {
        DATETIME: pd.Timestamp | np.datetime64 | str | int | float,
        DEMAND: float | int | np.number,
        LOAD: float | int | np.number,
        OCCUPATION: float,
        SOLAR_GHI: float | int | np.number,
        TEMP_OUT: float | int | np.number,
    }