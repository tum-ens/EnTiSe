import pandas as pd
import numpy as np


class Columns:
    """Column names used in timeseries data."""
    ID = "id"  # Object ID
    AREA = "area"  # Area of the object (m²)
    DATETIME = "datetime"  # Timestamps
    DEMAND = "demand"  # e.g. heating or cooling demand (Wh)
    FLH = "full_load_hours"  # full load hours (h; [0, 8760])
    GAIN = "gain"  # Gain (W)
    GENERATION = "generation"  # e.g. pv, wind (Wh)
    LOAD = "load"  # Load values (e.g., electricity load) (W)
    OCCUPATION = "occupation"  # Occupancy data
    ORIENTATION = "orientation"  # Orientation (degrees; 180 = south)
    POWER = "power"  # power (W)
    SHADING = "shading"  # Shading [0, 1]
    SOLAR_DHI = "solar_dhi"  # Diffuse horizontal irradiance (W/m2)
    SOLAR_DNI = "solar_dni"  # Direct normal irradiance (W/m2)
    SOLAR_GHI = "solar_ghi"  # Global horizontal irradiance (W/m2)
    TEMP_IN = "temp_in"  # Indoor temperature (ºC)
    TEMP_OUT = "temp_out"  # Outdoor temperature (ºC)
    TILT = "tilt"  # Tilt (degrees; 0 = horizontal)
    TRANSMITTANCE = "transmittance"  # Transmittance (W/m2/K)
    DTYPES = {
        ID: object,
        AREA: float | int | np.number,
        DATETIME: pd.Timestamp | np.datetime64 | str | int | float,
        DEMAND: float | int | np.number,
        GAIN: float | int | np.number,
        LOAD: float | int | np.number,
        OCCUPATION: float,
        ORIENTATION: float | int | np.number,
        SHADING: float | int | np.number,
        SOLAR_DHI: float | int | np.number,
        SOLAR_DNI: float | int | np.number,
        SOLAR_GHI: float | int | np.number,
        TEMP_IN: float | int | np.number,
        TEMP_OUT: float | int | np.number,
        TILT: float | int | np.number,
        TRANSMITTANCE: float | int | np.number,
    }