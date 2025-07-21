import datetime as dt

import numpy as np
import pandas as pd


class Columns:
    """Column names used in timeseries data."""

    ID = "id"  # Object ID
    AREA = "area"  # Area of the object (m²)
    DATE = "date"  # Date
    DATETIME = "datetime"  # Timestamps
    DAY = "day"  # day
    DAY_OF_WEEK = f"{DAY}_of_week"  # Day of the week (0-6; 6: Sunday)
    DAY_CALENDAR = f"{DAY}_calendar"  # Calendar day
    COP = "cop"  # Coefficient of Performance
    DEMAND = "demand"  # e.g. heating or cooling demand (Wh)
    DURATION = "duration"  # (s)
    DURATION_SIGMA = f"{DURATION}_sigma"
    EVENT = "event"  # event
    FLOW_RATE = "flow_rate"  # l/s
    FLOW_RATE_SIGMA = f"{FLOW_RATE}_sigma"
    FLH = "full_load_hours"  # full load hours (h; [0, 8760])
    GAIN = "gain"  # Gain (W)
    GENERATION = "generation"  # e.g. pv, wind (Wh)
    LOAD = "load"  # Load values (e.g., electricity load) (W)
    MONTH = "month"
    OCCUPATION = "occupation"  # Occupancy data
    ORIENTATION = "orientation"  # Orientation (degrees; 180 = south)
    POWER = "power"  # power (W)
    PROBABILITY = "probability"  # probability (of event; [0, 1])
    PROBABILITY_DAY = f"{PROBABILITY}_day"  # Probabilty of event per day [0, inf)
    ROUGHNESS_LENGTH = "roughness_length"  # Roughness length
    SHADING = "shading"  # Shading [0, 1]
    SOLAR_DHI = "diffuse_horizontal_irradiance"  # Diffuse horizontal irradiance (W/m2)
    SOLAR_DNI = "direct_normal_irradiance"  # Direct normal irradiance (W/m2)
    SOLAR_GHI = "global_horizontal_irradiance"  # Global horizontal irradiance (W/m2)
    SURFACE_AIR_PRESSURE = "surface_air_pressure"  # Surface pressure (Pa)
    TEMP = "temperature"  #  Temperature (K)
    TEMP_AIR = f"air_{TEMP}"  # Air temperature (K)
    TEMP_IN = f"indoor_{TEMP}"  # Indoor temperature (K)
    TEMP_OUT = f"outdoor_{TEMP}"  # Outdoor temperature (K)
    TEMP_SOIL = f"soil_{TEMP}"  # Soil temperature (K)
    TEMP_WATER = f"water_{TEMP}"  # Water temperature (K)
    TEMP_WATER_COLD = f"cold_{TEMP_WATER}"  # Cold water temperature (K)
    TEMP_WATER_GROUND = f"groundwater_{TEMP}"  # Groundwater temperature (K)
    TEMP_WATER_HOT = f"hot_{TEMP_WATER}"  # Hot water temperature (K)
    TILT = "tilt"  # Tilt (degrees; 0 = horizontal)
    TIME = "time"  # time of day
    TRANSMITTANCE = "transmittance"  # Transmittance (W/m2/K)
    WIND_SPEED = "wind_speed"  # Wind speed (m/s)
    WIND_DIRECTION = "wind_from_direction"  # Wind direction (º)
    DTYPES = {
        ID: object,
        AREA: float | int | np.number,
        DATE: dt.datetime,
        DATETIME: pd.Timestamp | np.datetime64 | str | int | float,
        DAY: int,
        DAY_OF_WEEK: int,
        DAY_CALENDAR: int,
        COP: float | int | np.number,
        DEMAND: float | int | np.number,
        DURATION: float | int | np.number,
        DURATION_SIGMA: float | int | np.number,
        EVENT: str,
        FLOW_RATE: float | int | np.number,
        FLOW_RATE_SIGMA: float | int | np.number,
        GAIN: float | int | np.number,
        LOAD: float | int | np.number,
        MONTH: int | np.number,
        OCCUPATION: float,
        ORIENTATION: float | int | np.number,
        PROBABILITY: float | np.number,
        PROBABILITY_DAY: float | np.number,
        SHADING: float | int | np.number,
        SOLAR_DHI: float | int | np.number,
        SOLAR_DNI: float | int | np.number,
        SOLAR_GHI: float | int | np.number,
        TEMP_IN: float | int | np.number,
        TEMP_OUT: float | int | np.number,
        TEMP_WATER: float | int | np.number,
        TEMP_WATER_COLD: float | int | np.number,
        TEMP_WATER_HOT: float | int | np.number,
        TILT: float | int | np.number,
        TIME: str,
        TRANSMITTANCE: float | int | np.number,
        WIND_SPEED: float | int | np.number,
        WIND_DIRECTION: float | int | np.number,
    }
