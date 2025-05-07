import numpy as np


class Objects:
    """Keys used for identifying objects and their attributes.
    Sorted alphabetically for easy reference (except for ID).
    """
    ID = "id"  # Unique identifier for an object
    ACTIVE = "active"
    ACTIVE_COOLING = f"{ACTIVE}_cooling"  # Activate cooling?
    ACTIVE_GAINS_INTERNAL = f"{ACTIVE}_gains_internal"  # Activate internal gains?
    ACTIVE_GAINS_SOLAR = f"{ACTIVE}_gains_solar"  # Activate solar gains?
    ACTIVE_HEATING = f"{ACTIVE}_heating"  # Activate heating?
    AREA = "area"  # Area (m2)
    CAPACITANCE = "capacitance"  # Capacitance (J/K)
    COLUMN = "column"
    DEMAND = "demand"
    FILE = "file"  # File
    GAINS_INTERNAL = "gains_internal"  # Internal gains (W)
    GAINS_INTERNAL_COL = f'{GAINS_INTERNAL}_{COLUMN}'  # column in which the internal gains are (if dataframe is provided)
    GAINS_INTERNAL_PER_PERSON = f'{GAINS_INTERNAL}_per_person'  # Internal gains per person (W)
    GAINS_SOLAR = "gains_solar"  # Solar gains (W)
    INHABITANTS = "inhabitants"  # Number of inhabitants
    LAT = "latitude"  # Latitude (degrees)
    LOAD = "load"
    LOAD_BASE = f"{LOAD}_base"  # Base load
    LOAD_MAX = f"{LOAD}_max"
    LON = "longitude"  # Longitude (degrees)
    OCCUPATION = "occupation"  # Occupancy (for timeseries files)
    ORIENTATION = "orientation"  # Orientation (degrees; 180 = south)
    POWER_COOLING = "power_cooling"  # Power cooling (W)
    POWER_HEATING = "power_heating"  # Power heating (W)
    RESISTANCE = "resistance"  # Resistance (K/W)
    TEMP_INIT = "temp_init"  # Initial temperature (C)
    TEMP_MAX = "temp_max"  # Maximum temperature (C)
    TEMP_MIN = "temp_min"  # Minimum temperature (C)
    TEMP_SET = "temp_set"  # Set temperature (C)
    THERMAL_INERTIA = "thermal_inertia"  # Thermal inertia [0, 1]
    TILT = "tilt"  # Tilt (degrees; 0 = horizontal)
    TRANSMITTANCE = "transmittance"  # Transmittance (W/m2/K)
    VENTILATION = "ventilation"  # Ventilation losses (W/K)
    VERBOSE = "verbose"  # Verbose
    WEATHER = "weather"  # Weather data
    WINDOWS = "windows"  # Windows
    DTYPES = {
        ID: object,
        ACTIVE_HEATING: bool,
        ACTIVE_COOLING: bool,
        ACTIVE_GAINS_SOLAR: bool,
        AREA: int | float | np.number,
        CAPACITANCE: int | float | np.number,
        DEMAND: int | float | np.number,
        FILE: str,
        GAINS_INTERNAL: int | float | np.number | str,
        GAINS_INTERNAL_COL: str,
        GAINS_SOLAR: int | float | np.number | str,
        LAT: int | float | np.number,
        LOAD: int | float | np.number,
        LOAD_BASE: int | float | np.number,
        LOAD_MAX: int | float | np.number,
        LON: int | float | np.number,
        OCCUPATION: str,
        ORIENTATION: int | float | np.number,
        POWER_COOLING: int | float | np.number,
        POWER_HEATING: int | float | np.number,
        RESISTANCE: int | float | np.number,
        TEMP_INIT: int | float | np.number,
        TEMP_MAX: int | float | np.number,
        TEMP_MIN: int | float | np.number,
        TEMP_SET: int | float | np.number,
        THERMAL_INERTIA: int | float | np.number,
        VENTILATION: int | float | np.number,
        VERBOSE: bool,
        WEATHER: str,
        WINDOWS: str,
        }
