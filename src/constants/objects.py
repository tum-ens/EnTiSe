import numpy as np


class Objects:
    """Keys used for identifying objects and their attributes.
    Sorted alphabetically for easy reference (except for ID).
    """
    ID = "id"  # Unique identifier for an object
    CAPACITANCE = "capacitance"  # Capacitance (J/K)
    FILE = "file"  # File
    LOAD_BASE = "load_base"  # Base load
    OCCUPATION = "occupation"  # Occupancy (for timeseries files)
    POWER_COOLING = "power_cooling"  # Power cooling (W)
    POWER_HEATING = "power_heating"  # Power heating (W)
    RESISTANCE = "resistance"  # Resistance (K/W)
    TEMP_INIT = "temp_init"  # Initial temperature (C)
    TEMP_MAX = "temp_max"  # Maximum temperature (C)
    TEMP_MIN = "temp_min"  # Minimum temperature (C)
    TEMP_SET = "temp_set"  # Set temperature (C)
    VERBOSE = "verbose"  # Verbose
    WEATHER = "weather"  # Weather data
    DTYPES = {
        ID: object,
        CAPACITANCE: int | float | np.number,
        FILE: str,
        LOAD_BASE: int | float | np.number,
        OCCUPATION: str,
        RESISTANCE: int | float | np.number,
        TEMP_INIT: int | float | np.number,
        VERBOSE: bool,
        WEATHER: str,
        }
