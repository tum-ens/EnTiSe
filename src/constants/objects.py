import numpy as np


class Objects:
    """Keys used for identifying objects and their attributes.
    Sorted alphabetically for easy reference (except for ID).
    """
    ID = "id"  # Unique identifier for an object
    CAPACITANCE = "capacitance"  # Capacitance identifier
    FILE = "file"  # File identifier
    LOAD_BASE = "load_base"  # Base load identifier
    OCCUPATION = "occupation"  # Occupancy identifier (for timeseries files)
    RESISTANCE = "resistance"  # Resistance identifier
    VERBOSE = "verbose"  # Verbose identifier
    WEATHER = "weather"  # Weather data identifier
    DTYPES = {
        ID: object,
        CAPACITANCE: int | float | np.number,
        FILE: str,
        LOAD_BASE: int | float | np.number,
        OCCUPATION: str,
        RESISTANCE: int | float | np.number,
        VERBOSE: bool,
        WEATHER: str,
        }
