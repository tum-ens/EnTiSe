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
    ALTITUDE = "altitude"  # Altitude (m)
    AREA = "area"  # Area (m2)
    AZIMUTH = "azimuth"  # Azimuth (degrees; 0/360 = North)
    CAPACITANCE = "capacitance"  # Capacitance (J/K)
    COLUMN = "column"
    DATETIMES = "datetimes"
    DEMAND = "demand"
    DHW_ACTIVITY = "dhw_activity"  # DHW activity filename
    DHW_DEMAND_PER_SIZE = "dhw_demand_per_size"  # DHW demand per size, i.e. m2
    DHW_DEMAND_PER_PERSON = "dhw_demand_per_person"  # DHW demand poer person
    DWELLING_SIZE = "dwelling_size"  # Size of dwelling (m2)
    FILE = "filename"  # File
    GAINS_INTERNAL = "gains_internal"  # Internal gains (W)
    GAINS_INTERNAL_COL = (
        f"{GAINS_INTERNAL}_{COLUMN}"  # column in which the internal gains are (if dataframe is provided)
    )
    GAINS_INTERNAL_PER_PERSON = f"{GAINS_INTERNAL}_per_person"  # Internal gains per person (W)
    GAINS_SOLAR = "gains_solar"  # Solar gains (W)
    GEN_MAX = "maximum_generation"  # Maximum generation (W)
    HOUSEHOLD_TYPE = "household_type"  # Type of household
    HOLIDAYS_LOCATION = (
        "holidays_location"  # Location from which to get the holidays from (e.g. BY,DE for Bavaria, Germany)
    )
    HUB_HEIGHT = "hub_height"  # Hub height (wind)
    INHABITANTS = "inhabitants"  # Number of inhabitants
    LAT = "latitude"  # Latitude (degrees)
    LOAD = "load"
    LOAD_BASE = f"{LOAD}_base"  # Base load
    LOAD_MAX = f"{LOAD}_max"
    LON = "longitude"  # Longitude (degrees)
    OCCUPATION = "occupation"  # Occupancy (for timeseries files)
    OCCUPANTS = "occupants"  # Number of occupants
    ORIENTATION = "orientation"  # Orientation (degrees; 180 = south)
    POWER = "power"  # Power (W)
    POWER_COOLING = f"{POWER}_cooling"  # Power cooling (W)
    POWER_HEATING = f"{POWER}_heating"  # Power heating (W)
    PV_ARRAYS = "pv_arrays"  # PV array configuration (dict)
    PV_INVERTER = "pv_inverter"  # PV inverter configuration (dict)
    RESISTANCE = "resistance"  # Resistance (K/W)
    SEASONAL_PEAK_DAY = "seasonal_peak_day"  # Day of year with peak demand
    SEASONAL_VARIATION = "seasonal_variation"  # Seasonal variation factor
    SEED = "seed"  # Seed to ensure reproducibility
    SOURCE = "source"  # Source of data or method
    TEMP_WATER_COLD = "temp_water_cold"  # Cold water temperature (C)
    TEMP_WATER_HOT = "temp_water_hot"  # Hot water temperature (C)
    TEMP_INIT = "temp_init"  # Initial temperature (C)
    TEMP_MAX = "temp_max"  # Maximum temperature (C)
    TEMP_MIN = "temp_min"  # Minimum temperature (C)
    TEMP_SET = "temp_set"  # Set temperature (C)
    THERMAL_INERTIA = "thermal_inertia"  # Thermal inertia [0, 1]
    TILT = "tilt"  # Tilt (degrees; 0 = horizontal)
    TRANSMITTANCE = "transmittance"  # Transmittance (W/m2/K)
    TURBINE_TYPE = "turbine_type"  # Turbine type (wind)
    VENTILATION = "ventilation"  # Ventilation losses (W/K)
    VERBOSE = "verbose"  # Verbose
    WEATHER = "weather"  # Weather data
    WIND_MODEL = "wind_model"  # Model chain (wind)
    WINDOWS = "windows"  # Windows
    DTYPES = {
        ID: object,
        ACTIVE_HEATING: bool,
        ACTIVE_COOLING: bool,
        ACTIVE_GAINS_SOLAR: bool,
        AREA: int | float | np.number,
        CAPACITANCE: int | float | np.number,
        DATETIMES: str,
        DEMAND: int | float | np.number,
        DHW_ACTIVITY: str,
        DHW_DEMAND_PER_SIZE: str,
        DWELLING_SIZE: int | float | np.number,
        FILE: str,
        GAINS_INTERNAL: int | float | np.number | str,
        GAINS_INTERNAL_COL: str,
        GAINS_SOLAR: int | float | np.number | str,
        HOUSEHOLD_TYPE: str,
        HOLIDAYS_LOCATION: str,
        INHABITANTS: int | float | np.number,
        LAT: int | float | np.number,
        LOAD: int | float | np.number,
        LOAD_BASE: int | float | np.number,
        LOAD_MAX: int | float | np.number,
        LON: int | float | np.number,
        OCCUPATION: str,
        OCCUPANTS: int,
        ORIENTATION: int | float | np.number,
        POWER_COOLING: int | float | np.number,
        POWER_HEATING: int | float | np.number,
        RESISTANCE: int | float | np.number,
        SEASONAL_PEAK_DAY: int,
        SEASONAL_VARIATION: float,
        SOURCE: str,
        TEMP_WATER_COLD: int | float | np.number,
        TEMP_WATER_HOT: int | float | np.number,
        TEMP_INIT: int | float | np.number,
        TEMP_MAX: int | float | np.number,
        TEMP_MIN: int | float | np.number,
        TEMP_SET: int | float | np.number,
        THERMAL_INERTIA: int | float | np.number,
        TILT: int | float | np.number,
        VENTILATION: int | float | np.number,
        VERBOSE: bool,
        WEATHER: str,
        WINDOWS: str,
    }
