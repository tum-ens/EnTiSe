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
    ALTITUDE = "altitude[m]"  # Altitude (m)
    AREA = "area[m2]"  # Area (m2)
    AZIMUTH = "azimuth[º]"  # Azimuth (degrees; 0/360 = North)
    CAPACITANCE = "capacitance[J K-1]"  # Capacitance (J/K)
    COLUMN = "column"
    CORRECTION_FACTOR = "correction_factor"  # Correction factor for calculations
    DATETIMES = "datetimes"
    DEMAND = "demand"
    DHW_ACTIVITY = "dhw_activity"  # DHW activity filename
    DHW_DEMAND_PER_SIZE = "dhw_demand_per_size[m2]"  # DHW demand per size, i.e. m2
    DHW_DEMAND_PER_PERSON = "dhw_demand_per_person"  # DHW demand per person
    DWELLING_SIZE = "dwelling_size[m2]"  # Size of dwelling (m2)
    FILE = "filename"  # File
    GAINS_INTERNAL = "gains_internal[W]"  # Internal gains (W)
    GAINS_INTERNAL_COL = f"{GAINS_INTERNAL}_{COLUMN}"  # column in which the internal gains are (dataframe provided)
    GAINS_INTERNAL_PER_PERSON = f"{GAINS_INTERNAL}_per_person[W]"  # Internal gains per person (W)
    GAINS_SOLAR = "gains_solar[W]"  # Solar gains (W)
    GEN_MAX = "maximum_generation[W]"  # Maximum generation (W)
    GRADIENT_SINK = "gradient_sink"  # Gradient of the sink temperature
    HEIGHT = "height[m]"  # Height (m)
    HOUSEHOLD_TYPE = "household_type"  # Type of household
    HOLIDAYS_LOCATION = "holidays_location"  # Location from which to get the holidays from (e.g. BY,DE)
    HP_SINK = "hp_sink"  # Heat pump sink type (floor, radiator, water)
    HP_SOURCE = "hp_source"  # Heat pump source type (air, soil, water)
    HP_SYSTEM = "hp_system"  # HP system configuration
    HUB_HEIGHT = "hub_height[m]"  # Hub height (wind)
    INHABITANTS = "inhabitants"  # Number of inhabitants
    LAT = "latitude[º]"  # Latitude (degree north)
    LOAD = "load[W]"
    LOAD_BASE = f"{LOAD}_base[W]"  # Base load
    LOAD_MAX = f"{LOAD}_max[W]"
    LON = "longitude[º]"  # Longitude (degree north)
    OCCUPATION = "occupation"  # Occupancy (for timeseries files)
    OCCUPANTS = "occupants"  # Number of occupants
    ORIENTATION = "orientation[º]"  # Orientation (degrees; 180 = south)
    POWER = "power[W]"  # Power (W)
    POWER_COOLING = f"{POWER}_cooling[W]"  # Power cooling (W)
    POWER_HEATING = f"{POWER}_heating[W]"  # Power heating (W)
    PV_ARRAYS = "pv_arrays"  # PV array configuration (dict)
    PV_INVERTER = "pv_inverter"  # PV inverter configuration (dict)
    RESISTANCE = "resistance[K W-1]"  # Resistance (K/W)
    SEASONAL_PEAK_DAY = "seasonal_peak_day"  # Day of year with peak demand
    SEASONAL_VARIATION = "seasonal_variation"  # Seasonal variation factor
    SEED = "seed"  # Seed to ensure reproducibility
    SOURCE = "source"  # Source of data or method
    TEMP = "temperature[C]"  # Temperature (°C)
    TEMP_WATER = f"water_{TEMP}[C]"  # Water temperature (°C)
    TEMP_WATER_COLD = f"cold_{TEMP_WATER}[C]"  # Cold water temperature (°C)
    TEMP_WATER_HOT = f"hot_{TEMP_WATER}[C]"  # Hot water temperature (°C)
    TEMP_INIT = f"init_{TEMP}[C]"  # Initial temperature (°C)
    TEMP_MAX = f"max_{TEMP}[C]"  # Maximum temperature (°C)
    TEMP_MIN = f"min_{TEMP}[C]"  # Minimum temperature (°C)
    TEMP_SET = f"set_{TEMP}[C]"  # Set temperature (°C)
    TEMP_SINK = f"sink_{TEMP}[C]"  # Heat pump sink temperature setting (°C)
    THERMAL_INERTIA = "thermal_inertia"  # Thermal inertia [0, 1]
    TILT = "tilt[º]"  # Tilt (degrees; 0 = horizontal)
    TRANSMITTANCE = "transmittance[W m-2 K-1]"  # Transmittance (W/m2/K)
    TURBINE_TYPE = "turbine_type"  # Turbine type (wind)
    VENTILATION = "ventilation[W K-1]"  # Ventilation losses (W/K)
    VENTILATION_COL = f"{VENTILATION}_{COLUMN}"
    VENTILATION_FACTOR = f"{VENTILATION}_factor[h-1]"  # 1/h
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
        CORRECTION_FACTOR: float | int | np.number,
        DATETIMES: str,
        DEMAND: int | float | np.number,
        DHW_ACTIVITY: str,
        DHW_DEMAND_PER_SIZE: str,
        DWELLING_SIZE: int | float | np.number,
        FILE: str,
        GAINS_INTERNAL: int | float | np.number | str,
        GAINS_INTERNAL_COL: str,
        GAINS_SOLAR: int | float | np.number | str,
        GRADIENT_SINK: int | float | np.number,
        HOUSEHOLD_TYPE: str,
        HOLIDAYS_LOCATION: str,
        HP_SINK: str,
        HP_SOURCE: str,
        HP_SYSTEM: str,
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
        TEMP_WATER: int | float | np.number,
        TEMP_WATER_COLD: int | float | np.number,
        TEMP_WATER_HOT: int | float | np.number,
        TEMP_INIT: int | float | np.number,
        TEMP_MAX: int | float | np.number,
        TEMP_MIN: int | float | np.number,
        TEMP_SET: int | float | np.number,
        TEMP_SINK: int | float | np.number,
        THERMAL_INERTIA: int | float | np.number,
        TILT: int | float | np.number,
        VENTILATION: int | float | np.number,
        VERBOSE: bool,
        WEATHER: str,
        WINDOWS: str,
    }
