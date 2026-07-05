from enum import Enum


class Constants(Enum):
    DEFAULT_HEIGHT = 2.5
    DEFAULT_AREA = 1
    DEFAULT_NIGHT_SCHEDULE = True  # GeoMA/PHT
    DEFAULT_NIGHT_SCHEDULE_START = "20:00"
    DEFAULT_NIGHT_SCHEDULE_END = "23:59"
    DEFAULT_LAMBDA = 0.05  # GeoMA/PHT
    DEFAULT_DETECTION_THRESHOLD = 0.3


class UnitConversion(Enum):
    CELSIUS2KELVIN = 273.15
    KELVIN2CELSIUS = -273.15


# --- Physical properties of dry air at typical building-simulation conditions.
# Single source of truth so ventilation / RC-solver / latent-cooling code all
# agree. Any refinement (e.g. bumping CP_AIR to 1005 J/(kg·K)) must land here.
CP_AIR: float = 1000.0  # J/(kg·K), specific heat capacity at constant pressure
RHO_AIR: float = 1.2  # kg/m³, air density at ~20 °C, sea level
