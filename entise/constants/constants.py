from enum import Enum


class Constants(Enum):
    DEFAULT_HEIGHT = 2.5
    DEFAULT_AREA = 1
    DEFAULT_NIGHT_SCHEDULE = True  # GeoMA
    DEFAULT_NIGHT_SCHEDULE_START = "20:00"
    DEFAULT_NIGHT_SCHEDULE_END = "23:59"
    DEFAULT_LAMBDA = 0.05  # GeoMA


class UnitConversion(Enum):
    CELSIUS2KELVIN = 273.15
    KELVIN2CELSIUS = -273.15
