from enum import Enum


class Units(Enum):
    # Length
    METER = "m"
    CENTIMETER = "cm"
    MILLIMETER = "mm"
    KILOMETER = "km"

    # Area
    SQUARE_METER = "m2"
    SQUARE_CENTIMETER = "cm2"
    SQUARE_MILLIMETER = "mm2"
    HECTARE = "ha"

    # Volume
    CUBIC_METER = "m3"
    LITER = "l"
    MILLILITER = "ml"

    # Temperature
    CELSIUS = "C"
    KELVIN = "K"
    FAHRENHEIT = "F"

    # Mass
    KILOGRAM = "kg"
    GRAM = "g"
    MILLIGRAM = "mg"

    # Time
    SECOND = "s"
    MINUTE = "min"
    HOUR = "h"

    # Speed
    METER_PER_SECOND = "m s-1"
    KILOMETER_PER_HOUR = "km h-1"

    # Pressure
    PASCAL = "Pa"
    BAR = "bar"
    ATMOSPHERE = "atm"

    # Volume flow
    CUBIC_METER_PER_SECOND = "m3 s-1"
    LITER_PER_SECOND = "l s-1"

    # Energy
    JOULE = "J"
    WATT_HOUR = "Wh"
    KILOWATT_HOUR = "kWh"
    MEGAWATT_HOUR = "MWh"
    GIGAWATT_HOUR = "GWh"
    CALORIE = "cal"

    # Power
    WATT = "W"
    KILOWATT = "kW"
    MEGAWATT = "MW"
    GIGAWATT = "GW"
