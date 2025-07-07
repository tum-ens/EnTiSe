from entise.constants import Columns as C

# String definitions and mapping
# Define heat pump source types
HP_AIR = "air"  # Air Source Heat Pump
HP_SOIL = "soil"  # Ground Source Heat Pump
HP_WATER = "water"  # Water Source Heat Pump
SOURCES = [HP_AIR, HP_SOIL, HP_WATER]

# Define heat sink types
FLOOR = "floor"
RADIATOR = "radiator"
WATER = "water"

# Temperature column names
SOURCE_MAP = {HP_AIR: C.TEMP, HP_SOIL: C.TEMP_SOIL, HP_WATER: C.TEMP_WATER_GROUND}

# Default values for optional keys
DEFAULT_HP = HP_AIR
DEFAULT_HEAT = RADIATOR
