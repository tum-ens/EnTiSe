from entise.constants._namespace import Namespace


class Types(metaclass=Namespace):
    """Valid timeseries types."""

    # Add new types here as needed, sorted alphabetically for easy reference.
    # BIOMASS = "biomass"  # Biomass energy
    COOLING = "cooling"  # Cooling demand or supply
    # CSP = "csp"  # Concentrated Solar Power
    DHW = "dhw"  # Domestic Hot Water
    ELECTRICITY = "electricity"  # Electricity demand or supply
    # GEOTHERMAL = "geothermal"  # Geothermal energy
    HEATING = "heating"  # Heating demand or supply
    HP = "hp"  # Heat Pump COP
    # HYDRO = "hydro"  # Hydroelectric power
    HVAC = "hvac"  # Heating, Ventilation, and Air Conditioning
    MOBILITY = "mobility"  # Transportation-related data
    OCCUPANCY = "occupancy"  # Occupancy data
    PV = "pv"  # Solar Photovoltaic
    # TIDAL = "tidal"  # Tidal energy
    # WAVE = "wave"  # Wave energy
    WIND = "wind"  # Wind energy


# Export a set of all valid types for easy reference
VALID_TYPES = set(Types)
