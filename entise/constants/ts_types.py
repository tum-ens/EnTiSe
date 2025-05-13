class Types:
    """Valid timeseries types."""
    BIOMASS = "biomass"  # Biomass energy
    COOLING = "cooling"  # Cooling demand or supply
    CSP = "csp"  # Concentrated Solar Power
    DHW = "dhw"  # Domestic Hot Water
    ELECTRICITY = "electricity"  # Electricity demand or supply
    GEOTHERMAL = "geothermal"  # Geothermal energy
    HEATING = "heating"  # Heating demand or supply
    HYDRO = "hydro"  # Hydroelectric power
    HVAC = "hvac"  # Heating, Ventilation, and Air Conditioning
    MOBILITY = "mobility"  # Transportation-related data
    OCCUPANCY = "occupancy"  # Occupancy data
    SOLAR_PV = "solar_pv"  # Solar Photovoltaic
    TIDAL = "tidal"  # Tidal energy
    WAVE = "wave"  # Wave energy
    WIND = "wind"  # Wind energy


# Export a set of all valid types for easy reference
VALID_TYPES = {value for key, value in Types.__dict__.items() if not key.startswith("__") and not callable(value)}
