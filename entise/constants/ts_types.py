class Types:
    """Valid timeseries types."""
    COOLING = "cooling"  # Cooling demand or supply
    DHW = "dhw"  # Domestic Hot Water
    ELECTRICITY = "electricity"  # Electricity demand or supply
    HEATING = "heating"  # Heating demand or supply
    HVAC = "hvac"  # Heating, Ventilation, and Air Conditioning
    MOBILITY = "mobility"  # Transportation-related data
    OCCUPANCY = "occupancy"  # Occupancy data


# Export a set of all valid types for easy reference
VALID_TYPES = {value for key, value in Types.__dict__.items() if not key.startswith("__") and not callable(value)}
