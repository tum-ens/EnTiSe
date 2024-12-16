class Types:
    """Valid timeseries types."""
    HVAC = "hvac"  # Heating, Ventilation, and Air Conditioning
    DHW = "dhw"  # Domestic Hot Water
    ELECTRICITY = "electricity"  # Electricity demand or supply
    MOBILITY = "mobility"  # Transportation-related data
    OCCUPANCY = "occupancy"  # Occupancy data


# Export a set of all valid types for easy reference
VALID_TYPES = {value for key, value in Types.__dict__.items() if not key.startswith("__") and not callable(value)}
