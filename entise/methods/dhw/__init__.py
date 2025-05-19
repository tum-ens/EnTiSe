"""
DHW (Domestic Hot Water) methods for time series generation.

This package contains methods for generating domestic hot water demand time series.
"""

# Import the main facade class
from entise.methods.dhw.probabilistic import ProbabilisticDHW

# Import the Jordan & Vajen method
from entise.methods.dhw.jordan_vajen import JordanVajen

# Import utility functions
from entise.methods.dhw.utils import (
    get_activity_data, get_demand_data, get_cold_water_temperature, calculate_timeseries,
    DEFAULT_TEMP_COLD, DEFAULT_TEMP_HOT, DEFAULT_SEASONAL_VARIATION, DEFAULT_SEASONAL_PEAK_DAY
)

__all__ = [
    # Main facade class
    'ProbabilisticDHW',

    # Jordan & Vajen method
    'JordanVajen',

    # Utility functions
    'get_activity_data',
    'get_demand_data',
    'get_cold_water_temperature',
    'calculate_timeseries',

    # Default values
    'DEFAULT_TEMP_COLD',
    'DEFAULT_TEMP_HOT',
    'DEFAULT_SEASONAL_VARIATION',
    'DEFAULT_SEASONAL_PEAK_DAY'
]
