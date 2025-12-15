"""
HVAC (Heating, Ventilation, and Air Conditioning) methods for time series generation.

This package contains methods for generating HVAC demand time series.
"""

# Import and expose the R1C1 class
from entise.methods.hvac.R1C1 import R1C1
from entise.methods.hvac.R5C1 import R5C1

# Expose the classes
__all__ = [
    "R1C1",
    "R5C1",
]
