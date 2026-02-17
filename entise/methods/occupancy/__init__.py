"""
Occupancy methods for time series generation.

This package contains methods for generating occupancy-related time series.
"""

from .geoma import GeoMA
from .pht import PHT

__all__ = ["GeoMA", "PHT"]
