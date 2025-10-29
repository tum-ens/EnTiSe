"""
Heat Pump methods for time series generation.

This package contains methods for generating heat pump COP time series.
"""

# Import and expose the Ruhnau class
from entise.methods.hp.ruhnau import Ruhnau

# Expose the classes
__all__ = ["Ruhnau"]
