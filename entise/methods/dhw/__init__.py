"""
DHW (Domestic Hot Water) methods for time series generation.

This package contains methods for generating domestic hot water demand time series.
"""

# Import and expose the JordanVajen class
from entise.methods.dhw.jordan_vajen import JordanVajen

# Expose the classes
__all__ = ['JordanVajen']
