"""
DHW (Domestic Hot Water) methods for time series generation.

This package contains methods for generating domestic hot water demand time series.
"""

# Import the Jordan & Vajen method
from entise.methods.dhw.jordan_vajen import JordanVajen


__all__ = [
    # Methods
    'JordanVajen',
]
