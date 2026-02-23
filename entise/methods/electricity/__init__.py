"""
Electricity methods for time series generation.

This package contains methods for generating electricity demand or supply time series.
"""

from entise.methods.electricity.demandlib import Demandlib
from entise.methods.electricity.pylpg import PyLPG

__all__ = [
    "Demandlib",
    "PyLPG",
]