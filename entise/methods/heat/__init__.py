"""
Heating methods for time series generation.

This package contains methods for generating heating demand or supply time series.
"""

from entise.methods.heat.demandlib import Demandlib

# Try to import DistrictHeatSim only if the dependency is available
try:
    from entise.methods.heat.districtheatingsim import DistrictHeatSim

    __all__ = [
        "Demandlib",
        "DistrictHeatSim",
    ]
except ImportError:
    # DistrictHeatSim is not available, only export Demandlib
    __all__ = [
        "Demandlib",
    ]
