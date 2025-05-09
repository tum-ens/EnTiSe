"""
DHW (Domestic Hot Water) methods for time series generation.

This package contains methods for generating domestic hot water demand time series.
"""

# Import the main facade class
from entise.methods.dhw.probabilistic import ProbabilisticDHW

# Import the base class
from entise.methods.dhw.base import BaseProbabilisticDHW

# Import source-specific methods
from entise.methods.dhw.jordan_vajen import JordanVajenDwellingSizeDHW, JordanVajenWeekdayActivityDHW
from entise.methods.dhw.hendron_burch import HendronBurchOccupantsDHW
from entise.methods.dhw.iea_annex42 import IEAAnnex42HouseholdTypeDHW
from entise.methods.dhw.ashrae import ASHRAEWeekendActivityDHW
from entise.methods.dhw.vdi4655 import VDI4655ColdWaterTemperatureDHW
from entise.methods.dhw.user import UserDefinedDHW

__all__ = [
    # Main facade class
    'ProbabilisticDHW',
    
    # Base class
    'BaseProbabilisticDHW',
    
    # Jordan & Vajen methods
    'JordanVajenDwellingSizeDHW',
    'JordanVajenWeekdayActivityDHW',
    
    # Hendron & Burch methods
    'HendronBurchOccupantsDHW',
    
    # IEA Annex 42 methods
    'IEAAnnex42HouseholdTypeDHW',
    
    # ASHRAE methods
    'ASHRAEWeekendActivityDHW',
    
    # VDI 4655 methods
    'VDI4655ColdWaterTemperatureDHW',
    
    # User-defined methods
    'UserDefinedDHW'
]