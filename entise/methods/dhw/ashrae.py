"""
ASHRAE DHW (Domestic Hot Water) methods.

This module implements DHW methods based on ASHRAE Handbook:
"HVAC Applications chapter on Service Water Heating"
"""

import os
import logging
import numpy as np
import pandas as pd

from entise.methods.dhw.base import BaseProbabilisticDHW
from entise.constants import Columns as C, Objects as O, Keys as K, Types

logger = logging.getLogger(__name__)

class ASHRAEWeekendActivityDHW(BaseProbabilisticDHW):
    """
    Probabilistic DHW demand method with weekend activity profiles based on ASHRAE Handbook.
    
    This method uses different activity profiles for weekends.
    """
    name = "ASHRAEWeekendActivityDHW"

    def get_default_activity_file(self):
        """
        Get the default activity file path.
        
        This method overrides the base class method to provide a different default activity file for weekends.

        Returns:
        --------
        str
            Path to the default activity file
        """
        return os.path.join('entise', 'data', 'dhw', 'ashrae', 'dhw_activity_weekend.csv')

    def _calculate_daily_demand(self, obj, data):
        """
        This method must be implemented by a concrete subclass.
        """
        raise NotImplementedError("This class should be used as a mixin with a concrete demand calculation method.")