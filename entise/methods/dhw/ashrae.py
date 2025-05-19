"""
ASHRAE DHW (Domestic Hot Water) methods.

This module implements DHW methods based on ASHRAE Handbook:
"HVAC Applications chapter on Service Water Heating"

Source: ASHRAE. (2019). ASHRAE Handbook - HVAC Applications. Chapter 51: Service Water Heating.
American Society of Heating, Refrigerating and Air-Conditioning Engineers.
URL: https://www.ashrae.org/technical-resources/ashrae-handbook
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
    Mixin class for DHW methods that provides weekend activity profiles based on ASHRAE Handbook.

    This class is intended to be used as a mixin with another DHW class that implements
    the _calculate_daily_demand method. It provides weekend activity profiles for DHW events.

    Example usage:
    -------------
    class CombinedDHW(ASHRAEWeekendActivityDHW, HendronBurchOccupantsDHW):
        name = "CombinedDHW"
        # This class will use HendronBurchOccupantsDHW for demand calculation
        # and ASHRAEWeekendActivityDHW for activity profiles
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

        This class is a mixin and does not implement this method.
        It should be used with another class that provides this implementation.

        Raises:
        -------
        NotImplementedError
            If this method is called directly without being overridden by a subclass
        """
        raise NotImplementedError("This class should be used as a mixin with a concrete demand calculation method.")
