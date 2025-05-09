"""
Jordan & Vajen DHW (Domestic Hot Water) methods.

This module implements DHW methods based on Jordan & Vajen (2001):
"Realistic Domestic Hot-Water Profiles in Different Time Scales"
"""

import os
import logging
import numpy as np
import pandas as pd

from entise.methods.dhw.base import BaseProbabilisticDHW
from entise.constants import Columns as C, Objects as O, Keys as K, Types

logger = logging.getLogger(__name__)

class JordanVajenDwellingSizeDHW(BaseProbabilisticDHW):
    """
    Probabilistic DHW demand method based on dwelling size from Jordan & Vajen (2001).
    
    This method calculates daily DHW demand based on the size of the dwelling.
    """
    name = "JordanVajenDwellingSizeDHW"
    required_keys = BaseProbabilisticDHW.required_keys + [O.DWELLING_SIZE]
    optional_keys = BaseProbabilisticDHW.optional_keys + [O.DHW_DEMAND_FILE]

    def _calculate_daily_demand(self, obj, data):
        """
        Calculate daily demand based on dwelling size.

        Parameters:
        -----------
        obj : dict
            Object parameters
        data : dict
            Input data

        Returns:
        --------
        float
            Daily demand in liters
        """
        dwelling_size = obj[O.DWELLING_SIZE]
        dhw_demand_file = obj.get(O.DHW_DEMAND_FILE, None)
        
        # Get the demand file with fallback mechanism
        demand_file = self.get_data_file(
            'jordan_vajen', 
            'dhw_demand_by_dwelling.csv',
            dhw_demand_file
        )
        
        # Load demand data
        demand_data = pd.read_csv(demand_file)
        
        # Find closest dwelling size in data
        sizes = demand_data['dwelling_size'].values
        idx = np.abs(sizes - dwelling_size).argmin()
        m3_per_m2_a = demand_data.iloc[idx]['m3_per_m2_a']
        sigma = demand_data.iloc[idx]['sigma']

        # Calculate daily demand with random variation
        annual_demand_m3 = m3_per_m2_a * dwelling_size
        daily_demand_l = annual_demand_m3 * 1000 / 365

        # Add random variation based on sigma
        daily_demand_l = np.random.normal(daily_demand_l, daily_demand_l * sigma)

        return max(0, daily_demand_l)  # Ensure non-negative demand

    def get_default_activity_file(self):
        """
        Get the default activity file path.
        
        Returns:
        --------
        str
            Path to the default activity file
        """
        return os.path.join('entise', 'data', 'dhw', 'jordan_vajen', 'dhw_activity.csv')


class JordanVajenWeekdayActivityDHW(BaseProbabilisticDHW):
    """
    Probabilistic DHW demand method with weekday activity profiles from Jordan & Vajen (2001).
    
    This method uses weekday activity profiles for DHW events.
    """
    name = "JordanVajenWeekdayActivityDHW"

    def get_default_activity_file(self):
        """
        Get the default activity file path.
        
        Returns:
        --------
        str
            Path to the default activity file
        """
        return os.path.join('entise', 'data', 'dhw', 'jordan_vajen', 'dhw_activity.csv')

    def _calculate_daily_demand(self, obj, data):
        """
        This method must be implemented by a concrete subclass.
        """
        raise NotImplementedError("This class should be used as a mixin with a concrete demand calculation method.")