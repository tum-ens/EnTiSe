"""
Hendron & Burch DHW (Domestic Hot Water) methods.

This module implements DHW methods based on Hendron & Burch (2007):
"Development of Standardized Domestic Hot Water Event Schedules for Residential Buildings (NREL)"
"""

import os
import logging
import numpy as np
import pandas as pd

from entise.methods.dhw.base import BaseProbabilisticDHW
from entise.constants import Columns as C, Objects as O, Keys as K, Types

logger = logging.getLogger(__name__)

class HendronBurchOccupantsDHW(BaseProbabilisticDHW):
    """
    Probabilistic DHW demand method based on number of occupants from Hendron & Burch (2007).
    
    This method calculates daily DHW demand based on the number of occupants in the dwelling.
    """
    name = "HendronBurchOccupantsDHW"
    required_keys = BaseProbabilisticDHW.required_keys + [O.OCCUPANTS]
    optional_keys = BaseProbabilisticDHW.optional_keys + [O.DHW_DEMAND_FILE]

    def _calculate_daily_demand(self, obj, data):
        """
        Calculate daily demand based on number of occupants.

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
        occupants = obj[O.OCCUPANTS]
        dhw_demand_file = obj.get(O.DHW_DEMAND_FILE, None)
        
        # Get the demand file with fallback mechanism
        demand_file = self.get_data_file(
            'hendron_burch', 
            'dhw_demand_by_occupants.csv',
            dhw_demand_file
        )
        
        # Load demand data
        demand_data = pd.read_csv(demand_file)
        
        # Find closest number of occupants in data
        occupants_values = demand_data['occupants'].values
        idx = np.abs(occupants_values - occupants).argmin()
        liters_per_day = demand_data.iloc[idx]['liters_per_day']
        sigma = demand_data.iloc[idx]['sigma']

        # Add random variation based on sigma
        daily_demand_l = np.random.normal(liters_per_day, sigma)

        return max(0, daily_demand_l)  # Ensure non-negative demand

    def get_default_activity_file(self):
        """
        Get the default activity file path.
        
        Returns:
        --------
        str
            Path to the default activity file
        """
        # Use Jordan & Vajen activity profiles as default
        return os.path.join('entise', 'data', 'dhw', 'jordan_vajen', 'dhw_activity.csv')