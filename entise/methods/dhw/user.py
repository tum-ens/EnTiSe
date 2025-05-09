"""
User-defined DHW (Domestic Hot Water) methods.

This module implements user-defined DHW methods that allow users to provide their own data files.
"""

import os
import logging
import numpy as np
import pandas as pd

from entise.methods.dhw.base import BaseProbabilisticDHW
from entise.constants import Columns as C, Objects as O, Keys as K, Types

logger = logging.getLogger(__name__)

class UserDefinedDHW(BaseProbabilisticDHW):
    """
    User-defined DHW demand method with fallbacks.
    
    This method tries to use user-provided data files first,
    but falls back to default data files if necessary.
    """
    name = "UserDefinedDHW"
    optional_keys = BaseProbabilisticDHW.optional_keys + [
        O.DWELLING_SIZE, 
        O.OCCUPANTS, 
        O.HOUSEHOLD_TYPE,
        O.DHW_DEMAND_FILE, 
        O.DHW_ACTIVITY_FILE
    ]
    
    def _calculate_daily_demand(self, obj, data):
        """
        Calculate daily demand based on available parameters with fallbacks.

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
        dhw_demand_file = obj.get(O.DHW_DEMAND_FILE, None)
        
        if O.DWELLING_SIZE in obj:
            # Calculate based on dwelling size
            dwelling_size = obj[O.DWELLING_SIZE]
            
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
            
        elif O.OCCUPANTS in obj:
            # Calculate based on number of occupants
            occupants = obj[O.OCCUPANTS]
            
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
            daily_demand_l = demand_data.iloc[idx]['liters_per_day']
            sigma = demand_data.iloc[idx]['sigma']
            
        elif O.HOUSEHOLD_TYPE in obj:
            # Calculate based on household type
            household_type = obj[O.HOUSEHOLD_TYPE]
            
            # Get the demand file with fallback mechanism
            demand_file = self.get_data_file(
                'iea_annex42', 
                'dhw_demand_by_household_type.csv',
                dhw_demand_file
            )
            
            # Load demand data
            demand_data = pd.read_csv(demand_file)
            
            # Find the row for this household type
            try:
                row = demand_data[demand_data['household_type'] == household_type].iloc[0]
                daily_demand_l = row['liters_per_day']
                sigma = row['sigma']
            except (IndexError, KeyError):
                # If household type not found, use the first row as default
                logger.warning(f"Household type '{household_type}' not found in demand data. Using default values.")
                daily_demand_l = demand_data.iloc[0]['liters_per_day']
                sigma = demand_data.iloc[0]['sigma']
                
        else:
            # If no specific parameters are provided, use a default value
            logger.warning("No dwelling_size, occupants, or household_type specified. Using default demand value.")
            daily_demand_l = 100  # Default value: 100 liters per day
            sigma = 20  # Default standard deviation
        
        # Add random variation based on sigma
        daily_demand_l = np.random.normal(daily_demand_l, sigma)
        
        return max(0, daily_demand_l)  # Ensure non-negative demand

    def get_default_activity_file(self):
        """
        Get the default activity file path.
        
        This method first checks for a user-provided activity file.

        Returns:
        --------
        str
            Path to the default activity file
        """
        # First check if there's a user-provided activity file
        user_activity_file = os.path.join('entise', 'data', 'dhw', 'user', 'dhw_activity.csv')
        if os.path.exists(user_activity_file):
            return user_activity_file
            
        # Otherwise use Jordan & Vajen activity profiles as default
        return os.path.join('entise', 'data', 'dhw', 'jordan_vajen', 'dhw_activity.csv')