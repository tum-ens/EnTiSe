"""
IEA Annex 42 DHW (Domestic Hot Water) methods.

This module implements DHW methods based on IEA Annex 42:
"The Simulation of Building-Integrated Fuel Cell and Other Cogeneration Systems"

Source: IEA/ECBCS Annex 42. (2007). The Simulation of Building-Integrated Fuel Cell and 
Other Cogeneration Systems. International Energy Agency.
URL: https://www.iea-ebc.org/projects/project?AnnexID=42
"""

import os
import logging
import numpy as np
import pandas as pd

from entise.methods.dhw.base import BaseProbabilisticDHW
from entise.constants import Columns as C, Objects as O, Keys as K, Types

logger = logging.getLogger(__name__)

class IEAAnnex42HouseholdTypeDHW(BaseProbabilisticDHW):
    """
    Probabilistic DHW demand method based on household type from IEA Annex 42.

    This method calculates daily DHW demand based on the type of household.
    """
    name = "IEAAnnex42HouseholdTypeDHW"
    required_keys = BaseProbabilisticDHW.required_keys + [O.HOUSEHOLD_TYPE]
    optional_keys = BaseProbabilisticDHW.optional_keys + [O.DHW_DEMAND_FILE]

    def _calculate_daily_demand(self, obj, data):
        """
        Calculate daily demand based on household type.

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
        household_type = obj[O.HOUSEHOLD_TYPE]
        dhw_demand_file = obj.get(O.DHW_DEMAND_FILE, None)

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
            liters_per_day = row['liters_per_day']
            sigma = row['sigma']
        except (IndexError, KeyError):
            # If household type not found, use the first row as default
            logger.warning(f"Household type '{household_type}' not found in demand data. Using default values.")
            liters_per_day = demand_data.iloc[0]['liters_per_day']
            sigma = demand_data.iloc[0]['sigma']

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
