"""
Probabilistic DHW (Domestic Hot Water) method for time series generation.

This module implements a probabilistic method for generating domestic hot water demand time series
based on dwelling size using the Jordan & Vajen methodology.

This is a simplified version that only uses the JordanVajen class.
"""

import os
import logging

from entise.core.base import Method
from entise.constants import Columns as C, Objects as O, Keys as K, Types

# Import the JordanVajen class
from entise.methods.dhw.jordan_vajen import JordanVajen

logger = logging.getLogger(__name__)

class ProbabilisticDHW(Method):
    """
    Probabilistic DHW demand method.

    This method generates domestic hot water demand time series using the Jordan & Vajen
    methodology based on dwelling size.

    This class serves as a facade that forwards to the JordanVajen class.
    """
    types = [Types.DHW]
    name = "DHWprobabilistic"
    required_keys = [O.WEATHER]
    optional_keys = [
        O.DWELLING_SIZE,
        O.DHW_DEMAND_FILE, 
        O.DHW_ACTIVITY_FILE,
        O.TEMP_COLD,
        O.TEMP_HOT,
        O.SEASONAL_VARIATION,
        O.SEASONAL_PEAK_DAY,
        O.WEEKEND_ACTIVITY,
        O.SOURCE
    ]
    required_timeseries = [O.WEATHER]
    optional_timeseries = []
    output_summary = {
        f'{C.DEMAND}_{Types.DHW}_volume_total': 'total hot water demand in liters',
        f'{C.DEMAND}_{Types.DHW}_volume_avg': 'average hot water demand in liters',
        f'{C.DEMAND}_{Types.DHW}_volume_peak': 'peak hot water demand in liters',
        f'{C.DEMAND}_{Types.DHW}_energy_total': 'total energy demand for hot water in Wh',
        f'{C.DEMAND}_{Types.DHW}_energy_avg': 'average energy demand for hot water in Wh',
        f'{C.DEMAND}_{Types.DHW}_energy_peak': 'peak energy demand for hot water in Wh',
    }
    output_timeseries = {
        f'{C.LOAD}_{Types.DHW}_volume': 'hot water demand in liters',
        f'{C.LOAD}_{Types.DHW}_energy': 'energy demand for hot water in W',
    }

    def generate(self, obj, data, ts_type):
        """
        Generate DHW demand time series.

        Parameters:
        -----------
        obj : dict
            Object parameters
        data : dict
            Input data
        ts_type : str
            Time series type

        Returns:
        --------
        dict
            Dictionary with summary and time series data
        """
        # If dwelling_size is not provided, use a default value
        if O.DWELLING_SIZE not in obj:
            logger.warning("No dwelling_size provided. Using default value of 100 m².")
            obj_copy = obj.copy()
            obj_copy[O.DWELLING_SIZE] = 100  # Default dwelling size
            obj = obj_copy

        # Create JordanVajen instance and generate time series
        method = JordanVajen()
        return method.generate(obj, data, ts_type)
