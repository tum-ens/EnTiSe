"""
Jordan & Vajen DHW (Domestic Hot Water) method.

This module implements a DHW method based on Jordan & Vajen (2005):
"DHWcalc: PROGRAM TO GENERATE DOMESTIC HOT WATER PROFILES WITH STATISTICAL MEANS FOR USER DEFINED CONDITIONS"

Source: Jordan, U., & Vajen, K. (2005). DHWcalc: PROGRAM TO GENERATE DOMESTIC HOT WATER PROFILES WITH STATISTICAL MEANS FOR USER DEFINED CONDITIONS.
Universität Marburg.
URL: https://www.researchgate.net/publication/237651871_DHWcalc_PROGRAM_TO_GENERATE_DOMESTIC_HOT_WATER_PROFILES_WITH_STATISTICAL_MEANS_FOR_USER_DEFINED_CONDITIONS
"""

import os
import logging
import numpy as np
import pandas as pd

from entise.core.base import Method
from entise.constants import Columns as C, Objects as O, Keys as K, Types
from entise.methods.dhw.utils import (
    get_activity_data, get_demand_data, get_cold_water_temperature, calculate_timeseries,
    DEFAULT_TEMP_COLD, DEFAULT_TEMP_HOT, DEFAULT_SEASONAL_VARIATION, DEFAULT_SEASONAL_PEAK_DAY
)

logger = logging.getLogger(__name__)

DEFAULT_DWELLING_SIZE = 100  # m2


class JordanVajen(Method):
    """
    Jordan & Vajen DHW method based on dwelling size.

    This method calculates domestic hot water demand time series based on the size of the dwelling
    using the Jordan & Vajen (2005) methodology. It distributes daily demand according to activity
    profiles provided in the activity data file.

    Source: Jordan, U., & Vajen, K. (2005). DHWcalc: PROGRAM TO GENERATE DOMESTIC HOT WATER PROFILES 
    WITH STATISTICAL MEANS FOR USER DEFINED CONDITIONS. Universität Marburg.
    URL: https://www.researchgate.net/publication/237651871_DHWcalc_PROGRAM_TO_GENERATE_DOMESTIC_HOT_WATER_PROFILES_WITH_STATISTICAL_MEANS_FOR_USER_DEFINED_CONDITIONS
    """
    types = [Types.DHW]
    name = "JordanVajen"
    required_keys = [O.WEATHER, O.DWELLING_SIZE]
    optional_keys = [
        O.DHW_ACTIVITY_FILE, 
        O.DHW_DEMAND_FILE, 
        O.TEMP_COLD, 
        O.TEMP_HOT,
        O.SEASONAL_VARIATION, 
        O.SEASONAL_PEAK_DAY, 
        O.WEEKEND_ACTIVITY
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
        # Get parameters
        weather = data[O.WEATHER]
        dwelling_size = obj[O.DWELLING_SIZE]
        weekend_activity = obj.get(O.WEEKEND_ACTIVITY, False)
        use_seasonal_temp = obj.get(O.SOURCE, "").lower() == "vdi4655"

        # Get activity data
        activity_data = get_activity_data(obj, weekend_activity)

        # Calculate daily demand
        demand_data = get_demand_data('jordan_vajen', 'dhw_demand_by_dwelling.csv', obj)
        sizes = demand_data['dwelling_size'].values
        idx = np.abs(sizes - dwelling_size).argmin()
        m3_per_m2_a = demand_data.iloc[idx]['m3_per_m2_a']
        sigma = demand_data.iloc[idx]['sigma']

        # Calculate daily demand with random variation
        annual_demand_m3 = m3_per_m2_a * dwelling_size
        daily_demand_l = annual_demand_m3 * 1000 / 365
        daily_demand_l = np.random.normal(daily_demand_l, daily_demand_l * sigma)
        daily_demand_l = max(0, daily_demand_l)  # Ensure non-negative demand

        # Get cold water temperature
        cold_water_temp = get_cold_water_temperature(obj, weather, use_seasonal_temp)

        # Generate time series
        ts_volume, ts_energy = calculate_timeseries(
            weather, activity_data, daily_demand_l, cold_water_temp, obj
        )

        # Create output
        summary = {
            f'{C.DEMAND}_{Types.DHW}_volume_total': float(ts_volume.sum()),
            f'{C.DEMAND}_{Types.DHW}_volume_avg': float(ts_volume.mean()),
            f'{C.DEMAND}_{Types.DHW}_volume_peak': float(ts_volume.max()),
            f'{C.DEMAND}_{Types.DHW}_energy_total': float(ts_energy.sum()),
            f'{C.DEMAND}_{Types.DHW}_energy_avg': float(ts_energy.mean()),
            f'{C.DEMAND}_{Types.DHW}_energy_peak': float(ts_energy.max()),
        }

        timeseries = pd.DataFrame({
            f'{C.LOAD}_{Types.DHW}_volume': ts_volume,
            f'{C.LOAD}_{Types.DHW}_energy': ts_energy,
        }, index=weather.index)

        return {
            "summary": summary,
            "timeseries": timeseries
        }
