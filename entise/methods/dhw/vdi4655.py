"""
VDI 4655 DHW (Domestic Hot Water) methods.

This module implements DHW methods based on VDI 4655:
"Reference load profiles of single-family and multi-family houses for the use of CHP systems"

Source: VDI 4655. (2008). Reference load profiles of single-family and multi-family houses 
for the use of CHP systems. Verein Deutscher Ingenieure (Association of German Engineers).
URL: https://www.vdi.de/richtlinien/details/vdi-4655-reference-load-profiles-of-single-family-and-multi-family-houses-for-the-use-of-chp-systems
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from entise.methods.dhw.base import BaseProbabilisticDHW
from entise.constants import Columns as C, Objects as O, Keys as K, Types

logger = logging.getLogger(__name__)

class VDI4655ColdWaterTemperatureDHW(BaseProbabilisticDHW):
    """
    Mixin class for DHW methods that provides seasonal cold water temperature variations from VDI 4655.

    This class is intended to be used as a mixin with another DHW class that implements
    the _calculate_daily_demand method. It adjusts the energy demand based on monthly 
    cold water temperature variations.

    Example usage:
    -------------
    class CombinedDHW(VDI4655ColdWaterTemperatureDHW, IEAAnnex42HouseholdTypeDHW):
        name = "CombinedDHW"
        # This class will use IEAAnnex42HouseholdTypeDHW for demand calculation
        # and VDI4655ColdWaterTemperatureDHW for cold water temperature variations
    """
    name = "VDI4655ColdWaterTemperatureDHW"
    optional_keys = BaseProbabilisticDHW.optional_keys + [O.TEMP_COLD]

    def generate(self, obj, data, ts_type):
        """
        Generate DHW demand time series with seasonal cold water temperature variations.

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
        dhw_activity_file = obj.get(O.DHW_ACTIVITY_FILE, None)
        temp_hot = obj.get(O.TEMP_HOT, DEFAULT_TEMP_HOT)
        seasonal_variation = obj.get(O.SEASONAL_VARIATION, DEFAULT_SEASONAL_VARIATION)
        seasonal_peak_day = obj.get(O.SEASONAL_PEAK_DAY, DEFAULT_SEASONAL_PEAK_DAY)

        # Set default activity file if not provided
        if dhw_activity_file is None:
            dhw_activity_file = self.get_default_activity_file()

        # Load activity data
        activity_data = pd.read_csv(dhw_activity_file)

        # Calculate daily demand
        daily_demand = self._calculate_daily_demand(obj, data)

        # Load cold water temperature data
        cold_water_temp_file = self.get_data_file('vdi4655', 'cold_water_temperature.csv')
        cold_water_temp_data = pd.read_csv(cold_water_temp_file)

        # Generate time series
        ts_volume, ts_energy = self._generate_timeseries_with_variable_temp(
            weather, 
            activity_data, 
            daily_demand, 
            cold_water_temp_data,
            temp_hot,
            seasonal_variation,
            seasonal_peak_day
        )

        # Calculate summary
        total_volume = ts_volume.sum()
        total_energy = ts_energy.sum() / 1000  # Convert from Wh to kWh

        # Create output
        summary = {
            f'{C.DEMAND}_{Types.DHW}_volume': float(total_volume),
            f'{C.DEMAND}_{Types.DHW}_energy': float(total_energy),
        }

        timeseries = pd.DataFrame({
            f'{C.LOAD}_{Types.DHW}_volume': ts_volume,
            f'{C.LOAD}_{Types.DHW}_energy': ts_energy,
        }, index=weather.index)

        return {
            "summary": summary,
            "timeseries": timeseries
        }

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

    def _generate_timeseries_with_variable_temp(self, weather, activity_data, daily_demand, 
                                               cold_water_temp_data, temp_hot, 
                                               seasonal_variation, seasonal_peak_day):
        """
        Generate DHW demand time series with variable cold water temperature.

        Parameters:
        -----------
        weather : pd.DataFrame
            Weather data with datetime index
        activity_data : pd.DataFrame
            Activity data
        daily_demand : float
            Daily demand in liters
        cold_water_temp_data : pd.DataFrame
            Cold water temperature data by month
        temp_hot : float
            Hot water temperature in °C
        seasonal_variation : float
            Seasonal variation factor
        seasonal_peak_day : int
            Day of year with peak demand

        Returns:
        --------
        tuple
            (volume_timeseries, energy_timeseries)
        """
        # Create empty time series
        index = weather.index
        ts_volume = pd.Series(0.0, index=index)
        ts_energy = pd.Series(0.0, index=index)

        # Process each day
        for day_start in pd.date_range(index[0].date(), index[-1].date()):
            day_end = day_start + pd.Timedelta(days=1)

            # Get day of week (0 = Monday, 6 = Sunday)
            day_of_week = day_start.weekday()

            # Apply seasonal variation
            day_of_year = day_start.timetuple().tm_yday
            seasonal_factor = 1 + seasonal_variation * np.cos(2 * np.pi * (day_of_year - seasonal_peak_day) / 365)
            daily_demand_adjusted = daily_demand * seasonal_factor

            # Get activity data for this day of week
            day_activities = activity_data[activity_data['day'] == day_of_week]

            # Calculate total probability for normalization
            total_prob = day_activities['probability'].sum()

            # Get cold water temperature for this month
            month = day_start.month
            cold_water_temp = cold_water_temp_data[cold_water_temp_data['month'] == month]['temperature_c'].values[0]

            # Calculate energy factor for this month
            delta_t = temp_hot - cold_water_temp
            energy_factor = DEFAULT_DENSITY_WATER * DEFAULT_SPECIFIC_HEAT_WATER * delta_t / 3600  # Convert to Wh

            # Distribute daily demand according to activity probabilities
            for _, activity in day_activities.iterrows():
                # Calculate volume for this activity
                activity_volume = daily_demand_adjusted * activity['probability'] / total_prob

                # Calculate time for this activity
                activity_time = pd.Timestamp(
                    year=day_start.year,
                    month=day_start.month,
                    day=day_start.day,
                    hour=int(activity['time'].split(':')[0]),
                    minute=int(activity['time'].split(':')[1]),
                    second=int(activity['time'].split(':')[2]) if len(activity['time'].split(':')) > 2 else 0
                )

                # Skip if outside time range
                if activity_time < index[0] or activity_time > index[-1]:
                    continue

                # Find closest time in index
                closest_idx = index.get_indexer([activity_time], method='nearest')[0]

                # Add volume to time series
                ts_volume.iloc[closest_idx] += activity_volume

                # Add energy to time series
                ts_energy.iloc[closest_idx] += activity_volume * energy_factor

        return ts_volume, ts_energy

# Import default values from base module
from entise.methods.dhw.base import (
    DEFAULT_TEMP_COLD,
    DEFAULT_TEMP_HOT,
    DEFAULT_DENSITY_WATER,
    DEFAULT_SPECIFIC_HEAT_WATER,
    DEFAULT_SEASONAL_VARIATION,
    DEFAULT_SEASONAL_PEAK_DAY
)
