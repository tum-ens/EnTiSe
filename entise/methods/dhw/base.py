"""
Base class for probabilistic DHW (Domestic Hot Water) methods.

This module provides the abstract base class for all probabilistic DHW methods.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from entise.core.base import Method
from entise.constants import Columns as C, Objects as O, Keys as K, Types

logger = logging.getLogger(__name__)

# Default values for optional keys
DEFAULT_TEMP_COLD = 10  # °C
DEFAULT_TEMP_HOT = 60   # °C
DEFAULT_DENSITY_WATER = 1000  # kg/m³
DEFAULT_SPECIFIC_HEAT_WATER = 4186  # J/(kg·K)
DEFAULT_SEASONAL_VARIATION = 0.1  # ±10% seasonal variation
DEFAULT_SEASONAL_PEAK_DAY = 15  # Day of year with peak demand (January 15)

class BaseProbabilisticDHW(Method, ABC):
    """
    Base class for probabilistic DHW demand methods.
    
    This abstract base class provides common functionality for all probabilistic DHW methods.
    Subclasses must implement the _calculate_daily_demand method.
    """
    types = [Types.DHW]
    required_keys = [O.WEATHER]
    optional_keys = [
        O.DHW_ACTIVITY_FILE,
        O.TEMP_COLD,
        O.TEMP_HOT,
        O.SEASONAL_VARIATION,
        O.SEASONAL_PEAK_DAY
    ]
    required_timeseries = [O.WEATHER]
    optional_timeseries = []
    output_summary = {
        f'{C.DEMAND}_{Types.DHW}_volume': 'total hot water demand in liters',
        f'{C.DEMAND}_{Types.DHW}_energy': 'total energy demand for hot water in kWh',
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
        dhw_activity_file = obj.get(O.DHW_ACTIVITY_FILE, None)
        temp_cold = obj.get(O.TEMP_COLD, DEFAULT_TEMP_COLD)
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

        # Generate time series
        ts_volume, ts_energy = self._generate_timeseries(
            weather, 
            activity_data, 
            daily_demand, 
            temp_cold, 
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

    @abstractmethod
    def _calculate_daily_demand(self, obj, data):
        """
        Calculate daily demand based on the specific method.
        
        This method must be implemented by subclasses.

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
        pass

    def get_default_activity_file(self):
        """
        Get the default activity file path.
        
        Subclasses can override this method to provide a different default activity file.

        Returns:
        --------
        str
            Path to the default activity file
        """
        return os.path.join('entise', 'data', 'dhw', 'jordan_vajen', 'dhw_activity.csv')

    def _generate_timeseries(self, weather, activity_data, daily_demand, temp_cold, temp_hot, 
                            seasonal_variation, seasonal_peak_day):
        """
        Generate DHW demand time series.

        Parameters:
        -----------
        weather : pd.DataFrame
            Weather data with datetime index
        activity_data : pd.DataFrame
            Activity data
        daily_demand : float
            Daily demand in liters
        temp_cold : float
            Cold water temperature in °C
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

        # Get time step in seconds
        time_step = (index[1] - index[0]).total_seconds()

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

        # Calculate energy demand
        delta_t = temp_hot - temp_cold
        energy_factor = DEFAULT_DENSITY_WATER * DEFAULT_SPECIFIC_HEAT_WATER * delta_t / 3600  # Convert to Wh
        ts_energy = ts_volume * energy_factor

        return ts_volume, ts_energy

    @classmethod
    def get_user_data_path(cls, filename):
        """Get path to a user-provided data file."""
        user_path = os.path.join('entise', 'data', 'dhw', 'user', filename)
        if os.path.exists(user_path):
            return user_path
        return None
    
    @classmethod
    def get_data_file(cls, source, filename, user_file=None):
        """
        Get path to a data file with fallback mechanism.
        
        Parameters:
        -----------
        source : str
            Source directory (e.g., 'jordan_vajen')
        filename : str
            Filename (e.g., 'dhw_demand_by_dwelling.csv')
        user_file : str, optional
            User-provided filename
            
        Returns:
        --------
        str
            Path to the data file
        """
        # First try user-provided file
        if user_file:
            if os.path.exists(user_file):
                return user_file
            user_path = cls.get_user_data_path(user_file)
            if user_path:
                return user_path
        
        # Then try user directory with default filename
        user_path = cls.get_user_data_path(filename)
        if user_path:
            return user_path
        
        # Finally use the source-specific file
        return os.path.join('entise', 'data', 'dhw', source, filename)