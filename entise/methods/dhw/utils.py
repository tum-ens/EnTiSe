"""
Utility functions for DHW (Domestic Hot Water) methods.

This module provides common utility functions used by DHW methods.
"""

import os
import logging
import numpy as np
import pandas as pd

from entise.constants import Columns as C, Objects as O, Keys as K, Types

logger = logging.getLogger(__name__)

# Default values for optional keys
DEFAULT_TEMP_COLD = 10  # °C
DEFAULT_TEMP_HOT = 50   # °C
DEFAULT_DENSITY_WATER = 1000  # kg/m³
DEFAULT_SPECIFIC_HEAT_WATER = 4186  # J/(kg·K)
DEFAULT_SEASONAL_VARIATION = 0  # ±0% seasonal variation
DEFAULT_SEASONAL_PEAK_DAY = 15  # Day of year with peak demand (January 15)


def get_activity_data(obj, weekend_activity=False):
    """
    Get activity data based on parameters.
    
    Parameters:
    -----------
    obj : dict
        Object parameters
    weekend_activity : bool
        Whether to use weekend activity profiles
        
    Returns:
    --------
    pd.DataFrame
        Activity data
    """
    dhw_activity_file = obj.get(O.DHW_ACTIVITY_FILE)
    
    if dhw_activity_file and os.path.exists(dhw_activity_file):
        return pd.read_csv(dhw_activity_file)
    
    if weekend_activity:
        path = os.path.join('entise', 'data', 'dhw', 'ashrae', 'dhw_activity_weekend.csv')
    else:
        path = os.path.join('entise', 'data', 'dhw', 'jordan_vajen', 'dhw_activity.csv')
    
    return pd.read_csv(path)


def get_demand_data(source, filename, obj):
    """
    Get demand data from file with fallback mechanism.
    
    Parameters:
    -----------
    source : str
        Source directory (e.g., 'jordan_vajen')
    filename : str
        Filename (e.g., 'dhw_demand_by_dwelling.csv')
    obj : dict
        Object parameters
        
    Returns:
    --------
    pd.DataFrame
        Demand data
    """
    dhw_demand_file = obj.get(O.DHW_DEMAND_FILE)
    
    if dhw_demand_file and os.path.exists(dhw_demand_file):
        return pd.read_csv(dhw_demand_file)
    
    user_path = os.path.join('entise', 'data', 'dhw', 'user', filename)
    if os.path.exists(user_path):
        return pd.read_csv(user_path)
    
    return pd.read_csv(os.path.join('entise', 'data', 'dhw', source, filename))


def get_cold_water_temperature(obj, weather, use_seasonal_temp=False):
    """
    Get cold water temperature (constant or seasonal).
    
    Parameters:
    -----------
    obj : dict
        Object parameters
    weather : pd.DataFrame
        Weather data
    use_seasonal_temp : bool
        Whether to use seasonal temperature variations
        
    Returns:
    --------
    float or pd.Series
        Cold water temperature (constant or time series)
    """
    if not use_seasonal_temp:
        return obj.get(O.TEMP_COLD, DEFAULT_TEMP_COLD)
    
    # Use seasonal temperature from VDI4655
    temp_data = pd.read_csv(os.path.join('entise', 'data', 'dhw', 'vdi4655', 'cold_water_temperature.csv'))
    result = pd.Series(index=weather.index)
    
    for day_start in pd.date_range(weather.index[0].date(), weather.index[-1].date()):
        month = day_start.month
        temp = temp_data[temp_data['month'] == month]['temperature_c'].values[0]
        day_mask = (weather.index.date == day_start.date())
        result[day_mask] = temp
    
    return result


def calculate_timeseries(weather, activity_data, daily_demand, temp_cold, obj):
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
    temp_cold : float or pd.Series
        Cold water temperature in °C
    obj : dict
        Object parameters
        
    Returns:
    --------
    tuple
        (volume_timeseries, energy_timeseries)
    """
    # Get parameters
    temp_hot = obj.get(O.TEMP_HOT, DEFAULT_TEMP_HOT)
    seasonal_variation = obj.get(O.SEASONAL_VARIATION, DEFAULT_SEASONAL_VARIATION)
    seasonal_peak_day = obj.get(O.SEASONAL_PEAK_DAY, DEFAULT_SEASONAL_PEAK_DAY)
    
    # Create empty time series
    index = weather.index
    ts_volume = pd.Series(0.0, index=index)
    
    # Process each day
    for day_start in pd.date_range(index[0].date(), index[-1].date()):
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
    if isinstance(temp_cold, (int, float)):
        # Constant cold water temperature
        delta_t = temp_hot - temp_cold
        energy_factor = DEFAULT_DENSITY_WATER * DEFAULT_SPECIFIC_HEAT_WATER * delta_t / 3600  # Convert to Wh
        ts_energy = ts_volume * energy_factor
    else:
        # Variable cold water temperature
        ts_energy = pd.Series(0.0, index=index)
        for i in range(len(index)):
            delta_t = temp_hot - temp_cold.iloc[i]
            energy_factor = DEFAULT_DENSITY_WATER * DEFAULT_SPECIFIC_HEAT_WATER * delta_t / 3600
            ts_energy.iloc[i] = ts_volume.iloc[i] * energy_factor
    
    return ts_volume, ts_energy