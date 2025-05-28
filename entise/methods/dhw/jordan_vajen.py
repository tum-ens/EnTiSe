"""
Jordan & Vajen DHW (Domestic Hot Water) method.

This module implements a DHW method based on Jordan & Vajen (2005):
"DHWcalc: PROGRAM TO GENERATE DOMESTIC HOT WATER PROFILES WITH STATISTICAL MEANS FOR USER DEFINED CONDITIONS"

Source: Jordan, U., & Vajen, K. (2005). DHWcalc: PROGRAM TO GENERATE DOMESTIC HOT WATER PROFILES WITH STATISTICAL MEANS FOR USER DEFINED CONDITIONS.
Universität Marburg.
URL: https://www.researchgate.net/publication/237651871_DHWcalc_PROGRAM_TO_GENERATE_DOMESTIC_HOT_WATER_PROFILES_WITH_STATISTICAL_MEANS_FOR_USER_DEFINED_CONDITIONS
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import scipy.stats as stats

from entise.core.base import Method
from entise.constants import Columns as C, Objects as O, Types
import entise.methods.dhw.defaults as defaults
from entise.methods.dhw.jordanvajen.activity import _get_activity_data, _get_demand_data
from entise.methods.dhw.jordanvajen.temperature import _get_water_temperatures
from entise.methods.dhw.jordanvajen.calculation import _calculate_timeseries

logger = logging.getLogger(__name__)


class JordanVajen(Method):
    """
    Jordan & Vajen DHW method based on dwelling size.

    This method calculates domestic hot water demand time series based on the size of the dwelling
    using the Jordan & Vajen (2005) methodology. It distributes daily demand according to activity
    profiles provided in the activity data filename.

    Source: Jordan, U., & Vajen, K. (2005). DHWcalc: PROGRAM TO GENERATE DOMESTIC HOT WATER PROFILES
    WITH STATISTICAL MEANS FOR USER DEFINED CONDITIONS. Universität Marburg.
    URL: https://www.researchgate.net/publication/237651871_DHWcalc_PROGRAM_TO_GENERATE_DOMESTIC_HOT_WATER_PROFILES_WITH_STATISTICAL_MEANS_FOR_USER_DEFINED_CONDITIONS
    """
    types = [Types.DHW]
    name = "jordanvajen"
    required_keys = [O.DATETIMES, O.DWELLING_SIZE]
    optional_keys = [
        O.DHW_ACTIVITY,
        O.DHW_DEMAND_PER_SIZE,
        O.HOLIDAYS_LOCATION,
        O.TEMP_WATER_COLD,
        O.TEMP_WATER_HOT,
        O.SEASONAL_VARIATION,
        O.SEASONAL_PEAK_DAY,
        O.SEED,
    ]
    required_timeseries = [O.DATETIMES]
    optional_timeseries = [
        O.DHW_ACTIVITY,
        O.DHW_DEMAND_PER_SIZE,
        O.TEMP_WATER_COLD,
        O.TEMP_WATER_HOT,
    ]
    output_summary = {
        f'{C.DEMAND}_{Types.DHW}_volume_total': 'total hot water demand in liters',
        f'{C.DEMAND}_{Types.DHW}_volume_avg': 'average hot water demand in liters',
        f'{C.DEMAND}_{Types.DHW}_volume_peak': 'peak hot water demand in liters',
        f'{C.DEMAND}_{Types.DHW}_energy_total': 'total energy demand for hot water in Wh',
        f'{C.DEMAND}_{Types.DHW}_energy_avg': 'average energy demand for hot water in Wh',
        f'{C.DEMAND}_{Types.DHW}_energy_peak': 'peak energy demand for hot water in Wh',
        f'{C.DEMAND}_{Types.DHW}_power_avg': 'average power for hot water in W',
        f'{C.DEMAND}_{Types.DHW}_power_max': 'maximum power for hot water in W',
        f'{C.DEMAND}_{Types.DHW}_power_min': 'minimum power for hot water in W',
    }
    output_timeseries = {
        f'{C.LOAD}_{Types.DHW}_volume': 'hot water demand in liters',
        f'{C.LOAD}_{Types.DHW}_energy': 'energy demand for hot water in Wh',
        f'{C.LOAD}_{Types.DHW}_power': 'power demand for hot water in W',
        f'{Types.DHW}_{O.TEMP_WATER_COLD}': 'cold water temperature in degrees Celsius',
        f'{Types.DHW}_{O.TEMP_WATER_HOT}': 'hot water temperature in degrees Celsius',
    }

    def generate(self, obj, data, ts_type: str = Types.DHW):
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
        # Reproducible RNG
        seed = obj.get(O.SEED, None)
        rng = np.random.default_rng(seed)

        # Get parameters
        datetimes = obj[O.DATETIMES]
        datetimes = data[datetimes]
        datetimes[C.DATETIME] = (pd.to_datetime(datetimes[C.DATETIME], utc=True).dt
                                 .tz_convert(pd.to_datetime(datetimes[C.DATETIME].iloc[0]).tz))
        dwelling_size = obj[O.DWELLING_SIZE]

        # Get activity data
        activity_data = data.get(O.DHW_ACTIVITY, None)
        if activity_data is None:
            activity_data = _get_activity_data('jordan_vajen')

        # Obtain statistical data for yearly demand
        demand_data = _get_demand_data('jordan_vajen')
        sizes = demand_data['dwelling_size'].values
        idx = np.abs(sizes - dwelling_size).argmin()
        m3_per_m2_a = demand_data.iloc[idx]['m3_per_m2_a']
        sigma = demand_data.iloc[idx]['sigma']

        # Compute mean & std in litres/day
        mean_daily_l = m3_per_m2_a * dwelling_size * 1e3 / 365
        sd_daily_l = mean_daily_l * sigma

        # Define truncation bounds (no negatives)
        a, b = (0 - mean_daily_l) / sd_daily_l, np.inf

        # Build a date index for each simulation day
        start = datetimes[C.DATETIME].iloc[0].normalize()
        end = datetimes[C.DATETIME].iloc[-1].normalize()
        days = pd.date_range(start, end, freq='D')

        # 5) sample once **per day** from the truncated normal
        dist = stats.truncnorm(a, b, loc=mean_daily_l, scale=sd_daily_l)
        daily_demand_l = pd.Series(dist.rvs(size=len(days), random_state=rng), index=days)

        # Get cold water temperature
        water_temp = _get_water_temperatures(obj, data, datetimes)

        # Generate time series
        ts_volume, ts_energy = _calculate_timeseries(
            datetimes, activity_data, daily_demand_l, water_temp, obj
        )

        # Convert energy into power
        # Calculate time interval in hours
        time_diff = pd.Series(datetimes[C.DATETIME]).diff().dt.total_seconds() / 3600
        # Use the median time difference for the first element
        time_diff.iloc[0] = time_diff.median()
        # Calculate power as energy divided by time
        ts_power = ts_energy / time_diff.values

        # Create output
        summary = {
            f'{C.DEMAND}_{Types.DHW}_volume_total': int(ts_volume.sum().round(0)),
            f'{C.DEMAND}_{Types.DHW}_volume_avg': float(ts_volume.mean().round(3)),
            f'{C.DEMAND}_{Types.DHW}_volume_peak': float(ts_volume.max().round(3)),
            f'{C.DEMAND}_{Types.DHW}_energy_total': int(ts_energy.sum()),
            f'{C.DEMAND}_{Types.DHW}_energy_avg': int(ts_energy.mean().round(0)),
            f'{C.DEMAND}_{Types.DHW}_energy_peak': int(ts_energy.max()),
            f'{C.DEMAND}_{Types.DHW}_power_avg': int(ts_power.mean().round(0)),
            f'{C.DEMAND}_{Types.DHW}_power_max': int(ts_power.max()),
            f'{C.DEMAND}_{Types.DHW}_power_min': int(ts_power.min()),
        }

        timeseries = pd.DataFrame({
            f'{C.LOAD}_{Types.DHW}_volume': ts_volume,
            f'{C.LOAD}_{Types.DHW}_energy': ts_energy,
            f'{C.LOAD}_{Types.DHW}_power': ts_power,
        }, index=datetimes[C.DATETIME])

        return {
            "summary": summary,
            "timeseries": timeseries
        }


