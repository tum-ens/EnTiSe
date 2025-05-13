import pandas as pd

from entise.core.base import Method
from entise.constants import Columns as C, Objects as O, Keys as K, Types

class FixedSchedule(Method):
    """
    A simple occupancy method that generates a fixed schedule.

    This method creates an occupancy schedule where occupancy is 0 during working hours (8-18)
    and 1 otherwise. It's useful for simple simulations where a basic occupancy pattern is needed.

    The method requires weather data only to establish the time index for the occupancy schedule.
    """
    types = [Types.OCCUPANCY]
    name = "FixedSchedule"
    required_keys = [O.ID, O.WEATHER]
    required_timeseries = [O.WEATHER]
    output_summary = {
            f'Average_{O.OCCUPATION}': 'average occupancy over time',
    }
    output_timeseries = {
            f'{O.OCCUPATION}': 'pu occupancy per point in time',
    }

    def generate(self, obj, data, ts_type=None):
        """
        Generate a fixed occupancy schedule based on time of day.

        This method creates an occupancy schedule where occupancy is 0 during working hours (8-18)
        and 1 otherwise.

        Args:
            obj (dict): Object metadata and parameters.
            data (dict): Dictionary of available timeseries data.
            ts_type (str, optional): The timeseries type being processed.

        Returns:
            dict: A dictionary containing:
                - summary: Dictionary with average occupancy metrics
                - timeseries: DataFrame with occupancy values for each timestep
        """
        index = data[O.WEATHER].index
        occ = [0 if 8 <= t.hour < 18 else 1 for t in index]
        df = pd.DataFrame({f'{O.OCCUPATION}': occ}, index=index)
        df.index.name = C.DATETIME
        return {
            "summary": {
                f'average_{O.OCCUPATION}': df[f'{O.OCCUPATION}'].mean().round(2),
            },
            "timeseries": df
        }
