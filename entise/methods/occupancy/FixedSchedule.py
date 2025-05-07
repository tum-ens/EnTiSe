import pandas as pd

from entise.core.base import Method
from entise.constants import Columns as C, Objects as O, Keys as K, Types

class FixedSchedule(Method):
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
