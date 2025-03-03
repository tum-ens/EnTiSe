import logging

import pandas as pd

from src.core.base import TimeSeriesMethod
from src.core.registry import Meta
from src.utils.decorators import supported_types
from src.constants import Columns as C, Keys as K, Objects as O, Types

logger = logging.getLogger(__name__)


@supported_types(Types.HVAC)
class DegreeDay(TimeSeriesMethod, metaclass=Meta):
    """Timeseries generation method using degree days."""
    # Define required keys for each method
    required_keys = {O.WEATHER: O.DTYPES[O.WEATHER],
                     O.LOAD_BASE: O.DTYPES[O.LOAD_BASE],
                     }
    # Define required timeseries for each method
    required_timeseries = {O.WEATHER:
                               {K.COLS_REQUIRED: {
                                    C.DATETIME: C.DTYPES[C.DATETIME],
                                    C.T_OUT: C.DTYPES[C.T_OUT],
                                    },
                                K.COLS_OPTIONAL: {},
                                K.DTYPE: pd.DataFrame
                                },
                           }
    # Define dependencies for each method
    dependencies = []

    def generate(self, obj: dict, data: dict, ts_type: str, **kwargs) -> (dict, pd.DataFrame):
        """
        Generate HVAC timeseries using degree days.

        Parameters:
        - obj (dict): Objects metadata and parameters.
        - data (dict): Timeseries data.

        Returns:
        - pd.DataFrame: Timeseries for HVAC demand.
        """
        ts_type = Types.HVAC
        # Prepare inputs
        obj = self.prepare_inputs(obj, data, ts_type)

        # Example thermal model logic
        weather = data[obj[O.WEATHER]]
        base_load = obj[O.LOAD_BASE]
        t_out = weather[C.T_OUT]

        # Example logic
        heating_demand = base_load * (18 - t_out).clip(lower=0)

        summary = {
            "heating_demand": heating_demand.sum(),
        }
        df = pd.DataFrame({
            C.DATETIME: weather[C.DATETIME],
            C.DEMAND: heating_demand,
        })

        return summary, df
