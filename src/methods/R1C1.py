import logging

import pandas as pd

from src.core.base import TimeSeriesMethod
from src.core.registry import Meta
from src.utils.decorators import supported_types
from src.constants import Columns as C, Keys as K, Objects as O, Types, SEP

logger = logging.getLogger(__name__)


@supported_types(Types.HVAC)
class R1C1(TimeSeriesMethod, metaclass=Meta):
    """Timeseries generation method using a 1R1C-model."""
    # Define required keys for each method
    required_keys = {O.WEATHER: str,
                     O.RESISTANCE: int | float,
                     O.CAPACITANCE: int | float,  # TODO: It will probably be that there are required and optional keys (both for objects and timeseries). Think about how to integrate that in the best way. (e.g. occupancy does not have to be provided but can be)
                     }
    # Define required timeseries for each method
    required_timeseries = {O.WEATHER:
                               {K.COLUMNS: {C.DATETIME: pd.Timestamp,
                                            C.T_OUT: float,
                                            },
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
        t_out = weather[C.T_OUT]

        # Example logic
        heating_demand = (18 - t_out).clip(lower=0)

        summary = {
            "demand_total": heating_demand.sum(),
        }
        df = pd.DataFrame({
            C.DATETIME: weather[C.DATETIME],
            C.DEMAND: heating_demand,
        })

        return summary, df
