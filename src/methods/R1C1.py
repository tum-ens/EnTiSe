import logging

from numba import njit
import numpy as np
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
    required_keys = {O.WEATHER: O.DTYPES[O.WEATHER],
                     O.RESISTANCE: O.DTYPES[O.RESISTANCE],
                     O.CAPACITANCE: O.DTYPES[O.CAPACITANCE],
                     }
    optional_keys = None
    # Define required timeseries for each method
    required_timeseries = {O.WEATHER:
                               {K.COLS_REQUIRED: {
                                   C.DATETIME: C.DTYPES[C.DATETIME],
                                   C.T_OUT: C.DTYPES[C.T_OUT],
                                    },
                                K.DTYPE: pd.DataFrame
                                },
                           }
    optional_timeseries = None
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

        return self.calculate(obj, data)

    @staticmethod
    @njit(cache=True)
    def calculate(obj: dict, data: dict) -> (dict, pd.DataFrame):
        """
        Optimized calculation of the 1R1C thermal model for a single building.
        """
        # Unpack parameters
        weather = data[O.WEATHER]
        thermal_resistance = obj[O.RESISTANCE]
        thermal_capacitance = obj[O.CAPACITANCE]
        initial_temperature = 20
        outdoor_temperature = data[O.WEATHER][C.T_OUT].to_numpy(dtype=np.float32)
        timestep = data[O.WEATHER][C.DATETIME].diff().dt.total_seconds().mean()
        solar_gain = 0
        T_min = 18
        T_max = 22
        heating_power = np.inf
        cooling_power = np.inf

        n_steps = len(outdoor_temperature)
        indoor_temperature = np.zeros(n_steps, dtype=np.float32)
        heating_load = np.zeros(n_steps, dtype=np.float32)
        cooling_load = np.zeros(n_steps, dtype=np.float32)

        indoor_temperature[0] = initial_temperature

        for t in range(1, n_steps):
            temp_change = (
                                  (outdoor_temperature[t] - indoor_temperature[t - 1]) / thermal_resistance
                                  + solar_gain
                          ) * timestep / thermal_capacitance

            indoor_temperature[t] = indoor_temperature[t - 1] + temp_change

            if indoor_temperature[t] < T_min:
                heating_load[t] = min(heating_power, (T_min - indoor_temperature[t]) / timestep)
                indoor_temperature[t] = T_min
            elif indoor_temperature[t] > T_max:
                cooling_load[t] = min(cooling_power, (indoor_temperature[t] - T_max) / timestep)
                indoor_temperature[t] = T_max



        summary = {
            f'{C.DEMAND}_{Types.HEATING}': heating_load.sum(),
            f'{C.DEMAND}_{Types.COOLING}': cooling_load.sum(),
        }
        df = pd.DataFrame({
            C.DATETIME: weather[C.DATETIME],
            f'{C.LOAD}_{Types.HEATING}': heating_load,
            f'{C.LOAD}_{Types.COOLING}': cooling_load,
        })

        return summary, df
