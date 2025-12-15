import logging

import numpy as np
import pandas as pd

from entise.constants import Constants as Const
from entise.constants import Objects as O
from entise.core.base_auxiliary import AuxiliaryMethod

log = logging.getLogger(__name__)

DEFAULT_VENTILATION = 65  # only for constant calculation (W/K)
DEFAULT_VENTILATION_FACTOR = 0.5  # 1/h
AIR_DENSITY = 1.2  # kg/m³
HEAT_CAPACITY = 1000  # J/kgK


class VentilationInactive(AuxiliaryMethod):
    """
    Represents a ventilation mechanism providing inactive ventilation calculations.

    This class processes weather data to produce inactive ventilation outputs.
    It fetches the required input data and computes the results as a DataFrame
    containing ventilation values initialized to zeros. The class is an
    extension of the `AuxiliaryMethod` class.
    """

    required_timeseries = [O.WEATHER]

    def get_input_data(self, obj, data):
        return {O.WEATHER: data[O.WEATHER]}

    def run(self, weather):
        return pd.DataFrame({O.VENTILATION: np.zeros(len(weather), dtype=np.float32)}, index=weather.index)


class VentilationConstant(AuxiliaryMethod):
    """
    Handles the ventilation calculations and manipulations in relation to weather data.
    This class is responsible for processing, generating, and managing ventilation
    data, which could be derived from constants or referenced time series data. The
    class extends `AuxiliaryMethod` to utilize its auxiliary functionalities and is
    designed to be used in scenarios where ventilation needs to be modeled or
    analyzed.
    """

    required_keys = [O.VENTILATION]
    required_timeseries = [O.WEATHER]

    def generate(self, obj, data):
        ventilation = obj.get(O.VENTILATION, DEFAULT_VENTILATION)
        is_string = False

        try:
            ventilation = float(ventilation)
        except ValueError:
            is_string = True

        if is_string or isinstance(ventilation, str):
            # If a string key is given, assume it's a reference to a time series
            return VentilationTimeSeries().generate(obj, data)

        return self.run(**self.get_input_data(obj, data))

    def get_input_data(self, obj, data):
        return {
            "ventilation": obj.get(O.VENTILATION, DEFAULT_VENTILATION),
            "weather": data[O.WEATHER],
        }

    def run(self, ventilation, weather):
        return pd.DataFrame({O.VENTILATION: np.full(len(weather), ventilation, dtype=np.float32)}, index=weather.index)


class VentilationTimeSeries(AuxiliaryMethod):
    """
    Represents the processing of ventilation time series data for further computations.

    This class is designed to handle, manipulate, and process ventilation time series
    data provided as input. It validates the data, ensures it conforms to the expected
    formats, and executes transformations or auxiliary methods as necessary. It inherits
    from the `AuxiliaryMethod` base class and relies heavily on specific keys and ventilation
    structures for its operations.
    """

    required_keys = [O.VENTILATION_COL]
    optional_keys = [O.ID, O.AREA, O.HEIGHT]
    required_timeseries = [O.VENTILATION]

    def generate(self, obj, data):
        ventilation = obj.get(O.VENTILATION)
        is_numeric = False

        try:
            ventilation = float(ventilation)
            is_numeric = True
        except ValueError:
            pass

        if is_numeric and not isinstance(ventilation, str):
            return VentilationConstant().generate(obj, data)

        return self.run(**self.get_input_data(obj, data))

    def get_input_data(self, obj, data):
        ventilation_key = obj.get(O.VENTILATION)
        ventilation_ts = data.get(ventilation_key)
        input_data = {
            O.ID: obj.get(O.ID, None),
            O.AREA: obj.get(O.AREA, Const.DEFAULT_AREA.value),
            O.HEIGHT: obj.get(O.HEIGHT, Const.DEFAULT_HEIGHT.value),
            O.VENTILATION_COL: obj.get(O.VENTILATION_COL, None),
            O.VENTILATION: ventilation_ts,
            O.VENTILATION_FACTOR: obj.get(O.VENTILATION_FACTOR, DEFAULT_VENTILATION_FACTOR),
        }
        return input_data

    def run(self, **kwargs):
        object_id = kwargs[O.ID]
        area = kwargs[O.AREA]
        height = kwargs[O.HEIGHT]
        ventilation = kwargs[O.VENTILATION]
        col = kwargs[O.VENTILATION_COL]
        col = col if isinstance(col, str) else str(object_id)
        unit = col.split("[")[-1].split("]")[0]
        try:
            ts = ventilation.loc[:, col]
        except KeyError as err:
            raise Warning(
                f"Neither explicit (column name) or implicit (column id) are specified." f"Given input column: {col}"
            ) from err
        match unit:
            case "1/h":
                ventilation = ts * area * height * AIR_DENSITY * HEAT_CAPACITY / 3600
            case "m3/h":
                ventilation = ts * AIR_DENSITY * HEAT_CAPACITY / 3600
            case "W/K":
                ventilation = ts
            case _:
                log.warning('No unit given. "1/h" assumed as unit.')
                ventilation = ts * area * height * AIR_DENSITY * HEAT_CAPACITY / 3600
        return pd.DataFrame({O.VENTILATION: ventilation}, index=ventilation.index)


__all__ = [
    name
    for name, obj in globals().items()
    if isinstance(obj, type)
    and issubclass(obj, AuxiliaryMethod)
    and obj is not AuxiliaryMethod  # exclude the base class
]
