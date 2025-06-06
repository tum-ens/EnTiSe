from abc import ABC
import logging

import numpy as np
import pandas as pd
import pvlib

from entise.core.base import Method
from entise.core.base_auxiliary import AuxiliaryMethod, BaseSelector
from entise.constants import Keys, SEP, Objects as O, Types, Columns

log = logging.getLogger(__name__)

DEFAULT_GAINS_INTERNAL = 500


class InternalInactive(AuxiliaryMethod):
    """
    Represents an internal mechanism providing inactive gain calculations.

    This class processes weather data to produce inactive internal gain outputs.
    It fetches the required input data and computes the results as a DataFrame
    containing internal gain values initialized to zeros. The class is an
    extension of the `AuxiliaryMethod` class.
    """
    required_timeseries = [O.WEATHER]

    def get_input_data(self, obj, data):
        return {O.WEATHER: data[O.WEATHER]}

    def run(self, weather):
        return pd.DataFrame({O.GAINS_INTERNAL: np.zeros(len(weather), dtype=np.float32)}, index=weather.index)


class InternalConstant(AuxiliaryMethod):
    """
    Handles the internal gains calculations and manipulations in relation to weather data.
    This class is responsible for processing, generating, and managing internal gains
    data, which could be derived from constants or referenced time series data. The
    class extends `AuxiliaryMethod` to utilize its auxiliary functionalities and is
    designed to be used in scenarios where internal gains need to be modeled or
    analyzed.
    """
    required_keys = [O.GAINS_INTERNAL]
    required_timeseries = [O.WEATHER]

    def generate(self, obj, data):
        gains_internal = obj.get(O.GAINS_INTERNAL, DEFAULT_GAINS_INTERNAL)
        try:
            gains_internal = float(gains_internal)
        except ValueError:
            pass
        finally:
            if isinstance(gains_internal, str):
                # If a string key is given, assume it's a reference to a time series
                return InternalTimeSeries().generate(obj, data)
        return self.run(**self.get_input_data(obj, data))

    def get_input_data(self, obj, data):
        return {
            O.GAINS_INTERNAL: obj.get(O.GAINS_INTERNAL, DEFAULT_GAINS_INTERNAL),
            O.WEATHER: data[O.WEATHER],
        }

    def run(self, gains_internal, weather):
        return pd.DataFrame(
            {O.GAINS_INTERNAL: np.full(len(weather), gains_internal, dtype=np.float32)},
            index=weather.index
        )


class InternalTimeSeries(AuxiliaryMethod):
    """
    Represents the processing of internal time series data for further computations.

    This class is designed to handle, manipulate, and process internal time series
    data provided as input. It validates the data, ensures it conforms to the expected
    formats, and executes transformations or auxiliary methods as necessary. It inherits
    from the `AuxiliaryMethod` base class and relies heavily on specific keys and internal
    structures for its operations.
    """
    required_keys = [O.GAINS_INTERNAL_COL]
    optional_keys = [O.ID]
    required_timeseries = [O.GAINS_INTERNAL]

    def generate(self, obj, data):
        gains_internal = obj.get(O.GAINS_INTERNAL)
        try:
            gains_internal = float(gains_internal)
        except ValueError:
            pass
        finally:
            if isinstance(gains_internal, O.DTYPES[O.GAINS_INTERNAL]) and not isinstance(gains_internal, str):
                return InternalConstant().generate(obj, data)
        return self.run(**self.get_input_data(obj, data))

    def get_input_data(self, obj, data):
        gains_key = obj.get(O.GAINS_INTERNAL)
        gains_ts = data.get(gains_key)
        input_data = {
            O.ID: obj.get(O.ID, None),
            O.GAINS_INTERNAL_COL: obj.get(O.GAINS_INTERNAL_COL, None),
            O.GAINS_INTERNAL: gains_ts,
        }
        return input_data

    def run(self, **kwargs):
        object_id = kwargs[O.ID]
        col = kwargs[O.GAINS_INTERNAL_COL]
        internal_gains = kwargs[O.GAINS_INTERNAL]
        col = col if isinstance(col, str) else str(object_id)
        try:
            internal_gains = internal_gains.loc[:, col]
        except KeyError:
            log.error('Internal gains column "%s" does not exist', col)
            raise Warning(f'Neither explicit (column name) or implicit (column id) are specified.'
                          f'Given input column: {col}')
        return pd.DataFrame({O.GAINS_INTERNAL: internal_gains}, index=internal_gains.index)


class InternalOccupancy(Method):
    """
    Class responsible for handling internal gains due to occupancy.

    The InternalOccupancy class calculates the internal gains generated by
    occupants in a given building or setting based on the number of people,
    gains per person, and occupancy data over time. The result is returned as
    a timeseries data structure that reflects the internal gains.
    """
    name = "InternalGainsOccupancy"
    required_keys = [O.INHABITANTS, O.GAINS_INTERNAL_PER_PERSON]
    required_timeseries = [Types.OCCUPANCY]
    output_timeseries = {}

    def generate(self, obj, data, ts_type=None):
        num_people = obj.get(O.INHABITANTS)
        gain_per_person = obj[O.GAINS_INTERNAL_PER_PERSON]
        occ = data[Types.OCCUPANCY]

        internal_gains = occ * gain_per_person * num_people

        return pd.DataFrame({O.GAINS_INTERNAL: internal_gains}, index=occ.index)


__all__ = [
    name for name, obj in globals().items()
    if isinstance(obj, type)
    and issubclass(obj, AuxiliaryMethod)
    and obj is not AuxiliaryMethod  # exclude the base class
]