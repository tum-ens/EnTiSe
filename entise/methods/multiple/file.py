import pandas as pd
from entise.core.base import Method
from entise.constants import Objects as O, VALID_TYPES


class FileLoader(Method):
    """
    Loads a timeseries from the input data using a provided key in `O.FILE`.
    Useful for injecting external time series into the simulation pipeline.
    """
    types = VALID_TYPES  # Adapt as needed
    name = "file"

    required_keys = [O.ID, O.FILE]
    required_timeseries = [O.FILE]

    def generate(self, obj, data, ts_type=None):
        key = obj.get(O.FILE)
        if key not in data:
            raise ValueError(f"FileLoader expected timeseries key '{key}' to be present in input data.")

        df = data[key]

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a DataFrame for key '{key}', but got {type(df).__name__}.")

        return {
            "summary": {},
            "timeseries": df,
        }
