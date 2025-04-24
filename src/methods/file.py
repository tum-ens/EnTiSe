import logging

import pandas as pd

from src.constants import Keys as K, Objects as O, VALID_TYPES
from src.core.base import TimeSeriesMethod
from src.core.registry import Meta
from src.utils.decorators import supported_types

logger = logging.getLogger(__name__)


@supported_types(*VALID_TYPES)
class File(TimeSeriesMethod, metaclass=Meta):
    """Timeseries generation method to load and return a timeseries from a file."""

    # Define required keys for each method
    required_keys = {O.FILE: str}
    # Define required timeseries for each method
    required_timeseries = {
        O.FILE: {
            K.COLUMNS: {},  # Columns change depending on the file type
            K.DTYPE: pd.DataFrame,
        },
    }
    # Define dependencies for each method
    dependencies = []

    def generate(self, obj: dict, data: dict, ts_type: str, **kwargs) -> (dict, pd.DataFrame):
        """
        Generate a timeseries by loading it from a file.

        Parameters:
        - obj (dict): Object metadata and parameters.
        - data (dict): Dictionary of available timeseries data.
        - ts_type (str): The timeseries type being processed.
        - **kwargs: Additional keyword arguments.

        Returns:
        - (dict, pd.DataFrame): Summary metrics (empty for this method) and the loaded timeseries.
        """
        # Prepare inputs
        obj = self.prepare_inputs(obj, data, ts_type)

        # Retrieve the file data
        file_key = obj.get(f"{O.FILE}")
        df = data[file_key]

        return {}, df
