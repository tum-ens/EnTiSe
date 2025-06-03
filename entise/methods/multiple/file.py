"""File loader module for importing external time series.

This module implements a method for loading time series data from external sources
using a file key. It follows the Method pattern established in the project architecture.

The module provides functionality to:
- Process input parameters for file loading
- Validate and prepare the input data
- Return the loaded time series data

The main class, FileLoader, inherits from the Method base class and implements the
required interface for integration with the EnTiSe framework.
"""

import pandas as pd

from entise.core.base import Method
from entise.constants import Objects as O, VALID_TYPES


class FileLoader(Method):
    """Implements a method for loading time series data from external sources.

    This class provides functionality to load time series data from external sources
    using a file key. It is useful for injecting external time series into the
    simulation pipeline.

    The class follows the Method pattern defined in the EnTiSe framework, implementing
    the required interface for time series generation methods.

    Attributes:
        types (list): List of time series types this method can generate (all valid types).
        name (str): Name identifier for the method.
        required_keys (list): Required input parameters (id, file).
        required_timeseries (list): Required time series inputs (file).
        output_summary (dict): Empty dictionary as no summary is provided.
        output_timeseries (dict): Empty dictionary as the output format depends on the input.

    Example:
        >>> from entise.methods.multiple.file import FileLoader
        >>> from entise.core.generator import TimeSeriesGenerator
        >>> 
        >>> # Create a generator and add objects
        >>> gen = TimeSeriesGenerator()
        >>> gen.add_objects(objects_df)  # DataFrame with file parameters
        >>> 
        >>> # Generate time series
        >>> summary, timeseries = gen.generate(data)  # data contains external time series
    """
    types = VALID_TYPES  # Adapt as needed
    name = "file"

    required_keys = [O.ID, O.FILE]
    required_timeseries = [O.FILE]
    optional_keys = []
    optional_timeseries = []
    output_summary = {}
    output_timeseries = {}

    def generate(self, 
                obj: dict = None, 
                data: dict = None, 
                ts_type: str = None,
                *,
                file: str = None):
        """Load time series data from external sources.

        This method implements the abstract generate method from the Method base class.
        It processes the input parameters, loads the time series data, and returns it.

        Args:
            obj (dict, optional): Dictionary containing file parameters. Defaults to None.
            data (dict, optional): Dictionary containing input data. Defaults to None.
            ts_type (str, optional): Time series type to generate. Defaults to None.
            file (str, optional): Key to use for loading the time series. Defaults to None.

        Returns:
            dict: Dictionary containing:
                - "summary" (dict): Empty dictionary as no summary is provided.
                - "timeseries" (pd.DataFrame): The loaded time series data.

        Raises:
            ValueError: If required data is missing.
            TypeError: If the loaded data is not a DataFrame.

        Example:
            >>> fileloader = FileLoader()
            >>> # Using explicit parameters
            >>> result = fileloader.generate(file="external_input", data={"external_input": df})
            >>> # Or using dictionaries
            >>> obj = {"file": "external_input"}
            >>> data = {"external_input": df}
            >>> result = fileloader.generate(obj=obj, data=data)
            >>> timeseries = result["timeseries"]
        """
        # Process keyword arguments
        processed_obj, processed_data = self._process_kwargs(
            obj, data,
            file=file
        )

        # Continue with existing implementation
        processed_obj, processed_data = get_input_data(processed_obj, processed_data, ts_type)

        # Load the time series
        timeseries = load_timeseries(processed_obj, processed_data)

        return {
            "summary": {},
            "timeseries": timeseries,
        }


def get_input_data(obj: dict, data: dict, method_type: str = None) -> tuple[dict, dict]:
    """Process and validate input data for file loading.

    This function extracts required parameters from the input dictionaries,
    performs data validation, and prepares the data for loading.

    Args:
        obj (dict): Dictionary containing file parameters such as the file key.
        data (dict): Dictionary containing input data such as external time series.
        method_type (str, optional): Method type to use for prefixing. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - obj_out (dict): Processed object parameters.
            - data_out (dict): Processed data with required format for loading.

    Raises:
        ValueError: If required parameters are missing.
    """
    # Process object parameters
    obj_out = {
        O.ID: Method.get_with_backup(obj, O.ID),
        O.FILE: Method.get_with_method_backup(obj, O.FILE, method_type),
    }

    # Validate required parameters
    if obj_out[O.FILE] is None:
        raise ValueError(f"Required parameter '{O.FILE}' not found in object parameters")

    # Process data
    data_out = {
        obj_out[O.FILE]: Method.get_with_backup(data, obj_out[O.FILE]),
    }

    # Validate required data
    if data_out[obj_out[O.FILE]] is None:
        raise ValueError(f"FileLoader expected timeseries key '{obj_out[O.FILE]}' to be present in input data.")

    return obj_out, data_out


def load_timeseries(obj: dict, data: dict) -> pd.DataFrame:
    """Load time series data from the input data.

    This function loads a time series from the input data using the provided key.

    Args:
        obj (dict): Dictionary containing processed file parameters such as:
            - file: Key to use for loading the time series.
        data (dict): Dictionary containing processed data such as:
            - The time series data to load.

    Returns:
        pd.DataFrame: The loaded time series data.

    Raises:
        TypeError: If the loaded data is not a DataFrame.
    """
    key = obj[O.FILE]
    df = data[key]

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame for key '{key}', but got {type(df).__name__}.")

    return df
