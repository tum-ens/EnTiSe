from abc import ABC, abstractmethod
import logging

import pandas as pd

from src.constants import Keys, SEP, Objects as O
from src.utils.validation import Validator

logger = logging.getLogger(__name__)


class TimeSeriesMethod(ABC):
    """
    Abstract base class for timeseries generation methods.
    Subclasses must define the `generate` method.
    """
    # Define required keys for each method (which keys have to be in obj with which dtype)
    required_keys = dict()
    # Define required timeseries for each method (which timeseries have to be in data with which columns and dtype)
    required_timeseries = dict()
    # Define dependencies for each method (on which method does this method depend)
    dependencies = []
    # Placeholder for future implementation of available outputs for each method (not implemented yet)
    available_outputs = {
        Keys.SUMMARY: dict(),
        Keys.TIMESERIES: dict()
    }

    @abstractmethod
    def generate(self, obj: dict, data: dict, ts_type: str, dependencies: dict = None, **kwargs):
        """
        Generate a timeseries for the given object.

        Parameters:
        - obj (dict): Objects metadata and parameters.
        - *args: Additional positional arguments for the subclass method.
        - **kwargs: Additional keyword arguments for the subclass method.

        Returns:
        - pd.DataFrame: Generated timeseries.
        """
        raise NotImplementedError("Method 'generate' must be implemented in the subclass.")

    def prepare_inputs(self, obj: dict, data: dict, ts_type: str) -> dict:
        """
        Reduce object to relevant keys and validate inputs.

        Parameters:
        - obj (dict): The input object dictionary.
        - data (dict): The input data dictionary.
        - ts_type (str): The timeseries type being processed.

        Returns:
        - dict: The reduced and validated object.
        """

        # Check for verbose flag; default to True
        verbose = obj.get(O.VERBOSE, True)

        # Reduce obj to only the relevant keys
        relevant_obj = self.get_relevant_objects(obj, ts_type)

        # Initialize Validator
        validator = Validator()

        # Enable or disable caching based on verbosity
        if verbose:
            validator.disable_cache()
        else:
            validator.enable_cache()

        # Validate inputs
        validator.validate_object(relevant_obj, self.required_keys)
        validator.validate_timeseries(data, self.required_timeseries)

        return relevant_obj

    @staticmethod
    def resolve_column(ts_key: str, column: str, ts_type: str, data: dict) -> pd.Series:
        """
        Resolve the correct column from the data based on the naming convention.

        Parameters:
        - ts_key (str): The timeseries key in the data dictionary.
        - column (str): The base column name.
        - ts_type (str): The timeseries type prefix.
        - data (dict): The data dictionary.

        Returns:
        - pd.Series: The resolved column.

        Raises:
        - ValueError: If the column is not found.
        """
        # Check for prefixed column first
        ts_data = data.get(ts_key)
        if ts_data is None:
            raise ValueError(f"Timeseries key '{ts_key}' not found in data.")

        prefixed_col = f"{ts_type}{SEP}{column}"
        if prefixed_col in ts_data.columns:
            return ts_data[prefixed_col]

        # Fall back to shared column
        if column in ts_data.columns:
            return ts_data[column]

        raise ValueError(f"Neither '{prefixed_col}' nor '{column}' found in timeseries '{ts_key}'.")

    def get_relevant_objects(self, obj: dict, ts_type: str = None):
        """
        Get only the relevant objects for this method.

        Parameters:
        - obj (dict): Objects metadata and parameters.
        - ts_type (str): Timeseries type for which keys are resolved.

        Returns:
        - dict: Relevant objects for this method.
        """
        if ts_type is None:
            return obj

        # Collect all prefixed keys for the specific ts_type
        relevant_objs = {k.split(SEP, 1)[1]: v for k, v in obj.items() if k.startswith(f'{ts_type}{SEP}')}

        # Determine shared keys that are to be used
        shared_keys_set = set(self.required_keys.keys()) - set(relevant_objs.keys())
        shared_objs = {k: v for k, v in obj.items() if k in shared_keys_set and k not in relevant_objs}

        # Update relevant objects with shared keys
        relevant_objs.update(shared_objs)

        # Log details of the operation
        logger.debug(
            f"Method '{self.__class__.__name__}':\n"
            f"- Required keys: {self.required_keys}\n"
            f"- Relevant objects: {relevant_objs}\n"
            f"- Shared keys: {shared_keys_set}"
        )

        return relevant_objs

    @classmethod
    def get_requirements(cls):
        """
        Get the requirements for this method.

        Returns:
        - dict: Required keys and timeseries for this method.
        """
        return {
            Keys.KEYS: cls.required_keys,
            Keys.TIMESERIES: cls.required_timeseries
        }

    @classmethod
    def get_dependencies(cls):
        """
        Get the dependencies for this method.

        Returns:
        - list: Required dependencies for this method.
        """
        return cls.dependencies

    @classmethod
    def get_available_outputs(cls):
        """
        Get the available outputs for this method.

        Returns:
        - dict: Available outputs for this method.
        """
        return cls.available_outputs
