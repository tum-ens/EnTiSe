_author__ = "Markus Doepfert"
__license__ = "MIT"
__maintainer__ = "Markus Doepfert"
__email__ = "markus.doepfert@tum.de"
__status__ = "Stable"
__date__ = "2025-04-14"
__credits__ = []
__description__ = "Abstract base class for timeseries generation methods."
__url__ = ""
__dependencies__ = ["pandas"]

from abc import ABC, abstractmethod
import logging
import pandas as pd

from src.constants import Keys, SEP, Objects as O
from src.utils.validation import Validator

logger = logging.getLogger(__name__)


class TimeSeriesMethod(ABC):
    """
    Abstract base class for timeseries generation methods.

    Subclasses must implement the `generate` method and define the class-level attributes:
    - `required_keys`: Dictionary specifying the required object keys and their types.
    - `required_timeseries`: Dictionary specifying required timeseries data and columns.
    - `dependencies`: List of dependencies required for this method.
    - `available_outputs`: Dictionary specifying available outputs (summary and timeseries).
    """

    # Define required and optional keys for each method (key names and their types)
    required_keys = dict()
    optional_keys = dict()

    # Define required and optional timeseries data (key names, expected columns, and their types)
    required_timeseries = dict()
    optional_timeseries = dict()

    # List of dependencies for this method
    dependencies = []

    # Available outputs (placeholders for summary and timeseries outputs)
    available_outputs = {
        Keys.SUMMARY: dict(),
        Keys.TIMESERIES: dict()
    }

    @abstractmethod
    def generate(self, obj: dict, data: dict, ts_type: str, dependencies: dict = None, **kwargs):
        """
        Abstract method for generating a timeseries.

        Args:
            obj (dict): The input object's metadata and parameters.
            data (dict): The input data dictionary containing timeseries.
            ts_type (str): The type of timeseries being generated.
            dependencies (dict, optional): Dependencies required by the method.
            **kwargs: Additional keyword arguments for subclass-specific implementations.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.

        Returns:
            pd.DataFrame: The generated timeseries data.
        """
        raise NotImplementedError("Method 'generate' must be implemented in the subclass.")

    def prepare_inputs(self, obj: dict, data: dict, ts_type: str) -> dict:
        """
        Prepare and validate the inputs for timeseries generation.

        Args:
            obj (dict): The input object metadata and parameters.
            data (dict): The input data dictionary.
            ts_type (str): The type of timeseries being processed.

        Returns:
            dict: A reduced and validated object dictionary.

        Raises:
            ValueError: If validation of the inputs fails.
        """
        # Check for verbose flag; default to False
        verbose = obj.get(O.VERBOSE, False)

        # Reduce object to relevant keys
        relevant_obj = self.get_relevant_objects(obj, ts_type)

        # Initialize and configure the validator
        validator = Validator()
        if verbose == 'once':
            validator.enable_cache()
        elif verbose:
            validator.disable_cache()
        else:
            return relevant_obj

        # Validate the object and timeseries data
        validator.validate_object(relevant_obj, self.required_keys, self.optional_keys)
        validator.validate_timeseries(data, self.required_timeseries, self.optional_timeseries)

        return relevant_obj

    @staticmethod
    def resolve_column(ts_key: str, column: str, ts_type: str, data: dict) -> pd.Series:
        """
        Resolve a column from the input data, checking for prefixed and shared columns.

        Args:
            ts_key (str): The key identifying the timeseries data.
            column (str): The column name to resolve.
            ts_type (str): The type of timeseries (used for prefixing).
            data (dict): The input data dictionary.

        Returns:
            pd.Series: The resolved column data.

        Raises:
            ValueError: If the specified column is not found.
        """
        # Extract timeseries data based on the key
        ts_data = data.get(ts_key)
        if ts_data is None:
            raise ValueError(f"Timeseries key '{ts_key}' not found in data.")

        # Check for prefixed column
        prefixed_col = f"{ts_type}{SEP}{column}"
        if prefixed_col in ts_data.columns:
            return ts_data[prefixed_col]

        # Fallback to shared column
        if column in ts_data.columns:
            return ts_data[column]

        raise ValueError(f"Neither '{prefixed_col}' nor '{column}' found in timeseries '{ts_key}'.")

    def get_relevant_objects(self, obj: dict, ts_type: str = None) -> dict:
        """
        Retrieve only the relevant object keys for the method.

        Args:
            obj (dict): The input object metadata and parameters.
            ts_type (str, optional): The timeseries type being processed.

        Returns:
            dict: A dictionary containing the relevant object keys.
        """
        if ts_type is None:
            return obj

        # Extract relevant prefixed keys
        relevant_objs = {
            k.split(SEP, 1)[1]: v
            for k, v in obj.items()
            if k.startswith(f"{ts_type}{SEP}")
        }

        # Collect shared keys not present in prefixed keys

        all_required_keys = set(self.required_keys.keys())
        all_optional_keys = set(self.optional_keys.keys())
        all_expected_keys = all_required_keys.union(all_optional_keys)
        shared_keys_set = all_expected_keys - set(relevant_objs.keys())
        shared_objs = {
            k: v
            for k, v in obj.items()
            if k in shared_keys_set and k not in relevant_objs
        }

        # Combine relevant objects and shared keys
        relevant_objs.update(shared_objs)

        # Log the process for debugging
        logger.debug(
            f"Method '{self.__class__.__name__}':\n"
            f"- Required keys: {self.required_keys}\n"
            f"- Relevant objects: {relevant_objs}\n"
            f"- Shared keys: {shared_keys_set}"
        )

        return relevant_objs

    @classmethod
    def get_requirements(cls) -> dict:
        """
        Retrieve the requirements for this method.

        Returns:
            dict: A dictionary containing required keys and timeseries.
        """
        return {
            Keys.KEYS_REQUIRED: cls.required_keys,
            Keys.KEYS_OPTIONAL: cls.optional_keys,
            Keys.TIMESERIES_REQUIRED: cls.required_timeseries,
            Keys.TIMESERIES_OPTIONAL: cls.optional_timeseries,
        }

    @classmethod
    def get_dependencies(cls) -> list:
        """
        Retrieve the dependencies for this method.

        Returns:
            list: A list of method dependencies.
        """
        return cls.dependencies

    @classmethod
    def get_available_outputs(cls) -> dict:
        """
        Retrieve the available outputs for this method.

        Returns:
            dict: A dictionary containing available summary and timeseries outputs.
        """
        return cls.available_outputs
