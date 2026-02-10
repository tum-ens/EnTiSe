import logging
import math
import re
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, List, Tuple, Type

import pandas as pd

from entise.constants import SEP, VALID_TYPES, Keys
from entise.constants import Columns as C

logger = logging.getLogger(__name__)

# Global registry
method_registry: Dict[str, Type["Method"]] = {}


class MethodMeta(ABCMeta):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        if cls.__name__ == "Method" or getattr(cls, "register", True) is False:
            return
        for t in getattr(cls, "types", []):
            if t not in VALID_TYPES:
                raise ValueError(f"Invalid type '{t}' in {cls.__name__}. Must be one of {VALID_TYPES}.")
        method_registry[cls.name.lower()] = cls


class Method(ABC, metaclass=MethodMeta):
    types: List[str]
    name: str = ""
    required_keys: List[str]
    optional_keys: List[str] = []
    required_data: List[str] = []
    optional_data: List[str] = []
    output_summary: Dict[str, pd.DataFrame] = {}
    output_timeseries: Dict[str, pd.DataFrame]

    @abstractmethod
    def generate(self, obj: dict, data: dict, results: dict, ts_type: str) -> Dict[str, Any]:
        raise NotImplementedError("Method 'generate' must be implemented.")

    @staticmethod
    def resolve_column(ts_key: str, column: str, ts_type: str, data: dict) -> pd.Series:
        """Resolve a column from the timeseries data, checking for both method-specific and generic column names."""
        ts_data = data.get(ts_key)
        if ts_data is None:
            raise ValueError(f"Timeseries key '{ts_key}' not found in data.")

        prefixed_col = f"{ts_type}{SEP}{column}"
        if prefixed_col in ts_data.columns:
            return ts_data[prefixed_col]
        if column in ts_data.columns:
            return ts_data[column]

        raise ValueError(f"Neither '{prefixed_col}' nor '{column}' found in timeseries '{ts_key}'.")

    def get_relevant_objects(self, obj: dict, ts_type: str = None) -> dict:
        """Extract relevant objects for the method based on the timeseries type."""
        if ts_type is None:
            return obj

        relevant_objs = {k.split(SEP, 1)[1]: v for k, v in obj.items() if k.startswith(f"{ts_type}{SEP}")}

        expected_keys = set(self.required_keys + self.optional_keys)
        shared_keys_set = expected_keys - set(relevant_objs.keys())
        shared_objs = {k: v for k, v in obj.items() if k in shared_keys_set and k not in relevant_objs}

        relevant_objs.update(shared_objs)
        logger.debug(
            f"Method '{self.__class__.__name__}':\n"
            f"- Required keys: {self.required_keys}\n"
            f"- Relevant objects: {relevant_objs}\n"
            f"- Shared keys: {shared_keys_set}"
        )

        return relevant_objs

    def _process_kwargs(self, obj=None, data=None, **kwargs) -> Tuple[dict, dict]:
        """Process keyword arguments into obj and data dictionaries.

        Args:
            obj (dict, optional): Initial object dictionary. Defaults to None.
            data (dict, optional): Initial data dictionary. Defaults to None.
            **kwargs: Keyword arguments to process.

        Returns:
            Tuple[dict, dict]: Processed obj and data dictionaries.
        """
        local_obj = {} if obj is None else obj.copy()
        local_data = {} if data is None else data.copy()

        # Get all possible keys for this method
        all_obj_keys = set(self.required_keys + self.optional_keys)
        all_data_keys = set(self.required_data + self.optional_data)

        for param_name, value in kwargs.items():
            if value is None:
                continue

            # Check if the parameter name is in the required or optional keys
            if param_name in all_obj_keys:
                local_obj[param_name] = value
            elif param_name in all_data_keys:
                local_data[param_name] = value

        return local_obj, local_data

    @staticmethod
    def _strip_weather_height(weather: pd.DataFrame) -> pd.DataFrame:
        """Strip height information from weather DataFrame columns.

        Args:
            weather (pd.DataFrame): Weather data DataFrame.

        Returns:
            pd.DataFrame: Weather DataFrame with height information stripped.
        """
        # Remove height suffix from column names
        weather.columns = [re.sub(r"@\d+(\.\d+)?m$", "", col) for col in weather.columns]
        return weather

    @staticmethod
    def _obtain_weather_info(weather: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Process weather DataFrame to ensure it has the required columns.

        Args:
            weather (pd.DataFrame): Weather data DataFrame.

        Returns:
            weather (pd.DataFrame): Processed weather DataFrame.
            info (dict): Information dictionary with column information.
        """
        pattern = re.compile(
            r"^(?P<name>[a-z0-9_]+)\[(?P<unit>[^\]]+)\](?:@(?P<height>[0-9]+(?:\.[0-9]+)?m))?$",
            re.IGNORECASE,
        )

        def parse_column(col: str):
            """Return (name, unit, height_m) or (None, None, None) if cannot parse."""
            m = pattern.match(col)
            if m:
                n = m.group("name")
                u = m.group("unit")
                h = m.group("height")
                height_m = float(h[:-1]) if h else None  # strip trailing 'm'
                return n, u, height_m

            return None, None, None

        info: dict = {}

        for col in weather.columns:
            if col == C.DATETIME:
                continue

            name, unit, height_m = parse_column(col)

            if name is None:
                # Unparsed; keep as-is but record minimal info
                info[col] = {"name": col, "unit": None, "height": None}
            else:
                # Store parsed information using original column name as key
                info[col] = {"name": name, "unit": unit, "height": height_m}

        return weather, info

    @staticmethod
    def get_with_backup(obj, key, backup=None):
        """Get a value from a dictionary with a backup value if not found or None.

        Args:
            obj (dict): Dictionary to get value from
            key (str): Key to look for
            backup: Default value if key doesn't exist or value is None/NaN

        Returns:
            The value from the dictionary, or the backup value if not found
        """
        value = obj.get(key, backup)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return backup
        return value

    @classmethod
    def get_with_method_backup(cls, obj, key, method_type, backup=None):
        """Get a value from a dictionary, checking for method-specific key first.

        Args:
            obj (dict): Dictionary to get value from
            key (str): Base key to look for
            method_type (str): Method type to use for prefixing
            backup: Default value if neither method-specific nor base key exists

        Returns:
            The value from the dictionary, or the backup value if not found

        Example:
            # Will check for "pv:altitude" first, then "altitude"
            altitude = cls.get_with_method_backup(obj, "altitude", "pv", backup=None)
        """
        # First try method-specific key (e.g., "pv:altitude")
        method_key = f"{method_type}{SEP}{key}"
        value = cls.get_with_backup(obj, method_key)

        # If not found, try the base key (e.g., "altitude")
        if value is None:
            value = cls.get_with_backup(obj, key, backup)

        return value

    @classmethod
    def get_requirements(cls) -> dict:
        """Return a dictionary of the method's requirements, including required and optional keys and timeseries."""
        return {
            Keys.KEYS_REQUIRED: cls.required_keys,
            Keys.KEYS_OPTIONAL: cls.optional_keys,
            Keys.DATA_REQUIRED: cls.required_data,
            Keys.DATA_OPTIONAL: cls.optional_data,
        }
