from abc import ABC, ABCMeta, abstractmethod
import logging
import math
from typing import List, Dict, Any, Type, Optional, Tuple

import pandas as pd

from entise.constants import Keys, SEP, Objects as O, VALID_TYPES

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
    required_timeseries: List[str] = []
    optional_timeseries: List[str] = []
    output_summary: Dict[str, pd.DataFrame] = {}
    output_timeseries: Dict[str, pd.DataFrame]

    @abstractmethod
    def generate(self, obj: dict, data: dict, ts_type: str) -> Dict[str, Any]:
        raise NotImplementedError("Method 'generate' must be implemented.")

    @staticmethod
    def resolve_column(ts_key: str, column: str, ts_type: str, data: dict) -> pd.Series:
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
        if ts_type is None:
            return obj

        relevant_objs = {
            k.split(SEP, 1)[1]: v
            for k, v in obj.items()
            if k.startswith(f"{ts_type}{SEP}")
        }

        expected_keys = set(self.required_keys + self.optional_keys)
        shared_keys_set = expected_keys - set(relevant_objs.keys())
        shared_objs = {
            k: v for k, v in obj.items()
            if k in shared_keys_set and k not in relevant_objs
        }

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
        all_data_keys = set(self.required_timeseries + self.optional_timeseries)

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
        return {
            Keys.KEYS_REQUIRED: cls.required_keys,
            Keys.KEYS_OPTIONAL: cls.optional_keys,
            Keys.TIMESERIES_REQUIRED: cls.required_timeseries,
            Keys.TIMESERIES_OPTIONAL: cls.optional_timeseries,
        }
