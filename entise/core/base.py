from abc import ABC, ABCMeta, abstractmethod
import logging
from typing import List, Dict, Any, Type

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

    @classmethod
    def get_requirements(cls) -> dict:
        return {
            Keys.KEYS_REQUIRED: cls.required_keys,
            Keys.KEYS_OPTIONAL: cls.optional_keys,
            Keys.TIMESERIES_REQUIRED: cls.required_timeseries,
            Keys.TIMESERIES_OPTIONAL: cls.optional_timeseries,
        }
