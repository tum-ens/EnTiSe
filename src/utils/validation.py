# Imports
import logging
from types import UnionType
from typing import Union, get_origin, get_args


import pandas as pd

from src.constants import Keys

logger = logging.getLogger(__name__)


class Validator:
    """Centralized input validation for objects and timeseries."""

    _cache_enabled = True  # Toggle caching on or off
    _validated_objects = set()  # Cache for validated objects
    _validated_timeseries = set()  # Cache for validated timeseries

    @classmethod
    def enable_cache(cls):
        """Enable caching for validations."""
        cls._cache_enabled = True
        logger.info("Validation cache enabled.")

    @classmethod
    def disable_cache(cls):
        """Disable caching for validations."""
        cls._cache_enabled = False
        cls._validated_objects.clear()
        cls._validated_timeseries.clear()
        logger.info("Validation cache disabled and cleared.")

    @staticmethod
    def _generate_cache_key(data: dict) -> int:
        """Generate a unique hashable key for caching."""
        from hashlib import sha256
        import json

        def serialize_data(value):
            if isinstance(value, pd.DataFrame):
                return value.to_json()
            elif isinstance(value, pd.Timestamp):
                return value.isoformat()  # Convert to ISO 8601 string
            elif isinstance(value, dict):
                return {k: serialize_data(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [serialize_data(v) for v in value]
            return value

        serialized = {key: serialize_data(value) for key, value in data.items()}
        return int(sha256(json.dumps(serialized, sort_keys=True).encode()).hexdigest(), 16)

    @classmethod
    def validate_object(cls, obj: dict, required_keys: dict):
        """
        Validate required keys in the object dictionary.

        Parameters:
        - obj (dict): The object metadata.
        - required_keys (dict): Expected keys with their types.

        Raises:
        - ValueError: If any required key is missing or has the wrong type.
        """
        cache_key = cls._generate_cache_key(obj)
        if cls._cache_enabled and cache_key in cls._validated_objects:
            logger.debug(f"Object validation skipped for cache key: {cache_key}")
            return

        # Check for missing keys
        missing_keys = [key for key in required_keys if key not in obj]
        if missing_keys:
            logger.error(f"Missing required keys: {missing_keys}")
            raise ValueError(f"Missing required keys: {missing_keys}")

        # Check for invalid types
        for key, expected_type in required_keys.items():
            value = obj[key]

            # Handle Union types (e.g., int | float)
            if isinstance(expected_type, UnionType):
                allowed_types = expected_type.__args__
                if not isinstance(value, allowed_types):
                    allowed_types_str = ", ".join(t.__name__ for t in allowed_types)
                    raise ValueError(
                        f"Key '{key}' must be one of {allowed_types_str}, got '{type(value).__name__}'."
                    )
            # Single type check
            elif not isinstance(value, expected_type):
                raise ValueError(
                    f"Key '{key}' must be of type '{expected_type.__name__}', got '{type(value).__name__}'."
                )

        if cls._cache_enabled:
            cls._validated_objects.add(cache_key)
            logger.debug(f"Object validation passed and cached for key: {cache_key}")

    @classmethod
    def validate_timeseries(cls, data: dict, required_timeseries: dict):
        """
        Validate timeseries data.

        Parameters:
        - data (dict): Timeseries data dictionary.
        - required_timeseries (dict): Schema with required timeseries, columns, and types.

        Raises:
        - ValueError: If any required timeseries or column is missing or has the wrong type.
        """
        cache_key = cls._generate_cache_key(data)
        if cls._cache_enabled and cache_key in cls._validated_timeseries:
            logger.debug(f"Timeseries validation skipped for cache key: {cache_key}")
            return

        for ts_key, schema in required_timeseries.items():
            if ts_key not in data:
                logger.error(f"Missing timeseries '{ts_key}' in data.")
                raise ValueError(f"Missing timeseries '{ts_key}' in data.")

            ts = data[ts_key]
            if not isinstance(ts, schema.get(Keys.DTYPE, pd.DataFrame)):
                logger.error(f"Timeseries '{ts_key}' must be of type '{schema.get(Keys.DTYPE)}'.")
                raise ValueError(f"Timeseries '{ts_key}' must be of type '{schema.get(Keys.DTYPE)}'.")

            for col, col_dtype in schema.get(Keys.COLUMNS, {}).items():
                if col not in ts.columns:
                    logger.error(f"Missing column '{col}' in timeseries '{ts_key}'.")
                    raise ValueError(f"Timeseries '{ts_key}' is missing required column '{col}'.")
                if not pd.api.types.is_dtype_equal(ts[col].dtype, col_dtype):
                    logger.error(f"Invalid type for column '{col}' in timeseries '{ts_key}'.")
                    raise ValueError(f"Column '{col}' in timeseries '{ts_key}' must be of type '{col_dtype}'.")

        if cls._cache_enabled:
            cls._validated_timeseries.add(cache_key)
            logger.debug(f"Timeseries validation passed and cached for key: {cache_key}")
