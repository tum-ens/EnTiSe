import logging
from types import UnionType
from typing import Union, Dict, Any

import pandas as pd
from src.constants import Keys

logger = logging.getLogger(__name__)


class Validator:
    """
    A centralized validator for objects and timeseries data.

    This class validates:
        - Object metadata to ensure required keys and types are present.
        - Timeseries data to ensure required structure, columns, and data types are valid.

    It supports caching to optimize repeated validation of identical inputs.
    """

    _cache_enabled: bool = True  # Toggle caching on/off
    _validated_objects: set = set()  # Cache for validated objects
    _validated_timeseries: set = set()  # Cache for validated timeseries

    @classmethod
    def enable_cache(cls):
        """
        Enable caching for validations.

        When enabled, previously validated objects and timeseries will not be revalidated.
        """
        cls._cache_enabled = True
        logger.info("Validation cache enabled.")

    @classmethod
    def disable_cache(cls):
        """
        Disable caching for validations and clear existing cache.

        When disabled, every validation will run even if inputs were previously validated.
        """
        cls._cache_enabled = False
        cls._validated_objects.clear()
        cls._validated_timeseries.clear()
        logger.info("Validation cache disabled and cleared.")

    @staticmethod
    def _generate_cache_key(data: Dict[str, Any]) -> int:
        """
        Generate a unique hashable key for caching validations.

        Args:
            data (Dict[str, Any]): Input data to generate a cache key.

        Returns:
            int: A hashable key based on the input data.
        """
        from hashlib import sha256
        import json

        def serialize_data(value: Any) -> Any:
            if isinstance(value, pd.DataFrame):
                return value.to_json()
            elif isinstance(value, pd.Timestamp):
                return value.isoformat()
            elif isinstance(value, dict):
                return {k: serialize_data(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [serialize_data(v) for v in value]
            return value

        serialized = {key: serialize_data(value) for key, value in data.items()}
        return int(sha256(json.dumps(serialized, sort_keys=True).encode()).hexdigest(), 16)

    @classmethod
    def validate_object(cls, obj: Dict[str, Any], required_keys: Dict[str, Any], optional_keys: Dict[str, Any] = None,
                        raise_dtype_errors: bool = False) -> None:
        """
        Validate the required keys and their types in an object.

        Args:
            obj (Dict[str, Any]): The object metadata to validate.
            required_keys (Dict[str, Any]): Expected keys and their types.
            optional_keys (Dict[str, Any]): Optional keys and their types.
            raise_dtype_errors (bool): Whether to raise errors for dtype mismatches.

        Raises:
            ValueError: If any required key is missing or has an incorrect type.
        """
        cache_key = cls._generate_cache_key(obj)
        if cls._cache_enabled and cache_key in cls._validated_objects:
            logger.debug(f"Object validation skipped for cache key: {cache_key}")
            return

        cls.validate_object_keys(obj, required_keys, raise_dtype_errors=raise_dtype_errors)
        if optional_keys:
            cls.validate_object_keys(obj, optional_keys, check_for_missing=False, raise_dtype_errors=raise_dtype_errors)

        if cls._cache_enabled:
            cls._validated_objects.add(cache_key)
            logger.debug(f"Object validation passed and cached for key: {cache_key}")

    @classmethod
    def validate_object_keys(cls, obj, keys: Dict[str, Any], check_for_missing: bool = True,
                             raise_dtype_errors: bool = False) -> None:
        """
        Validate the keys and types in an object.
        Args:
            obj (Dict[str, Any]): The object metadata to validate.
            keys (Dict[str, Any]): Expected keys and their types.
            check_for_missing (bool): Whether to check for missing keys.
            raise_dtype_errors (bool): Whether to raise errors on dtypes.

        Raises:
            ValueError: If any required key is missing or has an incorrect type.
        """

        # Check for missing keys
        if check_for_missing:
            missing_keys = [key for key in keys if key not in obj]
            if missing_keys:
                logger.error(f"Missing required keys: {missing_keys}")
                raise ValueError(f"Missing required keys: {missing_keys}")

        # Check for invalid types
        for key, expected_type in keys.items():
            value = obj[key]
            if isinstance(expected_type, UnionType):
                allowed_types = expected_type.__args__
                if not isinstance(value, allowed_types):
                    allowed_types_str = ", ".join(t.__name__ for t in allowed_types)
                    logger.info(f"Key '{key}' must be one of {allowed_types_str}, got '{type(value).__name__}'.")
                    if raise_dtype_errors:
                        raise ValueError(f"Key '{key}' must be one of {allowed_types_str}, got '{type(value).__name__}'.")
            elif not isinstance(value, expected_type):
                logger.info(f"Key '{key}' must be of type '{expected_type.__name__}', got '{type(value).__name__}'.")
                if raise_dtype_errors:
                    raise ValueError(f"Key '{key}' must be of type '{expected_type.__name__}', got '{type(value).__name__}'.")

    @classmethod
    def validate_timeseries(cls, data: Dict[str, Any], required_timeseries: Dict[str, Dict],
                            optional_timeseries: Dict[str, Dict] = None, raise_dtype_errors: bool = False) -> None:
        """
        Validate the structure and content of timeseries data.

        Args:
            data (Dict[str, Any]): Timeseries data dictionary.
            required_timeseries (Dict[str, Dict]): Schema specifying required keys, columns, and types.
            optional_timeseries (Dict[str, Dict]): Schema specifying optional keys, columns, and types
            raise_dtype_errors (bool): Whether to raise errors for dtype mismatches.

        Raises:
            ValueError: If any timeseries key, column, or data type is invalid.
        """
        cache_key = cls._generate_cache_key(data)
        if cls._cache_enabled and cache_key in cls._validated_timeseries:
            logger.debug(f"Timeseries validation skipped for cache key: {cache_key}")
            return

        cls.validate_timeseries_keys(data, required_timeseries, raise_dtype_errors=raise_dtype_errors)
        if optional_timeseries:
            cls.validate_timeseries_keys(data, optional_timeseries, check_for_missing=False,
                                         raise_dtype_errors=raise_dtype_errors)

        if cls._cache_enabled:
            cls._validated_timeseries.add(cache_key)
            logger.debug(f"Timeseries validation passed and cached for key: {cache_key}")

    @classmethod
    def validate_timeseries_keys(cls, data: Dict[str, Any], timeseries: Dict[str, Dict],
                                 check_for_missing: bool = True, raise_dtype_errors: bool = False) -> None:
        """
        Validate the keys, columns, and types in timeseries data.

        Args:
            data (Dict[str, Any]): Timeseries data dictionary.
            timeseries (Dict[str, Dict]): Schema specifying keys, columns, and types.
            check_for_missing (bool): Whether to check for missing timeseries.
            raise_dtype_errors (bool): Whether to raise errors on dtypes.

        Raises:
            ValueError: If any timeseries key, column, or data type is invalid

        """

        # Check for missing timeseries
        for ts_key, schema in timeseries.items():
            if ts_key not in data and check_for_missing:
                logger.error(f"Missing timeseries '{ts_key}' in data.")
                raise ValueError(f"Missing timeseries '{ts_key}' in data.")

            # Check for invalid types
            ts = data[ts_key]
            if not isinstance(ts, schema.get(Keys.DTYPE, pd.DataFrame)):
                logger.error(f"Timeseries '{ts_key}' must be of type '{schema.get(Keys.DTYPE)}'.")
                if raise_dtype_errors:
                    raise ValueError(f"Timeseries '{ts_key}' must be of type '{schema.get(Keys.DTYPE)}'.")

            for col, col_dtype in schema.get(Keys.COLS_REQUIRED, {}).items():
                if col not in ts.columns:
                    logger.error(f"Missing column '{col}' in timeseries '{ts_key}'.")
                    raise ValueError(f"Timeseries '{ts_key}' is missing required column '{col}'.")
                if not pd.api.types.is_dtype_equal(type(ts[col].iat[0]), col_dtype):
                    logger.info(f"Column '{col}' in timeseries '{ts_key}' must be of type '{col_dtype}'.")
                    if raise_dtype_errors:
                        raise ValueError(f"Column '{col}' in timeseries '{ts_key}' must be of type '{col_dtype}'.")

            for col, col_dtype in schema.get(Keys.COLS_OPTIONAL, {}).items():
                if col in ts.columns and not pd.api.types.is_dtype_equal(type(ts[col].iat[0]), col_dtype):
                    logger.info(f"Column '{col}' in timeseries '{ts_key}' must be of type '{col_dtype}'.")
                    if raise_dtype_errors:
                        raise ValueError(f"Column '{col}' in timeseries '{ts_key}' must be of type '{col_dtype}'.")
