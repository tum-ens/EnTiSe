import logging
import re
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Tuple, Type

import pandas as pd

from entise.constants import SEP, VALID_TYPES, Keys
from entise.constants import Columns as C
from entise.constants import Objects as O

logger = logging.getLogger(__name__)

# Global registry
method_registry: Dict[str, Type["Method"]] = {}

_UNIT = re.compile(r"\[[^\]]*\]")
_HEIGHT = re.compile(r"@\d+(?:\.\d+)?m$")

# Site metadata of an object: read without a type prefix by every method.
SITE_KEYS = (O.ID, O.TIMEZONE)


def input_name(key: str) -> str:
    """Return the Python name of an input key: the key without its units and height.

    Example:
        ``area[m2]`` becomes ``area`` and ``air_temperature[C]@2m`` becomes ``air_temperature``.
    """
    return _UNIT.sub("", _HEIGHT.sub("", key))


def _scopes(ts_type: str | Sequence[str] | None) -> Tuple[str, ...]:
    """Return the type prefixes of a call: none, one type, or several."""
    if ts_type is None:
        return ()
    return (ts_type,) if isinstance(ts_type, str) else tuple(ts_type)


def _is_missing(value: Any) -> bool:
    """Whether an object value counts as not given: None, or a missing scalar such as NaN or pd.NA."""
    return value is None or (pd.api.types.is_scalar(value) and bool(pd.isna(value)))


def _same_value(a: Any, b: Any) -> bool:
    """Whether two object values are equal; values that cannot be compared are not."""
    if a is b:
        return True
    if isinstance(a, (pd.Series, pd.DataFrame)) and type(a) is type(b):
        return a.equals(b)
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def _check_input_names(cls) -> None:
    """Raise if two inputs of a method share a Python name or an alias is ambiguous."""
    keys = list(
        dict.fromkeys(
            [
                *getattr(cls, "required_keys", []),
                *getattr(cls, "optional_keys", []),
                *getattr(cls, "required_data", []),
                *getattr(cls, "optional_data", []),
            ]
        )
    )
    owner = {key: key for key in keys}
    for key in keys:
        name = input_name(key)
        if owner.setdefault(name, key) != key:
            raise TypeError(f"{cls.__name__}: inputs '{owner[name]}' and '{key}' both have the Python name '{name}'.")
    for alias, key in getattr(cls, "input_aliases", {}).items():
        if key not in keys:
            raise TypeError(f"{cls.__name__}: alias '{alias}' names '{key}', which is not a declared input.")
        if owner.get(alias, key) != key:
            raise TypeError(f"{cls.__name__}: alias '{alias}' for '{key}' is already a name of '{owner[alias]}'.")


class MethodMeta(ABCMeta):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        if cls.__name__ == "Method":
            return
        _check_input_names(cls)
        if getattr(cls, "register", True) is False:
            return
        for t in getattr(cls, "types", []):
            if t not in VALID_TYPES:
                raise ValueError(f"Invalid type '{t}' in {cls.__name__}. Must be one of {VALID_TYPES}.")
        method_registry[cls.name.lower()] = cls


class Method(ABC, metaclass=MethodMeta):
    types: List[str]
    name: str = ""
    required_keys: List[str]
    # A list, or a mapping of each key to its default (see ``key_defaults``).
    optional_keys: List[str] | Dict[str, Any] = []
    required_data: List[str] = []
    optional_data: List[str] = []
    output_summary: Dict[str, pd.DataFrame] = {}
    output_timeseries: Dict[str, pd.DataFrame]
    # Python names kept from earlier releases, mapped to the input key they set.
    input_aliases: Dict[str, str] = {}

    @abstractmethod
    def generate(
        self, obj: dict = None, data: dict = None, results: dict = None, ts_type: str = None, **inputs
    ) -> Dict[str, Any]:
        """Generate the time series of one object.

        Args:
            obj (dict, optional): Object keys.
            data (dict, optional): Input tables.
            results (dict, optional): Results of the time series already generated for the object.
            ts_type (str, optional): Time series type to generate.
            **inputs: Any declared input, by its key or by its Python name (see ``input_names``).
                Keywords override the same keys in ``obj``.
        """
        raise NotImplementedError("Method 'generate' must be implemented.")

    @classmethod
    def input_names(cls) -> Dict[str, str]:
        """Return every name ``generate`` accepts for an input, mapped to the input's key.

        Each declared input is accepted by its key, by its Python name (see ``input_name``),
        and by the names in ``input_aliases``.
        """
        keys = dict.fromkeys([*cls.required_keys, *cls.optional_keys, *cls.required_data, *cls.optional_data])
        names = {key: key for key in keys}
        for key in keys:
            names.setdefault(input_name(key), key)
        names.update(cls.input_aliases)
        return names

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

    @classmethod
    def key_defaults(cls) -> Dict[str, Any]:
        """Return each declared key mapped to its default, in declaration order.

        Required keys have no default. ``optional_keys`` either maps each key to its
        default or lists the keys, which then have none.
        """
        defaults = dict.fromkeys(cls.required_keys)
        optional = cls.optional_keys
        defaults.update(optional if isinstance(optional, Mapping) else dict.fromkeys(optional))
        return defaults

    @classmethod
    def get_relevant_objects(cls, obj: Mapping, ts_type: str | Sequence[str] | None = None) -> dict:
        """Resolve the declared keys of the method from one object.

        Each key in ``required_keys`` and ``optional_keys`` takes the first value it finds,
        in this order:

        1. the key prefixed with the type, ``<ts_type>:<key>``;
        2. the key itself;
        3. its default (see ``key_defaults``).

        ``None`` and NaN count as missing, while ``False`` and ``0`` are values. The site
        keys ``id`` and ``timezone`` are always read without a prefix. Keys without a value
        are left out, and undeclared keys are never returned. Values are returned as given:
        a table reference such as ``"table|column"`` is resolved by the method.

        Args:
            obj: The object's keys.
            ts_type: The type the method generates, or several types that one call
                generates together. Without a type, only unprefixed keys are read.

        Returns:
            dict: The site keys first, then the declared keys in declaration order.

        Raises:
            ValueError: If a key is given under two of the types with different values.
        """
        scopes = _scopes(ts_type)
        resolved = {}
        for key in SITE_KEYS:
            value = cls.get_with_backup(obj, key)
            if value is not None:
                resolved[key] = value
        for key, default in cls.key_defaults().items():
            if key in SITE_KEYS:
                continue
            value = cls._scoped_value(obj, key, scopes)
            if value is None:
                value = cls.get_with_backup(obj, key, default)
            if value is not None:
                resolved[key] = value
        return resolved

    @classmethod
    def _scoped_value(cls, obj: Mapping, key: str, scopes: Tuple[str, ...]) -> Any:
        """Return the value of a key given under any of the scopes, or None."""
        found = None
        for scope in scopes:
            scoped = f"{scope}{SEP}{key}"
            value = cls.get_with_backup(obj, scoped)
            if value is None:
                continue
            if found is None:
                found = (scoped, value)
            elif not _same_value(found[1], value):
                raise ValueError(f"'{found[0]}' ({found[1]!r}) and '{scoped}' ({value!r}) give different values.")
        return None if found is None else found[1]

    def _process_kwargs(self, obj: dict = None, data: dict = None, ts_type: str = None, **inputs) -> Tuple[dict, dict]:
        """Apply the keyword inputs of a direct call to copies of obj and data.

        See ``_process_inputs``. A method with an input that another type produces,
        such as the electricity series of the occupancy methods, calls
        ``_process_inputs`` instead.

        Returns:
            Tuple[dict, dict]: Processed obj and data dictionaries.
        """
        local_obj, local_data, results = self._process_inputs(obj, data, None, ts_type, **inputs)
        if results:
            raise TypeError(f"{type(self).__name__} takes results of other types; call _process_inputs.")
        return local_obj, local_data

    def _process_inputs(
        self, obj: dict = None, data: dict = None, results: dict = None, ts_type: str = None, **inputs
    ) -> Tuple[dict, dict, dict]:
        """Apply the keyword inputs of a direct call to copies of obj, data and results.

        ``obj`` may be a dict, a row of an objects table (a Series) or None; the copy is a
        dict. A keyword sets the input it names (see ``input_names``) and overrides the
        same key in ``obj``, including the key prefixed with ``ts_type``. A ``None`` value
        counts as not given. For a data input:

        - a string is the name of a table in ``data``; ``obj`` refers to it;
        - a table (DataFrame or mapping) goes into ``data`` under the input's key, and
          ``obj`` refers to it by that key;
        - for an input that another type produces, such as ``electricity``, the table
          becomes that type's result in ``results``.

        Any other value goes into ``obj``, or for an input that only ``data`` holds,
        into ``data`` as a table.

        Args:
            obj (dict or pd.Series, optional): Initial object. Defaults to None.
            data (dict, optional): Initial data dictionary. Defaults to None.
            results (dict, optional): Results of the types generated before. Defaults to None.
            ts_type (str, optional): Time series type the method generates. Defaults to None.
            **inputs: Inputs by key or Python name.

        Returns:
            Tuple[dict, dict, dict]: Processed obj, data and results. ``results`` is the
            given mapping unless a keyword sets one of its entries.

        Raises:
            TypeError: If a keyword names no input of the method, or two keywords name the same input.
        """
        local_obj = {} if obj is None else dict(obj)
        local_data = {} if data is None else dict(data)
        if not inputs:
            return local_obj, local_data, results

        names = self.input_names()
        unknown = [name for name in inputs if name not in names]
        if unknown:
            raise TypeError(
                f"{type(self).__name__}.generate() got an unexpected keyword argument '{unknown[0]}'. "
                f"Inputs of method '{self.name}': {self._describe_inputs(names)}."
            )

        obj_keys = set(self.key_defaults())
        data_keys = set(self.required_data + self.optional_data)
        given: Dict[str, str] = {}
        for name, value in inputs.items():
            if value is None:
                continue
            key = names[name]
            if key in given:
                raise TypeError(
                    f"{type(self).__name__}.generate() got multiple values for input '{key}': "
                    f"'{given[key]}' and '{name}'."
                )
            given[key] = name

            if key in data_keys and key in VALID_TYPES:
                # Another type produces this input; the method reads it from the results.
                table = local_data[value] if isinstance(value, str) else value
                results = {**(results or {}), key: {Keys.TIMESERIES: table}}
                continue
            if key in data_keys and not isinstance(value, str):
                if key not in obj_keys or isinstance(value, (pd.DataFrame, Mapping)):
                    local_data[key] = value
                    value = key  # the object refers to the table by its key
            local_obj[key] = value
            for scope in _scopes(ts_type):
                local_obj.pop(f"{scope}{SEP}{key}", None)

        return local_obj, local_data, results

    @staticmethod
    def _describe_inputs(names: Dict[str, str]) -> str:
        """List the Python names of the inputs, each with its key where that differs."""
        described = [name if name == key else f"{name} ({key})" for name, key in names.items() if name.isidentifier()]
        return ", ".join(sorted(described, key=str.lower))

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
    def get_with_backup(obj: dict, key: str, backup: Any = None):
        """Get a value from a dictionary with a backup value if not found or None.

        Args:
            obj (dict): Dictionary to get value from
            key (str): Key to look for
            backup: Default value if key doesn't exist or value is None/NaN

        Returns:
            The value from the dictionary, or the backup value if not found
        """
        value = obj.get(key, backup)
        if _is_missing(value):
            return backup
        return value

    @classmethod
    def get_with_method_backup(cls, obj: dict, key: str, method_type: str, backup: Any = None):
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
            Keys.KEYS_OPTIONAL: list(cls.optional_keys),
            Keys.DATA_REQUIRED: cls.required_data,
            Keys.DATA_OPTIONAL: cls.optional_data,
        }
