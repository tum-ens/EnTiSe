from src.constants import VALID_TYPES
from abc import ABCMeta
method_registry = {}


class Meta(ABCMeta, type):
    """Metaclass to automatically register timeseries methods."""

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        if cls.__name__ != "TimeSeriesMethod":  # Skip the base class itself
            # Validate supported_types during registration
            for t in getattr(cls, "supported_types", []):
                if t not in VALID_TYPES:
                    raise ValueError(f"Invalid type '{t}' in {cls.__name__}. Must be one of {VALID_TYPES}.")

            # Register the method
            method_registry[cls.__name__.lower()] = cls


def register_method(name: str, method):
    """
    Register a method in the registry.

    Parameters:
    - name (str): Name of the method.
    - method (class): Method class to register.
    """
    name = name.lower()  # Normalize name
    if name in method_registry:
        raise ValueError(f"Method '{name}' is already registered.")
    # Validate dependencies
    for dep in getattr(method, "dependencies", []):
        if dep not in VALID_TYPES:
            raise ValueError(f"Dependency '{dep}' in '{name}' is invalid.")
    method_registry[name] = method
    print(f"Registered method: {name}")  # Replace with logging if desired


def get_method(name: str):
    """
    Retrieve a method from the registry by name. Lazily loads methods if the registry is empty.

    Parameters:
    - name (str): Name of the method.

    Returns:
    - class: The registered method class.
    """
    name = name.lower()  # Normalize name to lowercase
    if name not in method_registry:
        raise ValueError(f"Method '{name}' not found in registry.")
    return method_registry[name]


def get_methods_by_type(timeseries_type: str):
    """
    Retrieve all methods that support a given timeseries type.

    Parameters:
    - timeseries_type (str): The type of timeseries to search for.

    Returns:
    - list: List of methods that support the given timeseries type.
    """
    return [
        method for method in method_registry.values()
        if timeseries_type in getattr(method, "supported_types", [])
    ]
