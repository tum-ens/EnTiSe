from src.constants import VALID_TYPES
from abc import ABCMeta

method_registry = {}


class Meta(ABCMeta, type):
    """
    Metaclass to automatically register timeseries methods.

    Any class that inherits from this metaclass and is not the base `TimeSeriesMethod`
    will be automatically registered in the `method_registry`.

    Attributes:
        method_registry (dict): A global registry mapping method names (lowercased)
            to their respective classes.
    """

    def __init__(cls, name, bases, dct):
        """
        Initialize the metaclass and register methods automatically.

        Args:
            name (str): The name of the class being initialized.
            bases (tuple): Base classes of the class being initialized.
            dct (dict): Class attributes and methods.

        Raises:
            ValueError: If any `supported_types` specified in the class are invalid.
        """
        super().__init__(name, bases, dct)
        if cls.__name__ != "TimeSeriesMethod":  # Skip the base class itself
            # Validate supported_types during registration
            for t in getattr(cls, "supported_types", []):
                if t not in VALID_TYPES:
                    raise ValueError(
                        f"Invalid type '{t}' in {cls.__name__}. Must be one of {VALID_TYPES}."
                    )

            # Register the method
            method_registry[cls.__name__.lower()] = cls


def register_method(name: str, method):
    """
    Register a method in the global method registry.

    Args:
        name (str): The name of the method to register.
        method (class): The class representing the method to be registered.

    Raises:
        ValueError: If the method name is already registered.
        ValueError: If any dependencies specified in the method are invalid.

    Example:
        >>> class MockMethod:
        ...     dependencies = ["dependency1"]
        ... register_method("mock_method", MockMethod)
        Registered method: mock_method
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
    Retrieve a method from the registry by its name.

    Args:
        name (str): The name of the method to retrieve.

    Returns:
        class: The registered method class.

    Raises:
        ValueError: If the method is not found in the registry.

    Example:
        >>> get_method("mock_method")
        <class '__main__.MockMethod'>
    """
    name = name.lower()  # Normalize name to lowercase
    if name not in method_registry:
        raise ValueError(f"Method '{name}' not found in registry.")
    return method_registry[name]


def get_methods_by_type(timeseries_type: str):
    """
    Retrieve all registered methods that support a given timeseries type.

    Args:
        timeseries_type (str): The type of timeseries to filter for.

    Returns:
        list: A list of method classes that support the specified timeseries type.

    Example:
        >>> class MockMethod:
        ...     supported_types = ["hvac"]
        ... register_method("mock_method", MockMethod)
        >>> get_methods_by_type("hvac")
        [<class '__main__.MockMethod'>]
    """
    return [
        method for method in method_registry.values()
        if timeseries_type in getattr(method, "supported_types", [])
    ]
