def supported_types(*types):
    """
    Decorator to register timeseries methods for specific types.

    Parameters:
    - *types: One or multiple timeseries types (strings).
    """

    def wrapper(cls):
        from src.constants import VALID_TYPES

        # Validate each type
        for t in types:
            if t not in VALID_TYPES:
                raise ValueError(f"Invalid timeseries type: {t}. Must be one of {VALID_TYPES}.")

        # Attach supported types to the class
        cls.supported_types = types
        return cls

    return wrapper

