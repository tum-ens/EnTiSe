def supported_types(*types):
    """
    Decorator to register timeseries methods for specific types.

    This decorator validates the provided timeseries types and assigns them to
    the `supported_types` attribute of the decorated class. It ensures that all
    specified types are part of the predefined `VALID_TYPES` constant.

    Args:
        *types (str): One or more strings representing supported timeseries types.

    Returns:
        class: The decorated class with the `supported_types` attribute attached.

    Raises:
        ValueError: If any of the provided types are not part of `VALID_TYPES`.

    Example:
        >>> from src.constants import VALID_TYPES
        >>> VALID_TYPES = ["HVAC", "DHW"]

        >>> @supported_types("HVAC", "DHW")
        ... class HeatingMethod:
        ...     pass

        >>> HeatingMethod.supported_types
        ('HVAC', 'DHW')
    """

    def wrapper(cls):
        from src.constants import VALID_TYPES

        # Validate each provided type
        for t in types:
            if t not in VALID_TYPES:
                raise ValueError(
                    f"Invalid timeseries type: {t}. Must be one of {VALID_TYPES}."
                )

        # Attach supported types to the class
        cls.supported_types = types
        return cls

    return wrapper
