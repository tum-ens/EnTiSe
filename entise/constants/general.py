class Keys:
    """General-purpose keys used in the tool.
    Sorted alphabetically for easy reference.
    """

    OPTIONAL = "optional"  # Key for specifying optional keys
    REQUIRED = "required"  # Key for specifying required keys

    COLUMNS = "columns"
    DATA = "data"
    KEYS = "keys"  # Key for specifying required keys
    TIMESERIES = "timeseries"  # Key for timeseries data

    DTYPE = "dtype"  # Key for specifying data type
    SUMMARY = "summary"  # Key for summary metrics

    COLS_REQUIRED = f"{REQUIRED}_{COLUMNS}"  # Key for specifying required columns
    COLS_OPTIONAL = f"{OPTIONAL}_{COLUMNS}"  # Key for specifying optional columns
    DATA_REQUIRED = f"{REQUIRED}_{DATA}"  # Key for specifying required data
    DATA_OPTIONAL = f"{OPTIONAL}_{DATA}"  # Key for specifying optional data
    KEYS_REQUIRED = f"{REQUIRED}_{KEYS}"  # Key for specifying required keys
    KEYS_OPTIONAL = f"{OPTIONAL}_{KEYS}"  # Key for specifying optional keys


SEP = ":"  # Separator for any type of keys
