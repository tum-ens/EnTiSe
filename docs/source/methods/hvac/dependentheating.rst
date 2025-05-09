DependentHeating
=========================

Description
-----------

Abstract base class for timeseries generation methods.

Subclasses must implement the `generate` method and define the class-level attributes:
- `required_keys`: Dictionary specifying the required object keys and their types.
- `required_timeseries`: Dictionary specifying required timeseries data and columns.
- `dependencies`: List of dependencies required for this method.
- `available_outputs`: Dictionary specifying available outputs (summary and timeseries).

Requirements
-------------

- **Required Keys**:

  - `weather`: str


- **Required Timeseries**:

  - `weather`:
    
      - columns: {'datetime': <class 'pandas._libs.tslibs.timestamps.Timestamp'>}
    
      - dtype: <class 'pandas.core.frame.DataFrame'>
    

  - `electricity`:
    
      - columns: {'load': <class 'float'>}
    
      - dtype: <class 'pandas.core.frame.DataFrame'>
    


Dependencies
-------------


- electricity


Methods
-------


- **generate**:
  Generate the heating timeseries.

Parameters:
- obj (dict): Objects-specific metadata.
- data (dict): Timeseries data.
- dependencies (dict): Generated timeseries for dependencies.

Returns:
- (dict, pd.DataFrame): Metrics and the generated timeseries.

- **get_relevant_objects**:
  Retrieve only the relevant object keys for the method.

Args:
    obj (dict): The input object metadata and parameters.
    ts_type (str, optional): The timeseries type being processed.

Returns:
    dict: A dictionary containing the relevant object keys.

- **prepare_inputs**:
  Prepare and validate the inputs for timeseries generation.

Args:
    obj (dict): The input object metadata and parameters.
    data (dict): The input data dictionary.
    ts_type (str): The type of timeseries being processed.

Returns:
    dict: A reduced and validated object dictionary.

Raises:
    ValueError: If validation of the inputs fails.

- **resolve_column**:
  Resolve a column from the input data, checking for prefixed and shared columns.

Args:
    ts_key (str): The key identifying the timeseries data.
    column (str): The column name to resolve.
    ts_type (str): The type of timeseries (used for prefixing).
    data (dict): The input data dictionary.

Returns:
    pd.Series: The resolved column data.

Raises:
    ValueError: If the specified column is not found.
