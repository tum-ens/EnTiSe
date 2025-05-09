File
=========================

Description
-----------

Timeseries generation method to load and return a timeseries from a file.

Requirements
-------------

Required Keys
~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Key
     - Type
   
   * - ``file``
     - ``str``
   


Required Timeseries
~~~~~~~~~~~~~~~~~~~


**Timeseries Key:** ``file``






**DataFrame Type:** ``DataFrame``






Dependencies
-------------


- None


Methods
-------


**generate**:

  .. code-block:: none

     Generate a timeseries by loading it from a file.

   Args:
   - obj (dict): Object metadata and parameters.
   - data (dict): Dictionary of available timeseries data.
   - ts_type (str): The timeseries type being processed.
   - kwargs: Additional keyword arguments.

   Returns:
   - (dict, pd.DataFrame): Summary metrics (empty for this method) and the loaded timeseries.


**get_relevant_objects**:

  .. code-block:: none

     Retrieve only the relevant object keys for the method.

   Args:
       obj (dict): The input object metadata and parameters.
       ts_type (str, optional): The timeseries type being processed.

   Returns:
       dict: A dictionary containing the relevant object keys.


**prepare_inputs**:

  .. code-block:: none

     Prepare and validate the inputs for timeseries generation.

   Args:
       obj (dict): The input object metadata and parameters.
       data (dict): The input data dictionary.
       ts_type (str): The type of timeseries being processed.

   Returns:
       dict: A reduced and validated object dictionary.

   Raises:
       ValueError: If validation of the inputs fails.


**resolve_column**:

  .. code-block:: none

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

