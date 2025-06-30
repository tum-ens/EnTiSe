file
=========================


**Method Key:** ``file``

.. note::
   This is the key required to call this method when using bulk generation with TimeSeriesGenerator.


Description
-----------

Load time series data from external sources.

This method implements the abstract generate method from the Method base class.
It processes the input parameters, loads the time series data, and returns it.

Args:
    obj (dict, optional): Dictionary containing file parameters. Defaults to None.
    data (dict, optional): Dictionary containing input data. Defaults to None.
    ts_type (str, optional): Time series type to generate. Defaults to None.
    file (str, optional): Key to use for loading the time series. Defaults to None.

Returns:
    dict: Dictionary containing:
        - "summary" (dict): Empty dictionary as no summary is provided.
        - "timeseries" (pd.DataFrame): The loaded time series data.

Raises:
    ValueError: If required data is missing.
    TypeError: If the loaded data is not a DataFrame.

Example:
    >>> fileloader = FileLoader()
    >>> # Using explicit parameters
    >>> result = fileloader.generate(file="external_input", data={"external_input": df})
    >>> # Or using dictionaries
    >>> obj = {"file": "external_input"}
    >>> data = {"external_input": df}
    >>> result = fileloader.generate(obj=obj, data=data)
    >>> timeseries = result["timeseries"]

Requirements
-------------

Required Keys
~~~~~~~~~~~~~


.. list-table::
   :widths: auto
   :header-rows: 1

   * - Key
     - Type

   * - ``id``
     - ``str``

   * - ``filename``
     - ``str``




Required Timeseries
~~~~~~~~~~~~~~~~~~~



**Timeseries Key:** ``filename``












Dependencies
-------------


- None


Methods
-------


**generate**:


  .. code-block:: python

         def generate(self,
                   obj: dict = None,
                   data: dict = None,
                   ts_type: str = None,
                   *,
                   file: str = None):
           """Load time series data from external sources.

           This method implements the abstract generate method from the Method base class.
           It processes the input parameters, loads the time series data, and returns it.

           Args:
               obj (dict, optional): Dictionary containing file parameters. Defaults to None.
               data (dict, optional): Dictionary containing input data. Defaults to None.
               ts_type (str, optional): Time series type to generate. Defaults to None.
               file (str, optional): Key to use for loading the time series. Defaults to None.

           Returns:
               dict: Dictionary containing:
                   - "summary" (dict): Empty dictionary as no summary is provided.
                   - "timeseries" (pd.DataFrame): The loaded time series data.

           Raises:
               ValueError: If required data is missing.
               TypeError: If the loaded data is not a DataFrame.

           Example:
               >>> fileloader = FileLoader()
               >>> # Using explicit parameters
               >>> result = fileloader.generate(file="external_input", data={"external_input": df})
               >>> # Or using dictionaries
               >>> obj = {"file": "external_input"}
               >>> data = {"external_input": df}
               >>> result = fileloader.generate(obj=obj, data=data)
               >>> timeseries = result["timeseries"]
           """
           # Process keyword arguments
           processed_obj, processed_data = self._process_kwargs(
               obj, data,
               file=file
           )

           # Continue with existing implementation
           processed_obj, processed_data = get_input_data(processed_obj, processed_data, ts_type)

           # Load the time series
           timeseries = load_timeseries(processed_obj, processed_data)

           return {
               "summary": {},
               "timeseries": timeseries,
           }
