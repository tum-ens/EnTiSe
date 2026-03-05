file
====

Overview
--------

Implements a method for loading time series data from external sources.

This class provides functionality to load time series data from external sources
using a file key. It is useful for injecting external time series into the
simulation pipeline.

The class follows the Method pattern defined in the EnTiSe framework, implementing
the required interface for time series generation methods.

Attributes:
types (list): List of time series types this method can generate (all valid types).
name (str): Name identifier for the method.
required_keys (list): Required input parameters (id, file).
required_data (list): Required time series inputs (file).
output_summary (dict): Empty dictionary as no summary is provided.
output_timeseries (dict): Empty dictionary as the output format depends on the input.

Example:
>>> from entise.methods.multiple.file import FileLoader
>>> from entise.core.generator import Generator
>>>
>>> # Create a generator and add objects
>>> gen = Generator()
>>> gen.add_objects(objects_df)  # DataFrame with file parameters
>>>
>>> # Generate time series
>>> summary, timeseries = gen.generate(data)  # data contains external time series


Key facts
---------

- Method key: ``file``

- Supported types:

  
  - ``occupancy``
  
  - ``pv``
  
  - ``hvac``
  
  - ``hp``
  
  - ``wind``
  
  - ``mobility``
  
  - ``heating``
  
  - ``dhw``
  
  - ``electricity``
  
  - ``cooling``
  


Requirements
------------

Required keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``id``
- ``filename``



Optional keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- None


Required data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``filename``



Optional data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- None


Outputs
-------

Summary metrics
~~~~~~~~~~~~~~~


- None


Timeseries columns
~~~~~~~~~~~~~~~~~~


- None


Public methods
--------------


- generate

  .. code-block:: python

         def generate(
           self, obj: dict = None, data: dict = None, results: dict = None, ts_type: str = None, *, file: str = None
       ):
           """Load time series data from external sources.

           This method implements the abstract generate method from the Method base class.
           It processes the input parameters, loads the time series data, and returns it.

           Args:
               obj (dict, optional): Dictionary containing file parameters. Defaults to None.
               data (dict, optional): Dictionary containing input data. Defaults to None.
               results (dict, optional): Dictionary with results from previously generated time series
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
           processed_obj, processed_data = self._process_kwargs(obj, data, file=file)

           # Continue with existing implementation
           processed_obj, processed_data = get_input_data(processed_obj, processed_data, ts_type)

           # Load the time series
           timeseries = load_timeseries(processed_obj, processed_data)

           return {
               "summary": {},
               "timeseries": timeseries,
           }

