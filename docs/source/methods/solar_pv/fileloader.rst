FileLoader
=========================


**Method Key:** ``file``

.. note::
   This is the key required to call this method when using bulk generation with TimeSeriesGenerator.


Description
-----------



Usage
-----

This method can be used in two ways:

1. **Through TimeSeriesGenerator**:

   .. code-block:: python

      from entise.core.generator import TimeSeriesGenerator

      # Initialize the generator
      gen = TimeSeriesGenerator()

      # Add objects
      gen.add_objects({
          "id": "external_data_1",
          "file": "external_input",
      })

      # Generate timeseries
      summary, df = gen.generate(data)  # data contains external time series

2. **Direct Import**:

   .. code-block:: python

      from entise.methods.multiple import FileLoader

      # Create an instance
      file_loader = FileLoader()

      # Using dictionaries
      result = file_loader.generate(
          obj={"id": "external_data_1", "file": "external_input"},
          data={"external_input": df}
      )

      # Using named parameters
      result = file_loader.generate(
          file="external_input",
          data={"external_input": df}
      )

      # Combining both approaches
      obj = {"id": "external_data_1"}
      result = file_loader.generate(
          obj=obj,
          file="external_input",  # This overrides any "file" value in obj
          data={"external_input": df}
      )

      # Access results
      summary = result["summary"]
      timeseries = result["timeseries"]

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

   * - ``file``
     - ``str``




Required Timeseries
~~~~~~~~~~~~~~~~~~~



**Timeseries Key:** ``file``












Dependencies
-------------


- None


Methods
-------


**generate**:


  .. code-block:: python

         def generate(self, obj, data, ts_type=None):
           key = obj.get(O.FILE)
           if key not in data:
               raise ValueError(f"FileLoader expected timeseries key '{key}' to be present in input data.")

           df = data[key]

           if not isinstance(df, pd.DataFrame):
               raise TypeError(f"Expected a DataFrame for key '{key}', but got {type(df).__name__}.")

           return {
               "summary": {},
               "timeseries": df,
           }
