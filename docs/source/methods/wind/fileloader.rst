FileLoader
=========================


**Method Key:** ``file``

.. note::
   This is the key required to call this method when using bulk generation with TimeSeriesGenerator.


Description
-----------



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


