FixedSchedule
=========================


**Method Key:** ``FixedSchedule``

.. note::
   This is the key required to call this method when using bulk generation with TimeSeriesGenerator.


Description
-----------

Generate a fixed occupancy schedule based on time of day.

This method creates an occupancy schedule where occupancy is 0 during working hours (8-18)
and 1 otherwise.

Args:
    obj (dict): Object metadata and parameters.
    data (dict): Dictionary of available timeseries data.
    ts_type (str, optional): The timeseries type being processed.

Returns:
    dict: A dictionary containing:
        - summary: Dictionary with average occupancy metrics
        - timeseries: DataFrame with occupancy values for each timestep

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
   
   * - ``weather``
     - ``str``
   



Required Timeseries
~~~~~~~~~~~~~~~~~~~



**Timeseries Key:** ``weather``












Dependencies
-------------


- None


Methods
-------


**generate**:


  .. code-block:: python

         def generate(self, obj, data, ts_type=None):
           """
           Generate a fixed occupancy schedule based on time of day.

           This method creates an occupancy schedule where occupancy is 0 during working hours (8-18)
           and 1 otherwise.

           Args:
               obj (dict): Object metadata and parameters.
               data (dict): Dictionary of available timeseries data.
               ts_type (str, optional): The timeseries type being processed.

           Returns:
               dict: A dictionary containing:
                   - summary: Dictionary with average occupancy metrics
                   - timeseries: DataFrame with occupancy values for each timestep
           """
           index = data[O.WEATHER].index
           occ = [0 if 8 <= t.hour < 18 else 1 for t in index]
           df = pd.DataFrame({f'{O.OCCUPATION}': occ}, index=index)
           df.index.name = C.DATETIME
           return {
               "summary": {
                   f'average_{O.OCCUPATION}': df[f'{O.OCCUPATION}'].mean().round(2),
               },
               "timeseries": df
           }


