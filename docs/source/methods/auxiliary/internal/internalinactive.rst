InternalInactive
=========================


**Method Key:** ``InternalInactive``

.. note::
   This is the class name. For auxiliary methods, the key is determined by the selector class.


Description
-----------



Requirements
-------------

Required Keys
~~~~~~~~~~~~~


None



Required Timeseries
~~~~~~~~~~~~~~~~~~~



**Timeseries Key:** ``weather``












Dependencies
-------------


- None


Methods
-------


**get_input_data**:


  .. code-block:: python

         def get_input_data(self, obj, data):
           return {O.WEATHER: data[O.WEATHER]}



**run**:


  .. code-block:: python

         def run(self, weather):
           return pd.DataFrame({O.GAINS_INTERNAL: np.zeros(len(weather), dtype=np.float32)}, index=weather.index)


