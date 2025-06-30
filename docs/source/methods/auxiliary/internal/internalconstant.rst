internalconstant
=========================


**Method Key:** ``internalconstant``

.. note::
   This is the class name. For auxiliary methods, the key is determined by the selector class.


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

   * - ``gains_internal``
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

         def generate(self, obj, data):
           gains_internal = obj.get(O.GAINS_INTERNAL, DEFAULT_GAINS_INTERNAL)
           try:
               gains_internal = float(gains_internal)
           except ValueError:
               pass
           finally:
               if isinstance(gains_internal, str):
                   # If a string key is given, assume it's a reference to a time series
                   return InternalTimeSeries().generate(obj, data)
           return self.run(**self.get_input_data(obj, data))



**get_input_data**:


  .. code-block:: python

         def get_input_data(self, obj, data):
           return {
               O.GAINS_INTERNAL: obj.get(O.GAINS_INTERNAL, DEFAULT_GAINS_INTERNAL),
               O.WEATHER: data[O.WEATHER],
           }



**run**:


  .. code-block:: python

         def run(self, gains_internal, weather):
           return pd.DataFrame(
               {O.GAINS_INTERNAL: np.full(len(weather), gains_internal, dtype=np.float32)},
               index=weather.index
           )
