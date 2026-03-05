internalconstant
================

Overview
--------

Handles the internal gains calculations and manipulations in relation to weather data.
This class is responsible for processing, generating, and managing internal gains
data, which could be derived from constants or referenced time series data. The
class extends `AuxiliaryMethod` to utilize its auxiliary functionalities and is
designed to be used in scenarios where internal gains need to be modeled or
analyzed.


Key facts
---------

- Method key: ``InternalConstant``


Requirements
------------

Required keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``gains_internal[W]``



Optional keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- None


Required data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``weather``



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

         def generate(self, obj, data):
           gains_internal = obj.get(O.GAINS_INTERNAL, DEFAULT_GAINS_INTERNAL)
           try:
               gains_internal = float(gains_internal)
           except ValueError:
               pass
           if isinstance(gains_internal, str):
               # If a string key is given, assume it's a reference to a time series
               return InternalTimeSeries().generate(obj, data)
           return self.run(**self.get_input_data(obj, data))


- get_input_data

  .. code-block:: python

         def get_input_data(self, obj, data):
           return {
               "gains_internal": obj.get(O.GAINS_INTERNAL, DEFAULT_GAINS_INTERNAL),
               "weather": data[O.WEATHER],
           }


- run

  .. code-block:: python

         def run(self, gains_internal, weather):
           return pd.DataFrame(
               {O.GAINS_INTERNAL: np.full(len(weather), gains_internal, dtype=np.float32)}, index=weather.index
           )

