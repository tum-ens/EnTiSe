ventilationconstant
===================

Overview
--------

Handles the ventilation calculations and manipulations in relation to weather data.
This class is responsible for processing, generating, and managing ventilation
data, which could be derived from constants or referenced time series data. The
class extends `AuxiliaryMethod` to utilize its auxiliary functionalities and is
designed to be used in scenarios where ventilation needs to be modeled or
analyzed.


Key facts
---------

- Method key: ``VentilationConstant``


Requirements
------------

Required keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``ventilation[W K-1]``



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
           ventilation = obj.get(O.VENTILATION, DEFAULT_VENTILATION)
           is_string = False

           try:
               ventilation = float(ventilation)
           except ValueError:
               is_string = True

           if is_string or isinstance(ventilation, str):
               # If a string key is given, assume it's a reference to a time series
               return VentilationTimeSeries().generate(obj, data)

           return self.run(**self.get_input_data(obj, data))


- get_input_data

  .. code-block:: python

         def get_input_data(self, obj, data):
           return {
               "ventilation": obj.get(O.VENTILATION, DEFAULT_VENTILATION),
               "weather": data[O.WEATHER],
           }


- run

  .. code-block:: python

         def run(self, ventilation, weather):
           return pd.DataFrame({O.VENTILATION: np.full(len(weather), ventilation, dtype=np.float32)}, index=weather.index)

