ventilationinactive
===================

Overview
--------

Represents a ventilation mechanism providing inactive ventilation calculations.

This class processes weather data to produce inactive ventilation outputs.
It fetches the required input data and computes the results as a DataFrame
containing ventilation values initialized to zeros. The class is an
extension of the `AuxiliaryMethod` class.


Key facts
---------

- Method key: ``VentilationInactive``


Requirements
------------

Required keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- None


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


- get_input_data

  .. code-block:: python

         def get_input_data(self, obj, data):
           return {O.WEATHER: data[O.WEATHER]}


- run

  .. code-block:: python

         def run(self, weather):
           return pd.DataFrame({O.VENTILATION: np.zeros(len(weather), dtype=np.float32)}, index=weather.index)

