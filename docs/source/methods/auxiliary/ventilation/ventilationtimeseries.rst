ventilationtimeseries
=====================

Overview
--------

Represents the processing of ventilation time series data for further computations.

This class is designed to handle, manipulate, and process ventilation time series
data provided as input. It validates the data, ensures it conforms to the expected
formats, and executes transformations or auxiliary methods as necessary. It inherits
from the `AuxiliaryMethod` base class and relies heavily on specific keys and ventilation
structures for its operations.


Key facts
---------

- Method key: ``VentilationTimeSeries``


Requirements
------------

Required keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``ventilation_column``



Optional keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``id``
- ``area[m2]``
- ``height[m]``



Required data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``ventilation[W K-1]``



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
           ventilation = obj.get(O.VENTILATION)
           is_numeric = False

           try:
               ventilation = float(ventilation)
               is_numeric = True
           except ValueError:
               pass

           if is_numeric and not isinstance(ventilation, str):
               return VentilationConstant().generate(obj, data)

           return self.run(**self.get_input_data(obj, data))


- get_input_data

  .. code-block:: python

         def get_input_data(self, obj, data):
           ventilation_key = obj.get(O.VENTILATION)
           ventilation_ts = data.get(ventilation_key)
           input_data = {
               O.ID: obj.get(O.ID, None),
               O.AREA: obj.get(O.AREA, Const.DEFAULT_AREA.value),
               O.HEIGHT: obj.get(O.HEIGHT, Const.DEFAULT_HEIGHT.value),
               O.VENTILATION_COL: obj.get(O.VENTILATION_COL, None),
               O.VENTILATION: ventilation_ts,
               O.VENTILATION_FACTOR: obj.get(O.VENTILATION_FACTOR, DEFAULT_VENTILATION_FACTOR),
           }
           return input_data


- run

  .. code-block:: python

         def run(self, **kwargs):
           object_id = kwargs[O.ID]
           area = kwargs[O.AREA]
           height = kwargs[O.HEIGHT]
           ventilation = kwargs[O.VENTILATION]
           col = kwargs[O.VENTILATION_COL]
           col = col if isinstance(col, str) else str(object_id)
           unit = col.split("[")[-1].split("]")[0]
           try:
               ts = ventilation.loc[:, col]
           except KeyError as err:
               raise Warning(
                   f"Neither explicit (column name) or implicit (column id) are specified." f"Given input column: {col}"
               ) from err
           match unit:
               case "1/h":
                   ventilation = ts * area * height * AIR_DENSITY * HEAT_CAPACITY / 3600
               case "m3/h":
                   ventilation = ts * AIR_DENSITY * HEAT_CAPACITY / 3600
               case "W/K":
                   ventilation = ts
               case _:
                   log.warning('No unit given. "1/h" assumed as unit.')
                   ventilation = ts * area * height * AIR_DENSITY * HEAT_CAPACITY / 3600
           return pd.DataFrame({O.VENTILATION: ventilation}, index=ventilation.index)

