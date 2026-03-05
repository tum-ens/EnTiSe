internaltimeseries
==================

Overview
--------

Represents the processing of internal time series data for further computations.

This class is designed to handle, manipulate, and process internal time series
data provided as input. It validates the data, ensures it conforms to the expected
formats, and executes transformations or auxiliary methods as necessary. It inherits
from the `AuxiliaryMethod` base class and relies heavily on specific keys and internal
structures for its operations.


Key facts
---------

- Method key: ``InternalTimeSeries``


Requirements
------------

Required keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``gains_internal_column``



Optional keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``id``



Required data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``gains_internal[W]``



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
           gains_internal = obj.get(O.GAINS_INTERNAL)
           try:
               gains_internal = float(gains_internal)
           except ValueError:
               pass
           if not isinstance(gains_internal, str):
               return InternalConstant().generate(obj, data)
           return self.run(**self.get_input_data(obj, data))


- get_input_data

  .. code-block:: python

         def get_input_data(self, obj, data):
           gains_key = obj.get(O.GAINS_INTERNAL)
           gains_ts = data.get(gains_key)
           input_data = {
               O.ID: obj.get(O.ID, None),
               O.GAINS_INTERNAL_COL: obj.get(O.GAINS_INTERNAL_COL, None),
               O.GAINS_INTERNAL: gains_ts,
           }
           return input_data


- run

  .. code-block:: python

         def run(self, **kwargs):
           object_id = kwargs[O.ID]
           col = kwargs[O.GAINS_INTERNAL_COL]
           internal_gains = kwargs[O.GAINS_INTERNAL]
           col = col if isinstance(col, str) else str(object_id)
           try:
               internal_gains = internal_gains.loc[:, col]
           except KeyError:
               log.error('Internal gains column "%s" does not exist', col)
               raise Warning(
                   f"Neither explicit (column name) or implicit (column id) are specified." f"Given input column: {col}"
               )
           return pd.DataFrame({O.GAINS_INTERNAL: internal_gains}, index=internal_gains.index)

