internaltimeseries
=========================


**Method Key:** ``internaltimeseries``

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

   * - ``gains_internal_column``
     - ``str``




Required Timeseries
~~~~~~~~~~~~~~~~~~~



**Timeseries Key:** ``gains_internal``












Dependencies
-------------


- None


Methods
-------


**generate**:


  .. code-block:: python

         def generate(self, obj, data):
           gains_internal = obj.get(O.GAINS_INTERNAL)
           try:
               gains_internal = float(gains_internal)
           except ValueError:
               pass
           finally:
               if isinstance(gains_internal, O.DTYPES[O.GAINS_INTERNAL]) and not isinstance(gains_internal, str):
                   return InternalConstant().generate(obj, data)
           return self.run(**self.get_input_data(obj, data))



**get_input_data**:


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



**run**:


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
               raise Warning(f'Neither explicit (column name) or implicit (column id) are specified.'
                             f'Given input column: {col}')
           return pd.DataFrame({O.GAINS_INTERNAL: internal_gains}, index=internal_gains.index)
