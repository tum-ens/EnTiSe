InternalOccupancy
=========================


**Method Key:** ``InternalGainsOccupancy``

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
   
   * - ``inhabitants``
     - ``str``
   
   * - ``gains_internal_per_person``
     - ``str``
   



Required Timeseries
~~~~~~~~~~~~~~~~~~~



**Timeseries Key:** ``occupancy``












Dependencies
-------------


- None


Methods
-------


**generate**:


  .. code-block:: python

         def generate(self, obj, data, ts_type=None):
           num_people = obj.get(O.INHABITANTS)
           gain_per_person = obj[O.GAINS_INTERNAL_PER_PERSON]
           occ = data[Types.OCCUPANCY]

           internal_gains = occ * gain_per_person * num_people

           return pd.DataFrame({O.GAINS_INTERNAL: internal_gains}, index=occ.index)


