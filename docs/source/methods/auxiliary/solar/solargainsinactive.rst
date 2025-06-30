solargainsinactive
=========================


**Method Key:** ``solargainsinactive``

.. note::
   This is the class name. For auxiliary methods, the key is determined by the selector class.


Description
-----------

Processes weather data to compute a DataFrame with solar gains.

This function calculates a DataFrame containing a single column of zeros that
represents the solar gains. The DataFrame uses the same index as the provided
weather data. This is primarily used in scenarios where solar gains need to
be initialized or simulated with a default value.

Args:
    weather: A pandas DataFrame representing weather data. The index should
        be a time-based index, and the length of the DataFrame determines the
        number of rows in the resultant DataFrame.

Returns:
    pandas.DataFrame: A DataFrame with a single column named 'O.GAINS_SOLAR',
    filled with zeros of type `np.float32`. The index corresponds to the input
    weather DataFrame's index.

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
           """
           Processes input data and extracts specified information for further usage.

           Args:
               obj: A reference to an object that may hold contextual information or be used
                   in processing, nature of its usage to be defined by implementation.
               data: Dictionary or mapping that contains information from which specific
                   values, such as weather data, are extracted.

           Returns:
               Dictionary containing the extracted weather data under a predefined key.
           """
           return {O.WEATHER: data[O.WEATHER]}



**run**:


  .. code-block:: python

         def run(self, weather):
           """
           Processes weather data to compute a DataFrame with solar gains.

           This function calculates a DataFrame containing a single column of zeros that
           represents the solar gains. The DataFrame uses the same index as the provided
           weather data. This is primarily used in scenarios where solar gains need to
           be initialized or simulated with a default value.

           Args:
               weather: A pandas DataFrame representing weather data. The index should
                   be a time-based index, and the length of the DataFrame determines the
                   number of rows in the resultant DataFrame.

           Returns:
               pandas.DataFrame: A DataFrame with a single column named 'O.GAINS_SOLAR',
               filled with zeros of type `np.float32`. The index corresponds to the input
               weather DataFrame's index.
           """
           return pd.DataFrame({O.GAINS_SOLAR: np.zeros(len(weather), dtype=np.float32)}, index=weather.index)
