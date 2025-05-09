.. _auxiliary_internal:

Internal Gains
=============

Description
----------

Internal gains auxiliary methods calculate the heat gains from internal sources such as occupants, appliances, and lighting. These gains are used by HVAC methods to calculate the heating and cooling demand.

Available Strategies
------------------

EnTiSe provides several strategies for calculating internal gains:

InternalConstant
~~~~~~~~~~~~~~~

A simple strategy that uses a constant value for internal gains.

**Requirements**:

- ``gains_internal``: Constant internal gains value in Watts.

**Example**:

.. code-block:: python

    object_row = {
        "id": "building1",
        "gains_internal": 500.0,  # 500 W constant internal gains
    }

InternalTimeSeries
~~~~~~~~~~~~~~~~

A strategy that uses a time series for internal gains.

**Requirements**:

- ``gains_internal_ts``: Name of the time series containing internal gains data.
- Time series with a column named ``internal_gains`` or ``<ts_type>_internal_gains``.

**Example**:

.. code-block:: python

    object_row = {
        "id": "building1",
        "gains_internal_ts": "internal_gains",
    }
    
    # Time series data
    internal_gains_df = pd.DataFrame({
        "datetime": pd.date_range(start="2025-01-01", periods=24, freq="h"),
        "internal_gains": [300.0, 350.0, 400.0, ...],  # W
    })
    
    data = {
        "internal_gains": internal_gains_df
    }

InternalOccupancy
~~~~~~~~~~~~~~~

A strategy that calculates internal gains based on occupancy and per-person gains.

**Requirements**:

- ``gains_internal_per_person``: Internal gains per person in Watts.
- Occupancy time series with a column named ``occupancy`` or ``<ts_type>_occupancy``.

**Example**:

.. code-block:: python

    object_row = {
        "id": "building1",
        "gains_internal_per_person": 100.0,  # 100 W per person
    }
    
    # Occupancy time series will be automatically used if available