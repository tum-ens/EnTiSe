.. _getting_started:

Getting Started
===============

Welcome to EnTiSe, the framework for generating timeseries data for HVAC, Electricity, Mobility, and Occupancy types. This guide will walk you through the basics of setting up and using EnTiSe for the first time.

Prerequisites
-------------

Before getting started, ensure you have:

- Python 3.8 or higher installed.
- EnTiSe installed. If not, refer to the `Installation` guide.

Quickstart Example
------------------

Here’s a simple example to help you get started with EnTiSe quickly.

1. **Import the Required Components**

Start by importing the necessary classes:

.. code-block:: python

    from entise.core.generator import TimeSeriesGenerator
    import pandas as pd

2. **Prepare Input Data**

Define objects (metadata) and timeseries data that EnTiSe will process:

.. code-block:: python

    # Define an object for HVAC
    object_row = {
        "id": "building1",
        "hvac": "1R1C",
        "resistance": 2.0,
        "capacitance": 1e5,
        "temp_init": 20.0,
        "temp_min": 20.0,
        "temp_max": 24.0,
        "power_heating": 3000.0,
        "power_cooling": 3000.0,
    }

    # Mock weather data
    index = pd.date_range(start="2025-01-01", periods=24, freq="h", tz="UTC")
    weather_df = pd.DataFrame({
        "datetime": index,
        "temp_out": [0.0] * 24,
    }, index=index)

    # Combine into a dictionary
    data = {"weather": weather_df}

3. **Create and Run the Generator**

Use the `TimeSeriesGenerator` to process your objects and data:

.. code-block:: python

    # Initialize the generator
    gen = TimeSeriesGenerator()

    # Add the object(s)
    gen.add_objects(object_row)

    # Generate timeseries
    summary, df = gen.generate(data, workers=1)

4. **View the Results**

The `summary` will contain summary metrics, and `timeseries` will hold the generated data:

.. code-block:: python

    print("Summary Metrics:")
    print(summary)

    print("\nTimeseries Data:")
    print(df)

### Expected Output:

.. code-block:: text

    Summary Metrics:
                                 hvac_energy_demand  hvac_power_demand
    building1                                  72.0               3000

    Timeseries Data:
                             hvac_energy_demand  hvac_power_demand  hvac_temperature
    2025-01-01 00:00:00+00:00                3.0             3000.0              20.0
    2025-01-01 01:00:00+00:00                3.0             3000.0              20.0
    2025-01-01 02:00:00+00:00                3.0             3000.0              20.0
    ...                                       ...               ...               ...

Understanding the Workflow
--------------------------

Here’s a breakdown of how EnTiSe processes your data:

1. **Objects**: Define the metadata for the timeseries you want to generate.
2. **Input Data**: Provide the required timeseries data (e.g., weather data, occupancy data).
3. **Method Selection**: EnTiSe selects the appropriate method based on the object properties.
4. **Strategy Selection**: For auxiliary calculations (like solar gains or internal gains), EnTiSe selects the most appropriate strategy based on the available data.
5. **Pipeline Processing**: Methods are executed in a pipeline, with dependencies automatically resolved.
6. **Outputs**: EnTiSe generates summary metrics and detailed timeseries for each object.

Available Methods
-----------------

For a list of available methods for HVAC, Electricity, Mobility, and Occupancy, refer to the :ref:`methods` section in this documentation.

Next Steps
----------

- Explore the :ref:`workflow` section to understand the full lifecycle of timeseries generation.
- Check out the :ref:`examples` for practical applications.
