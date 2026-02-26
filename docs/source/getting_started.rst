.. _getting_started:

Getting Started
===============

This guide will walk you through the basics of setting up and using EnTiSe for the first time.

Prerequisites
-------------

Before getting started, ensure you have:

- Python 3.10 to 3.13 installed.
- EnTiSe installed. If not, refer to the :ref:`installation` guide.

Quickstart Example
------------------

EnTiSe was designed to allow users to generate time series quickly for multiple objects.
Therefore, the tool focuses on batch processing so that users do not have to deal with aspects such as
multiprocessing but can focus on defining the time series they need. However, the tool also allows to use
the methods directly if they need more control. Below you find an example for both ways to get you started quickly.

Batch Processing
~~~~~~~~~~~~~~~~

In batch processing you define all the parameters for all time series types beforehand and then create them all at once.
This is a simple example for one object and one time series type but you can extend it to thousands and millions of time series.
In the :ref:`examples`, we defined several objects in a csv file and then load them into the generator. The same applies to the input data.

1. **Import the Required Components**

Start by importing the necessary classes:

.. code-block:: python

    from entise import Generator
    import pandas as pd

2. **Prepare Input Data**

Define objects (metadata) and data that EnTiSe will process. Objects describe the characteristics of the time series you want to generate, while data provides the necessary input for the generation process, i.e. all data that does not fit into a dataframe but is required for the generation process. In this example, we define a simple object for an HVAC system and mock weather data.

.. code-block:: python

    # Define an object for HVAC
    object_row = {
        "id": "building1",  # Unique identifier for the object which always needs to be specified
        "hvac": "1R1C",  # This specifies the method to use for generating the HVAC time series
        "resistance": 2.0,  # This is the thermal resistance in degrees Celsius per watt
        "capacitance": 1e5,  # This is the thermal capacitance in joules per degree Celsius
        "temp_init": 20.0,  # This is the initial indoor temperature in degrees Celsius
        "temp_min": 20.0,  # This is the minimum allowed indoor temperature in degrees Celsius
        "temp_max": 24.0,  # This is the maximum allowed indoor temperature in degrees Celsius
        "power_heating": 3000.0,  # This is the maximum heating power in watts
        "power_cooling": 3000.0,  # This is the maximum cooling power in watts
        "weather": "weather",  # This references the key in the data dictionary that contains the weather DataFrame
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

Use the `Generator` to process your objects and data:

.. code-block:: python

    # Initialize the generator
    gen = Generator()

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


Direct Method Access
~~~~~~~~~~~~~~~~~~~~

You can also call a method directly when you want fine-grained control or to embed EnTiSe in an existing pipeline. In this case you can either provide the parameters directly or use the same objects and data as in the batch processing example. Below is an example for the PVLib method using the direct method access.

.. code-block:: python

    from entise.methods.pv import PVLib

    pv = PVLib()
    result = pv.generate(
        latitude=48.1,  # Latitude of the location
        longitude=11.6,  # Longitude of the location
        power=5000,  # Maximum power output of the solar system in watts
        weather=weather_df,  # DataFrame with solar radiation data
    )

    summary = result["summary"]
    timeseries = result["timeseries"]


Understanding the Workflow
--------------------------

- Define objects and provide input data (e.g., weather).
- EnTiSe selects methods and strategies based on your inputs.
- A dependency resolver ensures correct execution order.
- You get a summary and a detailed timeseries DataFrame.

See :ref:`workflow` for the full conceptual guide, decision charts, and troubleshooting.

Available Methods
-----------------

For a list of available methods, refer to the :ref:`methods` section in this documentation.
