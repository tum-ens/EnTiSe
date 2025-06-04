.. _workflow:

Workflow
========

This section provides an overview of the EnTiSe workflow, detailing the lifecycle of generating timeseries data from object definitions and input data. Follow these steps to seamlessly integrate EnTiSe into your projects.

Overview
--------

EnTiSe processes data in the following stages:

1. **Object Definition**: Define the metadata for the timeseries generation.
2. **Input Data Preparation**: Provide the required timeseries input data.
3. **Dependency Resolution**: Ensure all methods are executed in the correct order.
4. **Timeseries Generation**: Generate timeseries data and compute summary metrics.
5. **Output Collection**: Collect and export results for further analysis.

Detailed Steps
--------------

1. Define Objects
~~~~~~~~~~~~~~~~~

Objects represent the entities for which timeseries data will be generated. Each object should have:
- A unique identifier (``Objects.ID``).
- References to the methods and input data required for processing.

Example:

.. code-block:: python

    objects = [
        {
            Objects.ID: "building1",
            Types.HVAC: "degree_day",
            Objects.WEATHER: "weather",
        }
    ]

2. Prepare Input Data
~~~~~~~~~~~~~~~~~~~~~

Input data provides the raw timeseries required for processing. Ensure your data includes:
- All required columns specified by the methods.
- Compatible formats (e.g., ``pandas.DataFrame``).

Example:

.. code-block:: python

    import pandas as pd

    weather_data = pd.DataFrame({
        Keys.DATETIME: ["2023-01-01 11:00:00", "2023-01-01 11:30:00", "2023-01-01 12:00:00"],
        Keys.TEMP_OUT: [15, 10, 5],
    })

    data = {"weather": weather_data}

3. Add Objects and Input Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``TimeSeriesGenerator`` to add your objects and input data:

.. code-block:: python

    from entise.core.generator import TimeSeriesGenerator

    # Initialize the generator
    generator = TimeSeriesGenerator()

    # Add objects
    generator.add_objects(objects)

4. Dependency Resolution
~~~~~~~~~~~~~~~~~~~~~~~~

EnTiSe ensures that methods with dependencies are executed in the correct order. This is handled automatically by the ``DependencyResolver``:

- **Example**: If ``Method B`` depends on the output of ``Method A``, EnTiSe ensures ``Method A`` runs first.

5. Generate Timeseries
~~~~~~~~~~~~~~~~~~~~~~

Generate timeseries data with the specified number of worker processes:

.. code-block:: python

    # With multiple workers (parallel processing)
    summary, df = generator.generate(data, workers=4)

    # With a single worker (sequential processing, easier for debugging)
    summary, df = generator.generate(data, workers=1)

6. Collect and Export Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once processing is complete, you can access and export your results:

- **Access Summary Metrics**:

.. code-block:: python

    print(summary)

- **Export Timeseries**:

.. code-block:: python

    # Export to CSV
    df.to_csv("output/timeseries.csv")

    # Export to Excel
    df.to_excel("output/timeseries.xlsx")

    # Export to Feather (fast binary format)
    df.to_feather("output/timeseries.feather")

Alternative Workflow: Direct Method Access
------------------------------------------

In addition to the batch processing workflow with TimeSeriesGenerator, EnTiSe also supports a direct method access workflow:

1. **Import Method**:

   .. code-block:: python

      from entise.methods.pv import PVLib
      from entise.methods.hvac import R1C1

2. **Create Method Instance**:

   .. code-block:: python

      pvlib = PVLib()
      rc_model = R1C1()

3. **Define Parameters and Data**:

   .. code-block:: python

      obj = {
          "id": "system_1",
          "latitude": 48.1,
          "longitude": 11.6,
      }

      data = {
          "weather": weather_df,
      }

4. **Generate Timeseries**:

   .. code-block:: python

      # Using dictionaries
      result = pvlib.generate(obj, data)

      # Using named parameters directly
      result = pvlib.generate(
          latitude=48.1,
          longitude=11.6,
          weather=weather_df
      )

      # Combining both approaches
      result = pvlib.generate(
          obj=obj,  # Use values from obj dictionary
          weather=weather_df,  # Provide weather data directly
          power=6000  # Override any "power" value in obj
      )

      summary = result["summary"]
      timeseries = result["timeseries"]

This approach gives you more direct control over the generation process and is useful for:

- Working with individual methods
- Integrating EnTiSe methods into custom workflows
- Debugging and testing specific methods

Visual Overview
---------------

The diagram below illustrates the EnTiSe workflow:

.. mermaid::

   graph TD
       A[Define Objects] --> B[Prepare Input Data]
       B --> C[Add to Generator]
       C --> D[Method Selection]
       D --> E[Strategy Selection]
       E --> F[Pipeline Processing]
       F --> G[Generate Timeseries]
       G --> H[Collect and Export Results]

Best Practices
--------------

- **Data Validation**: Ensure your objects and data conform to the required schema for the methods you're using.
- **Strategy Selection**: Understand the different strategies available for auxiliary calculations and their requirements.
- **Pipeline Design**: Consider the dependencies between methods when designing your pipeline.
- **Method Reuse**: Organize and reuse methods across timeseries types to simplify your workflow.
- **Export Options**: Take advantage of multiple export formats (``CSV``, ``Excel``, ``Feather``) to integrate with your tools.

Next Steps
----------

- Explore the :ref:`methods` section to understand available timeseries generation techniques.
- Check out the :ref:`examples` for practical applications.
